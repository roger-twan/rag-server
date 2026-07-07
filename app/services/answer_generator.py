from collections.abc import AsyncIterator

from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.langsmith import configure_langsmith_tracing
from app.db import postgres
from app.services.prompts import (
    fallback_prompt_template,
    prompt_template,
    rewrite_prompt_template,
)
from app.services.retriever import retrieve

configure_langsmith_tracing()

# Lazy initialization - only create LLM when needed
_llm = None


def _get_llm():
    """Get LLM instance based on configured provider."""
    global _llm
    if _llm is not None:
        return _llm

    provider = settings.LLM_PROVIDER.lower()
    temperature = 0.7

    if provider == "google":
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )
    elif provider == "openai":
        _llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )
    elif provider == "deepseek":
        _llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")

    return _llm


def _trace_config(
    run_name: str,
    *,
    conversation_id: str | None = None,
    prompt_name: str | None = None,
    chunk_count: int | None = None,
    source_count: int | None = None,
    has_context: bool | None = None,
) -> dict:
    tags = [
        "rag-server",
        f"env:{settings.ENVIRONMENT}",
        f"provider:{settings.LLM_PROVIDER.lower()}",
    ]
    if prompt_name:
        tags.append(f"prompt:{prompt_name}")

    metadata = {
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER.lower(),
    }
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id
    if prompt_name is not None:
        metadata["prompt_name"] = prompt_name
    if chunk_count is not None:
        metadata["chunk_count"] = chunk_count
    if source_count is not None:
        metadata["source_count"] = source_count
    if has_context is not None:
        metadata["has_context"] = has_context

    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
    }


def _format_history(messages: list) -> str:
    if not messages:
        return ""
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


async def _rewrite_query(query: str, conversation_id: str) -> str:
    history = postgres.get_recent_messages(conversation_id)
    if not history:
        return query

    chain = rewrite_prompt_template | _get_llm()
    response = await chain.ainvoke(
        {
            "history": _format_history(history),
            "question": query,
        },
        config=_trace_config(
            "rag_query_rewrite",
            conversation_id=conversation_id,
            prompt_name="rewrite",
        ),
    )
    rewritten = response.content.strip()
    return rewritten or query


def _format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "")
        title = metadata.get("title", "")
        path = metadata.get("path") or metadata.get("url") or metadata.get("repo") or ""
        label = " | ".join(part for part in [source, title, path] if part)
        header = f"[{index}] {label}" if label else f"[{index}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def _format_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    seen = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        key = (
            metadata.get("source", ""),
            metadata.get("path") or metadata.get("url") or metadata.get("repo") or "",
            metadata.get("title", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": key[0],
                "path": key[1],
                "title": key[2],
                "chunk_id": chunk["chunk_id"],
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
    return sources


def _chunk_content(chunk) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content) if content is not None else ""


async def generate_answer(
    query: str,
    conversation_id: str | None = None,
    include_contexts: bool = False,
) -> dict:
    """
    Generate answer using RAG pipeline:
    1. Retrieve relevant documents
    2. Format context
    3. Generate answer with Gemini

    Args:
        query: User question

    Returns:
        Generated answer payload
    """
    conversation_id = postgres.ensure_conversation(conversation_id)
    rewritten_query = await _rewrite_query(query, conversation_id)
    user_message_id = postgres.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
        rewritten_query=rewritten_query,
    )

    # Retrieve relevant documents
    chunks = await retrieve(rewritten_query)

    if not chunks:
        fallback_chain = fallback_prompt_template | _get_llm()
        fallback_response = await fallback_chain.ainvoke(
            {"question": query},
            config=_trace_config(
                "rag_answer_fallback",
                conversation_id=conversation_id,
                prompt_name="fallback",
                chunk_count=0,
                source_count=0,
                has_context=False,
            ),
        )
        answer = fallback_response.content
        postgres.add_message(conversation_id=conversation_id, role="assistant", content=answer)
        payload = {
            "answer": answer,
            "conversation_id": conversation_id,
            "rewritten_query": rewritten_query,
            "sources": [],
        }
        if include_contexts:
            payload["retrieved_contexts"] = []
        return payload

    # Format context
    context = _format_context(chunks)

    # Create chain and generate answer
    chain = prompt_template | _get_llm()
    response = await chain.ainvoke(
        {
            "context": context,
            "question": query,
        },
        config=_trace_config(
            "rag_answer",
            conversation_id=conversation_id,
            prompt_name="answer",
            chunk_count=len(chunks),
            source_count=len(_format_sources(chunks)),
            has_context=True,
        ),
    )

    answer = response.content
    postgres.add_message(conversation_id=conversation_id, role="assistant", content=answer)
    postgres.add_retrieval_traces(
        message_id=user_message_id,
        query=rewritten_query,
        chunks=chunks,
    )

    payload = {
        "answer": answer,
        "conversation_id": conversation_id,
        "rewritten_query": rewritten_query,
        "sources": _format_sources(chunks),
    }
    if include_contexts:
        payload["retrieved_contexts"] = [chunk["text"] for chunk in chunks]
    return payload


async def generate_answer_stream(
    query: str, conversation_id: str | None = None
) -> AsyncIterator[dict]:
    conversation_id = postgres.ensure_conversation(conversation_id)
    rewritten_query = await _rewrite_query(query, conversation_id)
    user_message_id = postgres.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
        rewritten_query=rewritten_query,
    )

    chunks = await retrieve(rewritten_query)
    sources = _format_sources(chunks)
    yield {
        "event": "metadata",
        "conversation_id": conversation_id,
        "rewritten_query": rewritten_query,
        "sources": sources,
    }

    if chunks:
        chain = prompt_template | _get_llm()
        stream_input = {
            "context": _format_context(chunks),
            "question": query,
        }
        stream_config = _trace_config(
            "rag_answer_stream",
            conversation_id=conversation_id,
            prompt_name="answer",
            chunk_count=len(chunks),
            source_count=len(sources),
            has_context=True,
        )
    else:
        chain = fallback_prompt_template | _get_llm()
        stream_input = {"question": query}
        stream_config = _trace_config(
            "rag_answer_stream_fallback",
            conversation_id=conversation_id,
            prompt_name="fallback",
            chunk_count=0,
            source_count=0,
            has_context=False,
        )

    answer_parts = []
    async for chunk in chain.astream(stream_input, config=stream_config):
        token = _chunk_content(chunk)
        if not token:
            continue
        answer_parts.append(token)
        yield {"event": "token", "content": token}

    answer = "".join(answer_parts)
    postgres.add_message(conversation_id=conversation_id, role="assistant", content=answer)
    if chunks:
        postgres.add_retrieval_traces(
            message_id=user_message_id,
            query=rewritten_query,
            chunks=chunks,
        )

    yield {"event": "done", "answer": answer}
