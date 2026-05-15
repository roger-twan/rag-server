from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.db import postgres
from app.services.retriever import retrieve

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


# Create prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer strictly from the provided context. If the context is insufficient, say you don't have enough information. Be concise and natural.",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)


rewrite_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question into a standalone search query. Use chat history only to resolve references. Return only the rewritten query.",
        ),
        (
            "human",
            """Chat history:
{history}

Latest question: {question}

Standalone search query:""",
        ),
    ]
)


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
        }
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


async def generate_answer(query: str, conversation_id: str | None = None) -> dict:
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
        answer = "I don't have enough information to answer this question."
        postgres.add_message(conversation_id=conversation_id, role="assistant", content=answer)
        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "rewritten_query": rewritten_query,
            "sources": [],
        }

    # Format context
    context = _format_context(chunks)

    # Create chain and generate answer
    chain = prompt_template | _get_llm()
    response = await chain.ainvoke(
        {
            "context": context,
            "question": query,
        }
    )

    answer = response.content
    postgres.add_message(conversation_id=conversation_id, role="assistant", content=answer)
    postgres.add_retrieval_traces(
        message_id=user_message_id,
        query=rewritten_query,
        chunks=chunks,
    )

    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "rewritten_query": rewritten_query,
        "sources": _format_sources(chunks),
    }
