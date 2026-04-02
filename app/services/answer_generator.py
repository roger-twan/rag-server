from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings
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
            "Answer questions as yourself, speaking naturally in first person. Use the provided context to inform your answers, but don't mention that you used context or that you're referencing documents. Just speak authentically as a real person would.",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer as yourself:""",
        ),
    ]
)


async def generate_answer(query: str) -> str:
    """
    Generate answer using RAG pipeline:
    1. Retrieve relevant documents
    2. Format context
    3. Generate answer with Gemini

    Args:
        query: User question

    Returns:
        Generated answer text
    """
    # Retrieve relevant documents
    docs = await retrieve(query)

    if not docs:
        return "I don't have enough information to answer this question."

    # Format context
    context = "\n\n---\n\n".join(docs)

    # Create chain and generate answer
    chain = prompt_template | _get_llm()
    response = await chain.ainvoke(
        {
            "context": context,
            "question": query,
        }
    )

    return response.content
