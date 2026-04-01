from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.retriever import retrieve

# Initialize Gemini model via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.7,
)

# Create prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question based on the provided context.",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Provide a clear and concise answer based on the context above.""",
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
    chain = prompt_template | llm
    response = await chain.ainvoke(
        {
            "context": context,
            "question": query,
        }
    )

    return response.content
