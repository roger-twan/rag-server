from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

# Initialize OpenAI embeddings with text-embedding-3-small (1536 dimensions)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY,
)


async def embed_content(content: str | list[str]) -> list[list[float]]:
    """
    Embed content using OpenAI text-embedding-3-small.
    Returns list of embedding vectors.
    """
    if isinstance(content, str):
        content = [content]

    # LangChain's aembed_documents returns list of embeddings
    result = await embeddings.aembed_documents(content)
    return result
