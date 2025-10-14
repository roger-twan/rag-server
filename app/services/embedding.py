from google import genai
from google.genai import types

from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)


async def embed_content(content: str | list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    return response.embeddings
