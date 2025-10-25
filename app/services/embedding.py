from google.genai import types

from app.utils.gemini_client import gemini_client


async def embed_content(content: str | list[str]) -> list[list[float]]:
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    return response.embeddings
