from app.db.qdrant_client import qdrant_client
from app.services.embedding import embed_content


async def retrieve(query: str) -> list[str]:
    embedding = (await embed_content(query))[0]
    search_result = qdrant_client.search(
        collection_name="documents",
        query_vector=embedding.values,
        limit=5,
    )
    return [hit.payload["text"] for hit in search_result]
