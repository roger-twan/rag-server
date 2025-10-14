import uuid

from qdrant_client.models import PointStruct

from app.db.qdrant_client import qdrant_client
from app.services.chunker import chunk_text
from app.services.embedding import embed_content


async def ingest_document(
    content: str,
    metadata: dict,
) -> list[str]:
    chunks = chunk_text(content)
    embeddings = await embed_content(chunks)

    points = []
    point_ids = []

    for idx, chunk in enumerate(chunks):
        embedding = embeddings[idx]

        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        point = PointStruct(
            id=point_id,
            vector=embedding.values,
            payload={
                "text": chunk,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **metadata,
            },
        )
        points.append(point)

    qdrant_client.upsert(collection_name="documents", points=points)

    return point_ids


async def ingest_documents_batch(
    documents: list[dict],
) -> dict:
    total_points = 0
    total_documents = len(documents)

    for doc in documents:
        point_ids = await ingest_document(
            content=doc["content"],
            metadata=doc["metadata"],
        )
        total_points += len(point_ids)

    return {
        "total_documents": total_documents,
        "total_chunks": total_points,
        "status": "success",
    }
