import logging

from langchain_cohere import CohereRerank
from pinecone.exceptions import PineconeException

from app.core.config import settings
from app.db import postgres
from app.db.pinecone import NAMESPACES, bm25_encoder, get_pinecone_index
from app.services.embedding import embeddings

logger = logging.getLogger(__name__)

# Initialize Cohere reranker
cohere_reranker = CohereRerank(
    model="rerank-v3.5",
    cohere_api_key=settings.COHERE_API_KEY,
    top_n=5,
)


def _get_all_namespaces() -> list[str]:
    """Get list of all namespaces to search across."""
    return list(NAMESPACES.values())


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _candidate_to_chunk(candidate: dict, rerank_score=None) -> dict:
    metadata = candidate["metadata"]
    doc_id = metadata.get("doc_id", "")
    chunk_index = int(metadata.get("chunk_index", 0))
    neighbors = postgres.get_neighbor_chunks(doc_id, chunk_index, window=1) if doc_id else []
    text = "\n\n".join(chunk.text for chunk in neighbors) if neighbors else candidate["text"]

    return {
        "chunk_id": candidate["chunk_id"],
        "text": text,
        "score": candidate["score"],
        "rerank_score": rerank_score,
        "metadata": metadata,
    }


async def retrieve(
    query: str,
    top_k: int = 10,
    rerank_top_n: int = 5,
) -> list[dict]:
    """
    Hybrid retrieval: dense + sparse search across all namespaces with Cohere reranking.

    Args:
        query: Search query
        top_k: Number of initial results to retrieve per namespace
        rerank_top_n: Number of results after reranking

    Returns:
        List of structured chunks from top reranked documents
    """
    # Generate dense embedding
    dense_vector = await embeddings.aembed_query(query)

    sparse_vector = None
    if settings.ENABLE_SPARSE_SEARCH:
        # Generate sparse vector (BM25) - skip if encoder not fitted
        try:
            sparse_vector = bm25_encoder.encode_queries(query)
        except ValueError:
            sparse_vector = None

    # Search across all namespaces
    all_results = []
    index = get_pinecone_index()

    for namespace in _get_all_namespaces():
        # Hybrid query
        try:
            results = index.query(
                vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
        except PineconeException:
            results = index.query(
                vector=dense_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )

        for match in results.matches:
            metadata = match.metadata or {}
            all_results.append(
                {
                    "chunk_id": metadata.get("chunk_id") or match.id,
                    "text_preview": metadata.get("text_preview") or metadata.get("text", ""),
                    "score": match.score,
                    "metadata": metadata,
                }
            )

    # Sort by initial score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Take top results for reranking
    candidates = all_results[: min(top_k * 2, len(all_results))]

    if not candidates:
        return []

    db_chunks = postgres.get_chunks_by_ids([r["chunk_id"] for r in candidates])
    for candidate in candidates:
        chunk = db_chunks.get(candidate["chunk_id"])
        candidate["text"] = chunk.text if chunk is not None else candidate["text_preview"]

    # Rerank with Cohere
    docs = [r["text"] for r in candidates]
    try:
        reranked = cohere_reranker.rerank(query=query, documents=docs)
    except Exception as exc:
        logger.warning(
            "Cohere rerank failed; falling back to vector search order: %s: %s",
            type(exc).__name__,
            exc,
        )
        return [_candidate_to_chunk(candidate) for candidate in candidates[:rerank_top_n]]

    reranked_chunks = []
    seen_chunk_ids = set()
    for r in reranked:
        index = _get_attr(r, "index")
        relevance_score = _get_attr(r, "relevance_score", 0)
        if relevance_score <= 0.1 or index is None:
            continue

        candidate = candidates[index]
        chunk_id = candidate["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        reranked_chunks.append(_candidate_to_chunk(candidate, rerank_score=relevance_score))

        if len(reranked_chunks) >= rerank_top_n:
            break

    return reranked_chunks
