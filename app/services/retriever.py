from langchain_cohere import CohereRerank

from app.core.config import settings
from app.db.pinecone import NAMESPACES, bm25_encoder, get_pinecone_index
from app.services.embedding import embeddings

# Initialize Cohere reranker
cohere_reranker = CohereRerank(
    model="rerank-v3.5",
    cohere_api_key=settings.COHERE_API_KEY,
    top_n=5,
)


def _get_all_namespaces() -> list[str]:
    """Get list of all namespaces to search across."""
    return list(NAMESPACES.values())


async def retrieve(
    query: str,
    top_k: int = 10,
    rerank_top_n: int = 5,
) -> list[str]:
    """
    Hybrid retrieval: dense + sparse search across all namespaces with Cohere reranking.

    Args:
        query: Search query
        top_k: Number of initial results to retrieve per namespace
        rerank_top_n: Number of results after reranking

    Returns:
        List of text content from top reranked documents
    """
    # Generate dense embedding
    dense_vector = await embeddings.aembed_query(query)

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
        results = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

        for match in results.matches:
            all_results.append(
                {
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "metadata": match.metadata,
                }
            )

    # Sort by initial score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Take top results for reranking
    candidates = all_results[: min(top_k * 2, len(all_results))]

    if not candidates:
        return []

    # Rerank with Cohere
    docs = [r["text"] for r in candidates]
    reranked = cohere_reranker.rerank(query=query, documents=docs)

    # Return top reranked documents
    def _get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    reranked_texts = [
        docs[_get_attr(r, "index")]
        for r in reranked[:rerank_top_n]
        if _get_attr(r, "relevance_score", 0) > 0.1
    ]

    return reranked_texts
