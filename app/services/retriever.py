from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder

from app.core.config import settings
from app.db.pinecone import NAMESPACES, get_pinecone_index

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY,
)

# Initialize BM25 encoder for sparse vectors
bm25_encoder = BM25Encoder()

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

    # Generate sparse vector (BM25)
    sparse_vector = bm25_encoder.encode_queries(query)

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
    reranked_texts = [docs[r.index] for r in reranked[:rerank_top_n] if r.relevance_score > 0.1]

    return reranked_texts
