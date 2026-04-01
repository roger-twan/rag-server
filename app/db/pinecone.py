from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(host=settings.PINECONE_INDEX_HOST)

# Initialize BM25 encoder for sparse vectors
bm25_encoder = BM25Encoder()

# Namespaces for different data sources
NAMESPACES = {
    "github_notes": "github_notes",
    "github_repos": "github_repos",
    "website_rogerink": "website_rogerink",
}


def get_pinecone_index():
    """Get Pinecone index instance."""
    return index


def clear_namespace(namespace: str):
    """Clear all vectors in a namespace."""
    index.delete(delete_all=True, namespace=namespace)


def upsert_vectors(vectors: list[dict], namespace: str):
    """
    Upsert vectors to Pinecone.
    Each vector dict should have: id, values (dense), sparse_values (optional), metadata
    """
    index.upsert(vectors=vectors, namespace=namespace)


def query_hybrid(
    dense_vector: list[float],
    sparse_vector: dict | None = None,
    namespace: str = "",
    top_k: int = 10,
    alpha: float = 0.5,
    filter_dict: dict | None = None,
):
    """
    Hybrid query combining dense and sparse vectors.
    alpha: weight for dense vector (0.0 = sparse only, 1.0 = dense only)
    """
    # If sparse vector provided, use hybrid search
    if sparse_vector:
        # Pinecone's hybrid search using query_vectors parameter
        result = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=True,
        )
    else:
        # Dense-only search
        result = index.query(
            vector=dense_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=True,
        )
    return result


def encode_sparse(texts: list[str]) -> list[dict]:
    """Encode texts to sparse vectors using BM25."""
    return bm25_encoder.encode_documents(texts)
