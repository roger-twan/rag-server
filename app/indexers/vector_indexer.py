import hashlib
import uuid
from datetime import datetime
from typing import Any

from llama_index.core import Document

from app.db.pinecone import NAMESPACES, encode_sparse, get_pinecone_index, upsert_vectors
from app.services.chunker import chunk_text
from app.services.embedding import embed_content


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


async def prepare_vectors(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Convert LlamaIndex documents to Pinecone vectors.
    Chunks documents, creates embeddings (dense + sparse), returns vector dicts.
    """
    vectors = []

    for doc in documents:
        # Get text content
        text = doc.text
        metadata = doc.metadata or {}

        # Chunk the text
        chunks = chunk_text(text, chunk_size, chunk_overlap)

        # Create embeddings for chunks (async call)
        chunk_embeddings = await embed_content(chunks)

        # Create sparse vectors for chunks
        sparse_vectors = encode_sparse(chunks)

        for idx, chunk in enumerate(chunks):
            embedding = chunk_embeddings[idx]
            sparse_vector = sparse_vectors[idx] if idx < len(sparse_vectors) else None

            # Create unique ID
            point_id = str(uuid.uuid4())

            # Compute content hash
            content_hash = compute_content_hash(chunk)

            # Build vector dict
            vector = {
                "id": point_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "content_hash": content_hash,
                    "last_updated": datetime.utcnow().isoformat(),
                    **metadata,
                },
            }

            # Add sparse vector if available
            if sparse_vector:
                vector["sparse_values"] = sparse_vector

            vectors.append(vector)

    return vectors


async def index_documents(
    documents: list[Document],
    namespace: str,
    clear_existing: bool = False,
) -> dict[str, Any]:
    """
    Index LlamaIndex documents to Pinecone namespace.

    Args:
        documents: List of LlamaIndex Document objects
        namespace: Pinecone namespace to store vectors
        clear_existing: If True, clear namespace before indexing

    Returns:
        Dict with indexing statistics
    """
    index = get_pinecone_index()

    # Clear existing vectors if requested
    if clear_existing:
        index.delete(delete_all=True, namespace=namespace)

    # Prepare vectors (async)
    vectors = await prepare_vectors(documents)

    if not vectors:
        return {
            "namespace": namespace,
            "documents_indexed": 0,
            "total_chunks": 0,
            "status": "no_data",
        }

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        upsert_vectors(batch, namespace)

    return {
        "namespace": namespace,
        "documents_indexed": len(documents),
        "total_chunks": len(vectors),
        "status": "success",
    }


def get_namespace_for_source(source: str, repo_name: str | None = None) -> str:
    """Get the appropriate Pinecone namespace for a data source."""
    if source == "github_notes":
        return NAMESPACES["github_notes"]
    elif source == "github_repos":
        return NAMESPACES["github_repos"]
    elif source == "website_rogerink":
        return NAMESPACES["website_rogerink"]
    else:
        return source  # Use source directly as namespace
