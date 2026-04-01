import hashlib
from datetime import datetime
from typing import Any

from llama_index.core import Document

from app.db.pinecone import NAMESPACES, encode_sparse, get_pinecone_index, upsert_vectors
from app.services.chunker import chunk_text
from app.services.embedding import embed_content


def compute_doc_id(source: str, path: str) -> str:
    """
    Generate deterministic doc_id from source and path.
    Used to track all chunks belonging to a single document.
    """
    return hashlib.sha256(f"{source}:{path}".encode()).hexdigest()[:16]


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


async def prepare_vectors(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Convert LlamaIndex documents to Pinecone vectors with doc_id tracking.
    Chunks documents, creates embeddings (dense + sparse), returns vector dicts.
    Each chunk gets a deterministic ID based on doc_id + chunk_index.
    """
    vectors = []

    for doc in documents:
        # Get text content
        text = doc.text
        metadata = doc.metadata or {}

        # Generate doc_id from source and path (or repo)
        source = metadata.get("source", "unknown")
        path = metadata.get("path") or metadata.get("repo") or metadata.get("url", "unknown")
        doc_id = compute_doc_id(source, path)
        doc_hash = compute_content_hash(text)

        # Chunk the text
        chunks = chunk_text(text, chunk_size, chunk_overlap)

        # Create embeddings for chunks (async call)
        chunk_embeddings = await embed_content(chunks)

        # Create sparse vectors for chunks
        sparse_vectors = encode_sparse(chunks)

        for idx, chunk in enumerate(chunks):
            embedding = chunk_embeddings[idx]
            sparse_vector = sparse_vectors[idx] if idx < len(sparse_vectors) else None

            # Deterministic ID: doc_id + chunk_index
            point_id = f"{doc_id}_{idx}"

            # Compute chunk hash
            chunk_hash = compute_content_hash(chunk)

            # Build vector dict
            vector = {
                "id": point_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "doc_id": doc_id,  # Track parent document
                    "doc_hash": doc_hash,  # Full document content hash
                    "chunk_hash": chunk_hash,  # Chunk-level hash
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "last_updated": datetime.now(datetime.timezone.utc).isoformat(),
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
    Index LlamaIndex documents to Pinecone namespace with optional clearing.

    Args:
        documents: List of LlamaIndex Document objects
        namespace: Pinecone namespace to store vectors
        clear_existing: If True, clear namespace before indexing (use with caution)

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


async def get_document_chunks(doc_id: str, namespace: str) -> list[dict]:
    """
    Query all chunks belonging to a specific document.

    Args:
        doc_id: Document ID to query
        namespace: Pinecone namespace

    Returns:
        List of chunk metadata
    """
    index = get_pinecone_index()

    # Query with filter for doc_id
    results = index.query(
        vector=[0.0] * 1536,  # Dummy vector
        top_k=100,
        namespace=namespace,
        filter={"doc_id": {"$eq": doc_id}},
        include_metadata=True,
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in results.matches
    ]


async def delete_document(doc_id: str, namespace: str) -> dict[str, Any]:
    """
    Delete all chunks for a specific document.

    Args:
        doc_id: Document ID to delete
        namespace: Pinecone namespace

    Returns:
        Dict with deletion statistics
    """
    index = get_pinecone_index()

    # Find all chunks for this document
    chunks = await get_document_chunks(doc_id, namespace)
    ids_to_delete = [chunk["id"] for chunk in chunks]

    if not ids_to_delete:
        return {
            "doc_id": doc_id,
            "namespace": namespace,
            "chunks_deleted": 0,
            "status": "not_found",
        }

    # Delete by IDs
    index.delete(ids=ids_to_delete, namespace=namespace)

    return {
        "doc_id": doc_id,
        "namespace": namespace,
        "chunks_deleted": len(ids_to_delete),
        "status": "deleted",
    }


async def upsert_documents(
    documents: list[Document],
    namespace: str,
) -> dict[str, Any]:
    """
    Smart upsert documents with content hash comparison.
    Only updates documents that have changed content.

    Args:
        documents: List of LlamaIndex Document objects
        namespace: Pinecone namespace

    Returns:
        Dict with upsert statistics
    """
    index = get_pinecone_index()

    updated = 0
    unchanged = 0
    deleted = 0

    for doc in documents:
        # Get doc_id
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        path = metadata.get("path") or metadata.get("repo") or metadata.get("url", "unknown")
        doc_id = compute_doc_id(source, path)
        new_hash = compute_content_hash(doc.text)

        # Check existing chunks
        existing_chunks = await get_document_chunks(doc_id, namespace)

        if existing_chunks:
            # Get the doc_hash from first chunk
            old_hash = existing_chunks[0]["metadata"].get("doc_hash")

            if old_hash == new_hash:
                # Content unchanged, skip
                unchanged += 1
                continue

            # Content changed - delete old chunks
            ids_to_delete = [chunk["id"] for chunk in existing_chunks]
            index.delete(ids=ids_to_delete, namespace=namespace)
            deleted += len(ids_to_delete)

        # Insert new chunks
        vectors = await prepare_vectors([doc])
        upsert_vectors(vectors, namespace)
        updated += 1

    return {
        "namespace": namespace,
        "documents_updated": updated,
        "documents_unchanged": unchanged,
        "chunks_deleted": deleted,
        "total_chunks": sum(
            len(
                await get_document_chunks(
                    compute_doc_id(
                        d.metadata.get("source", "unknown"),
                        d.metadata.get("path")
                        or d.metadata.get("repo")
                        or d.metadata.get("url", "unknown"),
                    ),
                    namespace,
                )
            )
            for d in documents
        ),
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
