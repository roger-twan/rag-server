import hashlib
from datetime import datetime, timezone
from typing import Any

from llama_index.core import Document

from app.db import pinecone
from app.services import chunker, embedding


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
        chunks = chunker.chunk_text(text, chunk_size, chunk_overlap)

        # Create embeddings for chunks (async call)
        chunk_embeddings = await embedding.embed_content(chunks)

        # Create sparse vectors for chunks
        sparse_vectors = pinecone.encode_sparse(chunks)

        for idx, chunk in enumerate(chunks):
            embedding_values = chunk_embeddings[idx]
            sparse_vector = sparse_vectors[idx] if idx < len(sparse_vectors) else None

            # Deterministic ID: doc_id + chunk_index
            point_id = f"{doc_id}_{idx}"

            # Compute chunk hash
            chunk_hash = compute_content_hash(chunk)

            # Build vector dict
            vector = {
                "id": point_id,
                "values": embedding_values,
                "metadata": {
                    "text": chunk,
                    "doc_id": doc_id,  # Track parent document
                    "doc_hash": doc_hash,  # Full document content hash
                    "chunk_hash": chunk_hash,  # Chunk-level hash
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
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
    index = pinecone.get_pinecone_index()

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
        pinecone.upsert_vectors(batch, namespace)

    return {
        "namespace": namespace,
        "documents_indexed": len(documents),
        "total_chunks": len(vectors),
        "status": "success",
    }


async def get_document_chunks(doc_id: str, namespace: str) -> list[dict]:
    """
    Query Pinecone for all chunks belonging to a document.

    Args:
        doc_id: The document ID to query for
        namespace: The Pinecone namespace to query

    Returns:
        List of chunk dictionaries with id, metadata, etc.
    """
    index = pinecone.get_pinecone_index()

    # Query with doc_id filter to get all chunks for this document
    results = index.query(
        vector=[0.0] * 1024,  # Dummy vector, we're using filter
        top_k=1000,  # Get all chunks
        namespace=namespace,
        filter={"doc_id": {"$eq": doc_id}},
        include_metadata=True,
    )

    return [{"id": match.id, "metadata": match.metadata} for match in results.matches]


async def delete_document(doc_id: str, namespace: str) -> dict[str, Any]:
    """
    Delete all chunks for a document from Pinecone.

    Args:
        doc_id: The document ID to delete
        namespace: The Pinecone namespace

    Returns:
        Dict with deletion status
    """
    # Get all chunks for this document
    existing_chunks = await get_document_chunks(doc_id, namespace)

    if not existing_chunks:
        return {
            "doc_id": doc_id,
            "namespace": namespace,
            "chunks_deleted": 0,
            "status": "not_found",
        }

    # Delete by IDs
    ids_to_delete = [chunk["id"] for chunk in existing_chunks]
    index = pinecone.get_pinecone_index()
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
    index = pinecone.get_pinecone_index()

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
        pinecone.upsert_vectors(vectors, namespace)
        updated += 1

    # Calculate total chunks after all updates
    total_chunks = 0
    for d in documents:
        doc_id = compute_doc_id(
            d.metadata.get("source", "unknown"),
            d.metadata.get("path") or d.metadata.get("repo") or d.metadata.get("url", "unknown"),
        )
        chunks = await get_document_chunks(doc_id, namespace)
        total_chunks += len(chunks)

    return {
        "namespace": namespace,
        "documents_updated": updated,
        "documents_unchanged": unchanged,
        "chunks_deleted": deleted,
        "total_chunks": total_chunks,
        "status": "success",
    }


def get_namespace_for_source(source: str, repo_name: str | None = None) -> str:
    """Get the appropriate Pinecone namespace for a data source."""
    if source == "github_notes":
        return pinecone.NAMESPACES["github_notes"]
    elif source == "github_repos":
        return pinecone.NAMESPACES["github_repos"]
    elif source == "website_roger_ink":
        return pinecone.NAMESPACES["website_roger_ink"]
    else:
        return source  # Use source directly as namespace
