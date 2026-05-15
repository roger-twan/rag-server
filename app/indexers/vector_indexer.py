import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from llama_index.core import Document

from app.core.config import settings
from app.db import pinecone, postgres
from app.services import chunker, embedding
from app.services.markdown_chunker import chunk_markdown_document

logger = logging.getLogger(__name__)


def compute_doc_id(source: str, path: str) -> str:
    """
    Generate deterministic doc_id from source and path.
    Used to track all chunks belonging to a single document.
    """
    return hashlib.sha256(f"{source}:{path}".encode()).hexdigest()[:16]


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


def sanitize_metadata_for_pinecone(metadata: dict) -> dict:
    """
    Sanitize metadata for Pinecone compatibility.
    Pinecone only supports string, number, and boolean values.
    Converts None to empty string, lists to comma-separated strings.
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, list):
            sanitized[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, bool):
            sanitized[key] = value  # Booleans are fine
        elif isinstance(value, (int, float)):
            sanitized[key] = value  # Numbers are fine
        else:
            sanitized[key] = str(value)  # Convert everything else to string
    return sanitized


async def prepare_vectors(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Convert LlamaIndex documents to Pinecone vectors with doc_id tracking.
    Uses markdown-aware chunking for blog posts, simple chunking for others.
    """
    vectors, _ = await _prepare_vectors_and_records(documents, chunk_size, chunk_overlap)
    return vectors


async def _prepare_vectors_and_records(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 200,
) -> tuple[list[dict], list[dict]]:
    vectors = []
    document_records = []

    for doc in documents:
        text = doc.text
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        path = metadata.get("path") or metadata.get("repo") or metadata.get("url", "unknown")
        doc_id = compute_doc_id(source, path)
        doc_hash = compute_content_hash(text)

        # Choose chunking strategy based on source
        if source == "github_notes":
            # Use markdown-aware chunking for blog posts
            document_title = metadata.get("title", "")
            chunk_data_list = chunk_markdown_document(
                text=text,
                document_title=document_title,
                min_chunk_size=200,
                max_chunk_size=1500,
                enrich_context=True,
            )
            chunks = [c["text"] for c in chunk_data_list]
            chunk_metadata_list = [c["metadata"] for c in chunk_data_list]
        else:
            # Use simple chunking for other sources
            chunks = chunker.chunk_text(text, chunk_size, chunk_overlap)
            chunk_metadata_list = [{} for _ in chunks]

        # Create embeddings
        chunk_embeddings = await embedding.embed_content(chunks)
        sparse_vectors = pinecone.encode_sparse(chunks) if settings.ENABLE_SPARSE_SEARCH else []
        db_chunks = []

        for idx, chunk in enumerate(chunks):
            embedding_values = chunk_embeddings[idx]
            sparse_vector = sparse_vectors[idx] if idx < len(sparse_vectors) else None
            point_id = f"{doc_id}_{idx}"
            chunk_hash = compute_content_hash(chunk)
            chunk_meta = chunk_metadata_list[idx]
            text_preview = chunk[:500]

            # Sanitize metadata for Pinecone compatibility
            clean_metadata = sanitize_metadata_for_pinecone(metadata)
            clean_chunk_meta = sanitize_metadata_for_pinecone(chunk_meta)

            vector = {
                "id": point_id,
                "values": embedding_values,
                "metadata": {
                    "chunk_id": point_id,
                    "text_preview": text_preview,
                    "doc_id": doc_id,
                    "doc_hash": doc_hash,
                    "chunk_hash": chunk_hash,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    **clean_metadata,
                    **clean_chunk_meta,
                },
            }

            if sparse_vector:
                vector["sparse_values"] = sparse_vector

            vectors.append(vector)
            db_chunks.append(
                {
                    "id": point_id,
                    "chunk_index": idx,
                    "text": chunk,
                    "text_preview": text_preview,
                    "content_hash": chunk_hash,
                    "meta": {**clean_metadata, **clean_chunk_meta},
                }
            )

        document_records.append(
            {
                "doc_id": doc_id,
                "source": source,
                "path": path,
                "title": str(metadata.get("title", "")),
                "content_hash": doc_hash,
                "meta": sanitize_metadata_for_pinecone(metadata),
                "chunks": db_chunks,
            }
        )

    return vectors, document_records


def _persist_document_records(document_records: list[dict]) -> None:
    with postgres.session_scope() as session:
        for record in document_records:
            postgres.upsert_document(
                doc_id=record["doc_id"],
                source=record["source"],
                path=record["path"],
                title=record["title"],
                content_hash=record["content_hash"],
                meta=record["meta"],
                session=session,
            )
            postgres.replace_document_chunks(
                doc_id=record["doc_id"],
                chunks=record["chunks"],
                session=session,
            )


def _delete_vectors_by_ids(ids: list[str], namespace: str) -> None:
    if not ids:
        return
    index = pinecone.get_pinecone_index()
    index.delete(ids=ids, namespace=namespace)


def _upsert_vectors_after_postgres(vectors: list[dict], namespace: str) -> None:
    upserted_ids = []
    batch_size = 100
    try:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            pinecone.upsert_vectors(batch, namespace)
            upserted_ids.extend(vector["id"] for vector in batch)
    except Exception:
        logger.critical(
            "Pinecone upsert failed after PostgreSQL write; attempting compensating delete "
            "for %s vectors in namespace %s",
            len(upserted_ids),
            namespace,
            exc_info=True,
        )
        try:
            _delete_vectors_by_ids(upserted_ids, namespace)
        except Exception:
            logger.critical(
                "Compensating Pinecone delete failed for namespace %s; manual cleanup required",
                namespace,
                exc_info=True,
            )
        raise


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
    # Prepare vectors (async)
    vectors, document_records = await _prepare_vectors_and_records(documents)

    if not vectors:
        return {
            "namespace": namespace,
            "documents_indexed": 0,
            "total_chunks": 0,
            "status": "no_data",
        }

    _persist_document_records(document_records)
    index = pinecone.get_pinecone_index()
    if clear_existing:
        index.delete(delete_all=True, namespace=namespace)
    _upsert_vectors_after_postgres(vectors, namespace)

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
    postgres.delete_document(doc_id)

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

            if old_hash == new_hash and postgres.document_has_chunks(doc_id):
                # Content unchanged, skip
                unchanged += 1
                continue

        # Insert new chunks
        vectors, document_records = await _prepare_vectors_and_records([doc])
        _persist_document_records(document_records)
        _upsert_vectors_after_postgres(vectors, namespace)

        if existing_chunks:
            new_ids = {vector["id"] for vector in vectors}
            stale_ids = [chunk["id"] for chunk in existing_chunks if chunk["id"] not in new_ids]
            _delete_vectors_by_ids(stale_ids, namespace)
            deleted += len(stale_ids)
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
