from llama_index.core import Document

from app.indexers.vector_indexer import (
    compute_doc_id,
    delete_document,
    get_namespace_for_source,
    index_documents,
    upsert_documents,
)


async def ingest_document(
    content: str,
    metadata: dict,
    source: str = "default",
) -> dict:
    """
    Ingest a single document to Pinecone with smart upsert.

    Args:
        content: Document text content
        metadata: Document metadata (should include 'path' for tracking)
        source: Data source identifier (github_notes, github_repos, website_rogerink)

    Returns:
        Dict with ingestion result
    """
    # Create LlamaIndex document
    document = Document(text=content, metadata=metadata)

    # Get namespace for source
    namespace = get_namespace_for_source(source)

    # Use smart upsert (only updates if content changed)
    result = await upsert_documents(
        documents=[document],
        namespace=namespace,
    )

    return result


async def ingest_documents_batch(
    documents: list[Document],
    source: str,
    clear_existing: bool = False,
) -> dict:
    """
    Ingest a batch of LlamaIndex documents to Pinecone.

    Args:
        documents: List of LlamaIndex Document objects
        source: Data source identifier
        clear_existing: If True, clear namespace before indexing (use with caution)

    Returns:
        Dict with batch ingestion statistics
    """
    namespace = get_namespace_for_source(source)

    if clear_existing:
        # Use traditional index with clearing
        result = await index_documents(
            documents=documents,
            namespace=namespace,
            clear_existing=True,
        )
    else:
        # Use smart upsert (only updates changed documents)
        result = await upsert_documents(
            documents=documents,
            namespace=namespace,
        )

    return result


async def delete_single_document(
    source: str,
    path: str,
) -> dict:
    """
    Delete a specific document from Pinecone.

    Args:
        source: Data source identifier (github_notes, github_repos, website_rogerink)
        path: Document path or identifier

    Returns:
        Dict with deletion statistics
    """
    namespace = get_namespace_for_source(source)
    doc_id = compute_doc_id(source, path)

    result = await delete_document(doc_id, namespace)
    return result
