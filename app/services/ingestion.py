from llama_index.core import Document

from app.indexers.vector_indexer import get_namespace_for_source, index_documents


async def ingest_document(
    content: str,
    metadata: dict,
    source: str = "default",
) -> dict:
    """
    Ingest a single document to Pinecone.

    Args:
        content: Document text content
        metadata: Document metadata
        source: Data source identifier (github_notes, github_repos, website_rogerink)

    Returns:
        Dict with ingestion result
    """
    # Create LlamaIndex document
    document = Document(text=content, metadata=metadata)

    # Get namespace for source
    namespace = get_namespace_for_source(source)

    # Index document
    result = await index_documents(
        documents=[document],
        namespace=namespace,
        clear_existing=False,
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
        clear_existing: If True, clear namespace before indexing

    Returns:
        Dict with batch ingestion statistics
    """
    namespace = get_namespace_for_source(source)

    result = await index_documents(
        documents=documents,
        namespace=namespace,
        clear_existing=clear_existing,
    )

    return result
