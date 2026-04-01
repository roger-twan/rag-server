"""
Sync jobs for manual ingestion of website and all GitHub repos.
"""

import logging
from typing import Any

from app.loaders.github_loader import AllReposLoader
from app.loaders.website_loader import load_website_documents
from app.services.ingestion import ingest_documents_batch

logger = logging.getLogger(__name__)


async def sync_website() -> dict[str, Any]:
    """
    Manually sync website content (roger.ink).
    Fetches sitemap, loads all pages, and indexes to Pinecone.

    Returns:
        Dict with sync statistics
    """
    logger.info("Starting website sync for roger.ink...")

    try:
        # Load website documents
        documents = await load_website_documents()

        if not documents:
            logger.warning("No documents found on website")
            return {
                "status": "no_data",
                "source": "website_rogerink",
                "message": "No documents found to index",
            }

        # Index documents (clear existing)
        result = await ingest_documents_batch(
            documents=documents,
            source="website_rogerink",
            clear_existing=True,
        )

        logger.info(
            f"Successfully synced website: {result['total_chunks']} chunks from {result['documents_indexed']} pages"
        )

        return {
            "status": "success",
            "source": "website_rogerink",
            "message": f"Synced {result['documents_indexed']} pages with {result['total_chunks']} chunks",
            "details": result,
        }

    except Exception as e:
        logger.error(f"Failed to sync website: {e}")
        return {
            "status": "error",
            "source": "website_rogerink",
            "message": f"Failed to sync: {str(e)}",
        }


async def sync_all_github_repos() -> dict[str, Any]:
    """
    Batch sync all GitHub repos (except notes) to Pinecone.
    For each repo, extracts: description, README, package.json

    Returns:
        Dict with sync statistics
    """
    logger.info("Starting batch sync for all GitHub repos...")

    try:
        # Load documents from all repos
        documents = await AllReposLoader.load_all_documents()

        if not documents:
            logger.warning("No documents found in any repo")
            return {
                "status": "no_data",
                "source": "github_repos",
                "message": "No documents found to index",
            }

        # Index documents (clear existing)
        result = await ingest_documents_batch(
            documents=documents,
            source="github_repos",
            clear_existing=True,
        )

        logger.info(
            f"Successfully synced GitHub repos: {result['total_chunks']} chunks from {result['documents_indexed']} repos"
        )

        return {
            "status": "success",
            "source": "github_repos",
            "message": f"Synced {result['documents_indexed']} repos with {result['total_chunks']} chunks",
            "details": result,
        }

    except Exception as e:
        logger.error(f"Failed to sync GitHub repos: {e}")
        return {
            "status": "error",
            "source": "github_repos",
            "message": f"Failed to sync: {str(e)}",
        }
