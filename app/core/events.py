"""
FastAPI lifecycle events for startup and shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.pinecone import get_pinecone_index

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting up RAG server...")

    try:
        # Verify Pinecone connection
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        logger.info(f"Pinecone index connected: {stats}")

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        raise

    logger.info("RAG server startup complete")

    yield

    # Shutdown
    logger.info("Shutting down RAG server...")
    logger.info("RAG server shutdown complete")
