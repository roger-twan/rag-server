"""
FastAPI lifecycle events for startup and shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.pinecone import get_pinecone_index
from app.db.postgres import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting up RAG server...")

    try:
        # Ensure Postgres schema exists before ingestion or queries run.
        init_db()
        logger.info("Postgres database initialized")

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
