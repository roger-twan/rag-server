"""Tests for ingestion service."""

from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core import Document

from app.services.ingestion import (
    delete_single_document,
    ingest_document,
    ingest_documents_batch,
)


class TestIngestDocument:
    """Tests for ingest_document function."""

    @pytest.mark.asyncio
    async def test_creates_document_and_ingests(self):
        """Test that function creates document and calls upsert."""
        with patch(
            "app.indexers.vector_indexer.upsert_documents", new_callable=AsyncMock
        ) as mock_upsert:
            mock_upsert.return_value = {
                "documents_updated": 1,
                "documents_unchanged": 0,
            }

            result = await ingest_document(
                content="Test content",
                metadata={"path": "test.md", "title": "Test"},
                source="test_source",
            )

            assert mock_upsert.called, "Mock was not called!"
            call_kwargs = mock_upsert.call_args[1]
            call_docs = call_kwargs["documents"]
            assert len(call_docs) == 1
            assert call_docs[0].text == "Test content"
            assert result["documents_updated"] == 1

    @pytest.mark.asyncio
    async def test_uses_correct_namespace(self):
        """Test that function uses correct namespace for source."""
        with patch(
            "app.indexers.vector_indexer.upsert_documents", new_callable=AsyncMock
        ) as mock_upsert:
            await ingest_document(
                content="Content",
                metadata={"path": "file.md"},
                source="github_notes",
            )

            call_kwargs = mock_upsert.call_args[1]
            assert call_kwargs["namespace"] == "github_notes"


class TestIngestDocumentsBatch:
    """Tests for ingest_documents_batch function."""

    @pytest.mark.asyncio
    async def test_uses_smart_upsert_by_default(self):
        """Test that smart upsert is used when clear_existing is False."""
        docs = [Document(text="Content", metadata={"path": "file.md"})]

        with (
            patch(
                "app.indexers.vector_indexer.upsert_documents", new_callable=AsyncMock
            ) as mock_upsert,
            patch(
                "app.indexers.vector_indexer.index_documents", new_callable=AsyncMock
            ) as mock_index,
        ):
            mock_upsert.return_value = {"documents_updated": 1}

            await ingest_documents_batch(docs, "test_source", clear_existing=False)

            mock_upsert.assert_called_once()
            mock_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_index_with_clear_when_requested(self):
        """Test that index_documents is used when clear_existing is True."""
        docs = [Document(text="Content", metadata={"path": "file.md"})]

        with (
            patch(
                "app.indexers.vector_indexer.upsert_documents", new_callable=AsyncMock
            ) as mock_upsert,
            patch(
                "app.indexers.vector_indexer.index_documents", new_callable=AsyncMock
            ) as mock_index,
        ):
            mock_index.return_value = {"documents_indexed": 1}

            await ingest_documents_batch(docs, "test_source", clear_existing=True)

            mock_index.assert_called_once()
            mock_upsert.assert_not_called()
            call_kwargs = mock_index.call_args[1]
            assert call_kwargs["clear_existing"] is True

    @pytest.mark.asyncio
    async def test_passes_documents_to_upsert(self):
        """Test that documents are passed to upsert function."""
        docs = [
            Document(text="Content 1", metadata={"path": "file1.md"}),
            Document(text="Content 2", metadata={"path": "file2.md"}),
        ]

        with patch(
            "app.indexers.vector_indexer.upsert_documents", new_callable=AsyncMock
        ) as mock_upsert:
            mock_upsert.return_value = {"documents_updated": 2}

            await ingest_documents_batch(docs, "test_source")

            call_kwargs = mock_upsert.call_args[1]
            call_docs = call_kwargs["documents"]
            assert len(call_docs) == 2


class TestDeleteSingleDocument:
    """Tests for delete_single_document function."""

    @pytest.mark.asyncio
    async def test_computes_doc_id_and_deletes(self):
        """Test that function computes doc_id and calls delete."""
        with (
            patch(
                "app.indexers.vector_indexer.delete_document", new_callable=AsyncMock
            ) as mock_delete,
            patch("app.indexers.vector_indexer.compute_doc_id") as mock_compute_id,
        ):
            mock_compute_id.return_value = "computed_doc_id"
            mock_delete.return_value = {"status": "deleted", "chunks_deleted": 3}

            result = await delete_single_document("github_notes", "path/to/file.md")

            mock_compute_id.assert_called_once_with("github_notes", "path/to/file.md")
            mock_delete.assert_called_once_with("computed_doc_id", "github_notes")
            assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_uses_correct_namespace(self):
        """Test that correct namespace is determined from source."""
        with patch(
            "app.indexers.vector_indexer.delete_document", new_callable=AsyncMock
        ) as mock_delete:
            mock_delete.return_value = {"status": "deleted", "chunks_deleted": 1}

            await delete_single_document("website_roger_ink", "/about")

            call_args = mock_delete.call_args
            assert call_args[0][1] == "website_roger_ink"
