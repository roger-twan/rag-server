"""Tests for vector_indexer module."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document

from app.indexers.vector_indexer import (
    compute_content_hash,
    compute_doc_id,
    delete_document,
    get_document_chunks,
    get_namespace_for_source,
    prepare_vectors,
    upsert_documents,
)


class TestComputeDocId:
    """Tests for compute_doc_id function."""

    def test_deterministic_id(self):
        """Test that same source+path produces same doc_id."""
        id1 = compute_doc_id("github_notes", "path/to/file.md")
        id2 = compute_doc_id("github_notes", "path/to/file.md")
        assert id1 == id2
        assert len(id1) == 16

    def test_different_sources_produce_different_ids(self):
        """Test that different sources produce different doc_ids."""
        id1 = compute_doc_id("source1", "path/to/file.md")
        id2 = compute_doc_id("source2", "path/to/file.md")
        assert id1 != id2

    def test_different_paths_produce_different_ids(self):
        """Test that different paths produce different doc_ids."""
        id1 = compute_doc_id("github_notes", "path/to/file1.md")
        id2 = compute_doc_id("github_notes", "path/to/file2.md")
        assert id1 != id2


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        content = "test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")
        assert hash1 != hash2

    def test_empty_content(self):
        """Test hashing empty content."""
        hash_val = compute_content_hash("")
        assert hash_val == hashlib.sha256("".encode()).hexdigest()


class TestPrepareVectors:
    """Tests for prepare_vectors function."""

    @pytest.mark.asyncio
    async def test_creates_deterministic_ids(self):
        """Test that vectors get deterministic IDs."""
        doc = Document(
            text="This is test content that will be chunked into multiple pieces",
            metadata={"source": "test", "path": "test.md"},
        )

        with patch(
            "app.services.embedding.embed_content", new_callable=AsyncMock
        ) as mock_embed, patch("app.db.pinecone.encode_sparse") as mock_sparse:
            mock_embed.return_value = [[0.1] * 1536]  # Single chunk
            mock_sparse.return_value = [{"indices": [1, 2], "values": [0.5, 0.5]}]

            vectors = await prepare_vectors([doc], chunk_size=100, chunk_overlap=0)

        assert len(vectors) == 1
        # Check ID format: {doc_id}_{chunk_index}
        doc_id = compute_doc_id("test", "test.md")
        assert vectors[0]["id"] == f"{doc_id}_0"

    @pytest.mark.asyncio
    async def test_includes_doc_hash_in_metadata(self):
        """Test that doc_hash is included in metadata."""
        doc = Document(
            text="Test content for hashing",
            metadata={"source": "test", "path": "test.md"},
        )

        with patch(
            "app.services.embedding.embed_content", new_callable=AsyncMock
        ) as mock_embed, patch("app.db.pinecone.encode_sparse") as mock_sparse:
            mock_embed.return_value = [[0.1] * 1536]
            mock_sparse.return_value = [{"indices": [1], "values": [0.5]}]

            vectors = await prepare_vectors([doc])

        expected_hash = compute_content_hash(doc.text)
        assert vectors[0]["metadata"]["doc_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_includes_chunk_hash_in_metadata(self):
        """Test that chunk_hash is included in metadata."""
        doc = Document(
            text="Test content for chunk hashing",
            metadata={"source": "test", "path": "test.md"},
        )

        with patch(
            "app.services.embedding.embed_content", new_callable=AsyncMock
        ) as mock_embed, patch("app.db.pinecone.encode_sparse") as mock_sparse:
            mock_embed.return_value = [[0.1] * 1536]
            mock_sparse.return_value = [{"indices": [1], "values": [0.5]}]

            vectors = await prepare_vectors([doc])

        assert "chunk_hash" in vectors[0]["metadata"]
        assert len(vectors[0]["metadata"]["chunk_hash"]) == 64

    @pytest.mark.asyncio
    async def test_multiple_chunks_get_sequential_ids(self):
        """Test that multiple chunks get sequential IDs."""
        doc = Document(
            text="A" * 120,  # 120 chars -> 3 chunks with chunk_size=50, overlap=0
            metadata={"source": "test", "path": "test.md"},
        )

        with patch(
            "app.services.embedding.embed_content", new_callable=AsyncMock
        ) as mock_embed, patch("app.db.pinecone.encode_sparse") as mock_sparse:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
            mock_sparse.return_value = [
                {"indices": [1], "values": [0.5]},
                {"indices": [2], "values": [0.6]},
                {"indices": [3], "values": [0.7]},
            ]

            vectors = await prepare_vectors([doc], chunk_size=50, chunk_overlap=0)

        doc_id = compute_doc_id("test", "test.md")
        assert len(vectors) == 3
        assert vectors[0]["id"] == f"{doc_id}_0"
        assert vectors[1]["id"] == f"{doc_id}_1"
        assert vectors[2]["id"] == f"{doc_id}_2"

    @pytest.mark.asyncio
    async def test_uses_repo_when_path_missing(self):
        """Test that repo is used as path when path is missing."""
        doc = Document(
            text="Test content",
            metadata={"source": "github_repos", "repo": "my-repo"},
        )

        with patch(
            "app.services.embedding.embed_content", new_callable=AsyncMock
        ) as mock_embed, patch("app.db.pinecone.encode_sparse") as mock_sparse:
            mock_embed.return_value = [[0.1] * 1536]
            mock_sparse.return_value = [{"indices": [1], "values": [0.5]}]

            vectors = await prepare_vectors([doc])

        doc_id = compute_doc_id("github_repos", "my-repo")
        assert vectors[0]["id"].startswith(doc_id)


class TestGetDocumentChunks:
    """Tests for get_document_chunks function."""

    @pytest.mark.asyncio
    async def test_queries_with_doc_id_filter(self):
        """Test that function queries with correct doc_id filter."""
        mock_index = MagicMock()
        mock_results = MagicMock()
        mock_results.matches = []

        with patch("app.db.pinecone.get_pinecone_index", return_value=mock_index):
            mock_index.query.return_value = mock_results
            await get_document_chunks("test_doc_id", "test_namespace")

        mock_index.query.assert_called_once()
        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"doc_id": {"$eq": "test_doc_id"}}
        assert call_kwargs["namespace"] == "test_namespace"


class TestDeleteDocument:
    """Tests for delete_document function."""

    @pytest.mark.asyncio
    async def test_deletes_by_ids(self):
        """Test that function deletes vectors by IDs."""
        mock_index = MagicMock()
        mock_results = MagicMock()
        mock_match = MagicMock()
        mock_match.id = "chunk_id_1"
        mock_results.matches = [mock_match]

        with patch("app.db.pinecone.get_pinecone_index", return_value=mock_index):
            mock_index.query.return_value = mock_results
            result = await delete_document("doc_id", "namespace")

        mock_index.delete.assert_called_once_with(ids=["chunk_id_1"], namespace="namespace")
        assert result["status"] == "deleted"
        assert result["chunks_deleted"] == 1

    @pytest.mark.asyncio
    async def test_returns_not_found_when_no_chunks(self):
        """Test that function returns not_found status when no chunks exist."""
        mock_index = MagicMock()
        mock_results = MagicMock()
        mock_results.matches = []

        with patch("app.db.pinecone.get_pinecone_index", return_value=mock_index):
            mock_index.query.return_value = mock_results
            result = await delete_document("doc_id", "namespace")

        assert result["status"] == "not_found"
        assert result["chunks_deleted"] == 0
        mock_index.delete.assert_not_called()


class TestUpsertDocuments:
    """Tests for upsert_documents function."""

    @pytest.mark.asyncio
    async def test_skips_unchanged_documents(self):
        """Test that unchanged documents are skipped."""
        doc = Document(
            text="Test content",
            metadata={"source": "test", "path": "file.md"},
        )
        doc_hash = compute_content_hash(doc.text)

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk_1"
        mock_chunk.metadata = {"doc_hash": doc_hash}  # Same hash

        with patch(
            "app.indexers.vector_indexer.get_document_chunks", new_callable=AsyncMock
        ) as mock_get_chunks, patch(
            "app.indexers.vector_indexer.prepare_vectors"
        ) as mock_prepare, patch(
            "app.db.pinecone.upsert_vectors"
        ) as mock_upsert:
            mock_get_chunks.return_value = [{"id": "chunk_1", "metadata": {"doc_hash": doc_hash}}]

            result = await upsert_documents([doc], "namespace")

            assert result["documents_unchanged"] == 1
            assert result["documents_updated"] == 0
            mock_prepare.assert_not_called()
            mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_changed_documents(self):
        """Test that changed documents are updated."""
        doc = Document(
            text="New content",
            metadata={"source": "test", "path": "file.md"},
        )

        with patch(
            "app.indexers.vector_indexer.get_document_chunks", new_callable=AsyncMock
        ) as mock_get_chunks, patch("app.db.pinecone.get_pinecone_index"), patch(
            "app.indexers.vector_indexer.prepare_vectors", new_callable=AsyncMock
        ) as mock_prepare, patch(
            "app.db.pinecone.upsert_vectors"
        ) as mock_upsert:
            mock_get_chunks.return_value = [
                {"id": "old_chunk", "metadata": {"doc_hash": "old_hash"}}
            ]
            mock_prepare.return_value = [{"id": "new_chunk"}]

            result = await upsert_documents([doc], "namespace")

            assert result["documents_updated"] == 1
            assert result["chunks_deleted"] == 1  # Old chunk deleted
            mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_new_documents(self):
        """Test that new documents are inserted."""
        doc = Document(
            text="New document content",
            metadata={"source": "test", "path": "new_file.md"},
        )

        with patch(
            "app.indexers.vector_indexer.get_document_chunks", new_callable=AsyncMock
        ) as mock_get_chunks, patch(
            "app.indexers.vector_indexer.prepare_vectors", new_callable=AsyncMock
        ) as mock_prepare, patch(
            "app.db.pinecone.upsert_vectors"
        ) as mock_upsert:
            mock_get_chunks.return_value = []  # No existing chunks
            mock_prepare.return_value = [{"id": "chunk_1"}]

            result = await upsert_documents([doc], "namespace")

            assert result["documents_updated"] == 1
            mock_upsert.assert_called_once()


class TestGetNamespaceForSource:
    """Tests for get_namespace_for_source function."""

    def test_github_notes_source(self):
        assert get_namespace_for_source("github_notes") == "github_notes"

    def test_github_repos_source(self):
        assert get_namespace_for_source("github_repos") == "github_repos"

    def test_website_source(self):
        assert get_namespace_for_source("website_rogerink") == "website_rogerink"

    def test_unknown_source_uses_directly(self):
        assert get_namespace_for_source("custom_source") == "custom_source"
