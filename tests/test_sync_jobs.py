"""Tests for sync_jobs module."""

from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core import Document

from app.scripts.sync_jobs import sync_all_github_repos, sync_website


class TestSyncWebsite:
    """Tests for sync_website function."""

    @pytest.mark.asyncio
    async def test_loads_and_ingests_website_documents(self):
        """Test that website documents are loaded and ingested."""
        mock_docs = [
            Document(
                text="Page 1 content", metadata={"path": "/about", "source": "website_roger_ink"}
            ),
            Document(
                text="Page 2 content", metadata={"path": "/contact", "source": "website_roger_ink"}
            ),
        ]

        with patch(
            "app.scripts.sync_jobs.load_website_documents", new_callable=AsyncMock
        ) as mock_load, patch(
            "app.scripts.sync_jobs.ingest_documents_batch", new_callable=AsyncMock
        ) as mock_ingest:
            mock_load.return_value = mock_docs
            mock_ingest.return_value = {
                "documents_updated": 2,
                "documents_unchanged": 0,
                "chunks_deleted": 0,
                "total_chunks": 5,
            }

            result = await sync_website()

            mock_load.assert_called_once()
            mock_ingest.assert_called_once()
            call_kwargs = mock_ingest.call_args[1]
            assert call_kwargs["source"] == "website_roger_ink"
            assert call_kwargs["clear_existing"] is False
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_returns_no_data_when_no_documents(self):
        """Test that no_data status is returned when no documents found."""
        with patch(
            "app.scripts.sync_jobs.load_website_documents", new_callable=AsyncMock
        ) as mock_load:
            mock_load.return_value = []

            result = await sync_website()

            assert result["status"] == "no_data"
            assert result["source"] == "website_roger_ink"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        """Test that error status is returned on exception."""
        with patch(
            "app.scripts.sync_jobs.load_website_documents", new_callable=AsyncMock
        ) as mock_load:
            mock_load.side_effect = Exception("Network error")

            result = await sync_website()

            assert result["status"] == "error"
            assert "Network error" in result["message"]

    @pytest.mark.asyncio
    async def test_includes_stats_in_result(self):
        """Test that sync stats are included in result."""
        mock_docs = [Document(text="Content", metadata={"path": "/"})]

        with patch(
            "app.scripts.sync_jobs.load_website_documents", new_callable=AsyncMock
        ) as mock_load, patch(
            "app.scripts.sync_jobs.ingest_documents_batch", new_callable=AsyncMock
        ) as mock_ingest:
            mock_load.return_value = mock_docs
            mock_ingest.return_value = {
                "documents_updated": 1,
                "documents_unchanged": 3,
                "total_chunks": 4,
            }

            result = await sync_website()

            assert "details" in result
            assert result["details"]["documents_updated"] == 1
            assert result["details"]["documents_unchanged"] == 3


class TestSyncAllGithubRepos:
    """Tests for sync_all_github_repos function."""

    @pytest.mark.asyncio
    async def test_loads_and_ingests_repo_documents(self):
        """Test that repo documents are loaded and ingested."""
        mock_docs = [
            Document(text="Repo 1 README", metadata={"repo": "repo1", "source": "github_repos"}),
            Document(text="Repo 2 README", metadata={"repo": "repo2", "source": "github_repos"}),
        ]

        with patch(
            "app.scripts.sync_jobs.AllReposLoader.load_all_documents", new_callable=AsyncMock
        ) as mock_load, patch(
            "app.scripts.sync_jobs.ingest_documents_batch", new_callable=AsyncMock
        ) as mock_ingest:
            mock_load.return_value = mock_docs
            mock_ingest.return_value = {
                "documents_updated": 2,
                "documents_unchanged": 0,
                "chunks_deleted": 0,
            }

            await sync_all_github_repos()

            mock_load.assert_called_once()
            mock_ingest.assert_called_once()
            call_kwargs = mock_ingest.call_args[1]
            assert call_kwargs["source"] == "github_repos"
            assert call_kwargs["clear_existing"] is False

    @pytest.mark.asyncio
    async def test_returns_no_data_when_no_repos(self):
        """Test that no_data status is returned when no repos found."""
        with patch(
            "app.scripts.sync_jobs.AllReposLoader.load_all_documents", new_callable=AsyncMock
        ) as mock_load:
            mock_load.return_value = []

            result = await sync_all_github_repos()

            assert result["status"] == "no_data"
            assert result["source"] == "github_repos"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        """Test that error status is returned on exception."""
        with patch(
            "app.scripts.sync_jobs.AllReposLoader.load_all_documents", new_callable=AsyncMock
        ) as mock_load:
            mock_load.side_effect = Exception("GitHub API error")

            result = await sync_all_github_repos()

            assert result["status"] == "error"
            assert "GitHub API error" in result["message"]

    @pytest.mark.asyncio
    async def test_uses_smart_upsert(self):
        """Test that smart upsert is used (clear_existing=False)."""
        mock_docs = [Document(text="Content", metadata={"repo": "test"})]

        with patch(
            "app.scripts.sync_jobs.AllReposLoader.load_all_documents", new_callable=AsyncMock
        ) as mock_load, patch(
            "app.scripts.sync_jobs.ingest_documents_batch", new_callable=AsyncMock
        ) as mock_ingest:
            mock_load.return_value = mock_docs
            mock_ingest.return_value = {"documents_updated": 1}

            await sync_all_github_repos()

            call_kwargs = mock_ingest.call_args[1]
            assert call_kwargs["clear_existing"] is False
