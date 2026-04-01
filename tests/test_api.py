"""Tests for API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_returns_status_and_environment(self):
        """Test that root endpoint returns status and environment."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "env" in data
        assert data["message"] == "RAG Server is running"


class TestQueryEndpoint:
    """Tests for query endpoint."""

    @pytest.mark.asyncio
    async def test_query_returns_answer(self):
        """Test that query endpoint returns generated answer."""
        with patch("app.api.routes.generate_answer", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "This is the answer to your question."

            response = client.get("/api/query?q=What is Python?")

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is Python?"
            assert data["result"] == "This is the answer to your question."
            mock_generate.assert_called_once_with("What is Python?")

    @pytest.mark.asyncio
    async def test_query_without_question(self):
        """Test that query without q parameter still works."""
        with patch("app.api.routes.generate_answer", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Default answer."

            response = client.get("/api/query")

            # Should still work with empty query
            assert response.status_code in [200, 422]  # 422 if validation fails


class TestIngestWebsiteEndpoint:
    """Tests for website ingestion endpoint."""

    @pytest.mark.asyncio
    async def test_triggers_website_sync(self):
        """Test that endpoint triggers website sync."""
        with patch("app.api.routes.sync_website", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {
                "status": "success",
                "source": "website_roger_ink",
                "message": "Synced 5 pages",
            }

            response = client.post("/api/ingest/website")

            assert response.status_code == 200
            assert response.json()["status"] == "success"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_sync_result(self):
        """Test that endpoint returns sync result."""
        with patch("app.api.routes.sync_website", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {
                "status": "success",
                "source": "website_roger_ink",
                "message": "Synced 2 updated, 3 unchanged",
                "details": {
                    "documents_updated": 2,
                    "documents_unchanged": 3,
                    "total_chunks": 10,
                },
            }

            response = client.post("/api/ingest/website")

            data = response.json()
            assert data["status"] == "success"
            assert "details" in data
            assert data["details"]["documents_updated"] == 2


class TestIngestGithubReposEndpoint:
    """Tests for GitHub repos ingestion endpoint."""

    @pytest.mark.asyncio
    async def test_triggers_github_sync(self):
        """Test that endpoint triggers GitHub repos sync."""
        with patch("app.api.routes.sync_all_github_repos", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {
                "status": "success",
                "source": "github_repos",
                "message": "Synced 10 repos",
            }

            response = client.post("/api/ingest/github-all-repos")

            assert response.status_code == 200
            assert response.json()["status"] == "success"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_sync_stats(self):
        """Test that endpoint returns sync statistics."""
        with patch("app.api.routes.sync_all_github_repos", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {
                "status": "success",
                "source": "github_repos",
                "message": "Synced 5 updated, 10 unchanged",
                "details": {
                    "documents_updated": 5,
                    "documents_unchanged": 10,
                    "total_chunks": 50,
                },
            }

            response = client.post("/api/ingest/github-all-repos")

            data = response.json()
            assert data["status"] == "success"
            assert "details" in data
            assert data["details"]["documents_updated"] == 5


class TestWebhookEndpoint:
    """Tests for GitHub webhook endpoint."""

    @pytest.mark.asyncio
    async def test_handles_valid_webhook(self):
        """Test that valid webhook is processed."""
        payload = {
            "ref": "refs/heads/main",
            "repository": {"name": "notes"},
            "commits": [{"id": "abc123", "message": "Update file"}],
        }

        with patch("app.api.webhooks.verify_github_webhook", return_value=True), patch(
            "app.api.webhooks.is_notes_repo_push", return_value=True
        ), patch("app.api.webhooks.NotesRepoLoader.load_documents") as mock_load, patch(
            "app.api.webhooks.ingest_documents_batch", new_callable=AsyncMock
        ) as mock_ingest:
            mock_load.return_value = []
            mock_ingest.return_value = {"status": "success"}

            response = client.post(
                "/api/webhooks/github",
                json=payload,
                headers={"X-Hub-Signature-256": "sha256=valid", "X-GitHub-Event": "push"},
            )

            # Should accept the webhook
            assert response.status_code in [200, 202]

    @pytest.mark.asyncio
    async def test_rejects_invalid_signature(self):
        """Test that invalid signature is rejected."""
        with patch("app.api.webhooks.verify_github_webhook", return_value=False):
            response = client.post(
                "/api/webhooks/github",
                json={"test": "data"},
                headers={"X-Hub-Signature-256": "sha256=invalid"},
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ignores_non_notes_repo(self):
        """Test that pushes to other repos are ignored."""
        with patch("app.api.webhooks.verify_github_webhook", return_value=True), patch(
            "app.api.webhooks.is_notes_repo_push", return_value=False
        ):
            response = client.post(
                "/api/webhooks/github",
                json={"ref": "refs/heads/main", "repository": {"name": "other-repo"}},
                headers={"X-Hub-Signature-256": "sha256=valid"},
            )

            # Should return 200 but skip processing
            assert response.status_code == 200
            assert (
                "ignored" in response.json().get("message", "").lower()
                or response.json().get("status") == "ignored"
            )
