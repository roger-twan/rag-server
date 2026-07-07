from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services import retriever


class FakeEmbeddings:
    async def aembed_query(self, query: str) -> list[float]:
        return [0.1] * 1024


class FakeReranker:
    def __init__(self, results: list[dict]):
        self.results = results

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        return self.results


class FailingReranker:
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        raise RuntimeError("cohere connection failed")


@pytest.mark.asyncio
async def test_retrieve_uses_postgres_full_text_and_neighbor_chunks(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPARSE_SEARCH", False)

    match = SimpleNamespace(
        id="doc_0",
        score=0.95,
        metadata={
            "chunk_id": "doc_0",
            "text_preview": "preview",
            "doc_id": "doc",
            "chunk_index": 0,
            "source": "test",
        },
    )
    index = MagicMock()
    index.query.return_value = SimpleNamespace(matches=[match])

    with (
        patch("app.services.retriever.embeddings", FakeEmbeddings()),
        patch("app.services.retriever.get_pinecone_index", return_value=index),
        patch("app.services.retriever._get_all_namespaces", return_value=["namespace"]),
        patch("app.services.retriever.postgres.get_chunks_by_ids") as get_chunks,
        patch("app.services.retriever.postgres.get_neighbor_chunks") as get_neighbors,
        patch(
            "app.services.retriever.cohere_reranker",
            FakeReranker([{"index": 0, "relevance_score": 0.9}]),
        ),
    ):
        get_chunks.return_value = {"doc_0": SimpleNamespace(text="full text")}
        get_neighbors.return_value = [
            SimpleNamespace(text="neighbor before"),
            SimpleNamespace(text="full text"),
        ]

        results = await retriever.retrieve("question", top_k=1, rerank_top_n=1)

    assert results[0]["chunk_id"] == "doc_0"
    assert results[0]["text"] == "neighbor before\n\nfull text"
    assert results[0]["rerank_score"] == 0.9
    get_chunks.assert_called_once_with(["doc_0"])
    get_neighbors.assert_called_once_with("doc", 0, window=1)


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_legacy_pinecone_text(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPARSE_SEARCH", False)

    match = SimpleNamespace(
        id="legacy_0",
        score=0.9,
        metadata={"text": "legacy metadata text", "doc_id": "legacy", "chunk_index": 0},
    )
    index = MagicMock()
    index.query.return_value = SimpleNamespace(matches=[match])

    with (
        patch("app.services.retriever.embeddings", FakeEmbeddings()),
        patch("app.services.retriever.get_pinecone_index", return_value=index),
        patch("app.services.retriever._get_all_namespaces", return_value=["namespace"]),
        patch("app.services.retriever.postgres.get_chunks_by_ids", return_value={}),
        patch("app.services.retriever.postgres.get_neighbor_chunks", return_value=[]),
        patch(
            "app.services.retriever.cohere_reranker",
            FakeReranker([{"index": 0, "relevance_score": 0.9}]),
        ),
    ):
        results = await retriever.retrieve("question", top_k=1, rerank_top_n=1)

    assert results[0]["text"] == "legacy metadata text"


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_vector_order_when_rerank_fails(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPARSE_SEARCH", False)

    match = SimpleNamespace(
        id="doc_0",
        score=0.95,
        metadata={
            "chunk_id": "doc_0",
            "text_preview": "preview",
            "doc_id": "doc",
            "chunk_index": 0,
            "source": "test",
        },
    )
    index = MagicMock()
    index.query.return_value = SimpleNamespace(matches=[match])

    with (
        patch("app.services.retriever.embeddings", FakeEmbeddings()),
        patch("app.services.retriever.get_pinecone_index", return_value=index),
        patch("app.services.retriever._get_all_namespaces", return_value=["namespace"]),
        patch("app.services.retriever.postgres.get_chunks_by_ids") as get_chunks,
        patch("app.services.retriever.postgres.get_neighbor_chunks") as get_neighbors,
        patch("app.services.retriever.cohere_reranker", FailingReranker()),
    ):
        get_chunks.return_value = {"doc_0": SimpleNamespace(text="full text")}
        get_neighbors.return_value = [SimpleNamespace(text="full text")]

        results = await retriever.retrieve("question", top_k=1, rerank_top_n=1)

    assert results == [
        {
            "chunk_id": "doc_0",
            "text": "full text",
            "score": 0.95,
            "rerank_score": None,
            "metadata": {
                "chunk_id": "doc_0",
                "text_preview": "preview",
                "doc_id": "doc",
                "chunk_index": 0,
                "source": "test",
            },
        }
    ]
