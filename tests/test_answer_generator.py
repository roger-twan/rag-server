from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import answer_generator


class FakePrompt:
    def __init__(self, chain):
        self.chain = chain

    def __or__(self, llm):
        return self.chain


@pytest.mark.asyncio
async def test_rewrite_query_skips_llm_when_history_is_empty():
    with (
        patch("app.services.answer_generator.postgres.get_recent_messages", return_value=[]),
        patch("app.services.answer_generator._get_llm") as mock_get_llm,
    ):
        result = await answer_generator._rewrite_query("What database does it use?", "conv-1")

    assert result == "What database does it use?"
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_generate_answer_returns_payload_when_no_chunks_found():
    with (
        patch("app.services.answer_generator.postgres.ensure_conversation", return_value="conv-1"),
        patch("app.services.answer_generator.postgres.get_recent_messages", return_value=[]),
        patch("app.services.answer_generator.postgres.add_message", return_value="msg-1"),
        patch("app.services.answer_generator.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.answer_generator._get_llm"),
    ):
        mock_retrieve.return_value = []
        fake_chain = AsyncMock()
        fake_chain.ainvoke.return_value.content = (
            "I don't have enough information to answer this question."
        )
        with patch(
            "app.services.answer_generator.fallback_prompt_template", FakePrompt(fake_chain)
        ):
            result = await answer_generator.generate_answer("Unknown?", conversation_id="conv-1")

    assert result == {
        "answer": "I don't have enough information to answer this question.",
        "conversation_id": "conv-1",
        "rewritten_query": "Unknown?",
        "sources": [],
    }
    mock_retrieve.assert_awaited_once_with("Unknown?")


def test_format_sources_deduplicates_by_source_path_title():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "score": 0.9,
            "rerank_score": 0.8,
            "metadata": {"source": "github_notes", "path": "a.md", "title": "A"},
        },
        {
            "chunk_id": "chunk-2",
            "score": 0.7,
            "rerank_score": 0.6,
            "metadata": {"source": "github_notes", "path": "a.md", "title": "A"},
        },
    ]

    assert answer_generator._format_sources(chunks) == [
        {
            "source": "github_notes",
            "path": "a.md",
            "title": "A",
            "chunk_id": "chunk-1",
            "score": 0.9,
            "rerank_score": 0.8,
        }
    ]


def test_format_history_uses_role_and_content():
    messages = [
        SimpleNamespace(role="user", content="What is RAG?"),
        SimpleNamespace(role="assistant", content="Retrieval augmented generation."),
    ]

    assert answer_generator._format_history(messages) == (
        "user: What is RAG?\nassistant: Retrieval augmented generation."
    )


@pytest.mark.asyncio
async def test_generate_answer_uses_fallback_prompt_when_no_chunks_found():
    with (
        patch("app.services.answer_generator.postgres.ensure_conversation", return_value="conv-1"),
        patch("app.services.answer_generator.postgres.get_recent_messages", return_value=[]),
        patch("app.services.answer_generator.postgres.add_message", return_value="msg-1"),
        patch("app.services.answer_generator.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.answer_generator._get_llm"),
    ):
        mock_retrieve.return_value = []
        fake_chain = AsyncMock()
        fake_chain.ainvoke.return_value.content = "Hello! What would you like to know?"
        with patch(
            "app.services.answer_generator.fallback_prompt_template", FakePrompt(fake_chain)
        ):
            result = await answer_generator.generate_answer("hello", conversation_id="conv-1")

    assert result == {
        "answer": "Hello! What would you like to know?",
        "conversation_id": "conv-1",
        "rewritten_query": "hello",
        "sources": [],
    }
    mock_retrieve.assert_awaited_once_with("hello")
