import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from app.evals import run_eval


def test_load_cases_reads_jsonl(tmp_path: Path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "question": "What do you do?",
                "reference_answer": "Roger is a full-stack product engineer.",
                "reference_sources": ["website_roger_ink"],
                "should_answer": True,
                "tags": ["profile"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = run_eval.load_cases(dataset)

    assert cases[0]["id"] == "case-1"
    assert cases[0]["_line_number"] == 1


def test_score_case_rewards_supported_answer():
    case = {
        "should_answer": True,
        "reference_sources": ["website_roger_ink"],
    }
    result = {
        "answer": "Roger is a full-stack product engineer.",
        "sources": [{"source": "website_roger_ink", "path": "/about", "title": "About"}],
    }

    scores = run_eval.score_case(case, result)

    assert scores["answer_presence"] == 1.0
    assert scores["source_match"] == 1.0
    assert scores["abstention_correctness"] == 1.0
    assert scores["final_score"] == 1.0


def test_score_case_rewards_refusal_when_case_should_not_answer():
    case = {
        "should_answer": False,
        "reference_sources": [],
    }
    result = {
        "answer": "I don't have enough information to answer that.",
        "sources": [],
    }

    scores = run_eval.score_case(case, result)

    assert scores["is_refusal"] is True
    assert scores["abstention_correctness"] == 1.0
    assert scores["source_match"] == 1.0


@pytest.mark.asyncio
async def test_run_case_uses_generate_answer_and_scores_result():
    case = {
        "id": "case-1",
        "question": "What do you do?",
        "reference_answer": "Roger is a full-stack product engineer.",
        "reference_sources": ["website_roger_ink"],
        "should_answer": True,
        "tags": ["profile"],
    }

    with patch(
        "app.evals.run_eval.generate_answer", new_callable=AsyncMock
    ) as mock_generate_answer:
        mock_generate_answer.return_value = {
            "answer": "Roger is a full-stack product engineer.",
            "conversation_id": "conv-1",
            "rewritten_query": "What does Roger do?",
            "sources": [{"source": "website_roger_ink", "path": "/about", "title": "About"}],
        }

        result = await run_eval.run_case(case)

    mock_generate_answer.assert_awaited_once_with(
        "What do you do?",
        conversation_id=None,
        include_contexts=True,
    )
    assert result["id"] == "case-1"
    assert result["scores"]["final_score"] == 1.0


@pytest.mark.asyncio
async def test_run_case_records_error_result_when_generation_fails():
    case = {
        "id": "case-1",
        "question": "What do you do?",
        "reference_answer": "Roger is a full-stack product engineer.",
        "reference_sources": ["website_roger_ink"],
        "should_answer": True,
        "tags": ["profile"],
    }

    with patch(
        "app.evals.run_eval.generate_answer", new_callable=AsyncMock
    ) as mock_generate_answer:
        mock_generate_answer.side_effect = RuntimeError("rate limited")

        result = await run_eval.run_case(case)

    assert result["error"]["type"] == "RuntimeError"
    assert result["error"]["message"] == "rate limited"
    assert result["scores"]["final_score"] == 0.0


@pytest.mark.asyncio
async def test_run_cases_waits_between_cases():
    cases = [
        {
            "id": "case-1",
            "question": "One?",
            "reference_answer": "",
            "reference_sources": [],
            "should_answer": True,
            "tags": [],
        },
        {
            "id": "case-2",
            "question": "Two?",
            "reference_answer": "",
            "reference_sources": [],
            "should_answer": True,
            "tags": [],
        },
    ]

    with (
        patch("app.evals.run_eval.run_case", new_callable=AsyncMock) as run_case,
        patch("app.evals.run_eval.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        run_case.side_effect = [
            {"id": "case-1", "tags": [], "scores": {"final_score": 1.0}},
            {"id": "case-2", "tags": [], "scores": {"final_score": 1.0}},
        ]

        results = await run_eval.run_cases(cases, delay_seconds=6.5)

    sleep.assert_awaited_once_with(6.5)
    assert [result["id"] for result in results] == ["case-1", "case-2"]


def test_write_outputs_writes_json_files(tmp_path: Path):
    results = [{"id": "case-1", "scores": {"final_score": 1.0}}]
    summary = {"case_count": 1}

    run_eval.write_outputs(results, summary, tmp_path)

    assert (tmp_path / "results.jsonl").exists()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_check_database_ready_raises_actionable_error_when_postgres_is_down():
    with patch("app.evals.run_eval.postgres.engine.connect") as mock_connect:
        mock_connect.side_effect = OperationalError("SELECT 1", {}, Exception("down"))

        with pytest.raises(RuntimeError, match="docker compose up -d postgres"):
            run_eval.check_database_ready()


def test_prepare_conversation_uses_database_safe_uuid():
    case = {
        "id": "conversation-001",
        "messages": [
            {"role": "user", "content": "What do you do?"},
            {"role": "assistant", "content": "I represent Roger."},
            {"role": "user", "content": "What projects are you best suited for?"},
        ],
    }

    with (
        patch(
            "app.evals.run_eval.postgres.ensure_conversation", return_value="conv-1"
        ) as ensure_conversation,
        patch("app.evals.run_eval.postgres.add_message") as add_message,
    ):
        conversation_id = run_eval._prepare_conversation(case)

    generated_id = ensure_conversation.call_args.args[0]
    assert len(generated_id) == 36
    assert conversation_id == "conv-1"
    assert add_message.call_count == 2
