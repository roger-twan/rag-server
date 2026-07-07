import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest

from app.evals import langsmith_eval


def _case():
    return {
        "id": "profile-role-001",
        "question": "What do you do?",
        "messages": [{"role": "user", "content": "Hi"}],
        "reference_answer": "Roger is a full-stack product engineer.",
        "reference_sources": ["website_roger_ink"],
        "should_answer": True,
        "tags": ["profile", "you-form"],
    }


def test_case_to_example_maps_inputs_outputs_and_metadata():
    example = langsmith_eval._case_to_example(_case(), "dataset")

    assert isinstance(example["id"], UUID)
    assert example["inputs"] == {
        "question": "What do you do?",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert example["outputs"]["should_answer"] is True
    assert example["metadata"]["case_id"] == "profile-role-001"
    assert example["metadata"]["tags"] == ["profile", "you-form"]


def test_sync_dataset_creates_dataset_and_upserts_examples(tmp_path: Path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(_case()) + "\n", encoding="utf-8")
    client = Mock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="dataset-1")

    result = langsmith_eval.sync_dataset(
        dataset_path=dataset,
        dataset_name="dataset",
        client=client,
    )

    client.create_dataset.assert_called_once()
    client.create_examples.assert_called_once()
    assert client.create_examples.call_args.kwargs["dataset_name"] == "dataset"
    assert len(client.create_examples.call_args.kwargs["examples"]) == 1
    assert result == {"dataset_name": "dataset", "examples_synced": 1}


def test_evaluators_score_run_outputs_against_example_outputs():
    run = SimpleNamespace(
        outputs={
            "answer": "Roger is a full-stack product engineer.",
            "sources": [{"source": "website_roger_ink", "path": "/about", "title": "About"}],
        }
    )
    example = SimpleNamespace(
        id="example-1",
        inputs={"question": "What do you do?"},
        outputs={
            "reference_answer": "Roger is a full-stack product engineer.",
            "reference_sources": ["website_roger_ink"],
            "should_answer": True,
        },
        metadata={"case_id": "profile-role-001", "tags": ["profile"]},
    )

    assert langsmith_eval.answer_presence(run, example)["score"] == 1.0
    assert langsmith_eval.source_match(run, example)["score"] == 1.0
    assert langsmith_eval.abstention_correctness(run, example)["score"] == 1.0
    assert langsmith_eval.final_score(run, example)["score"] == 1.0


@pytest.mark.asyncio
async def test_target_runs_generate_answer_with_prepared_conversation():
    with (
        patch("app.evals.langsmith_eval._prepare_conversation", return_value="conv-1") as prepare,
        patch(
            "app.evals.langsmith_eval.generate_answer", new_callable=AsyncMock
        ) as generate_answer,
    ):
        generate_answer.return_value = {"answer": "Hello", "sources": []}

        result = await langsmith_eval._target(
            {
                "question": "What do you do?",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )

    prepare.assert_called_once()
    generate_answer.assert_awaited_once_with("What do you do?", conversation_id="conv-1")
    assert result == {"answer": "Hello", "sources": []}


@pytest.mark.asyncio
async def test_run_experiment_calls_langsmith_aevaluate():
    client = Mock()

    with patch("app.evals.langsmith_eval.aevaluate", new_callable=AsyncMock) as aevaluate:
        await langsmith_eval.run_experiment(
            dataset_name="dataset",
            experiment_prefix="experiment",
            delay_seconds=0,
            client=client,
        )

    aevaluate.assert_awaited_once()
    assert aevaluate.call_args.kwargs["data"] == "dataset"
    assert aevaluate.call_args.kwargs["experiment_prefix"] == "experiment"
    assert aevaluate.call_args.kwargs["client"] is client
