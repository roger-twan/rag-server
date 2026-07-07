from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import Example, Run

from app.core.config import settings
from app.core.langsmith import configure_langsmith_tracing
from app.evals.run_eval import (
    DEFAULT_DATASET,
    DEFAULT_DELAY_SECONDS,
    _prepare_conversation,
    load_cases,
    score_case,
)
from app.services.answer_generator import generate_answer

DEFAULT_LANGSMITH_DATASET = "rag-server-personal-qa"
DEFAULT_EXPERIMENT_PREFIX = "rag-server-rag-eval"


def _case_to_example(case: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    inputs = {"question": case["question"]}
    if case.get("messages"):
        inputs["messages"] = case["messages"]

    return {
        "id": uuid5(NAMESPACE_URL, f"{dataset_name}:{case['id']}"),
        "inputs": inputs,
        "outputs": {
            "reference_answer": case.get("reference_answer", ""),
            "reference_sources": case.get("reference_sources", []),
            "should_answer": case["should_answer"],
        },
        "metadata": {
            "case_id": case["id"],
            "tags": case.get("tags", []),
        },
        "split": "default",
    }


def _get_or_create_dataset(client: Client, dataset_name: str):
    if client.has_dataset(dataset_name=dataset_name):
        return client.read_dataset(dataset_name=dataset_name)

    return client.create_dataset(
        dataset_name,
        description="RAG evaluation dataset for Roger's personal AI Q&A system.",
        metadata={
            "source": str(DEFAULT_DATASET),
            "system": "rag-server",
        },
    )


def sync_dataset(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    dataset_name: str = DEFAULT_LANGSMITH_DATASET,
    client: Client | None = None,
) -> dict[str, Any]:
    configure_langsmith_tracing()
    client = client or Client()
    cases = load_cases(dataset_path)
    _get_or_create_dataset(client, dataset_name)
    examples = [_case_to_example(case, dataset_name) for case in cases]
    client.create_examples(dataset_name=dataset_name, examples=examples)
    return {
        "dataset_name": dataset_name,
        "examples_synced": len(examples),
    }


def _example_to_case(example: Example) -> dict[str, Any]:
    outputs = example.outputs or {}
    metadata = example.metadata or {}
    inputs = example.inputs or {}
    return {
        "id": metadata.get("case_id") or str(example.id),
        "question": inputs["question"],
        "messages": inputs.get("messages", []),
        "reference_answer": outputs.get("reference_answer", ""),
        "reference_sources": outputs.get("reference_sources", []),
        "should_answer": outputs.get("should_answer", True),
        "tags": metadata.get("tags", []),
    }


async def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    case = {
        "id": "langsmith-example",
        "question": inputs["question"],
        "messages": inputs.get("messages", []),
    }
    conversation_id = _prepare_conversation(case)
    return await generate_answer(inputs["question"], conversation_id=conversation_id)


def _scores_from_run(run: Run, example: Example) -> dict[str, Any]:
    return score_case(_example_to_case(example), run.outputs or {})


def answer_presence(run: Run, example: Example) -> dict[str, Any]:
    scores = _scores_from_run(run, example)
    return {
        "key": "answer_presence",
        "score": scores["answer_presence"],
    }


def source_match(run: Run, example: Example) -> dict[str, Any]:
    scores = _scores_from_run(run, example)
    return {
        "key": "source_match",
        "score": scores["source_match"],
    }


def abstention_correctness(run: Run, example: Example) -> dict[str, Any]:
    scores = _scores_from_run(run, example)
    return {
        "key": "abstention_correctness",
        "score": scores["abstention_correctness"],
        "value": {"is_refusal": scores["is_refusal"]},
    }


def final_score(run: Run, example: Example) -> dict[str, Any]:
    scores = _scores_from_run(run, example)
    return {
        "key": "final_score",
        "score": scores["final_score"],
    }


async def _delayed_target(delay_seconds: float):
    first_run = True

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal first_run
        if first_run:
            first_run = False
        elif delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        return await _target(inputs)

    return target


async def run_experiment(
    *,
    dataset_name: str = DEFAULT_LANGSMITH_DATASET,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
    limit: int | None = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    client: Client | None = None,
) -> Any:
    configure_langsmith_tracing()
    client = client or Client()
    data = dataset_name
    if limit is not None:
        data = list(client.list_examples(dataset_name=dataset_name, limit=limit))

    target = await _delayed_target(delay_seconds)
    return await aevaluate(
        target,
        data=data,
        evaluators=[answer_presence, source_match, abstention_correctness, final_score],
        experiment_prefix=experiment_prefix,
        description="RAG evaluation experiment for Roger's personal AI Q&A system.",
        metadata={
            "llm_provider": settings.LLM_PROVIDER,
            "dataset_name": dataset_name,
            "delay_seconds": delay_seconds,
        },
        max_concurrency=0,
        client=client,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync and run LangSmith RAG evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync-dataset")
    sync_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sync_parser.add_argument("--dataset-name", default=DEFAULT_LANGSMITH_DATASET)

    run_parser = subparsers.add_parser("run-experiment")
    run_parser.add_argument("--dataset-name", default=DEFAULT_LANGSMITH_DATASET)
    run_parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)

    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if args.command == "sync-dataset":
        result = sync_dataset(dataset_path=args.dataset, dataset_name=args.dataset_name)
        print(json.dumps(result, indent=2))
        return 0

    await run_experiment(
        dataset_name=args.dataset_name,
        experiment_prefix=args.experiment_prefix,
        limit=args.limit,
        delay_seconds=args.delay_seconds,
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
