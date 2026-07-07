from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.config import settings
from app.db import postgres
from app.services.answer_generator import generate_answer

DEFAULT_DATASET = Path("evals/rag_questions.jsonl")
DEFAULT_OUTPUT_DIR = Path("evals/results")
DEFAULT_DELAY_SECONDS = 6.5

REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
    "available context is insufficient",
    "context is insufficient",
    "can't answer",
    "cannot answer",
    "don't know",
    "do not know",
)


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    cases = []
    with dataset_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case["_line_number"] = line_number
            cases.append(case)
    return cases


def _is_refusal(answer: str) -> bool:
    normalized = answer.lower()
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def _source_values(source: dict[str, Any]) -> set[str]:
    values = {
        str(source.get("source") or ""),
        str(source.get("path") or ""),
        str(source.get("title") or ""),
    }
    return {value.lower() for value in values if value}


def _source_match(reference_sources: list[str], actual_sources: list[dict[str, Any]]) -> float:
    if not reference_sources:
        return 1.0 if not actual_sources else 0.0

    actual_values = set()
    for source in actual_sources:
        actual_values.update(_source_values(source))

    matched = 0
    for reference in reference_sources:
        needle = reference.lower()
        if any(needle in value or value in needle for value in actual_values):
            matched += 1

    return matched / len(reference_sources)


def score_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    should_answer = bool(case["should_answer"])
    is_refusal = _is_refusal(answer)
    answer_present = bool(answer.strip())

    abstention_correctness = 1.0 if is_refusal != should_answer else 0.0
    if not should_answer:
        abstention_correctness = 1.0 if is_refusal else 0.0

    source_match = _source_match(case.get("reference_sources", []), sources)
    answer_presence = 1.0 if answer_present else 0.0

    final_score = mean([abstention_correctness, source_match, answer_presence])

    return {
        "answer_presence": answer_presence,
        "source_match": source_match,
        "abstention_correctness": abstention_correctness,
        "final_score": final_score,
        "is_refusal": is_refusal,
    }


def _prepare_conversation(case: dict[str, Any]) -> str | None:
    messages = case.get("messages") or []
    if not messages:
        return None

    conversation_id = postgres.ensure_conversation(str(uuid4()))
    for message in messages[:-1]:
        postgres.add_message(
            conversation_id=conversation_id,
            role=message["role"],
            content=message["content"],
        )
    return conversation_id


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    conversation_id = _prepare_conversation(case)
    base_result = {
        "id": case["id"],
        "question": case["question"],
        "tags": case.get("tags", []),
        "should_answer": case["should_answer"],
        "reference_answer": case.get("reference_answer", ""),
        "reference_sources": case.get("reference_sources", []),
        "conversation_id": conversation_id,
    }

    try:
        result = await generate_answer(
            case["question"],
            conversation_id=conversation_id,
            include_contexts=True,
        )
    except Exception as exc:
        return {
            **base_result,
            "answer": "",
            "rewritten_query": None,
            "sources": [],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "scores": {
                "answer_presence": 0.0,
                "source_match": 0.0,
                "abstention_correctness": 0.0,
                "final_score": 0.0,
                "is_refusal": False,
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    latency_ms = round((time.perf_counter() - started) * 1000)
    scores = score_case(case, result)

    return {
        **base_result,
        "answer": result.get("answer", ""),
        "rewritten_query": result.get("rewritten_query"),
        "sources": result.get("sources", []),
        "retrieved_contexts": result.get("retrieved_contexts", []),
        "conversation_id": result.get("conversation_id"),
        "latency_ms": latency_ms,
        "scores": scores,
    }


async def run_cases(
    cases: list[dict[str, Any]],
    limit: int | None = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
) -> list[dict[str, Any]]:
    selected_cases = cases[:limit] if limit else cases
    results = []
    for index, case in enumerate(selected_cases, start=1):
        if index > 1 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        print(f"[{index}/{len(selected_cases)}] {case['id']}: {case['question']}")
        results.append(await run_case(case))
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_keys = [
        "answer_presence",
        "source_match",
        "abstention_correctness",
        "final_score",
    ]
    averages = {
        key: round(mean(result["scores"][key] for result in results), 4) if results else 0.0
        for key in score_keys
    }
    tag_counts = Counter(tag for result in results for tag in result.get("tags", []))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": settings.LLM_PROVIDER,
        "case_count": len(results),
        "averages": averages,
        "latency_ms_avg": (
            round(mean(result["latency_ms"] for result in results), 2) if results else 0.0
        ),
        "tag_counts": dict(tag_counts.most_common()),
        "failed_case_ids": [
            result["id"] for result in results if result["scores"]["final_score"] < 1.0
        ],
        "error_case_ids": [result["id"] for result in results if result.get("error")],
    }


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def check_database_ready() -> None:
    try:
        with postgres.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        raise RuntimeError(
            "PostgreSQL is not reachable. Start the local database with "
            "`docker compose up -d postgres`, then rerun "
            "`uv run python -m app.evals.run_eval`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Seconds to wait between cases. Defaults to 6.5 to stay under Cohere trial limits.",
    )
    parser.add_argument("--fail-under", type=float, default=None)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    check_database_ready()
    cases = load_cases(args.dataset)
    results = await run_cases(cases, limit=args.limit, delay_seconds=args.delay_seconds)
    summary = summarize(results)
    write_outputs(results, summary, args.output_dir)

    print(json.dumps(summary, indent=2))
    if args.fail_under is not None and summary["averages"]["final_score"] < args.fail_under:
        return 1
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
