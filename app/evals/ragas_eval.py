from __future__ import annotations

import argparse
import asyncio
import json
import os
import warnings
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

from app.core.config import settings
from app.evals.run_eval import (
    DEFAULT_DATASET,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_OUTPUT_DIR,
    load_cases,
    run_cases,
    write_outputs,
)

DEFAULT_RAGAS_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ragas"


def _configure_openai_for_ragas() -> None:
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def _openai_client_for_ragas() -> OpenAI:
    _configure_openai_for_ragas()
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required to run RAGAS evaluation.")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _result_to_sample(result: dict[str, Any]) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=result["question"],
        response=result.get("answer", ""),
        reference=result.get("reference_answer", ""),
        retrieved_contexts=result.get("retrieved_contexts", []),
    )


def _legacy_metric_classes():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
            category=DeprecationWarning,
        )
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    return Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall


def _build_metrics():
    openai_client = _openai_client_for_ragas()
    evaluator_llm = llm_factory(settings.RAGAS_LLM_MODEL, client=openai_client)
    evaluator_embeddings = embedding_factory(
        "openai",
        model=settings.RAGAS_EMBEDDING_MODEL,
        client=openai_client,
    )
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall = _legacy_metric_classes()
    return [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]


def _metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
    }
    keys = set()
    for row in rows:
        keys.update(row)
    return sorted(key for key in keys if key not in excluded)


def _summarize_ragas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_columns = _metric_columns(rows)
    averages = {}
    for column in metric_columns:
        values = [row[column] for row in rows if isinstance(row.get(column), int | float)]
        averages[column] = round(mean(values), 4) if values else None

    return {
        "case_count": len(rows),
        "metrics": averages,
    }


def run_ragas_evaluation(
    *,
    results: list[dict[str, Any]],
    output_dir: Path = DEFAULT_RAGAS_OUTPUT_DIR,
) -> dict[str, Any]:
    samples = [_result_to_sample(result) for result in results]
    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(
        dataset,
        metrics=_build_metrics(),
        raise_exceptions=False,
    )
    dataframe = ragas_result.to_pandas()
    rows = dataframe.to_dict(orient="records")
    summary = _summarize_ragas(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "ragas_results.csv", index=False)
    (output_dir / "ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_results(results_path: Path) -> list[dict[str, Any]]:
    with results_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS metrics for RAG eval results.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAGAS_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if args.results_jsonl:
        results = load_results(args.results_jsonl)
    else:
        cases = load_cases(args.dataset)
        results = await run_cases(cases, limit=args.limit, delay_seconds=args.delay_seconds)
        local_summary = {
            "case_count": len(results),
            "source": str(args.dataset),
        }
        write_outputs(results, local_summary, DEFAULT_OUTPUT_DIR)

    summary = run_ragas_evaluation(results=results, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
