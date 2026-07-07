import json
from pathlib import Path
from unittest.mock import Mock, patch

from app.evals import ragas_eval


def _result():
    return {
        "question": "What do you do?",
        "answer": "Roger is a full-stack product engineer.",
        "reference_answer": "Roger is a full-stack product engineer.",
        "retrieved_contexts": ["Roger is a full-stack product engineer."],
    }


def test_result_to_sample_maps_rag_fields():
    sample = ragas_eval._result_to_sample(_result())

    assert sample.user_input == "What do you do?"
    assert sample.response == "Roger is a full-stack product engineer."
    assert sample.reference == "Roger is a full-stack product engineer."
    assert sample.retrieved_contexts == ["Roger is a full-stack product engineer."]


def test_summarize_ragas_averages_metric_columns():
    rows = [
        {"user_input": "q1", "faithfulness": 1.0, "answer_relevancy": 0.8},
        {"user_input": "q2", "faithfulness": 0.5, "answer_relevancy": 0.6},
    ]

    summary = ragas_eval._summarize_ragas(rows)

    assert summary == {
        "case_count": 2,
        "metrics": {
            "answer_relevancy": 0.7,
            "faithfulness": 0.75,
        },
    }


def test_load_results_reads_jsonl(tmp_path: Path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(_result()) + "\n", encoding="utf-8")

    assert ragas_eval.load_results(results_path) == [_result()]


def test_run_ragas_evaluation_writes_outputs(tmp_path: Path):
    fake_dataframe = Mock()
    fake_dataframe.to_dict.return_value = [
        {
            "user_input": "What do you do?",
            "response": "Roger is a full-stack product engineer.",
            "reference": "Roger is a full-stack product engineer.",
            "retrieved_contexts": ["Roger is a full-stack product engineer."],
            "faithfulness": 1.0,
        }
    ]
    fake_result = Mock()
    fake_result.to_pandas.return_value = fake_dataframe

    with (
        patch("app.evals.ragas_eval._build_metrics", return_value=[]),
        patch("app.evals.ragas_eval.evaluate", return_value=fake_result) as evaluate,
    ):
        summary = ragas_eval.run_ragas_evaluation(results=[_result()], output_dir=tmp_path)

    evaluate.assert_called_once()
    fake_dataframe.to_csv.assert_called_once_with(tmp_path / "ragas_results.csv", index=False)
    assert summary == {"case_count": 1, "metrics": {"faithfulness": 1.0}}
    assert (tmp_path / "ragas_summary.json").exists()
