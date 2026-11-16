import json

from backend.evaluation.evaluator import Evaluator


def test_evaluator_calculates_and_persists_metrics(tmp_path):
    evaluator = Evaluator(tmp_path / "evaluation.jsonl")
    contexts = [
        {
            "text": "The retention period is 30 days.",
            "metadata": {"document_name": "policy.pdf"},
        }
    ]

    metrics = evaluator.evaluate(
        "What is the retention period?",
        "The retention period is 30 days.",
        contexts,
        expected_source_names=["policy.pdf"],
    )

    assert set(metrics) == {
        "context_precision",
        "context_recall",
        "answer_relevancy",
        "faithfulness",
        "retrieval_hit_rate",
    }
    assert metrics["retrieval_hit_rate"] == 1.0
    with (tmp_path / "evaluation.jsonl").open(encoding="utf-8") as file:
        record = json.loads(file.readline())
    assert record["metrics"] == metrics
