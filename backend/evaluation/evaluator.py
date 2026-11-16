import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"\b[\w.-]+\b", text) if len(term) > 2}


class Evaluator:
    """Lightweight lexical evaluation; use labeled/LLM judges for semantic evaluation."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[dict[str, Any]],
        expected_source_names: list[str] | None = None,
    ) -> dict[str, float]:
        query_terms = _terms(query)
        context_terms = set().union(*(_terms(chunk["text"]) for chunk in contexts)) if contexts else set()
        answer_terms = _terms(answer)
        expected = set(expected_source_names or [])
        retrieved = {
            chunk.get("metadata", {}).get("document_name")
            for chunk in contexts
        }
        relevant = [
            bool(query_terms & _terms(chunk["text"]))
            for chunk in contexts
        ]
        metrics = {
            "context_precision": sum(relevant) / len(relevant) if relevant else 0.0,
            "context_recall": (
                len(query_terms & context_terms) / len(query_terms) if query_terms else 0.0
            ),
            "answer_relevancy": (
                len(query_terms & answer_terms) / len(query_terms) if query_terms else 0.0
            ),
            "faithfulness": (
                len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0
            ),
            "retrieval_hit_rate": len(expected & retrieved) / len(expected) if expected else None,
        }
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "metrics": metrics,
        }
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return metrics
