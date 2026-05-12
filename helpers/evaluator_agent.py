"""Evaluation agent for metrics and manual review artifacts."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .types import NewsSample


class EvaluatorAgent:
    """Evaluate classification quality and build manual-review artifacts."""

    def accuracy(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Compute classification accuracy for paired gold/prediction labels."""
        if not y_true:
            return 0.0
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    def build_manual_explanation_set(
        self,
        samples: Sequence[NewsSample],
        predictions: Sequence[int],
        explanations: Sequence[str],
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """Build a compact row set for manual explanation quality scoring."""
        rows = []
        for sample, pred, explanation in zip(samples, predictions, explanations):
            rows.append(
                {
                    "text": sample.text,
                    "label": str(sample.label),
                    "prediction": str(pred),
                    "explanation": explanation,
                    "manual_score_1_to_5": "",
                }
            )
            if len(rows) >= limit:
                break
        return rows
