"""Shared data-contract helpers for the agent handoff files.

The source of truth is docs/data_contracts.md. This module keeps the repeated
column lists, label validation, and common row shaping behind one interface so
agent modules do not each reimplement the same contract details.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

# Used by Nadi (`agents/nadi_classifier.py`) to reject bad generated labels and
# by Sabina (`agents/sabina_evaluator.py`) to score per-label accuracy.
LABELS = ("up", "down", "neutral")

# Used by Sabina's prediction validation; probabilities are rounded, so the
# contract allows a small tolerance instead of requiring exact sums.
PROBABILITY_TOLERANCE = 0.02


def combine_binary_probs(p_up: float, p_down: float) -> dict[str, float]:
    """Fold the up-head and down-head positive-class probabilities into one
    three-way distribution.

    `neutral` is the mass left over when neither head fires, so it is the product
    of the two heads disagreeing with their own class. Renormalising makes the
    three sum to 1, which keeps the maximum always >= 1/3 regardless of how
    confident either head is.

    No divide-by-zero guard: for p in [0, 1] the raw total is
    p_up + p_down + (1 - p_up)(1 - p_down), which is >= 1 everywhere (it hits its
    minimum of exactly 1 at the corners), so it can never be 0.
    """
    raw = {
        "up": p_up,
        "down": p_down,
        "neutral": (1.0 - p_up) * (1.0 - p_down),
    }
    total = sum(raw.values())
    return {label: value / total for label, value in raw.items()}

# Handoff 2: Nadi writes these columns in predictions_test.csv; Sabina reads
# them, and Nadi's LLM-code guardrail checks generated scripts against them.
PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_up", "prob_down", "prob_neutral", "split",
]

# Handoff 4: Jack writes these columns in sample_for_explanation.csv; Freddi
# reads them before generating explanations.
EXPLANATION_SAMPLE_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "prob_up", "prob_down", "prob_neutral",
]

# Handoff 5: Freddi writes these columns in explanations.csv; Jack reads them
# when building the final outputs.
EXPLANATION_OUTPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "explanation", "manual_score",
]

# Freddi copies these input columns straight through to explanations.csv.
EXPLANATION_PASSTHROUGH_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label", "confidence",
]

# Handoff 6: Jack writes these columns in final_results.csv for the notebook and
# any Streamlit app to read.
FINAL_RESULTS_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence", "explanation", "manual_score",
]


def read_prediction_rows(path: str | os.PathLike[str]) -> list[dict]:
    """Read Nadi's predictions_test.csv for Sabina and enforce column order."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if reader.fieldnames != PREDICTION_COLUMNS:
        raise ValueError(
            "predictions_test.csv columns do not match data contract: "
            f"{reader.fieldnames}"
        )
    if not rows:
        raise ValueError("predictions_test.csv must contain at least one held-out row")
    return rows


def validate_prediction_rows(rows: Iterable[dict]) -> None:
    """Validate Nadi prediction rows before Sabina scores them."""
    for row in rows:
        article_id = row.get("article_id", "")
        label = row.get("label", "")
        predicted = row.get("predicted_label", "")
        if row.get("split") not in ("val", "test"):
            raise ValueError(f"{article_id}: split must be val or test")
        if label not in LABELS:
            raise ValueError(f"{article_id}: invalid label {label!r}")
        if predicted not in LABELS:
            raise ValueError(f"{article_id}: invalid predicted_label {predicted!r}")

        probs = [float(row[f"prob_{label_name}"]) for label_name in LABELS]
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError(f"{article_id}: confidence must be between 0 and 1")
        if any(prob < 0 or prob > 1 for prob in probs):
            raise ValueError(f"{article_id}: probabilities must be between 0 and 1")
        if abs(sum(probs) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(f"{article_id}: prob_* columns must sum to about 1")
        if abs(confidence - max(probs)) > PROBABILITY_TOLERANCE:
            raise ValueError(f"{article_id}: confidence must equal max prob_*")


def _test_rows(predictions):
    """Return test rows Jack uses for explanations and final outputs."""
    if "split" not in predictions.columns:
        raise ValueError("predictions file has no split column (Handoff 2)")
    test = predictions[predictions["split"] == "test"]
    if test.empty:
        raise ValueError("predictions file has no split=test rows")
    return test


def build_explanation_sample(predictions, sample_size: int = 300):
    """Shape Jack's sample_for_explanation.csv rows from Nadi predictions."""
    predictions = _test_rows(predictions)
    sample = (predictions[[
        "article_id", "article_title", "predicted_label", "label",
        "confidence", "prob_up", "prob_down", "prob_neutral",
    ]].rename(columns={"label": "actual_label"}))
    n = min(len(sample), sample_size)
    if n < len(sample):
        sample = sample.sample(n=n, random_state=42)
    return sample


def write_explanation_sample(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    sample_size: int = 300,
) -> int:
    """Write Jack's sample_for_explanation.csv for Freddi."""
    import pandas as pd

    preds = pd.read_csv(predictions_path)
    sample = build_explanation_sample(preds, sample_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    return len(sample)


def build_final_results(predictions, explanations):
    """Shape Jack's final_results.csv from Nadi predictions and Freddi output."""
    predictions = _test_rows(predictions)
    expl = explanations[["article_id", "explanation", "manual_score"]]
    final = predictions.merge(expl, on="article_id", how="left")[FINAL_RESULTS_COLUMNS]
    final["explanation"] = final["explanation"].fillna("")
    final["manual_score"] = final["manual_score"].astype("Int64")
    return final
