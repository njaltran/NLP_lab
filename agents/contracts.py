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

LABELS = ("up", "down", "neutral")
PROBABILITY_TOLERANCE = 0.02

PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_up", "prob_down", "prob_neutral", "split",
]

EXPLANATION_SAMPLE_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "prob_up", "prob_down", "prob_neutral",
]

EXPLANATION_OUTPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "explanation", "manual_score",
]

EXPLANATION_PASSTHROUGH_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label", "confidence",
]

FINAL_RESULTS_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence", "explanation", "manual_score",
]


def read_prediction_rows(path: str | os.PathLike[str]) -> list[dict]:
    """Read predictions_test.csv and enforce the Handoff 2 column order."""
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
    """Validate the Handoff 2 fields needed before scoring or finalizing."""
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
    """Return only test rows from a predictions DataFrame."""
    if "split" not in predictions.columns:
        raise ValueError("predictions file has no split column (Handoff 2)")
    test = predictions[predictions["split"] == "test"]
    if test.empty:
        raise ValueError("predictions file has no split=test rows")
    return test


def build_explanation_sample(predictions, sample_size: int = 300):
    """Return Jack's Handoff 4 sample DataFrame from Nadi predictions."""
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
    """Write sample_for_explanation.csv and return the number of rows written."""
    import pandas as pd

    preds = pd.read_csv(predictions_path)
    sample = build_explanation_sample(preds, sample_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    return len(sample)


def build_final_results(predictions, explanations):
    """Return Handoff 6 final_results rows from predictions and explanations."""
    predictions = _test_rows(predictions)
    expl = explanations[["article_id", "explanation", "manual_score"]]
    final = predictions.merge(expl, on="article_id", how="left")[FINAL_RESULTS_COLUMNS]
    final["explanation"] = final["explanation"].fillna("")
    final["manual_score"] = final["manual_score"].astype("Int64")
    return final
