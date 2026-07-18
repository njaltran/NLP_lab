"""Tests for the reusable per-head FinBERT training engine
(agents/finbert_finetuner.py). Nadi calls this directly to train the up-head
and down-head checkpoints (ADR 0001, ADR 0002).

The fast tests only check pure, model-free logic (row filtering, label
derivation). Tests that actually train are marked `slow` and skipped by
default.
"""

import json

import pandas as pd
import pytest
from transformers import AutoConfig

from agents.finbert_finetuner import rows_for_head, train_finbert

PROCESSED_DATA = "mock_data/processed_data.csv"


def test_up_head_keeps_only_up_and_neutral_rows():
    frame = pd.read_csv(PROCESSED_DATA)

    filtered = rows_for_head(frame, "up")

    assert set(filtered["label"]) <= {"up", "neutral"}
    assert (filtered["label"] == "up").sum() == (frame["label"] == "up").sum()
    assert (filtered["label"] == "neutral").sum() == (frame["label"] == "neutral").sum()
    assert "down" not in set(filtered["label"])


def test_down_head_keeps_only_down_and_neutral_rows():
    frame = pd.read_csv(PROCESSED_DATA)

    filtered = rows_for_head(frame, "down")

    assert set(filtered["label"]) <= {"down", "neutral"}
    assert (filtered["label"] == "down").sum() == (frame["label"] == "down").sum()
    assert (filtered["label"] == "neutral").sum() == (frame["label"] == "neutral").sum()
    assert "up" not in set(filtered["label"])


def test_filtering_preserves_other_columns():
    frame = pd.read_csv(PROCESSED_DATA)

    filtered = rows_for_head(frame, "up")

    assert list(filtered.columns) == list(frame.columns)


def test_invalid_head_raises():
    frame = pd.read_csv(PROCESSED_DATA)

    with pytest.raises(ValueError, match="head must be 'up' or 'down'"):
        rows_for_head(frame, "neutral")


@pytest.mark.slow
def test_train_finbert_publishes_a_binary_head_checkpoint(tmp_path):
    """End-to-end smoke: a tiny training run for the up-head publishes a
    genuine 2-class checkpoint (up vs neutral) and a matching report."""
    out_dir = tmp_path / "up"

    result = train_finbert(
        data_path=PROCESSED_DATA,
        out_dir=str(out_dir),
        head="up",
        learning_rate=5e-6,
        focus_weight_multiplier=1.0,
        epochs=1,
        limit=8,
    )

    assert result["model_dir"] == str(out_dir)
    report = result["report"]
    assert report["head"] == "up"
    assert set(report["val_class_accuracy"]) == {"up", "neutral"}
    assert 0.0 <= report["best_val_accuracy"] <= 1.0

    config = AutoConfig.from_pretrained(out_dir)
    id2label = {int(k): v.lower() for k, v in config.id2label.items()}
    assert id2label == {0: "neutral", 1: "up"}

    report_path = out_dir / "training_report.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text())["head"] == "up"
