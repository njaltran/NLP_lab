"""Tests for the reusable per-head FinBERT training engine
(agents/finbert_finetuner.py). Nadi calls this directly to train the up-head
and down-head checkpoints (ADR 0001, ADR 0002).

The fast tests only check pure, model-free logic (row filtering, label
derivation). Tests that actually train are marked `slow` and skipped by
default.
"""

import pandas as pd
import pytest

from agents.finbert_finetuner import rows_for_head

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
