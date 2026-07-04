"""Tests for the fine-tuning script's data split (agents/finetune_finbert.py).

The fast tests only check the split logic (no model download). The smoke test
that actually trains is marked `slow` and is skipped by default.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from agents.finetune_finbert import split_frames

PROCESSED_DATA = "mock_data/processed_data.csv"


def test_split_is_time_ordered_no_leakage():
    """train dates come before val dates come before test dates — so nothing
    from the future ever reaches training."""
    df = pd.read_csv(PROCESSED_DATA)
    train, val, test = split_frames(df)

    assert len(train) > 0 and len(test) > 0
    train_max = pd.to_datetime(train["date"]).max()
    test_min = pd.to_datetime(test["date"]).min()
    assert train_max <= test_min

    if len(val) > 0:
        val_min = pd.to_datetime(val["date"]).min()
        val_max = pd.to_datetime(val["date"]).max()
        assert train_max <= val_min
        assert val_max <= test_min


def test_missing_split_column_raises():
    """Without Aurora's split column the script must stop with a clear error."""
    df = pd.DataFrame({"date": ["2019-01-01"], "article_title": ["x"], "label": ["up"]})
    with pytest.raises(SystemExit):
        split_frames(df)


@pytest.mark.slow
def test_finetune_smoke_writes_report(tmp_path):
    """End-to-end smoke: a tiny training run completes and writes a valid
    finetune_report.json. Marked slow because it downloads/loads FinBERT."""
    out_dir = tmp_path / "model"
    result = subprocess.run(
        [sys.executable, "agents/finetune_finbert.py",
         "--data", PROCESSED_DATA, "--out-dir", str(out_dir),
         "--epochs", "1", "--limit", "8"],
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, result.stderr

    report_path = tmp_path / "finetune_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "test_accuracy" in report
    assert set(report["test_class_accuracy"]) == {"up", "down", "neutral"}
