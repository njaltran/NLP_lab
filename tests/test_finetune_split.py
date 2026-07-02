"""Tests for the fine-tune increment's plumbing: Aurora's time-based split and
the finetune script's frame splitting. Model training itself is exercised by the
opt-in smoke test at the bottom (deselected by default via the `slow` marker).
"""

import pandas as pd
import pytest

from agents.aurora_processing import _assign_split
from agents.finetune_finbert import _split_frames


def _frame(dates):
    labels = ["up", "down", "neutral"]
    return pd.DataFrame({
        "date": dates,
        "article_title": [f"headline {i}" for i in range(len(dates))],
        "label": [labels[i % 3] for i in range(len(dates))],
    })


def test_split_is_date_monotonic():
    dates = [f"2019-{m:02d}-15" for m in range(1, 11)]
    df = _assign_split(_frame(dates))
    train_max = pd.to_datetime(df[df["split"] == "train"]["date"]).max()
    test_min = pd.to_datetime(df[df["split"] == "test"]["date"]).min()
    assert train_max < test_min
    assert set(df["split"]) == {"train", "test"}


def test_dataset_end_drops_later_rows():
    dates = ["2019-06-01", "2019-12-30", "2020-03-01", "2020-06-01"]
    df = _assign_split(_frame(dates), dataset_end="2019-12-31")
    assert len(df) == 2
    assert pd.to_datetime(df["date"]).max() <= pd.Timestamp("2019-12-31")


def test_split_frames_validation_is_time_ordered():
    dates = [f"2019-{m:02d}-01" for m in range(1, 11)]
    df = _assign_split(_frame(dates))
    train, val, test = _split_frames(df)
    assert len(val) > 0
    # val comes after train in time, and both precede test
    assert pd.to_datetime(train["date"]).max() < pd.to_datetime(val["date"]).min()
    assert pd.to_datetime(val["date"]).max() <= pd.to_datetime(test["date"]).min()


def test_split_frames_requires_split_column():
    with pytest.raises(SystemExit, match="split"):
        _split_frames(pd.DataFrame({"date": ["2019-01-01"], "label": ["up"]}))


@pytest.mark.slow
def test_finetune_smoke(tmp_path, monkeypatch):
    """End-to-end 20-row training run. Slow (downloads FinBERT, trains 1 epoch):
    run explicitly with `uv run pytest -m slow tests/test_finetune_split.py`."""
    import sys
    from agents import finetune_finbert

    dates = ([f"2019-{m:02d}-{d:02d}" for m in range(1, 11) for d in (1, 15)])
    df = _assign_split(_frame(dates))
    data = tmp_path / "processed_data.csv"
    df.to_csv(data, index=False)
    out_dir = tmp_path / "finbert_finetuned"

    monkeypatch.setattr(sys, "argv", [
        "finetune_finbert.py", "--data", str(data), "--out-dir", str(out_dir),
        "--epochs", "1", "--batch-size", "4", "--limit", "20",
    ])
    finetune_finbert.main()

    import json
    report = json.loads((tmp_path / "finetune_report.json").read_text())
    assert 0.0 <= report["test_accuracy"] <= 1.0
    assert (out_dir / "config.json").exists()
