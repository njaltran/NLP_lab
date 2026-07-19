"""Tests for Nadi owning fine-tuning and per-head hyperparameter tuning
(ADR 0001). `next_head_training_params` is the pure scheduling function: given
one directional head's own prior attempts, pick the next learning rate and
focus weight. It has no notion of the OTHER head — each head's schedule is
independent (ADR 0002 decision to split the heads applies to tuning too).
"""

import json
import os
from pathlib import Path

import pytest

from agents.nadi_classifier import (
    TRAINING_LR_FLOOR,
    TRAINING_LR_START,
    TRAINING_MULTIPLIER_MAX,
    ClassifierAgent,
    _heads_needing_training,
    fine_tune,
    next_head_training_params,
)

PROCESSED_DATA = "mock_data/processed_data.csv"


def test_first_attempt_starts_at_the_base_learning_rate():
    params = next_head_training_params([], collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START
    assert params["focus_weight_multiplier"] == 1.25


def test_first_attempt_uses_a_higher_focus_weight_when_collapsed():
    params = next_head_training_params([], collapsed=True)

    assert params["focus_weight_multiplier"] == 1.5


def test_regression_backs_off_learning_rate_and_focus_weight():
    history = [{"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.5,
               "regressed": True}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START * 0.5
    assert params["focus_weight_multiplier"] == 1.25


def test_regression_floors_the_learning_rate():
    history = [{"learning_rate": TRAINING_LR_FLOOR, "focus_weight_multiplier": 1.5,
               "regressed": True}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_FLOOR


def test_no_regression_holds_learning_rate_and_steps_focus_weight_up():
    history = [{"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.25,
               "regressed": False}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START
    assert params["focus_weight_multiplier"] == 1.5


def test_focus_weight_never_exceeds_the_max():
    history = [{"learning_rate": TRAINING_LR_START,
               "focus_weight_multiplier": TRAINING_MULTIPLIER_MAX, "regressed": False}]

    params = next_head_training_params(history, collapsed=False)

    assert params["focus_weight_multiplier"] == TRAINING_MULTIPLIER_MAX


def test_only_the_most_recent_attempt_matters():
    """Older attempts don't influence the next pick — only what happened last."""
    history = [
        {"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.25, "regressed": True},
        {"learning_rate": TRAINING_LR_START * 0.5, "focus_weight_multiplier": 1.0, "regressed": False},
    ]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START * 0.5
    assert params["focus_weight_multiplier"] == 1.25


# --- Which heads train this pass -----------------------------------------------

def test_neither_head_trained_yet_trains_both_regardless_of_request():
    """Iteration 0 has no checkpoints at all — the baseline pass trains both
    heads even if Manager only flagged one (there's nothing to classify with
    otherwise)."""
    assert _heads_needing_training(requested=["up"], already_trained=set()) == {"up", "down"}
    assert _heads_needing_training(requested=[], already_trained=set()) == {"up", "down"}


def test_once_both_trained_only_the_flagged_head_retrains():
    already = {"up", "down"}

    assert _heads_needing_training(requested=["down"], already_trained=already) == {"down"}
    assert _heads_needing_training(requested=[], already_trained=already) == set()


def test_fine_tune_trains_at_the_same_max_length_the_classifier_infers_at(monkeypatch):
    """train_finbert's max_length must match CLASSIFIER_TEMPLATE's MAX_LENGTH --
    otherwise every head trains truncated at one length while inference
    truncates at another."""
    import agents.nadi_classifier as nc

    captured = {}

    def fake_train_finbert(**kwargs):
        captured.update(kwargs)
        return {"model_dir": kwargs["out_dir"],
                "report": {"val_class_accuracy": {kwargs["head"]: 0.5}}}

    monkeypatch.setattr(nc, "train_finbert", fake_train_finbert)

    nc.fine_tune({
        "processed_data_path": "mock_data/processed_data.csv",
        "finetuned_base_dir": "/tmp/does-not-matter",
        "heads_to_retrain": ["up"], "collapsed_heads": [],
        "up_model_dir": None, "down_model_dir": None,
    })

    assert captured["max_length"] == nc.MAX_LENGTH


# --- fine_tune node (integration; trains real tiny checkpoints) ---------------
#
# train_finbert's val split needs at least one row of a head's own class AND
# one neutral row to be meaningful (see agents/finbert_finetuner.py). The
# shared mock_data/processed_data.csv's 2-row val split is entirely `up`, so
# the down-head's filtered val split is empty there. Rather than resize a
# fixture other tests depend on, these tests use their own small dataset with
# a val split that covers both heads.

_HEAD_TEST_DATA_HEADER = "article_id,date,ticker,article_title,price_t,price_t1,pct_change,label,split\n"
_HEAD_TEST_DATA_ROWS = [
    ("t1", "2021-01-01", "AAPL", "headline one", 100, 103, 3.0, "up", "train"),
    ("t2", "2021-01-02", "AAPL", "headline two", 100, 97, -3.0, "down", "train"),
    ("t3", "2021-01-03", "AAPL", "headline three", 100, 100, 0.0, "neutral", "train"),
    ("t4", "2021-01-04", "AAPL", "headline four", 100, 104, 4.0, "up", "train"),
    ("t5", "2021-01-05", "AAPL", "headline five", 100, 96, -4.0, "down", "train"),
    ("t6", "2021-01-06", "AAPL", "headline six", 100, 100, 0.0, "neutral", "train"),
    ("t7", "2021-01-07", "AAPL", "headline seven", 100, 102, 2.0, "up", "val"),
    ("t8", "2021-01-08", "AAPL", "headline eight", 100, 98, -2.0, "down", "val"),
    ("t9", "2021-01-09", "AAPL", "headline nine", 100, 100, 0.0, "neutral", "val"),
]


@pytest.fixture
def two_head_data(tmp_path):
    path = tmp_path / "processed_data.csv"
    lines = [_HEAD_TEST_DATA_HEADER]
    lines += [",".join(str(v) for v in row) + "\n" for row in _HEAD_TEST_DATA_ROWS]
    path.write_text("".join(lines))
    return str(path)


@pytest.mark.slow
def test_fine_tune_bootstrap_trains_both_heads_and_records_history(tmp_path, two_head_data):
    state = {
        "processed_data_path": two_head_data,
        "finetuned_base_dir": str(tmp_path / "finbert_finetuned"),
        "heads_to_retrain": [],
        "collapsed_heads": [],
        "up_model_dir": None,
        "down_model_dir": None,
    }

    out = fine_tune(state)

    assert os.path.isdir(out["up_model_dir"])
    assert os.path.isdir(out["down_model_dir"])
    assert len(out["up_training_history"]) == 1
    assert len(out["down_training_history"]) == 1
    assert out["up_training_history"][0]["learning_rate"] == TRAINING_LR_START
    assert "regressed" in out["up_training_history"][0]


@pytest.mark.slow
def test_fine_tune_only_retrains_the_flagged_head(tmp_path, two_head_data):
    base = str(tmp_path / "finbert_finetuned")
    bootstrap = fine_tune({
        "processed_data_path": two_head_data, "finetuned_base_dir": base,
        "heads_to_retrain": [], "collapsed_heads": [],
        "up_model_dir": None, "down_model_dir": None,
    })
    down_mtime_before = os.path.getmtime(
        os.path.join(bootstrap["down_model_dir"], "training_report.json"))

    second = fine_tune({
        "processed_data_path": two_head_data, "finetuned_base_dir": base,
        "heads_to_retrain": ["up"], "collapsed_heads": [],
        "up_model_dir": bootstrap["up_model_dir"],
        "down_model_dir": bootstrap["down_model_dir"],
        "up_training_history": bootstrap["up_training_history"],
        "down_training_history": bootstrap["down_training_history"],
    })

    assert "up_training_history" in second
    assert "down_training_history" not in second
    assert "down_model_dir" not in second
    down_mtime_after = os.path.getmtime(
        os.path.join(bootstrap["down_model_dir"], "training_report.json"))
    assert down_mtime_after == down_mtime_before  # untouched this pass


@pytest.mark.slow
def test_classifier_agent_run_fine_tunes_when_flagged(tmp_path, two_head_data):
    """End-to-end: ClassifierAgent.run() with heads_to_retrain in the retune
    request trains checkpoints, then generates and runs a classifier.py that
    actually loads both heads and produces contract-shaped predictions --
    proving the two-head template (Task 6) is wired to fine_tune's output
    (Task 5), not just that checkpoints happen to exist on disk."""
    import pandas as pd

    retune_path = tmp_path / "retune_request.json"
    retune_path.write_text(json.dumps({"iteration": 1, "heads_to_retrain": ["up", "down"]}))

    agent = ClassifierAgent()
    pred_path = tmp_path / "predictions_test.csv"
    res = agent.run(
        processed_data=two_head_data,
        classifier_code=str(tmp_path / "classifier.py"),
        predictions=str(pred_path),
        retune_request=str(retune_path),
        finetuned_base_dir=str(tmp_path / "finbert_finetuned"),
    )

    assert os.path.isdir(res["up_model_dir"])
    assert os.path.isdir(res["down_model_dir"])

    code = Path(res["classifier_code_path"]).read_text()
    assert f"UP_MODEL_DIR = {res['up_model_dir']!r}" in code
    assert f"DOWN_MODEL_DIR = {res['down_model_dir']!r}" in code

    df = pd.read_csv(pred_path)
    assert set(df["predicted_label"]) <= {"up", "down", "neutral"}
    assert (df["prob_up"] + df["prob_down"] + df["prob_neutral"]).round(2).eq(1.0).all()
