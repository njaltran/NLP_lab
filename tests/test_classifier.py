"""Tests for Nadi's Classifier Agent.
"""

import inspect
import json
import os
import pytest
import pandas as pd
from agents.contracts import combine_binary_probs
from agents.nadi_classifier import ClassifierAgent, generate_code

PROCESSED_DATA = "mock_data/processed_data.csv"

@pytest.fixture
def outdir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path

def test_classifier_agent_run(outdir):
    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=PROCESSED_DATA,
        classifier_code=str(code_path),
        predictions=str(pred_path)
    )

    assert os.path.exists(res["classifier_code_path"])
    assert os.path.exists(res["predictions_path"])

    # Verify predictions_test.csv matches data contract
    df = pd.read_csv(res["predictions_path"])
    expected_cols = [
        "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
        "pct_change", "label", "predicted_label", "confidence",
        "prob_up", "prob_down", "prob_neutral", "split"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Held-out rows only: val (loop scoring) + test (final report); never train.
    assert df["split"].isin(["val", "test"]).all()
    assert (df["split"] == "test").any()
    assert set(df["predicted_label"]) <= {"up", "down", "neutral"}


def test_classifier_to_evaluator_integration(outdir):
    from agents.sabina_evaluator import EvaluatorAgent

    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"

    # 1. Run Nadi's Classifier Agent
    classifier = ClassifierAgent()
    classifier.run(
        processed_data=PROCESSED_DATA,
        classifier_code=str(code_path),
        predictions=str(pred_path)
    )

    # 2. Run Sabina's Evaluator Agent on the exact outputs
    evaluator = EvaluatorAgent(output_dir=str(outdir))
    eval_res = evaluator.run(
        predictions=str(pred_path),
        classifier_code=str(code_path)
    )

    # 3. Assert Evaluator Agent output matches data contracts Handoff 3
    assert os.path.exists(eval_res["output_path"])

    with open(eval_res["output_path"], "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "accuracy" in report
    assert "below_threshold" in report
    assert "class_accuracy" in report
    assert "proposal" in report
    assert report["proposal"]["recommended_action"] in ["retune", "proceed"]


def test_classifier_archives_each_iteration_separately(outdir):
    """Past retune attempts must survive classifier.py being overwritten."""
    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"
    agent = ClassifierAgent()

    first = agent.run(processed_data=PROCESSED_DATA, classifier_code=str(code_path),
                      predictions=str(pred_path))
    assert first["classifier_history_path"].endswith("classifier_iter0.py")

    retune_path = outdir / "retune_request.json"
    with open(retune_path, "w", encoding="utf-8") as f:
        json.dump({"iteration": 1, "heads_to_retrain": ["up", "down"]}, f)
    second = agent.run(processed_data=PROCESSED_DATA, classifier_code=str(code_path),
                       predictions=str(pred_path), retune_request=str(retune_path),
                       finetuned_base_dir=str(outdir / "finbert_finetuned"))
    assert second["classifier_history_path"].endswith("classifier_iter1.py")

    # Both archived copies exist even though classifier.py itself was overwritten.
    assert os.path.exists(first["classifier_history_path"])
    assert os.path.exists(second["classifier_history_path"])
    assert first["classifier_history_path"] != second["classifier_history_path"]


# --- Two-head codegen (ADR 0002, ADR 0003) -------------------------------------

def test_generate_code_defaults_to_pretrained(outdir):
    """With neither head trained, the generated code uses the pretrained
    FinBERT sentiment-translation branch."""
    code_path = outdir / "classifier.py"
    res = generate_code({"classifier_code_path": str(code_path)})
    content = code_path.read_text()
    assert "UP_MODEL_DIR = None" in content
    assert "DOWN_MODEL_DIR = None" in content
    assert res["classifier_metadata"]["model_name"] == "ProsusAI/finbert"


def test_generate_code_uses_two_head_template_when_both_present(outdir):
    """When both directional heads have published checkpoints, the generated
    code loads both and combines them via the shared combination rule."""
    up_dir = outdir / "up"
    down_dir = outdir / "down"
    up_dir.mkdir()
    down_dir.mkdir()
    code_path = outdir / "classifier.py"

    res = generate_code({
        "classifier_code_path": str(code_path),
        "up_model_dir": str(up_dir),
        "down_model_dir": str(down_dir),
    })

    content = code_path.read_text()
    assert f"UP_MODEL_DIR = {str(up_dir)!r}" in content
    assert f"DOWN_MODEL_DIR = {str(down_dir)!r}" in content
    # The combination rule's source is injected verbatim -- one definition,
    # shared by the trainer and every generated classifier (no drift).
    assert inspect.getsource(combine_binary_probs) in content
    assert str(up_dir) in res["classifier_metadata"]["model_name"]
    assert str(down_dir) in res["classifier_metadata"]["model_name"]


def test_generate_code_falls_back_to_pretrained_when_only_one_head_present(outdir):
    """A lone head (e.g. a stale directory, or mid-bootstrap) isn't enough --
    the two-head branch needs both, so this stays on pretrained rather than
    guessing at the missing head."""
    up_dir = outdir / "up"
    up_dir.mkdir()
    code_path = outdir / "classifier.py"

    res = generate_code({
        "classifier_code_path": str(code_path),
        "up_model_dir": str(up_dir),
        "down_model_dir": str(outdir / "does_not_exist"),
    })

    content = code_path.read_text()
    assert "UP_MODEL_DIR = None" in content
    assert "DOWN_MODEL_DIR = None" in content
    assert res["classifier_metadata"]["model_name"] == "ProsusAI/finbert"
