"""Tests for Nadi's Classifier Agent.
"""

import json
import os
import shutil
import pytest
import pandas as pd
from agents.nadi_classifier import ClassifierAgent

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
        
    assert (df["split"] == "test").all()
    assert set(df["predicted_label"]) <= {"up", "down", "neutral"}

def test_classifier_agent_retune(outdir):
    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"
    retune_path = outdir / "retune_request.json"

    retune_data = {
        "iteration": 1,
        "suggested_params": {
            "threshold": 0.65,
            "max_length": 64,
            "boost_factor": 1.5
        },
        "focus_labels": ["down"]
    }
    with open(retune_path, "w", encoding="utf-8") as f:
        json.dump(retune_data, f)

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=PROCESSED_DATA,
        classifier_code=str(code_path),
        predictions=str(pred_path),
        retune_request=str(retune_path)
    )

    # Check metadata in state
    assert res["classifier_metadata"]["fine_tuning_params"]["threshold"] == 0.65
    assert res["classifier_metadata"]["fine_tuning_params"]["max_length"] == 64
    assert res["classifier_metadata"]["fine_tuning_params"]["focus_labels"] == ["down"]
    assert res["classifier_metadata"]["fine_tuning_params"]["boost_factor"] == 1.5

    # Verify classifier.py was updated with new values
    with open(res["classifier_code_path"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "THRESHOLD = 0.65" in content
        assert "MAX_LENGTH = 64" in content
        assert "FOCUS_LABELS = ['down']" in content
        assert "BOOST_FACTOR = 1.5" in content

    # Verify this iteration's code was archived, named by retune_request's iteration
    assert res["classifier_history_path"].endswith("classifier_iter1.py")
    with open(res["classifier_history_path"], "r", encoding="utf-8") as f:
        assert f.read() == content


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
        json.dump({"iteration": 1, "suggested_params": {"threshold": 0.4}}, f)
    second = agent.run(processed_data=PROCESSED_DATA, classifier_code=str(code_path),
                       predictions=str(pred_path), retune_request=str(retune_path))
    assert second["classifier_history_path"].endswith("classifier_iter1.py")

    # Both archived copies exist even though classifier.py itself was overwritten.
    assert os.path.exists(first["classifier_history_path"])
    assert os.path.exists(second["classifier_history_path"])
    assert first["classifier_history_path"] != second["classifier_history_path"]

def test_generated_code_prefers_finetuned_model_when_present(outdir, monkeypatch):
    """generate_code points the script at outputs/finbert_finetuned when the dir
    exists, and falls back to the pretrained hub model (MODEL_DIR = None) when
    it doesn't. Template-level check — no model download."""
    import agents.nadi_classifier as nc
    from agents.nadi_classifier import generate_code

    # Isolate from the repo's real outputs/ (a genuine fine-tuned dir may exist there)
    monkeypatch.setattr(nc, "OUTPUT_DIR", str(outdir))

    # No fine-tuned dir -> fallback
    code_path = outdir / "classifier.py"
    res = generate_code({"classifier_code_path": str(code_path)})
    content = code_path.read_text()
    assert "MODEL_DIR = None" in content
    assert res["classifier_metadata"]["model_name"] == "ProsusAI/finbert"

    # Fine-tuned dir present -> preferred
    model_dir = outdir / "finbert_finetuned"
    model_dir.mkdir()
    res = generate_code({"classifier_code_path": str(code_path),
                         "model_dir": str(model_dir)})
    content = code_path.read_text()
    assert f"MODEL_DIR = {str(model_dir)!r}" in content
    assert res["classifier_metadata"]["model_name"] == str(model_dir)


def test_predictions_only_contain_test_split_rows(outdir):
    """With Aurora's split column in the input, the generated classifier predicts
    only split=test rows (train rows were seen in fine-tuning)."""
    df = pd.read_csv(PROCESSED_DATA)
    assert "split" in df.columns and (df["split"] == "train").any()

    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"
    ClassifierAgent().run(processed_data=PROCESSED_DATA,
                          classifier_code=str(code_path), predictions=str(pred_path))

    out = pd.read_csv(pred_path)
    assert len(out) == (df["split"] == "test").sum()
    assert (out["split"] == "test").all()
    assert set(out["article_id"]) == set(df[df["split"] == "test"]["article_id"])
    # split stays the last column per the contract
    assert list(out.columns)[-1] == "split"


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
