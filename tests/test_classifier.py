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
        "suggested_params": {
            "threshold": 0.65,
            "max_length": 64
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

    # Verify classifier.py was updated with new values
    with open(res["classifier_code_path"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "THRESHOLD = 0.65" in content
        assert "MAX_LENGTH = 64" in content
        assert "FOCUS_LABELS = ['down']" in content

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
