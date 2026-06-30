"""Tests for Nadi's Classifier Agent.
"""

import json
import os
import shutil
import pytest
import pandas as pd
import importlib.util
import socket
from agents.nadi_classifier import ClassifierAgent, generate_code

PROCESSED_DATA = "mock_data/processed_data.csv"

@pytest.fixture
def outdir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path

def _check_torch_and_model():
    # 1. Check if torch and transformers are installed
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False, "torch or transformers not installed"
    
    # 2. Check if the model can be loaded offline (is cached) or online (we have network)
    try:
        from transformers import AutoTokenizer
        # Try to load with local_files_only=True. If it succeeds, the model is cached!
        AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
        return True, "Model is cached locally"
    except Exception:
        # If it fails, check if we have internet connectivity to download it.
        try:
            # 2 second timeout for DNS resolve + connect
            socket.create_connection(("huggingface.co", 443), timeout=2.0)
            return True, "Hugging Face is reachable"
        except Exception:
            return False, "Model is not cached locally and Hugging Face is unreachable"

_CAN_RUN_INFERENCE, _REASON = _check_torch_and_model()
skip_inference = pytest.mark.skipif(not _CAN_RUN_INFERENCE, reason=_REASON)

def test_classifier_code_generation_default(outdir):
    """Test default code generation (offline)."""
    code_path = outdir / "classifier.py"
    state = {
        "classifier_code_path": str(code_path)
    }
    res = generate_code(state)
    assert os.path.exists(res["classifier_code_path"])
    
    # Verify default template parameters
    with open(res["classifier_code_path"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "THRESHOLD = 0.5" in content
        assert "MAX_LENGTH = 128" in content
        assert "FOCUS_LABELS = []" in content

def test_classifier_agent_retune(outdir):
    """Test code generation with retune parameters (offline)."""
    code_path = outdir / "classifier.py"
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

    state = {
        "classifier_code_path": str(code_path),
        "retune_request": retune_data
    }
    res = generate_code(state)

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

@skip_inference
def test_classifier_agent_run(outdir):
    """Test agent execution path (requires torch & model/network)."""
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

@skip_inference
def test_classifier_to_evaluator_integration(outdir):
    """Test integration from classifier execution to evaluator (requires torch & model/network)."""
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


def test_classifier_agent_run_missing_retune_request(outdir):
    """Test that a missing retune_request prints a warning and sets 'no retune applied' in state."""
    from unittest.mock import patch
    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"
    retune_path = outdir / "nonexistent_retune_request.json"

    agent = ClassifierAgent()
    with patch("subprocess.run") as mock_run:
        res = agent.run(
            processed_data=PROCESSED_DATA,
            classifier_code=str(code_path),
            predictions=str(pred_path),
            retune_request=str(retune_path)
        )
        assert mock_run.called
        assert res.get("retune_request") == "no retune applied"


def test_classifier_agent_run_malformed_retune_request(outdir):
    """Test that a malformed retune_request JSON raises an exception."""
    from unittest.mock import patch
    code_path = outdir / "classifier.py"
    pred_path = outdir / "predictions_test.csv"
    retune_path = outdir / "malformed_retune_request.json"

    # Write malformed json content
    with open(retune_path, "w", encoding="utf-8") as f:
        f.write("{invalid json")

    agent = ClassifierAgent()
    with patch("subprocess.run") as mock_run:
        with pytest.raises(Exception):
            agent.run(
                processed_data=PROCESSED_DATA,
                classifier_code=str(code_path),
                predictions=str(pred_path),
                retune_request=str(retune_path)
            )
