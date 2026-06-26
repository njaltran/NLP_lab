"""Regression tests for the Manager agent (Jack).

Freezes the manual checks from development: the full retune→sample→finalize
lifecycle, the accept/override gate, contract-faithful outputs, reducer
accumulation, and reproducible sampling. Runs offline against mock_data/
(no HF_TOKEN → rationale node uses its deterministic fallback).
"""

import json

import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver

import agents.jack_manager as jm

PRED = "mock_data/predictions_test.csv"
EXPL = "mock_data/explanations.csv"
BASE = {"target_accuracy": 0.60, "max_iterations": 5, "predictions_path": PRED}

RETUNE_REPORT = {
    "accuracy": 0.54,
    "below_threshold": True,
    "class_accuracy": {"up": 0.60, "down": 0.40, "neutral": 0.30},
    "misclassified_ids": ["FNSPID_00006", "FNSPID_00010"],
    "proposal": {
        "recommended_action": "retune",
        "reason": "accuracy 0.54 below target",
        "focus_labels": ["down", "neutral"],
        "suggested_params": {"threshold": 0.5, "max_length": 128},
        "code_notes": "",
    },
}


@pytest.fixture
def proceed_report():
    with open("mock_data/evaluation_report.json") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def outdir(tmp_path, monkeypatch):
    """Redirect every file write to a temp dir so tests never touch outputs/."""
    monkeypatch.setattr(jm, "OUTPUT_DIR", str(tmp_path))
    return tmp_path


def _graph():
    return jm.build_graph(MemorySaver())


# --- lifecycle ------------------------------------------------------------

def test_full_lifecycle(outdir, proceed_report):
    g, cfg = _graph(), {"configurable": {"thread_id": "t"}}

    r1 = g.invoke({**BASE, "evaluation_report": RETUNE_REPORT}, cfg)
    assert (r1["iteration"], r1["final_action"], r1["decision"]) == (1, "retune", "accept")
    assert (outdir / "retune_request.json").exists()

    r2 = g.invoke({**BASE, "evaluation_report": proceed_report}, cfg)
    assert (r2["iteration"], r2["final_action"]) == (2, "proceed")
    assert (outdir / "sample_for_explanation.csv").exists()

    r3 = g.invoke({**BASE, "evaluation_report": proceed_report, "explanations_path": EXPL}, cfg)
    assert r3["iteration"] == 2          # finalize is NOT a new loop iteration
    assert (outdir / "final_results.csv").exists()
    assert (outdir / "final_report.json").exists()
    assert len(r3["decision_log"]) == 6  # reducer accumulated: 2 entries × 3 invocations


# --- gate: accept vs override --------------------------------------------

def test_override_when_cap_forces_proceed():
    state = {
        "evaluation_report": {"accuracy": 0.54, "proposal": {"recommended_action": "retune"}},
        "target_accuracy": 0.60, "max_iterations": 5,
        "iteration": 4, "final_action": "retune",   # restored from prior round
    }
    out = jm.decide(state)
    assert (out["iteration"], out["final_action"]) == (5, "proceed")
    assert out["decision"] == "override"
    assert out["overrides"] == {"final_action": "proceed"}


def test_accept_when_gate_agrees_with_proposal():
    state = {
        "evaluation_report": {"accuracy": 0.70, "proposal": {"recommended_action": "proceed"}},
        "target_accuracy": 0.60, "max_iterations": 5,
    }
    out = jm.decide(state)
    assert out["decision"] == "accept"
    assert out["overrides"] == {}


# --- contract conformance -------------------------------------------------

def test_outputs_match_contract(outdir, proceed_report):
    g, cfg = _graph(), {"configurable": {"thread_id": "c"}}
    g.invoke({**BASE, "evaluation_report": proceed_report}, cfg)                       # sample
    g.invoke({**BASE, "evaluation_report": proceed_report, "explanations_path": EXPL}, cfg)  # finalize

    samp = pd.read_csv(outdir / "sample_for_explanation.csv")
    assert list(samp.columns) == [
        "article_id", "article_title", "predicted_label", "actual_label",
        "confidence", "prob_up", "prob_down", "prob_neutral"]
    assert set(samp.predicted_label) <= {"up", "down", "neutral"}

    fin = pd.read_csv(outdir / "final_results.csv")
    assert list(fin.columns) == [
        "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
        "pct_change", "label", "predicted_label", "confidence", "explanation", "manual_score"]
    assert fin.date.str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert set(fin.label) <= {"up", "down", "neutral"}

    rep = json.loads((outdir / "final_report.json").read_text())
    assert set(rep) == {
        "final_accuracy", "loop_iterations", "class_accuracy",
        "test_set_size", "explanations_generated", "manually_scored"}
    assert rep["test_set_size"] == len(fin)
    assert rep["explanations_generated"] == int((fin.explanation.fillna("") != "").sum())
    assert rep["manually_scored"] == int(fin.manual_score.notna().sum())


# --- public ManagerAgent.run() API ---------------------------------------

def test_manager_agent_run_lifecycle(outdir, tmp_path, proceed_report):
    """The file-path wrapper drives the same retune→sample→finalize loop, with
    one instance carrying iteration state across runs via its checkpointer."""
    retune_path = tmp_path / "eval_retune.json"
    retune_path.write_text(json.dumps(RETUNE_REPORT))
    proceed_path = "mock_data/evaluation_report.json"
    mgr = jm.ManagerAgent(predictions_path=PRED, thread_id="agent")

    r1 = mgr.run(evaluation_report=str(retune_path))
    assert (r1["iteration"], r1["final_action"]) == (1, "retune")
    assert (outdir / "retune_request.json").exists()

    r2 = mgr.run(evaluation_report=proceed_path)
    assert (r2["iteration"], r2["final_action"]) == (2, "proceed")
    assert (outdir / "sample_for_explanation.csv").exists()

    r3 = mgr.run(evaluation_report=proceed_path, explanations=EXPL)
    assert r3["iteration"] == 2          # finalize is NOT a new loop iteration
    assert (outdir / "final_results.csv").exists()
    assert (outdir / "final_report.json").exists()


# --- reproducible sampling ------------------------------------------------

def test_sampling_is_reproducible(outdir):
    state = {
        "predictions_path": PRED, "sample_size": 3, "iteration": 2,
        "decision": "accept", "final_action": "proceed", "notes": "x",
        "evaluation_report": {"accuracy": 0.67, "proposal": {}},
    }
    jm.proceed(state)
    first = pd.read_csv(outdir / "sample_for_explanation.csv")["article_id"].tolist()
    jm.proceed(state)
    second = pd.read_csv(outdir / "sample_for_explanation.csv")["article_id"].tolist()
    assert first == second and len(first) == 3
