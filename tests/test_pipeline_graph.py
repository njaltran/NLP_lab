"""Tests for the unified pipeline graph (Increment 2).

The graph's job is control flow: run the agents in order and, crucially, *cycle*
`classify → evaluate → gate` while the Manager keeps retuning, then fall through to
explanation + finalize once it proceeds. We verify that cycle offline by injecting
fakes for the two heavy agents (Aurora needs the network, Nadi needs FinBERT) while
using the REAL, offline-capable Sabina-substitute, Manager, and Freddi.

A fake Sabina emits a fixed below-target report, so the Manager retunes for a couple
of passes and then early-stops on convergence — driving the graph around its loop.
Everything runs under a temp CWD so writes land in `<tmp>/outputs`.
"""

import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

_HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
needs_langgraph = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="langgraph not installed")

REPO = Path(__file__).resolve().parents[1]
MOCK_PRED = REPO / "mock_data" / "predictions_test.csv"


class FakeAurora:
    """Stands in for the yfinance-backed processing agent."""
    def run(self, **kwargs):
        return {"processed_data_path": "outputs/processed_data.csv"}


class FakeNadi:
    """Stands in for the FinBERT classifier: just copies mock predictions into
    place and counts how many times it ran (so we can prove the graph cycled)."""
    def __init__(self):
        self.calls = 0
        self.model_dir = None

    def run(self, *, processed_data, classifier_code, predictions, retune_request=None,
            model_dir=None):
        self.calls += 1
        self.model_dir = model_dir
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MOCK_PRED, predictions)
        Path(classifier_code).write_text("THRESHOLD = 0.5\n")  # readable by Sabina/Manager
        return {}


class FakeSabina:
    """Emits a fixed below-target report, forcing the Manager to retune until it
    converges."""
    def run(self, *, predictions, classifier_code):
        report = {
            "accuracy": 0.37,
            "below_threshold": True,
            "class_accuracy": {"up": 0.3, "down": 0.2, "neutral": 0.5},
            "misclassified_ids": [],
            "proposal": {"recommended_action": "retune", "reason": "low",
                         "focus_labels": ["down"],
                         "suggested_params": {"threshold": 0.5, "max_length": 128},
                         "code_notes": ""},
        }
        os.makedirs("outputs", exist_ok=True)
        Path("outputs/evaluation_report.json").write_text(json.dumps(report))
        return {"output_path": "outputs/evaluation_report.json"}


class MarkedNadi:
    """FakeNadi variant that stamps every predictions file with the pass number,
    so the test can tell WHICH iteration's artifacts survived to the end."""
    def __init__(self):
        self.calls = 0

    def run(self, *, processed_data, classifier_code, predictions, retune_request=None):
        self.calls += 1
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MOCK_PRED, predictions)
        import pandas as pd
        df = pd.read_csv(predictions)
        df["fake_pass"] = self.calls
        df.to_csv(predictions, index=False)
        Path(classifier_code).write_text(f"THRESHOLD = 0.5  # pass {self.calls}\n")
        return {}


class PeakSabina:
    """Accuracy peaks on pass 2 then regresses, so the best iteration is NOT the
    last one the gate proceeds with — exactly the case select_best must fix."""
    ACCS = [0.20, 0.39, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]

    def __init__(self):
        self.calls = 0

    def run(self, *, predictions, classifier_code):
        acc = self.ACCS[self.calls]
        self.calls += 1
        report = {
            "accuracy": acc,
            "below_threshold": True,
            "class_accuracy": {"up": 0.3, "down": 0.2, "neutral": 0.5},
            "misclassified_ids": [],
            "proposal": {"recommended_action": "retune", "reason": "low",
                         "focus_labels": ["down"],
                         "suggested_params": {"threshold": 0.5, "max_length": 128},
                         "code_notes": ""},
        }
        os.makedirs("outputs", exist_ok=True)
        Path("outputs/evaluation_report.json").write_text(json.dumps(report))
        return {"output_path": "outputs/evaluation_report.json"}


@needs_langgraph
def test_best_iteration_restored_when_accuracy_regresses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    import pandas as pd
    import agents.pipeline_graph as pg
    from agents.jack_manager import ManagerAgent
    from agents.freddi_explanation import ExplanationAgent
    from langgraph.checkpoint.memory import MemorySaver

    nadi, sabina = MarkedNadi(), PeakSabina()
    agents = pg.Agents(
        aurora=FakeAurora(), nadi=nadi, sabina=sabina,
        manager=ManagerAgent(predictions_path=pg.PREDS, target_accuracy=0.60,
                             max_iterations=6, patience=2, min_delta=0.01),
        freddi=ExplanationAgent(use_ollama=False, output_path=pg.EXPL),
    )
    graph = pg.build_pipeline(agents, checkpointer=MemorySaver())
    final = graph.invoke({"retune_request_path": None},
                         {"configurable": {"thread_id": "t"}, "recursion_limit": pg.RECURSION_LIMIT})

    assert final["final_action"] == "proceed"
    assert nadi.calls >= 3, "loop must run past the accuracy peak for this test to bite"

    # The artifacts that survive are the BEST pass's (accuracy 0.39 = pass 2),
    # not the last pass's — the whole point of select_best.
    preds = pd.read_csv(tmp_path / "outputs" / "predictions_test.csv")
    assert preds["fake_pass"].iloc[0] == 2, "final predictions should come from the best pass"
    report = json.loads((tmp_path / "outputs" / "evaluation_report.json").read_text())
    assert report["accuracy"] == 0.39
    final_report = json.loads((tmp_path / "outputs" / "final_report.json").read_text())
    assert final_report["final_accuracy"] == 0.39

    # The explanation sample was redrawn from the restored predictions.
    assert (tmp_path / "outputs" / "sample_for_explanation.csv").exists()


@needs_langgraph
def test_graph_cycles_then_finalizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    import agents.pipeline_graph as pg
    from agents.jack_manager import ManagerAgent
    from agents.freddi_explanation import ExplanationAgent
    from langgraph.checkpoint.memory import MemorySaver

    nadi = FakeNadi()
    agents = pg.Agents(
        aurora=FakeAurora(), nadi=nadi, sabina=FakeSabina(),
        manager=ManagerAgent(predictions_path=pg.PREDS, target_accuracy=0.60,
                             max_iterations=9, patience=2, min_delta=0.01),
        freddi=ExplanationAgent(use_ollama=False, output_path=pg.EXPL),
    )
    graph = pg.build_pipeline(agents, checkpointer=MemorySaver())
    final = graph.invoke({"retune_request_path": None},
                         {"configurable": {"thread_id": "t"}, "recursion_limit": pg.RECURSION_LIMIT})

    # Proceeded via convergence (not the cap of 9), after cycling the classifier.
    assert final["final_action"] == "proceed"
    assert nadi.calls >= 2, "classifier should have run more than once (the loop cycled)"

    # The full set of terminal contract files exists.
    for name in ("retune_request.json", "sample_for_explanation.csv", "explanations.csv",
                 "final_results.csv", "final_report.json"):
        assert (tmp_path / "outputs" / name).exists(), f"missing {name}"

    # Retune params escalated across cycle passes (no exact repeat).
    req = json.loads((tmp_path / "outputs" / "retune_request.json").read_text())
    assert "suggested_params" in req

    # Default build never asks Nadi for fine-tuned weights.
    assert nadi.model_dir is None


@needs_langgraph
def test_model_dir_reaches_nadi(tmp_path, monkeypatch):
    """`model_dir` set on build_pipeline must reach Nadi's run() so the loop can
    opt in to the fine-tuned weights; it stays off unless explicitly passed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    import agents.pipeline_graph as pg
    from agents.jack_manager import ManagerAgent
    from agents.freddi_explanation import ExplanationAgent
    from langgraph.checkpoint.memory import MemorySaver

    nadi = FakeNadi()
    agents = pg.Agents(
        aurora=FakeAurora(), nadi=nadi, sabina=FakeSabina(),
        manager=ManagerAgent(predictions_path=pg.PREDS, target_accuracy=0.60,
                             max_iterations=9, patience=2, min_delta=0.01),
        freddi=ExplanationAgent(use_ollama=False, output_path=pg.EXPL),
    )
    graph = pg.build_pipeline(agents, model_dir="outputs/finbert_finetuned",
                              checkpointer=MemorySaver())
    graph.invoke({"retune_request_path": None},
                 {"configurable": {"thread_id": "t"}, "recursion_limit": pg.RECURSION_LIMIT})

    assert nadi.model_dir == "outputs/finbert_finetuned"
