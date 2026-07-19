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

import pandas as pd
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
        self.epochs = None

    def run(self, *, processed_data, classifier_code, predictions, retune_request=None,
            epochs=None):
        self.calls += 1
        self.epochs = epochs
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MOCK_PRED, predictions)
        Path(classifier_code).write_text("THRESHOLD = 0.5\n")  # readable by Sabina/Manager
        return {}


class FakeSabina:
    """Emits below-target reports, forcing the Manager to retune until it
    converges. `accs` optionally scripts one accuracy per pass (default: flat
    0.37), keyed by MarkedNadi's fake_pass stamp when present so re-scoring
    restored predictions reproduces that pass's accuracy. Records which
    eval_split each call scored."""
    def __init__(self, accs=None):
        self.accs = accs
        self.calls = 0
        self.eval_splits = []

    def run(self, *, predictions, classifier_code, eval_split="test"):
        self.eval_splits.append(eval_split)
        if self.accs:
            df = pd.read_csv(predictions)
            pass_no = int(df["fake_pass"].iloc[0]) if "fake_pass" in df.columns else self.calls + 1
            acc = self.accs[pass_no - 1]
        else:
            acc = 0.37
        self.calls += 1
        report = {
            "accuracy": acc,
            "below_threshold": True,
            "class_accuracy": {"up": 0.3, "down": 0.2, "neutral": 0.5},
            "misclassified_ids": [],
            "proposal": {"recommended_action": "retune", "reason": "low",
                         "focus_labels": ["down"]},
        }
        os.makedirs("outputs", exist_ok=True)
        Path("outputs/evaluation_report.json").write_text(json.dumps(report))
        return {"output_path": "outputs/evaluation_report.json"}


class MarkedNadi(FakeNadi):
    """FakeNadi that also stamps every predictions file with the pass number,
    so the test can tell WHICH iteration's artifacts survived to the end. Also
    stamps fake up/down head checkpoints (a marker file each) so the same can
    be verified for the head-checkpoint best/restore snapshot."""
    def run(self, **kwargs):
        out = super().run(**kwargs)
        df = pd.read_csv(kwargs["predictions"])
        df["fake_pass"] = self.calls
        df.to_csv(kwargs["predictions"], index=False)
        for head in ("up", "down"):
            head_dir = Path("outputs/finbert_finetuned") / head
            head_dir.mkdir(parents=True, exist_ok=True)
            (head_dir / "marker.txt").write_text(str(self.calls))
        return out


def _run_pipeline(nadi, sabina, *, aurora=None, max_iterations=9, **build_kwargs):
    """Build the graph around fakes for the heavy agents plus the real Manager
    and offline Freddi, and drive it once end to end."""
    import agents.pipeline_graph as pg
    from agents.jack_manager import ManagerAgent
    from agents.freddi_explanation import ExplanationAgent
    from langgraph.checkpoint.memory import MemorySaver

    agents = pg.Agents(
        aurora=aurora or FakeAurora(), nadi=nadi, sabina=sabina,
        manager=ManagerAgent(predictions_path=pg.PREDS, target_accuracy=0.60,
                             max_iterations=max_iterations, patience=2, min_delta=0.01),
        freddi=ExplanationAgent(use_ollama=False, output_path=pg.EXPL),
    )
    graph = pg.build_pipeline(agents, checkpointer=MemorySaver(), **build_kwargs)
    return graph.invoke({"retune_request_path": None},
                        {"configurable": {"thread_id": "t"}, "recursion_limit": pg.RECURSION_LIMIT})


@needs_langgraph
def test_best_iteration_restored_when_accuracy_regresses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    # Accuracy peaks on pass 2 then regresses, so the best iteration is NOT the
    # last one the gate proceeds with — exactly the case select_best must fix.
    nadi = MarkedNadi()
    final = _run_pipeline(nadi, FakeSabina(accs=[0.20, 0.39] + [0.30] * 4),
                          max_iterations=6)

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

    # The head checkpoints restored are the BEST pass's too, not the last
    # pass's -- each head is overwritten in place every retrain (ADR 0002),
    # so without evaluate()/select_best() copying them out, this would read
    # pass 6's (the last, regressed pass) marker instead.
    for head in ("up", "down"):
        marker = tmp_path / "outputs" / "finbert_finetuned" / head / "marker.txt"
        assert marker.read_text() == "2", f"{head}-head checkpoint should be from the best pass"


def test_clean_outputs_removes_stale_loop_artifacts(tmp_path, monkeypatch):
    """A fresh run must not inherit the previous run's loop files: a stale
    evaluation_report.json can trigger Nadi's LLM rewrite on iteration 0, and a
    stale best/ snapshot could win select_best. Non-loop artifacts (fine-tuned
    weights, finetune report) must survive."""
    monkeypatch.chdir(tmp_path)
    import agents.pipeline_graph as pg

    out = tmp_path / "outputs"
    (out / "best").mkdir(parents=True)
    (out / "classifier_history").mkdir()
    (out / "finbert_finetuned").mkdir()
    stale = ["classifier.py", "predictions_test.csv", "evaluation_report.json",
             "retune_request.json", "sample_for_explanation.csv", "explanations.csv",
             "decision.json", "final_results.csv", "final_report.json"]
    for name in stale:
        (out / name).write_text("stale")
    (out / "best" / "evaluation_report.json").write_text("stale")
    (out / "finetune_report.json").write_text("keep")
    (out / "finbert_finetuned" / "model.safetensors").write_text("keep")

    pg.clean_outputs()

    for name in stale:
        assert not (out / name).exists(), f"stale {name} survived"
    assert not (out / "best").exists()
    assert not (out / "classifier_history").exists()
    assert (out / "finetune_report.json").exists()
    assert (out / "finbert_finetuned" / "model.safetensors").exists()


@needs_langgraph
def test_failed_processing_keeps_previous_runs_outputs(tmp_path, monkeypatch):
    """Cleanup runs only after Aurora succeeds — a failed start must not wipe
    the previous run's deliverables."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "final_report.json").write_text("previous run")

    class FailingAurora:
        def run(self, **kwargs):
            raise FileNotFoundError("fnspid_raw.csv not found")

    with pytest.raises(FileNotFoundError):
        _run_pipeline(FakeNadi(), FakeSabina(), aurora=FailingAurora())
    assert (tmp_path / "outputs" / "final_report.json").read_text() == "previous run"


@needs_langgraph
def test_graph_cycles_then_finalizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    nadi, sabina = FakeNadi(), FakeSabina()
    final = _run_pipeline(nadi, sabina)

    # Proceeded via convergence (not the cap of 9), after cycling the classifier.
    assert final["final_action"] == "proceed"
    assert nadi.calls >= 2, "classifier should have run more than once (the loop cycled)"

    # Every loop evaluation scored the val split; the single final pass scored test.
    assert sabina.eval_splits[-1] == "test"
    assert set(sabina.eval_splits[:-1]) == {"val"}

    # The full set of terminal contract files exists.
    for name in ("retune_request.json", "sample_for_explanation.csv", "explanations.csv",
                 "final_results.csv", "final_report.json"):
        assert (tmp_path / "outputs" / name).exists(), f"missing {name}"

    # Jack flags a directional head for Nadi to retrain (ADR 0001).
    req = json.loads((tmp_path / "outputs" / "retune_request.json").read_text())
    assert req["heads_to_retrain"] == ["down"]


@needs_langgraph
def test_epochs_override_reaches_nadi(tmp_path, monkeypatch):
    """build_pipeline(epochs=...) must thread through classify() to Nadi's
    run() call; omitted (default None) must not override Nadi's own default."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    with_override = FakeNadi()
    _run_pipeline(with_override, FakeSabina(), epochs=2)
    assert with_override.epochs == 2

    without_override = FakeNadi()
    _run_pipeline(without_override, FakeSabina())
    assert without_override.epochs is None


@needs_langgraph
def test_pipeline_fails_loudly_when_cap_hit_while_collapsed(tmp_path, monkeypatch):
    """ADR 0004: if the iteration cap is hit while a class is still collapsed,
    the whole unified pipeline must abort (RuntimeError) rather than finalize a
    collapsed model -- no final_results.csv/final_report.json get written."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    class CollapsedSabina(FakeSabina):
        """Always reports a collapsed `up` class, never improving."""
        def run(self, *, predictions, classifier_code, eval_split="test"):
            self.eval_splits.append(eval_split)
            self.calls += 1
            report = {
                "accuracy": 0.65,
                "below_threshold": False,
                "class_accuracy": {"up": 0.0, "down": 0.40, "neutral": 0.95},
                "misclassified_ids": [],
                "proposal": {"recommended_action": "proceed"},
            }
            os.makedirs("outputs", exist_ok=True)
            Path("outputs/evaluation_report.json").write_text(json.dumps(report))
            return {"output_path": "outputs/evaluation_report.json"}

    with pytest.raises(RuntimeError, match="collapsed"):
        _run_pipeline(FakeNadi(), CollapsedSabina(), max_iterations=2)

    for name in ("final_results.csv", "final_report.json", "sample_for_explanation.csv"):
        assert not (tmp_path / "outputs" / name).exists(), f"{name} should not have been written"
