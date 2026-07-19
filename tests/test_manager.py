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
    # reducer accumulated: 2 entries per loop invocation + 1 from finalize,
    # which routes straight to its node (no decide pass).
    assert len(r3["decision_log"]) == 5


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
        "final_accuracy", "loop_iterations", "class_accuracy", "class_support",
        "test_set_size", "explanations_generated", "manually_scored"}
    assert rep["test_set_size"] == len(fin)
    assert rep["explanations_generated"] == int((fin.explanation.fillna("") != "").sum())
    assert rep["manually_scored"] == int(fin.manual_score.notna().sum())


def test_retune_request_carries_heads_to_retrain(outdir):
    """Nadi's fine_tune node reads `heads_to_retrain`/`collapsed_heads` from
    retune_request.json (ADR 0001) — Jack must write them."""
    g, cfg = _graph(), {"configurable": {"thread_id": "heads"}}
    g.invoke({**BASE, "evaluation_report": RETUNE_REPORT}, cfg)
    req = json.loads((outdir / "retune_request.json").read_text())
    assert req["heads_to_retrain"] == ["down"]
    assert req["collapsed_heads"] == []
    assert "suggested_params" not in req
    assert "code_notes" not in req


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


# --- convergence + adaptive retune (Increment 1) --------------------------

def _report(accuracy):
    """Minimal below/above-target report with a retune proposal."""
    return {
        "accuracy": accuracy,
        "proposal": {"recommended_action": "retune" if accuracy < 0.60 else "proceed",
                     "focus_labels": ["down"],
                     "suggested_params": {"threshold": 0.5, "max_length": 128}},
    }


def test_convergence_proceeds_before_cap_when_accuracy_flat():
    """A flat accuracy trend should trip early-stop and proceed, even though the
    iteration cap (5) has not been reached and accuracy is below target."""
    g, cfg = _graph(), {"configurable": {"thread_id": "conv"}}
    base = {"target_accuracy": 0.60, "max_iterations": 5, "patience": 2, "min_delta": 0.01,
            "predictions_path": PRED}
    actions = [g.invoke({**base, "evaluation_report": _report(0.37)}, cfg)["final_action"]
               for _ in range(4)]
    # it1 retune, it2 retune (history too short), it3 converged -> proceed.
    assert actions[0] == "retune" and actions[1] == "retune"
    assert actions[2] == "proceed"


def test_finalize_pass_does_not_duplicate_accuracy_history(proceed_report):
    """The finalize invocation must not re-append the accuracy the loop already
    recorded (bug seen live: 9 entries, 8 iterations) — it routes straight to
    the finalize node, bypassing decide."""
    g, cfg = _graph(), {"configurable": {"thread_id": "no-dup"}}
    g.invoke({**BASE, "evaluation_report": RETUNE_REPORT}, cfg)
    g.invoke({**BASE, "evaluation_report": proceed_report}, cfg)
    out = g.invoke({**BASE, "evaluation_report": proceed_report, "explanations_path": EXPL}, cfg)
    assert len(out["accuracy_history"]) == 2  # retune + proceed; finalize adds nothing


# Above-target accuracy with a collapsed class: Sabina recommends proceed (so no
# suggested_params), but the gate's floor must force a retune.
COLLAPSED_STATE = {
    "evaluation_report": {
        "accuracy": 0.65,
        "class_accuracy": {"up": 0.0, "down": 0.10, "neutral": 0.95},
        "proposal": {"recommended_action": "proceed"},
    },
    "target_accuracy": 0.60, "max_iterations": 5,
}


def test_class_collapse_blocks_cleared_target():
    """Aggregate accuracy above target must not clear the gate when a class has
    collapsed to (near) zero recall — that run shipped up=0.0 while 'passing'."""
    out = jm.decide(COLLAPSED_STATE)
    assert out["final_action"] == "retune"


def test_zero_support_class_does_not_block_cleared_target():
    """A label absent from the eval split is not a model collapse."""
    state = {
        "evaluation_report": {
            "accuracy": 0.75,
            "class_accuracy": {"up": 0.75, "down": 0.0, "neutral": 0.80},
            "class_support": {"up": 20, "down": 0, "neutral": 20},
            "proposal": {"recommended_action": "proceed"},
        },
        "target_accuracy": 0.60,
        "max_iterations": 5,
    }

    out = jm.decide(state)

    assert out["final_action"] == "proceed"


# --- weak/collapsed head selection (ADR 0001) ------------------------------

def test_weak_heads_filters_focus_labels_to_directional_heads():
    report = {"class_accuracy": {"up": 0.5, "down": 0.4, "neutral": 0.6},
              "proposal": {"focus_labels": ["down", "neutral"]}}
    assert jm._weak_heads(report, 0.05) == ["down"]


def test_weak_heads_defaults_to_both_when_only_neutral_flagged():
    """A retune with nothing to retrain would burn an iteration for free --
    if neither directional head stood out as weakest, retrain both."""
    report = {"class_accuracy": {"up": 0.5, "down": 0.4, "neutral": 0.6},
              "proposal": {"focus_labels": ["neutral"]}}
    assert set(jm._weak_heads(report, 0.05)) == {"up", "down"}


def test_weak_heads_defaults_to_both_when_focus_labels_missing():
    assert set(jm._weak_heads({"proposal": {}}, 0.05)) == {"up", "down"}


def test_weak_heads_prioritizes_a_collapsed_head_over_focus_labels():
    """Collapse is a sharper signal than Sabina's general "weakest within a
    margin" -- if a head has actually collapsed, it wins outright even when
    focus_labels (or its absence) would suggest something else."""
    report = {"class_accuracy": {"up": 0.0, "down": 0.5, "neutral": 0.6},
              "proposal": {"focus_labels": ["neutral"]}}
    assert jm._weak_heads(report, 0.05) == ["up"]


def test_collapsed_heads_lists_only_heads_below_floor():
    report = {"class_accuracy": {"up": 0.0, "down": 0.40, "neutral": 0.95}}
    assert jm._collapsed_heads(report, 0.05) == ["up"]


def test_collapsed_heads_ignores_zero_support():
    report = {
        "class_accuracy": {"up": 0.0, "down": 0.40, "neutral": 0.95},
        "class_support": {"up": 0, "down": 20, "neutral": 20},
    }
    assert jm._collapsed_heads(report, 0.05) == []


# --- hard collapse gate: fail loudly instead of shipping (ADR 0004) --------

def test_gate_fails_when_cap_hit_while_collapsed():
    """Previously the iteration cap could force `proceed` even with a
    collapsed class; now that must become `fail` instead."""
    state = {
        "evaluation_report": COLLAPSED_STATE["evaluation_report"],
        "target_accuracy": 0.60, "max_iterations": 1,
        "iteration": 0,  # +1 == max_iterations -> cap hit on this call
    }
    out = jm.decide(state)
    assert out["final_action"] == "fail"
    assert out["decision"] == "override"
    assert out["overrides"] == {"final_action": "fail"}


def test_gate_fails_when_converged_while_collapsed():
    g, cfg = _graph(), {"configurable": {"thread_id": "fail-converged"}}
    base = {"target_accuracy": 0.60, "max_iterations": 9, "patience": 1, "min_delta": 0.01,
            "predictions_path": PRED}
    report = {**COLLAPSED_STATE["evaluation_report"], "accuracy": 0.30}
    # Two flat passes trip convergence (patience=1) while still collapsed.
    g.invoke({**base, "evaluation_report": report}, cfg)
    with pytest.raises(RuntimeError, match="collapsed"):
        g.invoke({**base, "evaluation_report": report}, cfg)


def test_fail_node_writes_decision_but_no_proceed_artifacts(outdir):
    """fail() must record the audit trail but never produce the artifacts a
    healthy proceed would -- a collapsed model must not look finished."""
    state = {
        "iteration": 3, "decision": "override", "final_action": "fail",
        "overrides": {"final_action": "fail"}, "notes": "collapsed",
        "evaluation_report": COLLAPSED_STATE["evaluation_report"],
        "accuracy_history": [0.65],
    }
    with pytest.raises(RuntimeError):
        jm.fail(state)
    assert (outdir / "decision.json").exists()
    assert not (outdir / "sample_for_explanation.csv").exists()
    assert not (outdir / "retune_request.json").exists()
    assert not (outdir / "final_results.csv").exists()


def test_report_score_ranks_collapsed_below_healthy():
    """The best-iteration snapshot must rank on the same rule as the gate: a
    collapsed high-accuracy pass never beats a healthy lower-accuracy one."""
    healthy = {"accuracy": 0.39, "class_accuracy": {"up": 0.3, "down": 0.2, "neutral": 0.5}}
    collapsed = {"accuracy": 0.65, "class_accuracy": {"up": 0.0, "down": 0.0, "neutral": 1.0}}
    assert jm.report_score(healthy) > jm.report_score(collapsed)
    assert jm.report_score(healthy) == 0.39   # healthy score IS the accuracy


def test_report_score_ignores_zero_support_classes():
    report = {
        "accuracy": 0.65,
        "class_accuracy": {"up": 0.65, "down": 0.0, "neutral": 0.70},
        "class_support": {"up": 10, "down": 0, "neutral": 10},
    }

    assert jm.report_score(report) == 0.65


def test_collapse_forced_retune_selects_the_collapsed_head():
    """When the collapse floor forces a retune on a pass Sabina recommended to
    proceed, her proposal carries no focus_labels — the gate must still name
    the collapsed head rather than sending Nadi an empty selection."""
    out = jm.decide(COLLAPSED_STATE)
    assert out["decision"] == "override"
    assert out["heads_to_retrain"] == ["up"]
    assert out["collapsed_heads"] == ["up"]


def test_accuracy_history_accumulates_one_per_iteration():
    g, cfg = _graph(), {"configurable": {"thread_id": "hist"}}
    base = {"target_accuracy": 0.60, "max_iterations": 9, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    g.invoke({**base, "evaluation_report": _report(0.30)}, cfg)
    g.invoke({**base, "evaluation_report": _report(0.40)}, cfg)
    out = g.invoke({**base, "evaluation_report": _report(0.50)}, cfg)
    assert out["accuracy_history"] == [0.30, 0.40, 0.50]


def test_decision_json_carries_accuracy_history(outdir):
    """decision.json is overwritten every iteration, so accuracy_history is the
    only on-disk record of the trend — verify it lands in the written file."""
    g, cfg = _graph(), {"configurable": {"thread_id": "hist-disk"}}
    base = {"target_accuracy": 0.60, "max_iterations": 9, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    g.invoke({**base, "evaluation_report": _report(0.30)}, cfg)
    g.invoke({**base, "evaluation_report": _report(0.40)}, cfg)
    decision = json.loads((outdir / "decision.json").read_text())
    assert decision["accuracy_history"] == [0.30, 0.40]


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
