# Manager Adaptive Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Manager's fixed `_RETUNE_SCHEDULE` ladder with a confusion-aware hill-climb over `threshold`/`boost_factor`, and make the loop finalize on the best iteration seen rather than the last.

**Architecture:** All changes live in `agents/jack_manager.py` (the Manager agent) plus its tests and two guide notebooks. The hill-climb derives each retune's params from the neutral over/under-prediction signal in `predictions_test.csv`, perturbing the best-so-far params one lever at a time and skipping already-tried sets. A best-accuracy snapshot of the predictions file lets `proceed`/`finalize` emit the best iteration's results. No data contract, no other agent, and no dlt code change.

**Tech Stack:** Python 3.13 (uv-managed), pandas, LangGraph (StateGraph + MemorySaver checkpointer), pytest.

## Global Constraints

- No changes to `docs/data_contracts.md`, other agents, or `yahoo_finance_pipeline.py`. `retune_request.json` keeps `suggested_params` + `focus_labels`; `final_report.json` keeps its existing fields (values now reflect the best iteration).
- Label values stay lowercase `up`/`down`/`neutral`.
- Run everything from repo root so config resolves: `uv run pytest ...`.
- `max_length` is frozen (passthrough), NOT a lever.
- Lever bounds/steps (exact): `threshold` ∈ [0.15, 0.50] step 0.05; `boost_factor` ∈ [1.0, 2.0] step 0.15; `_SKEW_EPS = 0.03`; defaults `_DEFAULT_BOOST = 1.25`, `_DEFAULT_MAXLEN = 128`.
- Commit messages: state what changed and why; **no** AI co-author trailer.

---

## File Structure

- `agents/jack_manager.py` — remove `_RETUNE_SCHEDULE` and `_same`; add lever constants, `_norm`, `_neutral_skew`, `_best_preds_path`; rewrite `_next_params`; rewire `decide`; point `proceed`/`finalize` at the snapshot; extend `ManagerState`.
- `tests/test_manager.py` — update the two schedule-era tests; add hill-climb, confusion, best-retention, and finalize-on-best tests.
- `docs/jack_manager_guide.ipynb` — rewrite the retune-strategy narrative.
- `docs/langgraph_pipeline_guide.ipynb` — update the loop cells that describe param escalation.

---

## Task 1: Lever constants + pure helpers (`_norm`, `_neutral_skew`, `_best_preds_path`)

**Files:**
- Modify: `agents/jack_manager.py` (add constants + helpers near the current `_RETUNE_SCHEDULE` block, ~line 86–101; add `import shutil` at the top with the other stdlib imports)
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: nothing (pure helpers).
- Produces:
  - `_T_MIN=0.15, _T_MAX=0.50, _T_STEP=0.05, _B_MIN=1.0, _B_MAX=2.0, _B_STEP=0.15, _SKEW_EPS=0.03, _DEFAULT_BOOST=1.25, _DEFAULT_MAXLEN=128`
  - `_norm(p: dict) -> dict` → `{"threshold": float, "max_length": int, "boost_factor": float}`, defaults filled, floats rounded to 2 dp.
  - `_neutral_skew(predictions_path: str) -> float` → `(count(pred==neutral) - count(label==neutral)) / N`; returns `0.0` when the file is missing or empty.
  - `_best_preds_path() -> str` → `os.path.join(OUTPUT_DIR, "best_predictions.csv")` (computed at call time so tests that monkeypatch `OUTPUT_DIR` work).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_manager.py`:

```python
def test_norm_fills_defaults_and_rounds():
    assert jm._norm({"threshold": 0.451, "max_length": 128}) == {
        "threshold": 0.45, "max_length": 128, "boost_factor": 1.25}
    assert jm._norm({}) == {"threshold": 0.5, "max_length": 128, "boost_factor": 1.25}


def test_neutral_skew_over_under_and_balanced(tmp_path):
    def _csv(preds, labels):
        p = tmp_path / f"{'_'.join(preds)}.csv"
        rows = "\n".join(f"a{i},{pr},{la}" for i, (pr, la) in enumerate(zip(preds, labels)))
        p.write_text("article_id,predicted_label,label\n" + rows + "\n")
        return str(p)

    over = _csv(["neutral", "neutral", "up"], ["up", "down", "up"])   # 2 pred, 0 true
    under = _csv(["up", "down", "up"], ["neutral", "neutral", "up"])  # 0 pred, 2 true
    bal = _csv(["neutral", "up", "down"], ["neutral", "up", "down"])  # 1 pred, 1 true
    assert jm._neutral_skew(over) > jm._SKEW_EPS
    assert jm._neutral_skew(under) < -jm._SKEW_EPS
    assert abs(jm._neutral_skew(bal)) <= jm._SKEW_EPS


def test_neutral_skew_missing_file_is_zero():
    assert jm._neutral_skew("does/not/exist.csv") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manager.py::test_norm_fills_defaults_and_rounds tests/test_manager.py::test_neutral_skew_over_under_and_balanced tests/test_manager.py::test_neutral_skew_missing_file_is_zero -v`
Expected: FAIL — `AttributeError: module 'agents.jack_manager' has no attribute '_norm'`.

- [ ] **Step 3: Add `import shutil`**

At the top of `agents/jack_manager.py`, alongside the existing stdlib imports (`import json`, `import operator`, `import os`), add:

```python
import shutil
```

- [ ] **Step 4: Add constants + helpers**

Replace the `_RETUNE_SCHEDULE = [...]` block (currently ~lines 86–101, including its docstring comment) with:

```python
# --- adaptive retune levers ---------------------------------------------------
# Two levers the hill-climb walks (max_length is frozen, a weak lever here):
#   threshold   — confidence cutoff; rows BELOW it are forced to `neutral`.
#   boost_factor — scales softmax mass of each focus class before renorm.
_T_MIN, _T_MAX, _T_STEP = 0.15, 0.50, 0.05
_B_MIN, _B_MAX, _B_STEP = 1.0, 2.0, 0.15
_SKEW_EPS = 0.03            # |neutral over/under-prediction| below this = "balanced"
_DEFAULT_BOOST = 1.25       # Nadi's boost default when a param set omits it
_DEFAULT_MAXLEN = 128       # frozen max_length


def _best_preds_path() -> str:
    """Path of the best-iteration predictions snapshot. Computed at call time so
    tests that monkeypatch OUTPUT_DIR to a temp dir hit the right location."""
    return os.path.join(OUTPUT_DIR, "best_predictions.csv")


def _norm(p: dict) -> dict:
    """Canonical param set on the three keys we compare/store, defaults filled.
    Lets us compare tried sets by exact equality now that boost_factor is a real
    lever (two sets differing only in boost_factor are genuinely different)."""
    return {"threshold": round(float(p.get("threshold", _T_MAX)), 2),
            "max_length": int(p.get("max_length", _DEFAULT_MAXLEN)),
            "boost_factor": round(float(p.get("boost_factor", _DEFAULT_BOOST)), 2)}


def _neutral_skew(predictions_path: str) -> float:
    """(#predicted==neutral − #label==neutral) / N from the predictions CSV.
    Positive → neutral over-predicted; negative → under-predicted. 0.0 when the
    file is missing or empty (caller falls back to the boost lever)."""
    import pandas as pd

    if not os.path.exists(predictions_path):
        return 0.0
    df = pd.read_csv(predictions_path)
    if len(df) == 0:
        return 0.0
    return (int((df["predicted_label"] == "neutral").sum())
            - int((df["label"] == "neutral").sum())) / len(df)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_manager.py::test_norm_fills_defaults_and_rounds tests/test_manager.py::test_neutral_skew_over_under_and_balanced tests/test_manager.py::test_neutral_skew_missing_file_is_zero -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add agents/jack_manager.py tests/test_manager.py
git commit -m "Add lever constants and confusion helpers for adaptive retune"
```

---

## Task 2: Rewrite `_next_params` as a confusion-aware hill-climb

**Files:**
- Modify: `agents/jack_manager.py` (replace `_next_params` and delete the `_same` inner helper, currently ~lines 104–120)
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `_norm`, `_T_*`, `_B_*`, `_SKEW_EPS` from Task 1.
- Produces: `_next_params(base: dict, tried: list, skew: float, focus: list) -> dict | None`. Returns a `_norm`-shaped dict (the next params to try), or `None` when every candidate is out-of-bounds or already tried (caller then proceeds). `focus` is currently unused by the math (boost is a global lever) but is kept in the signature for the override payload / future per-class use.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_manager.py`:

```python
def test_next_params_over_neutral_lowers_threshold():
    base = {"threshold": 0.45, "max_length": 128, "boost_factor": 1.25}
    out = jm._next_params(base, tried=[jm._norm(base)], skew=0.2, focus=["down"])
    assert out["threshold"] == 0.40          # stepped DOWN by 0.05
    assert out["boost_factor"] == 1.25 and out["max_length"] == 128


def test_next_params_under_neutral_raises_threshold():
    base = {"threshold": 0.45, "max_length": 128, "boost_factor": 1.25}
    out = jm._next_params(base, tried=[jm._norm(base)], skew=-0.2, focus=["down"])
    assert out["threshold"] == 0.50          # stepped UP by 0.05


def test_next_params_balanced_boosts():
    base = {"threshold": 0.45, "max_length": 128, "boost_factor": 1.25}
    out = jm._next_params(base, tried=[jm._norm(base)], skew=0.0, focus=["down"])
    assert out["boost_factor"] == 1.40       # boost UP by 0.15
    assert out["threshold"] == 0.45


def test_next_params_falls_through_to_next_candidate_when_primary_tried():
    base = {"threshold": 0.45, "max_length": 128, "boost_factor": 1.25}
    threshold_down = jm._norm({"threshold": 0.40, "max_length": 128, "boost_factor": 1.25})
    # over-neutral wants threshold down first; that set is already tried -> boost up next
    out = jm._next_params(base, tried=[jm._norm(base), threshold_down], skew=0.2, focus=[])
    assert out["boost_factor"] == 1.40 and out["threshold"] == 0.45


def test_next_params_exhausted_returns_none():
    base = {"threshold": 0.15, "max_length": 128, "boost_factor": 2.0}  # threshold/boost at bounds
    tried = [jm._norm(base),
             jm._norm({"threshold": 0.20, "max_length": 128, "boost_factor": 2.0})]
    # threshold-down and boost-up clamp back to base (tried); threshold-up (0.20) is tried too
    out = jm._next_params(base, tried=tried, skew=0.2, focus=[])
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manager.py -k next_params -v`
Expected: FAIL — new `_next_params` signature not present (old one takes one positional arg).

- [ ] **Step 3: Rewrite `_next_params` (and delete `_same`)**

Replace the entire current `_next_params` function (its docstring + the inner `_same` + the loop) with:

```python
def _next_params(base: dict, tried: list, skew: float, focus: list) -> dict | None:
    """Pick the next retune params by perturbing `base` (the best params so far)
    one lever at a time. The neutral over/under-prediction `skew` chooses the lever
    and direction; candidates are tried in priority order and the first in-bounds,
    not-already-`tried` set wins. Returns None when nothing new is left to try (the
    caller then proceeds instead of re-running an identical classifier).

    A regressed move is already recorded in `tried`, so it is skipped here and the
    loop falls through to the next candidate — that is the whole feedback mechanism.
    `focus` is carried for the override payload; the boost lever is global, so the
    math does not need it."""
    b = _norm(base)
    thr, boost = b["threshold"], b["boost_factor"]

    if skew > _SKEW_EPS:                      # over-predicting neutral -> fewer forced neutrals
        moves = [("threshold", -1), ("boost_factor", +1), ("threshold", +1)]
    elif skew < -_SKEW_EPS:                   # under-predicting neutral -> more forced neutrals
        moves = [("threshold", +1), ("boost_factor", +1), ("threshold", -1)]
    else:                                     # neutral balanced -> lean on the focus-class boost
        moves = [("boost_factor", +1), ("threshold", -1), ("threshold", +1)]

    for lever, d in moves:
        cand = dict(b)
        if lever == "threshold":
            cand["threshold"] = round(min(max(thr + d * _T_STEP, _T_MIN), _T_MAX), 2)
        else:
            cand["boost_factor"] = round(min(max(boost + d * _B_STEP, _B_MIN), _B_MAX), 2)
        if cand not in tried:
            return cand
    return None
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_manager.py -k next_params -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Delete the two schedule-era tests**

Both assert the removed ladder and the old one-arg `_next_params`; `decide` still calls the old signature until Task 3, so leaving them in would error the suite between tasks. Delete `test_retune_params_adapt_and_do_not_repeat` and `test_second_retune_skips_schedule_entry_matching_sabinas_proposal` entirely. Their intent — "consecutive retunes don't repeat a param set" — is re-covered by a new test added in Task 3 (the shared-key bug the second one guarded no longer exists now that `tried_params` stores `_norm`-ed full sets compared by exact equality).

- [ ] **Step 6: Commit**

```bash
git add agents/jack_manager.py tests/test_manager.py
git commit -m "Rewrite _next_params as confusion-aware hill-climb; drop fixed ladder"
```

---

## Task 3: Rewire `decide` — best-retention snapshot + hill-climb + exhaustion→proceed

**Files:**
- Modify: `agents/jack_manager.py` (`ManagerState` TypedDict ~lines 26–61; `decide` ~lines 139–207)
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `_next_params`, `_neutral_skew`, `_norm`, `_best_preds_path` (Tasks 1–2); existing `_converged`.
- Produces: `decide(state) -> dict` now also returns, when accuracy improves, `best_accuracy: float`, `best_params: dict`, `best_class_accuracy: dict`; `tried_params` entries are `_norm`-ed; a retune with no untried candidate flips `final_action` to `"proceed"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_manager.py`:

```python
def test_decide_snapshots_and_tracks_best_on_improvement(outdir):
    g, cfg = _graph(), {"configurable": {"thread_id": "best-up"}}
    base = {"target_accuracy": 0.99, "max_iterations": 9, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    s1 = g.invoke({**base, "evaluation_report": _report(0.40)}, cfg)
    assert s1["best_accuracy"] == 0.40
    assert (outdir / "best_predictions.csv").exists()          # snapshot written
    s2 = g.invoke({**base, "evaluation_report": _report(0.55)}, cfg)
    assert s2["best_accuracy"] == 0.55                          # improved -> updated


def test_decide_keeps_best_on_regression(outdir):
    g, cfg = _graph(), {"configurable": {"thread_id": "best-keep"}}
    base = {"target_accuracy": 0.99, "max_iterations": 9, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    g.invoke({**base, "evaluation_report": _report(0.55)}, cfg)
    s2 = g.invoke({**base, "evaluation_report": _report(0.40)}, cfg)   # regress
    assert s2["best_accuracy"] == 0.55                          # unchanged
    assert jm._norm(s2["best_params"]) == jm._norm({"threshold": 0.5, "max_length": 128})


def test_decide_proceeds_when_search_exhausted(outdir):
    """Once every in-bounds lever move is tried, a below-target report proceeds
    instead of retuning forever. Flat accuracy keeps the hill-climb base fixed, so
    its handful of neighbours exhaust quickly."""
    g, cfg = _graph(), {"configurable": {"thread_id": "exhaust"}}
    base = {"target_accuracy": 0.99, "max_iterations": 99, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    actions = [g.invoke({**base, "evaluation_report": _report(0.37)}, cfg)["final_action"]
               for _ in range(30)]
    assert "proceed" in actions                                 # exhaustion stops the loop


def test_retune_params_adapt_and_do_not_repeat(outdir):
    """With accuracy improving, best advances each step so the hill-climb keeps
    finding fresh, previously-unused param sets (never re-running one)."""
    g, cfg = _graph(), {"configurable": {"thread_id": "adapt"}}
    base = {"target_accuracy": 0.99, "max_iterations": 9, "patience": 99, "min_delta": 0.0,
            "predictions_path": PRED}
    seen = [g.invoke({**base, "evaluation_report": _report(a)}, cfg)["tried_params"][-1]
            for a in (0.30, 0.35, 0.40, 0.45)]
    assert len(seen) == 4
    assert all(seen[i] != seen[i + 1] for i in range(len(seen) - 1))
    assert {"threshold", "max_length", "boost_factor"} <= set(seen[-1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manager.py -k "best or exhausted" -v`
Expected: FAIL — `KeyError: 'best_accuracy'` / loop never proceeds.

- [ ] **Step 3: Extend `ManagerState`**

In the `ManagerState` TypedDict, in the "loop progress" group (after the `overrides`/`notes` lines, before the convergence-config group), add:

```python
    # --- best-so-far (finalize on best, and the hill-climb base) ---
    best_accuracy: float         # highest accuracy seen; -1.0 sentinel until first eval
    best_params: dict            # params that produced best_accuracy (hill-climb base)
    best_class_accuracy: dict    # per-class accuracy of the best iteration
```

- [ ] **Step 4: Rewrite `decide`**

Replace the whole `decide` function body with:

```python
def decide(state: ManagerState) -> dict:
    """Threshold gate (deterministic). Tracks the best iteration (snapshotting its
    predictions), decides retune vs. proceed, and on retune hill-climbs the next
    params from the neutral-prediction skew. Returns only changed keys."""
    report = state["evaluation_report"]
    accuracy = report["accuracy"]
    proposal = report.get("proposal", {})
    pred_path = state.get("predictions_path", "mock_data/predictions_test.csv")

    iteration = state.get("iteration", 0)
    if state.get("final_action") != "proceed":
        iteration += 1

    history = state.get("accuracy_history", []) + [accuracy]
    patience = state.get("patience", 2)
    min_delta = state.get("min_delta", 0.01)
    prev_tried = state.get("tried_params", [])

    # --- best-so-far: the params behind THIS accuracy are the last retune's (or,
    # on the pre-retune first pass, Sabina's suggestion). Snapshot on improvement. ---
    params_used = _norm(prev_tried[-1] if prev_tried else proposal.get("suggested_params", {}))
    best_acc = state.get("best_accuracy", -1.0)
    best_params = state.get("best_params", params_used)
    out_best = {}
    if accuracy > best_acc:
        best_acc, best_params = accuracy, params_used
        if os.path.exists(pred_path):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.copyfile(pred_path, _best_preds_path())
        out_best = {"best_accuracy": best_acc, "best_params": best_params,
                    "best_class_accuracy": report.get("class_accuracy", {})}

    # --- gate ---
    cleared = accuracy >= state["target_accuracy"]
    cap_hit = iteration >= state["max_iterations"]
    converged = _converged(history, patience, min_delta)
    final_action = "proceed" if (cleared or cap_hit or converged) else "retune"

    # --- retune param selection (may flip to proceed when the search is exhausted) ---
    retune_out = {}
    exhausted = False
    if final_action == "retune":
        focus = proposal.get("focus_labels", [])
        if not prev_tried:                       # first retune trusts Sabina as-is
            used = _norm(proposal.get("suggested_params", {}))
            retune_out = {"decision": "accept", "overrides": {}, "tried_params": [used]}
        else:
            used = _next_params(best_params, prev_tried, _neutral_skew(pred_path), focus)
            if used is None:
                final_action, exhausted = "proceed", True
            else:
                retune_out = {"decision": "override", "tried_params": [used],
                              "overrides": {"suggested_params": used, "focus_labels": focus}}

    if cleared:
        why = "cleared target"
    elif converged:
        why = "converged (no improvement), proceeding"
    elif cap_hit:
        why = "cap hit, forcing proceed"
    elif exhausted:
        why = "search exhausted, proceeding"
    else:
        why = "below target, retuning"
    note = (f"iteration {iteration}: accuracy {accuracy:.2f} vs target "
            f"{state['target_accuracy']:.2f} — {why}")

    out = {
        "iteration": iteration,
        "accuracy": accuracy,
        "final_action": final_action,
        "notes": note,
        "decision_log": [note],
        "accuracy_history": [accuracy],
        **out_best,
    }
    if final_action == "retune":
        out.update(retune_out)
    else:
        recommended = proposal.get("recommended_action")
        if recommended is not None and recommended != final_action:
            out["decision"], out["overrides"] = "override", {"final_action": final_action}
        else:
            out["decision"], out["overrides"] = "accept", {}
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_manager.py -k "best or exhausted or next_params or adapt" -v`
Expected: PASS.

- [ ] **Step 6: Run the full Manager suite to catch regressions**

Run: `uv run pytest tests/test_manager.py -v`
Expected: PASS. If `test_convergence_proceeds_before_cap_when_accuracy_flat` or lifecycle tests fail, reconcile — they should still hold (gate logic unchanged; `tried_params` now `_norm`-ed).

- [ ] **Step 7: Commit**

```bash
git add agents/jack_manager.py tests/test_manager.py
git commit -m "Track best iteration and hill-climb retune params in decide"
```

---

## Task 4: Finalize (and sample) on the best iteration

**Files:**
- Modify: `agents/jack_manager.py` (`proceed` ~lines 238–253; `finalize` ~lines 256–281)
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `_best_preds_path`, and `best_accuracy`/`best_class_accuracy` from state (Task 3).
- Produces: `proceed` samples from the best snapshot when present; `finalize` reads the best snapshot and reports `best_accuracy`/`best_class_accuracy`.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_manager.py`:

```python
def _write_preds(path, confidence):
    """Minimal predictions_test.csv with the columns finalize/proceed select."""
    import pandas as pd
    rows = []
    for i, lab in enumerate(["up", "down", "neutral", "up"]):
        rows.append({"article_id": f"FN_{i}", "date": "2021-03-15", "ticker": "AAPL",
                     "article_title": f"h{i}", "price_t": 100.0, "price_t1": 101.0,
                     "pct_change": 1.0, "label": lab, "predicted_label": lab,
                     "confidence": confidence, "prob_up": 0.5, "prob_down": 0.3,
                     "prob_neutral": 0.2})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_finalizes_on_best_not_last(outdir, tmp_path):
    """iter2 is the best; iter3 regresses. Final output must reflect iter2."""
    g, cfg = _graph(), {"configurable": {"thread_id": "fob"}}
    p1, p2, p3 = (str(tmp_path / f"p{i}.csv") for i in (1, 2, 3))
    _write_preds(p1, 0.11); _write_preds(p2, 0.99); _write_preds(p3, 0.33)  # p2 distinguishable
    common = {"target_accuracy": 0.95, "max_iterations": 3, "patience": 99, "min_delta": 0.0}

    g.invoke({**common, "predictions_path": p1, "evaluation_report": _report(0.30)}, cfg)
    g.invoke({**common, "predictions_path": p2, "evaluation_report": _report(0.55)}, cfg)  # best
    g.invoke({**common, "predictions_path": p3, "evaluation_report": _report(0.40)}, cfg)  # cap -> proceed

    # finalize pass (explanations present)
    g.invoke({**common, "predictions_path": p3, "explanations_path": EXPL,
              "evaluation_report": _report(0.40)}, cfg)

    import pandas as pd, json as _json
    final = pd.read_csv(outdir / "final_results.csv")
    assert (final["confidence"] == 0.99).all()                 # iter2's snapshot, not iter3
    rep = _json.loads((outdir / "final_report.json").read_text())
    assert rep["final_accuracy"] == 0.55                       # best, not last (0.40)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manager.py::test_finalizes_on_best_not_last -v`
Expected: FAIL — `final_accuracy` is 0.40 and `confidence` is 0.33 (last iteration, not best).

- [ ] **Step 3: Point `proceed` at the best snapshot**

In `proceed`, replace the predictions read line:

```python
    preds = pd.read_csv(state.get("predictions_path", "mock_data/predictions_test.csv"))
```

with:

```python
    src = _best_preds_path() if os.path.exists(_best_preds_path()) else \
        state.get("predictions_path", "mock_data/predictions_test.csv")
    preds = pd.read_csv(src)
```

- [ ] **Step 4: Point `finalize` at the best snapshot and report best metrics**

In `finalize`, replace the predictions read line the same way:

```python
    src = _best_preds_path() if os.path.exists(_best_preds_path()) else \
        state.get("predictions_path", "mock_data/predictions_test.csv")
    preds = pd.read_csv(src)
```

Then in the `_write_json("final_report.json", {...})` call, replace the `final_accuracy` and `class_accuracy` lines:

```python
        "final_accuracy": report.get("accuracy"),
        ...
        "class_accuracy": report.get("class_accuracy", {}),
```

with (fall back to the report when no best was recorded, e.g. a single-pass proceed):

```python
        "final_accuracy": state.get("best_accuracy") if state.get("best_accuracy", -1.0) >= 0
            else report.get("accuracy"),
        ...
        "class_accuracy": state.get("best_class_accuracy") or report.get("class_accuracy", {}),
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_manager.py::test_finalizes_on_best_not_last -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/test_manager.py -v`
Expected: PASS. The lifecycle tests use the mock `PRED`; since a best snapshot is written on the first evaluation, `finalize` now reads `best_predictions.csv` (a copy of the mock) — same shape, so `test_outputs_match_contract` and `test_full_lifecycle` still hold. Reconcile any that assert `final_accuracy` equals the last report's accuracy (update the expectation to the best accuracy).

- [ ] **Step 7: Commit**

```bash
git add agents/jack_manager.py tests/test_manager.py
git commit -m "Finalize and sample on the best iteration, not the last"
```

---

## Task 5: Update the guide notebooks

**Files:**
- Modify: `docs/jack_manager_guide.ipynb`
- Modify: `docs/langgraph_pipeline_guide.ipynb`

**Interfaces:**
- Consumes: the final `agents/jack_manager.py` behaviour from Tasks 1–4.
- Produces: docs consistent with the code (Golden rule: code and docs must not disagree).

- [ ] **Step 1: Find the stale cells**

Run: `uv run jupyter nbconvert --to script --stdout docs/jack_manager_guide.ipynb | grep -n "RETUNE_SCHEDULE\|_next_params\|ladder\|escalat\|schedule"`
Run: `uv run jupyter nbconvert --to script --stdout docs/langgraph_pipeline_guide.ipynb | grep -n "RETUNE_SCHEDULE\|_next_params\|ladder\|escalat\|schedule"`
Expected: line numbers of the cells describing the old fixed ladder.

- [ ] **Step 2: Rewrite the `jack_manager_guide.ipynb` retune section**

Open the notebook (Jupyter, or edit the JSON), and in the markdown/code cells that describe the retune strategy, replace the fixed-ladder narrative with the hill-climb. The replacement must state, in the guide's existing voice:
- The two levers and their bounds/steps (`threshold` [0.15, 0.50]/0.05; `boost_factor` [1.0, 2.0]/0.15) and that `max_length` is frozen.
- The confusion signal: over-predicting `neutral` → lower `threshold`; under-predicting → raise it; balanced → boost the weakest focus class.
- That the loop keeps the best iteration and finalizes on it (`best_predictions.csv` snapshot), and that the search stops when candidates are exhausted.
Update any code cell that imports/prints `_RETUNE_SCHEDULE` or calls `_next_params` with the old one-arg signature to the new `_next_params(base, tried, skew, focus)` form (or remove the demo cell if it only illustrated the ladder).

- [ ] **Step 3: Rewrite the `langgraph_pipeline_guide.ipynb` loop cells**

In the cells that describe param escalation across the cycle, replace the "walks down a schedule" description with "hill-climbs from the best-so-far params using the neutral-prediction skew, and finalizes on the best iteration." Keep the graph-shape explanation unchanged.

- [ ] **Step 4: Execute both notebooks top to bottom to verify they run**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace docs/jack_manager_guide.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace docs/langgraph_pipeline_guide.ipynb`
Expected: both execute without error (no `AttributeError: _RETUNE_SCHEDULE`, no signature errors).

- [ ] **Step 5: Commit**

```bash
git add docs/jack_manager_guide.ipynb docs/langgraph_pipeline_guide.ipynb
git commit -m "Update guide notebooks for confusion-aware hill-climb retune"
```

---

## Final verification

- [ ] Run the full Manager suite: `uv run pytest tests/test_manager.py -v` → all pass.
- [ ] Grep the codebase for stragglers: `grep -rn "_RETUNE_SCHEDULE\|_same(" agents/ tests/ docs/` → no hits (both symbols removed).
- [ ] Confirm no contract drift: `git diff main -- docs/data_contracts.md mock_data/` → empty.
