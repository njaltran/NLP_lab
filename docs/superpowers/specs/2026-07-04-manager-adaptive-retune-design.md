# Manager adaptive retune — confusion-aware hill-climb

> **⚠️ SUPERSEDED (2026-07-05) — not implemented.** PR #38 (`feature/manager-adapt`)
> independently shipped both goals of this design, more thoroughly:
> - **Finalize-on-best** → `pipeline_graph.py` snapshots each new-best iteration to
>   `outputs/best/` (scored by `report_score`, collapse-penalized) and `select_best`
>   restores it, with a val/test split. Stronger than the manager-level
>   `best_predictions.csv` snapshot proposed here.
> - **Hill-climb from best** → `_next_params(tried, history)` does revert-and-perturb
>   (revert to best iteration's params, step one knob) plus collapse detection
>   (`_is_collapsed`, `min_class_accuracy` floor).
>
> The only additive idea left was confusion-driven *lever ordering* (neutral skew
> picks which knob to perturb first), judged marginal at ~0.25 accuracy. Rebuild
> abandoned; this doc is kept as a design record only.

**Date:** 2026-07-04
**Owner:** Jack (Manager agent)
**Branch:** `feature/manager-adaptive-retune`
**Status:** approved design, ready for implementation plan

## Problem

The Manager's retune loop escalates hyperparameters with a fixed, open-loop ladder
(`_RETUNE_SCHEDULE` in `agents/jack_manager.py`). Each retune walks one step down the
ladder — `threshold` down, `max_length` up, `boost_factor` up — regardless of whether
the previous step helped. Three concrete harms:

1. **Finalizes on the last iteration, not the best.** `finalize` reads whatever
   `predictions_test.csv` holds at proceed time. Since Nadi overwrites that file every
   retune, the loop can (and at ~0.25 accuracy routinely does) finalize on a *worse*
   result than an earlier step. This is the primary harm.
2. **One-directional and blind.** `threshold` only ever drops, so the loop always pushes
   rows away from the `neutral` fallback. If `neutral` is the correct majority behavior,
   the whole ladder drifts the wrong way with no path back.
3. **Discards Sabina's per-class signal.** After the first retune, `suggested_params` is
   fully replaced by canned ladder values; only `focus_labels` survives.

## Goal

Replace the fixed ladder with a **confusion-aware hill-climb** that derives each retune's
params from the actual error pattern, and make the loop **finalize on the best iteration
seen**, not the last.

## Non-goals

- No changes to any data contract (`docs/data_contracts.md`), any other agent, or the dlt
  pipeline. `retune_request.json` still carries `suggested_params` + `focus_labels`;
  `final_report.json` keeps the same fields (their *values* now reflect the best iteration).
- `max_length` is **not** a lever — it stays frozen at Sabina's/default value. It is a weak
  lever here (headlines rarely exceed 128 tokens) and lacks a clean truncation signal.
- No new tuning of the convergence gate (`_converged`, `patience`, `min_delta`) — it stays
  as-is and runs alongside the hill-climb.

## Design

### Levers, bounds, steps

| Lever | Range | Step | Effect |
|---|---|---|---|
| `threshold` | [0.15, 0.50] | 0.05 | confidence cutoff; rows below it are forced to `neutral` |
| `boost_factor` | [1.0, 2.0] | 0.15 | scales softmax mass of each `focus_labels` class before renorm |
| `max_length` | frozen | — | passthrough from Sabina's proposal / default |

### Confusion signal

Read `predictions_test.csv` (already read elsewhere by the Manager) and compare
`predicted_label` vs `label`:

```
N          = number of rows
neutral_skew = (count(predicted == "neutral") - count(label == "neutral")) / N
EPS        = 0.03

neutral_skew >  EPS   -> over-predicting neutral   -> threshold DOWN  (fewer rows forced to neutral)
neutral_skew < -EPS   -> under-predicting neutral  -> threshold UP    (more rows forced to neutral)
|neutral_skew| <= EPS -> neutral balanced          -> boost UP on focus_labels[0] (weakest class)
```

**Direction note:** `threshold` is the confidence cutoff — rows *below* it are forced to
`neutral` (see `jack_manager.py` module comment). So over-predicting `neutral` calls for a
*lower* threshold. (This corrects the direction shown in the initial brainstorming preview.)

### Hill-climb — `_next_params` rewrite

- **Base** to perturb is `best_params` (the params of the best iteration so far), not the
  last iteration's params.
- **First retune** keeps the current behavior: accept Sabina's `suggested_params` as-is.
  This seeds the first `best_params`.
- **Second retune onward:** build a list of **candidate moves in priority order** from the
  confusion signal:
  1. primary lever + direction (from the signal above),
  2. the other lever, direction UP,
  3. primary lever, opposite direction.

  Apply each candidate to `best_params`, clamp to bounds, and take the **first candidate
  that is in-bounds and not already in `tried_params`**.
- **Feedback is implicit.** A move that regressed is already recorded in `tried_params`, so
  it is skipped on the next step and the loop falls through to the next candidate. No
  separate backtrack bookkeeping is needed.
- **Exhaustion.** If no candidate is both in-bounds and untried, the Manager short-circuits
  to `proceed` (reason: `"search exhausted"`) rather than re-running an identical classify
  pass. The existing `_converged` plateau check still runs in parallel.

### Best-retention + finalize-on-best

New plain (latest-wins) state fields on `ManagerState`:

- `best_accuracy: float` — sentinel `-1.0` until the first evaluation.
- `best_params: dict`
- `best_class_accuracy: dict`

In `decide`, after computing this iteration's `accuracy`: if `accuracy > best_accuracy`,
snapshot the current predictions and update the three `best_*` fields.

```
BEST_PREDS = outputs/best_predictions.csv
if accuracy > best_accuracy and os.path.exists(predictions_path):
    shutil.copyfile(predictions_path, BEST_PREDS)
    best_accuracy = accuracy
    best_params   = params_used_this_iter
    best_class_accuracy = report["class_accuracy"]
```

where `params_used_this_iter` is:

- iteration 1 (`tried_params` empty — fresh classifier): `report["proposal"]["suggested_params"]`
  — Sabina's read of the current classifier, and a sane base to perturb from.
- iteration k > 1: `tried_params[-1]` — the params Nadi actually applied this pass.

This keeps the hill-climb base well-defined even when the *initial* (pre-retune) iteration
turns out to be the best. It is also consistent with the first-retune rule (which accepts
`suggested_params` as-is), so the first retune's `tried_params[-1]` equals the iteration-1
`best_params`.

The `os.path.exists` guard keeps `decide` unit tests (which run without a predictions file)
working — they simply skip the snapshot.

- **`proceed`** samples from `BEST_PREDS` when the snapshot exists, else from
  `predictions_path` (current behavior).
- **`finalize`** reads `BEST_PREDS` when it exists; sets `final_report.final_accuracy =
  best_accuracy` and `class_accuracy = best_class_accuracy`. `loop_iterations` is unchanged.

### Files touched

- `agents/jack_manager.py` — remove `_RETUNE_SCHEDULE`; rewrite `_next_params`; add
  confusion helper + candidate generator; wire best-tracking + snapshot into `decide`;
  point `proceed`/`finalize` at the snapshot; extend `ManagerState`.
- `tests/test_manager.py` — update for the new `_next_params` signature and add hill-climb /
  best-retention cases.
- `docs/jack_manager_guide.ipynb` — rewrite the retune-strategy section.
- `docs/langgraph_pipeline_guide.ipynb` — update loop cells describing param escalation.

## Testing

**Unit (`tests/test_manager.py`):**

- Confusion → lever/direction: three cases (over-neutral → threshold down, under-neutral →
  threshold up, balanced → boost up on weakest focus class).
- Candidate generator: clamps at bounds; skips params already in `tried_params`;
  exhaustion returns the "proceed" signal.
- Best-tracking: snapshot + `best_*` update on an improving accuracy; **no** snapshot or
  update on a regressing accuracy.
- `finalize` reads `best_predictions.csv` and reports `best_accuracy` / `best_class_accuracy`.

**Integration (`pipeline_graph`):**

- Inject fake Nadi/Sabina emitting a scripted accuracy sequence where iteration 2 is the
  best and iteration 3 regresses. Assert `final_results.csv` reflects iteration 2's
  predictions and `final_report.final_accuracy` equals iteration 2's accuracy.

## Risks

At ~0.25 accuracy (near-random for a 3-class problem), the confusion signal is noisy and the
hill-climb may chase noise between roughly equivalent param sets. **Finalize-on-best is the
robust guardrail** — even if the climb wanders, the output can never be worse than the best
iteration observed. Bounds plus the exhaustion short-circuit keep the loop finite.
