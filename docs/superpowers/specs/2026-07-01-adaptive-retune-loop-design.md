# Adaptive retune loop — design

**Date:** 2026-07-01
**Owner:** Jack (Manager agent)
**Status:** proposed — awaiting review

## Problem

The retune loop is open-loop. It accumulates state but never reads it back into a
decision, so every iteration repeats. Evidence from the 2026-06-30 end-to-end run:
five iterations, all at accuracy **0.37**, then the iteration cap force-proceeded.

Three concrete causes:

1. **`decision_log` is write-only.** It is appended by every node and read by none.
2. **`decide` sees only the current report** — this accuracy vs. target vs. the
   iteration counter. It cannot tell that accuracy is flat across iterations.
3. **The retune request is constant.** Sabina proposes `threshold 0.5 / max_length 128`
   (the defaults) every time, so Nadi regenerates a near-identical classifier and
   lands at the same accuracy. The loop spins without adapting.

## Goals

1. **Convergence:** stop the loop early when accuracy stops improving, instead of
   burning the full iteration budget.
2. **Adaptation:** vary the retune hyperparameters across iterations, driven by what
   has already been tried, so each pass is a genuinely different attempt.
3. **Auditability:** the accuracy trend and the params tried are recorded in state
   and surfaced in the decision record.

## Non-goals

- Folding the five agents into one cyclic LangGraph (tracked separately as
  "Increment 2"). This design keeps each agent an independent graph.
- Changing the FinBERT model, the data contracts, or any handoff file *format*.
- Touching Aurora, Freddi, or the dlt pipeline.

## Design

The cross-iteration intelligence lives in the **Manager**, which already holds the
loop state via its checkpointer and already has an override mechanism (it can rewrite
Sabina's proposal when it writes `retune_request.json`). Sabina stays stateless — it
still proposes a baseline from the current report; the Manager adapts it using history.
This confines the change to `jack_manager.py` plus two additive state fields, and
requires **no change to Nadi** because it already applies `threshold`, `max_length`,
and `focus_labels` from the retune request.

### State additions (`ManagerState` in `agents/jack_manager.py`)

Two reducer fields, appended each iteration (mirroring the existing `decision_log`):

| Field | Type | Meaning |
|---|---|---|
| `accuracy_history` | `Annotated[list[float], operator.add]` | one accuracy per completed iteration |
| `tried_params` | `Annotated[list[dict], operator.add]` | the `suggested_params` used for each retune |

### Convergence (`decide` node)

After appending the current accuracy to `accuracy_history`, force `proceed` when the
loop has stalled, even if still below target:

- **Rule:** with at least `patience` prior iterations, if the best accuracy over the
  last `patience` iterations has not improved by at least `min_delta`, treat the loop
  as converged and proceed.
- **Config:** `patience = 2`, `min_delta = 0.01`, set on `ManagerAgent.__init__`
  alongside `target_accuracy` / `max_iterations`.
- The existing cap (`max_iterations`) remains as the hard backstop.

`final_action = proceed` when `cleared OR cap_hit OR converged`. The `notes`/rationale
records which of the three fired, plus the accuracy trend.

### Adaptation (`write_retune` node)

When the gate resolves to retune, the Manager computes the next parameter set from a
deterministic escalation schedule keyed by the retune count, skipping any set already
present in `tried_params`:

1. lower `threshold` in steps (0.50 → 0.45 → 0.40 → 0.35) to reduce the
   forced-`neutral` bias seen in the run;
2. widen `max_length` (128 → 192 → 256) as a secondary lever;
3. keep `focus_labels` on the weakest class from the current report.

The chosen set is written into `retune_request.json`'s `suggested_params` and recorded
as an `override` in `decision.json` (the existing override path already supports this).
Because the values live inside the free-form `suggested_params` object, **no data
contract changes.**

## Data flow

Unchanged. Same files, same handoffs (`processed_data.csv` → `predictions_test.csv` →
`evaluation_report.json` → `retune_request.json` → …). Only the *values* inside
`suggested_params`, and the Manager's stop condition, differ across iterations.

## Error handling

The convergence and escalation logic are pure functions of state — no new I/O, no new
failure modes. Guards: empty/short `accuracy_history` disables convergence (loop runs
normally until it has `patience` samples); an exhausted escalation schedule falls back
to the last set (the cap still bounds the loop).

## Testing

- **New:** convergence forces `proceed` when accuracy is flat and below target
  (feed a repeating-accuracy history, assert early proceed before the cap).
- **New:** consecutive retunes emit **different** `suggested_params` (assert
  `tried_params` has no duplicates across a multi-iteration run).
- **New:** `accuracy_history` accumulates one entry per iteration.
- **Regression:** existing `test_manager.py` cases (accept/override, full lifecycle,
  reproducible sampling) must still pass. Convergence defaults (`patience = 2`) do not
  trigger on the single-retune lifecycle test, but the suite will be re-run and any
  assertion coupled to a fixed iteration count adjusted.

## Complexity & size

| Item | Complexity | LOC |
|---|---|---|
| State fields (`agents/jack_manager.py`) | trivial | ~4 |
| Convergence in `decide` | low | ~15 |
| Escalation in `write_retune` | medium | ~25 |
| Rationale/notes surfacing history | trivial | ~8 |
| New + updated tests | low–medium | ~40 |
| **Total** | **Medium** | **~90** |

**Sign-off:** `agents/jack_manager.py` is Jack's. The two new `ManagerState` fields are
Manager-internal (no other agent reads them). No handoff format changes, so no
downstream agent is affected — a courtesy heads-up to Nadi/Sabina suffices per
AGENTS.md, not a contract negotiation.

## Follow-up improvements (2026-07-02)

Increment 1 (convergence + escalation, above) shipped but the 2026-06-30 run still
plateaued at accuracy **0.37** across all 3 iterations — the loop stopped repeating
identical params, but each retune still wasn't moving the needle much. Five small
fixes, in the order applied:

| # | Change | File | Commit | Impact |
|---|---|---|---|---|
| 1 | `_next_threshold()`: retune proposes classifier's current `THRESHOLD` stepped down 0.05 (floored), instead of Sabina always suggesting fixed 0.5 | `sabina_evaluator.py` | `41bf309` | Each retune now explores a gate not already tried on the *first* retune too, not just later ones |
| 2 | `decision.json` now includes cumulative `accuracy_history` | `jack_manager.py` | `f8ed00c` | Auditability fix — the file is overwritten every iteration, so the per-iteration trend was previously lost once the loop moved past iteration 1 |
| 3 | `focus_labels` widened from "exact min score" to "within `FOCUS_MARGIN` (0.05) of weakest" | `sabina_evaluator.py` | `de318f9` | Fixes a real bug: with `up=0.30, down=0.28, neutral=0.45`, only `down` was ever boosted — `up`, nearly as broken, got no help. Also widened `_RETUNE_SCHEDULE` from 4 to 6 finer steps |
| 4 | Aligned Sabina's `THRESHOLD_FLOOR` (was 0.35) with the Manager schedule's floor (0.20) | `sabina_evaluator.py` | `57f7d02` | The first retune stopped exploring earlier than later retunes would — inconsistency, not by design |
| 5 | Nadi's focus-label boost (hardcoded `1.25x`) is now a retune param (`boost_factor`), escalated 1.25→1.75 in `_RETUNE_SCHEDULE` | `nadi_classifier.py`, `jack_manager.py` | `f712c82` | The one lever that changes *how hard* focus labels get pushed, not just which gate/labels are targeted. **Crosses into Nadi's owned file** — flagged in the commit, not yet signed off per AGENTS.md |

**Ceiling caveat (unchanged from the original problem statement):** all five fixes
reshuffle a frozen, pretrained FinBERT sentiment head — none of them retrain the
model. Sentiment (positive/negative/neutral) is a weak proxy for next-day price
direction, so these are expected to make each retune step count for more, not to
guarantee reaching the 0.60 target. Actually fine-tuning FinBERT on labeled
up/down/neutral data is the only change identified that could raise the ceiling
itself; it's a much larger scope change (~100+ LOC, new training path, Nadi's file)
and is not part of this increment.

## Deferred (future increments)

- **1b — cycle subgraph:** make `classifier → evaluator → gate` a LangGraph subgraph
  with an internal cycle (~40 LOC), to demonstrate a real graph cycle without the full
  rewrite.
- **Increment 2 — single cyclic graph:** fold all five agents into one graph
  (~250–300 LOC churn, full-team sign-off, sacrifices per-agent test isolation).
  Elegance over capability; not recommended for now.
