# Magnitude reframe — design spec

**Date:** 2026-07-05
**Branch:** `feature/optimize-performance` (off `feature/manager-adaptive-retune`)
**Author:** Jack (Manager owner), with team sign-off required (crosses all lanes)

## Problem

The pipeline predicts next-day **direction** (`up`/`down`/`neutral`) from news
headlines and cannot clear the accuracy gate honestly. Measured on this data
(`docs/optimization_log.md`):

- FinBERT sentiment: val 0.36 vs majority 0.475 — no edge.
- Mean FinBERT prob is near-identical across true labels — **~zero directional signal**.
- Trained bag-of-words NB: val 0.38 (±1%) / 0.58 (±2%) — still below majority.
- Ticker / weekday priors: collapse to majority.

Direction at a 1-day horizon is not predictable above the base rate here.

**But magnitude is.** Predicting **big move (|Δ|>1%) vs small move (≤1%)**, a trained
NB beats the val majority with balanced recall:

| target | val acc | val majority | recall (big/small) |
|---|---|---|---|
| direction (3-class) | 0.38 | 0.475 | — (below) |
| **magnitude (2-class, ±1%)** | **0.564** | **0.525** | 0.72 / 0.39 |

The big-move recall is consistent out-of-sample (val 0.72, test 0.65), confirming a
real volatility signal rather than noise. **News predicts volatility, not direction.**

## Goal

Reframe the whole pipeline to predict **magnitude** and clear the gate with genuine
skill (beat majority + no class collapse), using a **fine-tuned FinBERT** classifier
whose fine-tune hyperparameters are driven by the Manager's adaptive retune loop.

Success: val accuracy ≥ 0.525 AND both class recalls above the collapse floor, reached
by a genuine model (not one-class abstention).

## Decisions (locked with user)

1. Target: 3-class `up/down/neutral` → 2-class `big/small`.
2. Cut: reuse Aurora's `--threshold`; default ±1% (0.01) — the only cut with signal.
3. Classifier: **fine-tuned FinBERT, binary head**.
4. Gate: `target_accuracy = 0.525`, keep `accuracy ≥ target AND not collapsed`.
5. Adaptation: the retune loop tunes **fine-tune hyperparameters** (lr, epochs).

## Architecture

### A. Label reframe (data + contracts)

- `agents/aurora_processing.py` `_assign_label`:
  `"big" if abs(pct_change) > threshold*100 else "small"` (was 3-branch).
- `agents/contracts.py`: `LABELS = ["big", "small"]`;
  `PREDICTION_COLUMNS` prob fields `prob_up/prob_down/prob_neutral` → `prob_big/prob_small`.
- `docs/data_contracts.md`: update Handoff 1 (label values) and Handoff 2 (prob column
  rename), plus the label-vocabulary note.
- `mock_data/*.csv`: regenerate every handoff file's `label`, `predicted_label`, and prob
  columns to the 2-class shape. These are the integration fixtures — they must match the
  new contract exactly.
- `docs/architecture.md`: mission line "up/down/neutral" → "big/small (magnitude)".

This is a coordinated contract change (AGENTS.md golden rule 2): docs + mock_data +
every downstream owner updated in the same change.

### B. Classifier — fine-tuned FinBERT, tuned in the loop

**`agents/finetune_finbert.py` → binary.**
- `LABELS = ["big", "small"]`, `LABEL_TO_ID = {"big":0, "small":1}`, `num_labels=2`.
- `class_weights` denominator `2 * count` (was 3).
- Accept `--lr` / `--epochs` (already present) and expose `finetune(...)` as an importable
  function returning `(model_dir, best_val_accuracy)` so the loop can call it without a
  subprocess. Keep the CLI `main()` for standalone runs.
- Uses `mps` on this Mac (device picker already handles it), so training is feasible.

**`agents/nadi_classifier.py`.**
- The generated `CLASSIFIER_TEMPLATE` already reads `id2label` from a fine-tuned
  `MODEL_DIR`, so a 2-class model maps automatically. Remove the pretrained
  `SENTIMENT_TO_LABEL` path's 3-class assumptions; output `prob_big`/`prob_small`.
- `generate_code`/`classify`: on each pass, ensure a fine-tuned model exists for the
  current hyperparameters. Iteration 0 fine-tunes with defaults; on a retune, read
  `suggested_params.lr` / `suggested_params.epochs` from `retune_request.json`, re-fine-tune,
  and run predictions over the fresh model. Fine-tuning moves **into** the classify node.

**Cost guardrails (accepted):** `max_iterations` default 2–3; per-iteration epoch budget
2–3; coarse lr grid. Each iteration re-trains FinBERT on ~10k rows — minutes on `mps`.

### C. Gate / eval / explanation

- **Manager (`agents/jack_manager.py`, my lane):** `target_accuracy` default → 0.525.
  Gate logic unchanged (`accuracy ≥ target AND not collapsed`); the collapse floor now
  spans 2 classes. The retune proposal's `suggested_params` carry lr/epochs instead of the
  old threshold/boost knobs.
- **Evaluator (`agents/sabina_evaluator.py`):** per-class recall over `{big, small}`;
  `class_accuracy` dict keys change accordingly. `report_score` collapse check spans 2 classes.
- **Explanation (`agents/freddi_explanation.py`):** justify "big move expected" vs
  "quiet day" instead of direction wording.

## Data flow (unchanged shape, new payload)

```
process(±1% big/small labels) → classify(fine-tune@hyperparams → predict big/small)
  → evaluate(val recall big/small) → gate(≥0.525?) ─retune(new lr/epochs)→ classify [CYCLE]
                                                     └proceed→ select_best → evaluate_test
                                                              → explain → finalize
```

## Testing

- `tests/test_classifier.py`, `test_pipeline_graph.py`, `test_finetune_split.py`: update
  fixtures/labels to 2-class; the fake classifier in pipeline tests emits `big/small`.
- New: `finetune(...)` returns `(model_dir, best_val_accuracy)` — unit-test on `--limit`
  smoke rows.
- Integration: each agent reads the new `mock_data/` and produces contract-matching output
  before the real run (AGENTS.md golden rule 3).
- Baseline sanity: the NB magnitude result (val 0.564) is the signal floor; if fine-tuned
  FinBERT lands below it, that's surfaced in `finetune_report.json`, not hidden.

## Risks

- **Fine-tune may not beat 0.525.** The 3-class fine-tune got val 0.457. Mitigation: the
  retune loop tunes lr/epochs; NB floor documents the achievable edge; honest reporting if
  it misses.
- **Loop cost.** Fine-tuning per iteration is the expensive part — capped by guardrails above.
- **Contract breakage.** Prob-column rename touches every agent; the single-change rule
  (docs + mock_data + all agents together) is the mitigation.

## Out of scope

Multi-day horizon, same-day/intraday reaction, non-text features — all need price data not
in the current contract. Recorded as future reframes in `docs/optimization_log.md`.
