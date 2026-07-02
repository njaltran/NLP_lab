# Fine-tuning FinBERT on move labels — design

**Date:** 2026-07-02
**Owner:** Jack (proposal) — implementation lands in Nadi's classifier, so **requires Nadi's sign-off**; the split change touches Aurora's output, so **requires Aurora's sign-off** too.
**Status:** proposed — awaiting review

## Problem

The retune loop cannot improve accuracy because there is nothing to tune. Analysis of
the 2026-07-02 run (13,527 rows, 30 tickers, 2012-2020):

| Measurement | Value |
|---|---|
| corr(prob_up − prob_down, pct_change) | **−0.008** (zero) |
| Model accuracy | 0.368 |
| Statistical-independence baseline (same marginals, random pairing) | 0.356 |
| All-neutral majority baseline | **0.467** |
| Accuracy on high-confidence rows (conf ≥ 0.9, n=5,307) | 0.352 (worse than average) |

FinBERT's pretrained sentiment head is statistically independent of next-day price
direction on this dataset. The confusion matrix has no diagonal structure — every true
class receives the same ~25/47/28 prediction split. Threshold/boost/max_length tuning
reshuffles marginals of noise; the loop's threshold-lowering direction actually moves
predictions *away* from the majority class and below the 0.467 all-neutral baseline.

The only lever that can change the ceiling is replacing the frozen sentiment head with
a head trained on our actual labels.

## Goals

1. **Measure the real ceiling:** fine-tune FinBERT end-to-end on `up`/`down`/`neutral`
   move labels and report honest held-out accuracy.
2. **Leak-free evaluation:** time-based train/test split so no future information
   reaches training.
3. **Contract-compatible:** the fine-tuned classifier still emits `predictions_test.csv`
   per Handoff 2 — Sabina, Jack, and Freddi are unaffected downstream.

## Non-goals

- Hitting the 0.60 target. Expected outcome is ~0.45–0.52; if headlines reliably
  predicted next-day direction the strategy would be an arbitrage machine. Part of this
  increment's value is evidence for **renegotiating the target** with the team.
- Feature engineering beyond text (momentum, volume) — future increment.
- Per-retune-iteration retraining. Training runs once; the retune loop keeps tuning
  cheap inference-time params on top of the fine-tuned model.
- Touching the dlt pipeline.

## Design

### 1. Train/test split (Aurora's contract — needs sign-off)

`processed_data.csv` gains a `split` column (`train`/`test`), assigned **by date**:
train = rows before the cut, test = rows on/after it.

**COVID caveat (measured):** the natural 80/20 cut lands at **2020-01-17**, making the
test set the COVID crash — label distribution shifts from 47/27/26 (neutral/up/down)
overall to 28/35/38. Options:

- **(a) cut at 2019-01-01** — test = calendar 2019 + Jan-Jun 2020, dilutes but keeps
  some regime shift;
- **(b) cut at 2020-01-17 (80/20)** and report the shift explicitly;
- **(c) end the dataset at 2019-12-31** and cut inside it — cleanest science, drops data.

**Recommendation: (c)** for the headline result, with (b) reported as a secondary
"regime shift" number. Decision belongs to the team.

Downstream contract impact: Nadi trains on `split=train` rows only and predicts
`split=test` rows only, so `predictions_test.csv` stays all-test (Sabina's existing
validation already enforces this — no change on her side).

### 2. Training script (Nadi's module — needs sign-off)

New `agents/finetune_finbert.py` (standalone, run manually, not part of the loop):

- HF `Trainer` on `ProsusAI/finbert` with a fresh 3-label head
  (`num_labels=3`, `ignore_mismatched_sizes=True`).
- Tokenize `article_title`, `max_length=128`; class weights from train distribution
  (neutral-heavy) via weighted cross-entropy.
- 3 epochs, batch 16, lr 2e-5, MPS/CPU autodetect. ~13k rows → roughly 20-60 min on
  an M-series Mac. Early stopping on a small validation slice (last 10% of *train*
  dates, so validation is also time-ordered).
- Saves to `outputs/finbert_finetuned/` (gitignored — weights are ~440 MB).
- Emits `outputs/finetune_report.json`: train/val/test accuracy, per-class metrics,
  train date range, seed, epochs. Reproducibility record.

### 3. Inference integration (Nadi's template)

`CLASSIFIER_TEMPLATE` gains one line: `MODEL_DIR = {model_dir}` — when
`outputs/finbert_finetuned/` exists, load it instead of `ProsusAI/finbert`; otherwise
fall back to the pretrained model (current behavior, so nothing breaks before the
first training run). The label mapping comes from the fine-tuned config (`up`/`down`/
`neutral` directly — no more sentiment→direction translation for the fine-tuned path).

The retune loop keeps working unchanged on top: threshold/boost/max_length still apply
to the fine-tuned model's softmax.

### 4. Success criteria

The increment is done when `finetune_report.json` exists with a leak-free test
accuracy, whatever the number. Decision rules for the team afterwards:

- test accuracy ≥ 0.52 → adopt fine-tuned model as the loop's default;
- 0.467–0.52 → adopt, but reframe the target around the measured ceiling;
- < 0.467 (all-neutral baseline) → do not adopt; the honest recommendation becomes
  "aggregate headlines per ticker-day and/or add price features, or reframe the task."

## Data flow

Unchanged shape: `processed_data.csv` → `predictions_test.csv` → `evaluation_report.json`
→ loop. New artifacts (`finbert_finetuned/`, `finetune_report.json`) sit outside the
handoff chain; only Nadi's generated script reads the model dir.

## Error handling

- Missing `split` column → training script fails loudly with a pointer to the Aurora
  change; generated classifier falls back to pretrained (current behavior).
- Missing/corrupt fine-tuned model dir → same fallback, logged.
- No new failure modes in the loop itself.

## Testing

- Unit: split logic is date-monotonic (max train date < min test date).
- Unit: generated classifier prefers the fine-tuned dir when present, falls back when
  absent (template string assertions, no GPU needed).
- Integration (slow, opt-in marker): 50-row smoke train run completes and emits a
  valid `finetune_report.json`.
- Regression: full existing suite (27 tests) unaffected when no fine-tuned model exists.

## Complexity & size

| Item | Complexity | LOC |
|---|---|---|
| Aurora split column + date cut | low | ~15 |
| `agents/finetune_finbert.py` | medium | ~120 |
| Template/`generate_code` model-dir support | low | ~15 |
| Tests | low-medium | ~60 |
| Contract doc + mock_data updates | trivial | ~10 |
| **Total** | **Medium** | **~220** |

## Sign-off needed (per AGENTS.md golden rules 2 & 6)

| Owner | Change | Why |
|---|---|---|
| Aurora | `split` column in `processed_data.csv` | her output format |
| Nadi | training script + template change | his module |
| Team | split-date choice (a/b/c above) + target renegotiation | affects the headline result |

## Deferred

- Headline aggregation per ticker-day (multiple headlines → one prediction).
- Non-text features (momentum, volume) — would need a different model head.
- Per-iteration retraining driven by the retune loop.
