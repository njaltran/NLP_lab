# Accuracy-gate optimization log

Branch: `feature/optimize-performance` (off `feature/manager-adaptive-retune`).
Goal: clear the Manager accuracy gate (`target_accuracy = 0.516`) **genuinely** —
i.e. with a model that beats the majority-class baseline via real up/down/neutral
discrimination, NOT by predicting one class. Gate scores the **val** split.

All experiments calibrated offline from `outputs/predictions_test.csv` (FinBERT
probs + true label for both splits) and `data/processed_data.csv`. Bands are
simulable: the band changes only the label definition, so labels are recomputed
from `pct_change`.

## Baseline

`uv run main.py` → proceed at iteration 3, test acc 0.34, gate never cleared.

| split | n | base rates u/d/neutral | majority baseline |
|---|---|---|---|
| val  | 1164 | 0.27 / 0.255 / 0.475 | 0.475 |
| test | 2930 | 0.396 / 0.397 / 0.207 | 0.207 |

## Attempt 1 — confidence-threshold abstention  → REJECTED (gaming)
Route low-confidence FinBERT preds to neutral. val acc climbs 0.353 → 0.421 (thr
0.9) but is **capped by the val all-neutral rate 0.475 < target**. Just predicts
majority; no skill. Cannot clear at ±1% band.

## Attempt 2 — alternative sentiment→move mappings  → NO SIGNAL
Inverted (mean-reversion), binary margin, per-class mean-prob separation.
Mean FinBERT prob by true val label is near-identical:
- true=up:   up 0.366 / down 0.277 / neutral 0.357
- true=down: up 0.362 / down 0.258 / neutral 0.380

**FinBERT headline sentiment carries ~zero next-day directional signal.** No
mapping beats majority.

## Attempt 3 — widen band + abstain  → REJECTED (gaming, user call)
±2% band lifts neutral majority to 0.70; abstain-to-majority then "clears" at val
0.565. This clears the gate but only by (a) redefining a move as ±2% and (b)
predicting neutral 74% of the time. **It is beating the gate by predicting one
class — exactly what we must not do.** Reverted.

## Attempt 4 — trained bag-of-words Naive Bayes  → NO SIGNAL
Trained on the train split's move labels, eval val:

| band | NB val acc | majority | verdict |
|---|---|---|---|
| ±1% | 0.382 | 0.475 | below baseline |
| ±2% | 0.583 | 0.701 | below baseline |

A model trained directly on the labels still can't beat the base rate.

## Attempt 5 — non-text features (ticker, weekday priors)  → NO SIGNAL
Per-ticker and per-weekday majority priors both collapse to the global majority
(val acc == majority at both bands). No exploitable structure.

## Conclusion

**No approach produces genuine skill on this data.** FinBERT sentiment, a trained
text classifier, and metadata priors all land at or below the majority baseline at
every band tested. There is no band where a model that beats its majority also
clears the 0.516 gate — because no model beats the majority here at all.

This is the expected result for next-day equity direction from news headlines:
at a 1-day horizon the signal is dominated by noise (efficient-market behaviour).
The gate is genuinely unclearable with the current target (headline → next-day
up/down/neutral) without gaming the metric.

### Honest options (need a scope decision)
1. **Reframe the target** to something with signal — e.g. predict *magnitude /
   volatility* ("big move vs not"), a *multi-day* horizon, or same-day reaction to
   the news timestamp rather than next-day.
2. **Add genuinely predictive features** — prior-window returns / technicals
   (requires extending the data contract; the current classifier is headline-only).
3. **Accept the negative result**: keep the gate honest and report "no skill above
   baseline" as the finding. The pipeline still works end to end; the ML target is
   just not learnable as posed.

Files are reverted to honest defaults (band ±1%, classifier confidence 0.5).
