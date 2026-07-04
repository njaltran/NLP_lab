# Experiment: can we make the classifier predict up/down instead of collapsing to one class?

**Date:** 2026-07-04
**Setup:** fine-tuned FinBERT (`outputs/finbert_finetuned/`, held-out test 0.4631) run through the
**full 5-agent pipeline** on the clean, COVID-excluded split (`--dataset-end 2019-12-31`,
2,196 test rows), deterministic (no LLMs), 5 max iterations.

## Question
The pipeline reports ~0.50 accuracy but predicts **~90% neutral** (up 0.01 / down 0.15).
Does optimizing for **balanced accuracy** (or changing the decision rule) make it actually
predict up/down — i.e. improve per-class balance?

## What we tried

| Run | Gate metric | Decision rule | Overall acc | up / down / neutral | **Balanced acc** |
|---|---|---|---|---|---|
| **A** | raw accuracy | threshold (pipeline default) | 0.50 | 0.01 / 0.15 / 0.90 | **0.35** |
| **B** | **balanced accuracy** | threshold (pipeline default) | 0.50 | 0.01 / 0.15 / 0.90 | **0.35** |
| **C** | — | **argmax** (no neutral-fallback) | 0.31 | 0.12 / 0.70 / 0.24 | **0.35** |
| *ref* | *standalone fine-tune (argmax)* | *argmax* | *0.46* | *0.10 / 0.33 / 0.71* | *0.38* |
| *baseline* | *all-neutral* | — | *~0.52* | *0 / 0 / 1.0* | *0.33* |
| *baseline* | *random (3-class)* | — | *~0.33* | — | *0.33* |

*(Runs A and B: identical flags, only the gate metric differs. Run C: recomputed from B's
model probabilities by taking argmax instead of the threshold rule.)*

## Findings

1. **Changing the gate metric did nothing (A == B).** Optimizing balanced accuracy instead
   of raw accuracy produced the *exact same* classifier. The retune loop lowered the
   threshold (0.5 → 0.45 → 0.40) either way and **converged at the same point** because
   balanced accuracy barely moved (history `0.34 → 0.34 → 0.35 → 0.35`). The metric decides
   *when to stop* and *which iteration is "best,"* not what the model can do.

2. **The decision rule only moves *which* class dominates — not the skill.**
   The threshold rule collapses to **neutral** (0.90); argmax collapses to **down** (0.70).
   Overall accuracy swings a lot (0.50 → 0.31) because it just tracks how much you predict the
   majority class. But **balanced accuracy is 0.35 in every case.**

3. **Balanced accuracy ≈ 0.35 is the real, invariant ceiling — and it's barely above the
   0.33 random baseline.** No gate metric, threshold, or decision rule changes it, because the
   model genuinely **cannot tell the three classes apart** on one headline. Everything else is
   just relabelling which class it over-predicts.

## Why the classify() threshold rule over-collapses to neutral
`classify()` forces `neutral` whenever the top probability is below `THRESHOLD`. That fit the
pretrained **sentiment** head. But the fine-tuned model already predicts `neutral` as one of its
three classes, so the extra neutral-fallback **double-counts** it. For the fine-tuned head,
plain **argmax** is the natural rule — it stops the neutral collapse (though, per finding 3, it
does not raise balanced skill).

## Conclusion

The "collapse to one class" is not a bug in the metric or the decision rule — it's the model
**operating at near-chance discrimination**, and each rule just picks a different corner to
collapse into:
- optimize **raw accuracy** → collapse to **neutral** (the majority) → looks like 0.50.
- optimize/relax toward **balance** → spread out, but total skill stays 0.35.

So **you cannot get a genuinely balanced, accurate classifier by tuning the loop** — the
information isn't in the input. Raw accuracy (~0.50) is a mirage that reflects the majority-class
share; balanced accuracy (~0.35) is the honest skill.

## Recommendations

1. **Report balanced accuracy (or macro-F1) alongside raw accuracy** so the collapse can't hide
   behind 0.50. Cheap, honest reporting change on Sabina's side.
2. **For the fine-tuned model, use argmax** in `classify()` (Nadi) instead of threshold-neutral —
   fits the 3-class head and kills the fake 0.50. Doesn't need re-training.
3. **Don't expect the loop to fix balance** — the only real levers are *more input signal*:
   aggregate all headlines per ticker-day, add price/momentum features, or lengthen the horizon
   (T+5/T+20). Those change what the model can learn; metric/rule tweaks do not.
4. **Frame for the write-up:** "Balanced accuracy is invariant at ~0.35 (≈ chance) across gate
   metrics and decision rules, so the model is at its discrimination ceiling — the task/inputs,
   not the pipeline, are the limit."

## Reproduce
```bash
# A (raw):       EVALUATOR_OPTIMIZE=accuracy  uv run main.py --dataset-end 2019-12-31 \
#                    --model-dir outputs/finbert_finetuned --max-iterations 5 --no-ollama
# B (balanced):  EVALUATOR_OPTIMIZE=balanced  uv run main.py ... (same flags)
# C (argmax):    recompute argmax over prob_up/prob_down/prob_neutral in predictions_test.csv
```
`EVALUATOR_OPTIMIZE` was a temporary experiment toggle in `agents/sabina_evaluator.py`; the
`raw_accuracy` + `balanced_accuracy` fields it adds to the report are worth keeping.
