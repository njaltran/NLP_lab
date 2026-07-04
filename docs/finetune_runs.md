# FinBERT fine-tune runs — results log (2026-07-04)

Weights and data are gitignored (`outputs/`, `data/processed_data.csv`), so this
doc is the record. Everything below is reproducible from a clean checkout: the
script is seeded (default 42) and the data is regenerated deterministically by
the Processing agent.

## Reproduce

```bash
uv sync
uv run agents/aurora_processing.py            # regenerates data/processed_data.csv (split column required)
uv run agents/finetune_finbert.py --data data/processed_data.csv --epochs 6 --lr 1e-5
```

Weights land in `outputs/finbert_finetuned/`, metrics in `outputs/finetune_report.json`.
The loop only uses the weights when explicitly asked: `uv run main.py --model-dir outputs/finbert_finetuned`.

## Dataset (identical across all runs)

- 14,689 rows; train 10,595 / val 1,164 / test 2,930
- Split by date, not randomly: train 2012-01-16 → 2019-12-02, test 2020-02-08 → 2020-06-11
- Label mix shifts hard across the split (COVID window):
  train ≈ 49% neutral / 26% up / 25% down; test ≈ 21% neutral / 40% up / 40% down
- Baselines on this test set: all-neutral = 0.207, uniform random ≈ 0.333

## Runs (seed 42, batch 16, max_length 128, device mps)

| run | lr | epochs | best val (epoch) | test acc | up | down | neutral |
|---|---|---|---|---|---|---|---|
| pretrained FinBERT (no fine-tune, via loop) | — | — | — | 0.21 | 0.00 | 0.00 | 1.00 |
| A | 2e-5 | 3 | 0.4536 (1) | 0.2546 | 0.20 | 0.026 | 0.80 |
| B | 2e-5 | 10 | 0.4545 (1) | 0.2553 | 0.20 | 0.025 | 0.80 |
| C | 1e-5 | 6 | 0.4570 (1) | **0.2706** | 0.28 | 0.048 | 0.69 |

Run C's per-epoch trajectory (representative of all runs): val 0.457 at epoch 1,
then 0.33–0.41 for every later epoch while train loss keeps falling
(1.13 → 0.71) — textbook overfitting after the first pass.

## Findings

1. **Every run's best checkpoint is epoch 1.** Extra epochs (B: 10) never beat
   it; the best-val save logic means they cost GPU time and change nothing.
2. **Halving the lr (C) is the only knob that moved anything**: +1.5 points test
   accuracy over B, less neutral-collapse. Marginal, not a fix.
3. **Fine-tuned beats pretrained on this test window** (0.27 vs 0.21) because
   pretrained collapses to all-neutral and neutral is rare during the COVID
   crash. Note this *reverses* the 2026-07-02 verdict, which was measured on a
   pre-COVID, neutral-heavy window where all-neutral scored 0.516 — which model
   "wins" is decided by the test window's label mix, not the model.
4. **`down` is near-dead in every run** (recall ≤ 0.05): headlines don't encode
   next-day drops in a way FinBERT captures.
5. **Ceiling is the data, not the optimizer.** Three independent runs land on
   ~0.25–0.27. Random (0.33) beats all of them; the 0.60 target is out of reach
   with one-headline → next-day-move as the task.

## Open follow-ups

- Pre-COVID dataset cutoff (drop rows after ~2019-12-31) to remove the
  train/test regime shift — Aurora's lane.
- Renegotiate the 0.60 target, or extend the task (headline aggregation per
  ticker-day, non-text features).
