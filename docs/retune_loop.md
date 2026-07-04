# The retune loop (as of `feature/manager-adapt`)

Short report on what one pass of the pipeline looks like after adapting the
Manager to Nadi's merged classifier (PR #34) and Sabina's merged evaluator
(PR #30). File formats live in [`data_contracts.md`](./data_contracts.md);
agent roles in [`architecture.md`](./architecture.md).

**One pass** = Nadi classifies → Sabina evaluates → Jack gates. The gate has
three exits: retune (cycle back to Nadi), proceed (sample for Freddi), and
finalize (once Freddi's explanations are back).

## What changed

- **`code_notes` now flows Jack → Nadi.** Sabina's code observations ride
  `retune_request.json`, feeding Nadi's optional Ollama rewrite of
  `classify()`. The rewrite is validated on mock data before it is trusted;
  anything that fails falls back to the deterministic template.
- **`model_dir` is an opt-in flag.** `uv run main.py --model-dir
  outputs/finbert_finetuned` hands the fine-tuned weights folder through the
  pipeline to Nadi's node. Omitted (the default) means pretrained FinBERT —
  the current fine-tune underperforms the all-neutral baseline, so it stays
  off unless explicitly requested.

## Flowchart

```mermaid
flowchart TD
    A["Aurora: process<br/>FNSPID + yfinance<br/>writes processed_data.csv, split train/test by date"] --> N

    N["Nadi: classify<br/>FinBERT, or fine-tuned weights via model_dir<br/>on retune: template param swap, or Ollama rewrite<br/>from code_notes — validated on mock, else fallback<br/>writes predictions_test.csv"] --> S
    S["Sabina: evaluate<br/>deterministic metrics, LLM polishes reason + code_notes<br/>writes evaluation_report.json"] --> G

    G{"Jack: gate<br/>accuracy at least 0.60? cap hit? converged?"}
    G -->|retune| R
    R["Jack writes retune_request.json<br/>params from schedule, code_notes passed through"] --> N

    G -->|proceed| B
    B["select_best: if an earlier iteration scored higher,<br/>restore its snapshot from outputs/best/<br/>and redraw the explanation sample"] --> P
    P["Jack writes sample_for_explanation.csv"] --> F
    F["Freddi: explain<br/>Ollama justification per row<br/>writes explanations.csv"] --> Z
    Z["Jack: finalize<br/>writes final_results.csv + final_report.json"] --> E([END])
```

## Gate rules (unchanged)

- The first retune accepts Sabina's `suggested_params` as-is; every retune
  after that walks the Manager's `_RETUNE_SCHEDULE` (threshold ↓,
  boost_factor ↑), skipping combinations already tried. (`max_length` was
  dropped as a knob — no headline exceeds 128 tokens.) If the last iteration
  regressed more than 0.05 below the best, the Manager instead reverts to the
  best iteration's params and perturbs one knob.
- Aggregate accuracy can't clear the target while any class sits below the
  per-class recall floor (0.05) — an all-neutral collapse no longer "passes".
- Each `main.py` run starts by clearing the previous run's loop artifacts from
  `outputs/` (fine-tuned weights and the finetune report are kept).
- Proceed fires on any of: target accuracy (0.60) cleared, iteration cap (5)
  hit, or convergence — the best of the last 2 iterations gained less than
  0.01 over the best before them.
- On proceed, `select_best` restores the highest-accuracy iteration's
  artifacts (predictions/report/classifier snapshotted to `outputs/best/` on
  each new best), so a regressed final retune can't ship worse results than an
  earlier pass — accuracy did regress 0.39 → 0.23 on the 2026-07-04 run.
