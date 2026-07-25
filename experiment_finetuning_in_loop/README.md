# Experiment: fine-tuning inside the retune loop

> **This is not part of the architecture we presented.** It was built after the
> presentation as an improvement attempt, and we kept it in its own folder so the
> submitted pipeline stays exactly as it was during the project. Nothing in
> `agents/` or `main.py` is modified by anything in here.

## The gap this addresses

In the submitted pipeline, training happens once, offline
(`agents/finetune_finbert.py`), and each retune only adjusts inference
settings — `threshold` and `boost_factor` — on top of fixed weights.

That means the loop can rearrange predictions in response to the evaluator,
but it can never actually learn from that feedback. When Sabina reports that
the `down` class has collapsed to 0.05 recall, the model that produced those
predictions is the same model that will produce the next ones.

This experiment closes that gap: every retune trains the model on the class the
evaluator flagged.

## What changes

One node is added to the front of the Classifier Agent's internal graph:

```
submitted:   START → generate_code → run_classifier → END
here:        START → fine_tune → generate_code → run_classifier → END
```

- **First pass:** `fine_tune` does nothing. The pipeline classifies with
  pretrained FinBERT, giving the Manager a baseline to compare trained rounds
  against.
- **Every retune:** `fine_tune` continues training from the previous round's
  checkpoint, weighting the loss towards the evaluator's `focus_labels`, and
  hands the new checkpoint to `generate_code`.

## Files

| File | What it is |
|---|---|
| `train_head.py` | The training loop as a callable function. `agents/finetune_finbert.py` keeps its loop inside `main()`, so it can only be run from a terminal; this wraps the same steps and adds `parent_model_dir` so each round continues from the last. Data splitting, class weighting and scoring are imported from the real fine-tuner, not reimplemented. |
| `looping_classifier.py` | The `fine_tune` node, plus `LoopFineTuningClassifier`, which subclasses the real `ClassifierAgent` and overrides only `build_graph`. |
| `run_experiment.py` | Builds the normal five-agent bundle with the classifier swapped, then runs the same `build_pipeline` graph. |

`agents/pipeline_graph.py` already takes an `Agents` bundle so tests can inject
fakes — we use that same extension point, which is why no submitted file needed
to change.

## Why one epoch, and a lower learning rate

The offline fine-tuner uses **3 epochs at `lr=2e-5`**. This experiment uses
**1 epoch at `lr=5e-6`**, and that is the most important setting here.

`docs/finetune_runs.md` records three offline runs — 3, 10 and 6 epochs. **In
every one, validation accuracy peaked at epoch 1 and got worse afterwards**
while training loss kept falling: textbook overfitting on a small, weak-signal
dataset.

A loop that trains 3 epochs per retune would therefore compound that overfitting
on every pass — five retunes would mean fifteen epochs of a model that already
starts degrading after one. So:

- **1 epoch per round** — one pass over the training rows per retune
- **`lr=5e-6`** (about a quarter of the offline rate) — each round nudges the
  weights rather than moving them far, because the loop revisits the same rows
  repeatedly
- **`FOCUS_BOOST = 1.5`** — the loss weight of whatever class the evaluator
  flagged is multiplied by 1.5, so the round pushes hardest where the pipeline
  is actually failing

`--max-iterations` also defaults to **3** here rather than 5, because every
iteration is now a training run.

## Running it

From the repo root:

```bash
uv run experiment_finetuning_in_loop/run_experiment.py --no-ollama --dataset-end 2019-12-31
```

**This is slow.** Every retune is a training round rather than a fast inference
pass — budget roughly 5–15 minutes per iteration on a laptop GPU, so 20–45
minutes overall.

Everything this run writes goes to `experiment_finetuning_in_loop/outputs/`, not
the submitted pipeline's `outputs/`. That isolation matters: the pipeline clears
its loop artifacts at the start of every run, so without it an experiment run
would delete the committed results of the run we actually presented. The script
changes its working directory before building the graph, which redirects all the
contract paths at once; the input data still loads from the repo root.

Alongside the usual contract files, two extra artifacts are written so the run
can be read without repeating it:

- `outputs/finbert_loop_finetuned/` — the checkpoint after each round
- `outputs/loop_finetune_history.json` — one entry per training round:
  validation accuracy, per-class accuracy, the focus labels used, and which
  checkpoint the round started from

## What we expect, and why we are reporting it either way

We do **not** expect this to beat the 0.516 majority-class baseline. Every
result in the main submission points to the ceiling being the *data* — one
headline carries very little information about the next day's price move, and
Karaoglu & Gowda (2026) report the same across five models and six horizons.

The point of the experiment is architectural: it makes the loop genuinely
*iterative training* rather than iterative re-scoring, which is the more honest
version of the "self-correcting pipeline" claim. Whether the accuracy moves or
not, the outcome is worth reporting — and if it does not move, that is one more
piece of evidence for the same conclusion.
