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

**This will take some time!! (one and only run lasted around half an hour)** 

Everything this run writes goes to `experiment_finetuning_in_loop/outputs/`, not
the submitted pipeline's `outputs/`. That isolation matters because the pipeline clears
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

## Results

The experiment ran once 25-07-2026 and gave the following output (we have included only the final output file just to not make the file even heavier):

Settings: `--no-ollama --dataset-end 2019-12-31`, default cap of 3 iterations —
one baseline pass on pretrained FinBERT, then two training rounds.

| | Submitted pipeline | This experiment |
|---|---|---|
| Test accuracy | 0.50 | 0.37 |
| `up` recall | 0.36 | 0.33 |
| `down` recall | 0.05 | 0.27 |
| `neutral` recall | 0.76 | 0.43 |
| Balanced accuracy | 0.39 | 0.34 |

*(majority-class baseline 0.516; chance, on balanced accuracy, 0.333)*

**1. It fixed the class collapse, which is what it was built to do.** `down` recall went
from 0.05 to 0.27, and the model stopped hiding behind `neutral`. It predicted
26/3/71 percent up/down/neutral before and 31/25/43 after, against a true split of
26/22/52. The evaluator's feedback reached training and changed the model.

**2. The accuracy still went down, not up.** Spreading the predictions out removed the
inflation the 0.50 depended on — that number came from predicting `neutral` for 71%
of a test set that is only 52% neutral. Balanced accuracy, which cannot be inflated
that way, moved from 0.39 to 0.34: no better than chance either way.

**3. The loop overcorrects.** Round 1 was told `up` and `down` were weak and swung
almost entirely to `down` (recall 0.87, with `neutral` down at 0.02). Round 2 was told
`up` was weak and swung back to `up` (0.56). Validation accuracy went
`0.35 → 0.40 → 0.34` — it peaked mid-loop and then regressed. `FOCUS_BOOST = 1.5` is
too strong when each round continues from the previous checkpoint, because the push
accumulates across rounds. A smaller boost, or one that decays as rounds progress,
is the obvious next thing to try.

The honest summary: making the loop train genuinely changed the model's behaviour, but
not its skill. That matches everything else we measured — the pipeline's threshold
rule, a balanced-accuracy gate, argmax, the standalone fine-tune and this experiment
all land between 0.34 and 0.39 balanced accuracy. Six fairly different setups, one
narrow band, just above chance.

Caveats: this is a single run of two training rounds, stopped by the iteration cap
rather than by converging, so the oscillation might settle if it ran longer.
Explanations used the offline fallback (no `HF_TOKEN` set), which does not affect any
of the numbers above.



