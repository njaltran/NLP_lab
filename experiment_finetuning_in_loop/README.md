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

This experiment closes that gap: every retune trains the model again, continuing
from the previous round, instead of only re-running it with new settings.

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
  checkpoint and hands the new checkpoint to `generate_code`.

The training objective is identical every round — the same class weighting the
offline fine-tuner uses. That is deliberate: this experiment changes exactly one
thing, *when* training happens, so anything that moves can be attributed to it.

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

A run writes the usual contract files plus two extras: `outputs/finbert_loop_finetuned/`
(the checkpoint after each round) and `outputs/loop_finetune_history.json` (one entry
per training round — validation accuracy, per-class accuracy, and which checkpoint the
round started from).

Only three of those are committed, so the results below can be checked without
re-running anything, without adding much to the submission:

| Committed file | What it shows |
|---|---|
| `outputs/final_report.json` | the headline numbers in the table below |
| `outputs/final_results.csv` | every test prediction with its explanation |
| `outputs/loop_finetune_history.json` | what each training round scored |

The rest (checkpoints, intermediate predictions, the evaluator report) are
regenerated by running the script.

## Results

Run on 25-07-2026 with `--no-ollama --dataset-end 2019-12-31` and the default cap
of 3 iterations: one baseline pass on pretrained FinBERT, then two training rounds.

| | Submitted pipeline | This experiment |
|---|---|---|
| Test accuracy | 0.50 | 0.46 |
| `up` recall | 0.36 | 0.38 |
| `down` recall | 0.05 | 0.10 |
| `neutral` recall | 0.76 | 0.67 |
| Balanced accuracy | 0.390 | 0.383 |

*(majority-class baseline 0.516; chance, on balanced accuracy, 0.333)*

**Training inside the loop changes almost nothing.** Balanced accuracy moves from
0.390 to 0.383 — a difference of 0.007, which is noise. The model still predicts
`neutral` for 61% of a test set that is 52% neutral, and `down` recall only creeps
from 0.05 to 0.10. Raw accuracy drifts down from 0.50 to 0.46 for the same reason it
was high in the first place: slightly less over-prediction of the majority class.

**The two rounds converge rather than oscillate.** Round 1, training from pretrained
with the standard class weighting, overshoots toward `down` (recall 0.61). Round 2
continues from that checkpoint and settles closer to the real distribution
(0.24 / 0.32 / 0.53). Validation went 0.356 then 0.383; the gate saw
`0.35 → 0.41 → 0.39` and stopped at the iteration cap.

**An earlier version of this experiment also reweighted the loss** toward whichever
class the evaluator flagged, and it *did* move `down` recall to 0.27. That version is
withdrawn because it changed two things at once, and this clean run shows where the
credit belonged: the collapse was being fixed by the reweighting, not by training in
the loop. Isolating the variable turned a promising-looking result into an honest
null one.

The conclusion is the same one every other measurement in this project points to.
The pipeline's threshold rule, a balanced-accuracy gate, argmax, the standalone
fine-tune and this experiment all land between 0.34 and 0.39 balanced accuracy —
different setups, one narrow band, just above chance.

Caveats: a single run of two training rounds, stopped by the cap rather than by
converging. Explanations used the offline fallback (no `HF_TOKEN`), which does not
affect any number above.
