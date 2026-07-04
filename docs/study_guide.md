# Study Guide

A single map of the project: where the guides live, and the **"why" behind our
decisions** — the experiments and reasoning that justify how each agent works.
Use this to understand the project end to end, or to explain a choice to the prof.

> This is an index + rationale doc. The full detail lives in the linked guides,
> the sources of truth in `docs/`, and the code itself.

---

## 1. Guides & docs that already exist

### Agent walkthrough notebooks (plain-English, block-by-block)
- [Processing Agent — Code Guide](./aurora_processing_guide.ipynb) — Aurora. Mirrors `agents/aurora_processing.py` line by line.
- [Preprocessing & Processing — Logic & EDA](./processing_experiments.ipynb) — Aurora. The data exploration that justifies the label threshold and accuracy target.
- [Explanation Agent — Logic & Walkthrough](./explanation_agent_walkthrough.ipynb) — Freddi. How Ollama justifies each prediction.

> Not yet written: dedicated guides for **Nadi** (Classifier), **Sabina** (Evaluator), **Jack** (Manager). See §4 for status.

### Sources of truth (do not duplicate — link to these)
- [Architecture](./architecture.md) — flow, agent roles, the two evaluation axes.
- [Data contracts](./data_contracts.md) — every handoff file's columns/types.
- [Collaborating](./collaborating.md) — how the team works async.
- [AGENTS.md / CLAUDE.md](../CLAUDE.md) — the team working agreement & golden rules.

### Design specs
- [Adaptive retune loop](./superpowers/specs/2026-07-01-adaptive-retune-loop-design.md) — why/how the loop adapts across iterations.
- **Fine-tuning experiment** — no spec on this branch; the results (incl. the epoch-1 overfitting finding) are summarised in §3 below.

---

## 2. The pipeline in one picture

```
Aurora (Processing) → Nadi (Classifier) → Sabina (Evaluator) → Jack (Manager)
   processed_data.csv    predictions_test.csv   evaluation_report.json     │
                              ▲                                            │
                              └──────── retune loop (retune_request) ──────┘
                                                   │ proceed
                                          Freddi (Explanation) → final output
```
Two evaluation axes: **quantitative** = accuracy on the held-out test set (Sabina); **qualitative** = manual 1–5 scoring of ~30–50 explanations (team, via Freddi).

---

## 3. The "Reasons" — key decisions & the evidence behind them

### Processing (Aurora)
- **Headline-only text input.** FNSPID's article *body* is unavailable in the HuggingFace version, and FinBERT is designed for short financial text — so `article_title` is the model input. Decision forced by the data + a good fit for the model.
- **Labels = ±1% threshold.** >+1% = `up`, <−1% = `down`, else `neutral`. Chosen in the EDA notebook; loosening it (e.g. 0.5%) would distort the model's target and inflate the `neutral` class.
- **`price_t` / `price_t1` = next *trading* day.** T+1 is the next available trading day, so weekends/holidays are skipped automatically. Rows without at least two prices (edge of dataset, delisted) are dropped rather than guessed.
- **Outlier removal = 1st–99th percentile fences, not IQR×1.5.** IQR×1.5 was cutting ~10% of rows because stock returns are **fat-tailed**; percentile fences keep exactly 98% regardless of distribution shape.
- **Train/test split is by DATE, not random.** Earliest 80% of dates = `train`, rest = `test`. A model that predicts the future must be trained on the past and tested on the future it has never seen — random splitting would leak future info.
- **COVID cutoff (`dataset_end = 2019-12-31`).** Our data runs to mid-2020, and 2020 is 25% of rows (all Jan–Jun, the crash). Because 2020 is the *latest* data, a plain split lands it in the **test set** — so keeping it would measure accuracy *on the COVID crash*. Cutting it keeps 11,067 rows (8,871 train / 2,196 test) in a normal regime. Optional flag; the mock leaves it off.

### Classifier (Nadi) — split shipped; fine-tuning + agentic code-gen built & verified (pending PR), see §4
- **It's a code-*generation* agent.** Nadi doesn't just run a model — it writes `classifier.py` as a standalone script, runs it, and hands **both the code and the predictions** to Sabina (prof's note: Sabina reviews code, not only outputs). Each iteration's code is archived to `classifier_history/` so past attempts aren't lost when `classifier.py` is overwritten.
- **The retune loop tunes inference knobs, not the model.** `threshold`, `max_length`, `focus_labels`, `boost_factor` are re-applied each iteration on top of the *same* weights. Training happens once, offline — never inside the loop.
- **Predict TEST rows only** (shipped, on `main`). Train rows were seen by the model, so scoring them would leak/inflate accuracy. The classifier filters to `split == "test"`.
- **Fine-tuning — a negative result, now independently REPRODUCED.** `agents/finetune_finbert.py` trains a fresh 3-class head over FinBERT, class-weighted for the neutral-heavy data. Our own clean (COVID-excluded) run scored **0.4631** — matching the earlier documented 0.4608 almost exactly (per-class neutral 0.71 / down 0.33 / up 0.10). Two independent fine-tunes landing at 0.46 with the same shape = the finding is **robust, not a fluke.**
  - Still **BELOW the all-neutral baseline (~0.52).** Validation **peaks at epoch 1** then degrades → immediate overfitting → **the ceiling is the data, not the optimizer.** Beats pretrained (0.46 vs 0.37) so there's *some* signal, just not enough to beat "always neutral."
- **Model-aware loading (opt-in).** When `model_dir` is set *and* the folder exists, the classifier loads the fine-tuned weights (labels read straight from its config — no sentiment→direction map); otherwise it uses pretrained FinBERT. Opt-in on purpose: a stray folder must never silently swap the model.
- **Agentic code-gen — IMPLEMENTED as "guardrailed adaptation."** When `CLASSIFIER_USE_OLLAMA=true`, a local LLM (Ollama) **rewrites `classify()` from Sabina's feedback** (`code_notes`, `focus_labels`), but the output is kept **only if it survives a validation gauntlet** — `ast.parse` → run on mock data → output columns/values match the contract — otherwise it **falls back to the template.** Same team pattern as Sabina: *LLM proposes, rules validate.*
  - **Verified end-to-end** against a real Ollama server: llama3.2's generated code was rejected by the guardrail and it fell back safely; an injected valid variant was accepted — both branches proven.
  - **Why it exists:** (1) the project is an *agent-based system* and must be genuinely agentic; (2) a weak model leaves post-processing for the agent to try. **Honest caveat:** an agent can reshuffle predictions but **cannot invent signal that isn't there** (headline→next-day correlation ≈ 0) — so it's about *agentic behaviour + safe exploration*, not hitting 0.60.

### Evaluator (Sabina) — ⚠️ being edited, see §4
- **Accuracy, not regression error** — the task is 3-class classification (up/down/neutral), not a continuous number.
- **Reviews code + results, then proposes** the next action (retune vs proceed, focus labels, suggested params, code notes). She reads `classifier.py`; she does **not** re-run it (she trusts Nadi's output + reads the source for context).
- **Deterministic control + LLM narrative.** Metrics, `recommended_action`, `focus_labels`, and `suggested_params` are all computed by **code**; an LLM (Ollama) may rewrite **only** the human-readable `reason`/`code_notes`. Justification: the loop's *decisions* must be reproducible and must match Jack's gate — a stochastic model can't be allowed to flip retune-vs-proceed.
- **LLM is opt-in and safe-by-default.** Off unless `EVALUATOR_USE_OLLAMA=true`; local Ollama, no API key; if it times out or returns bad JSON, it **falls back to the deterministic proposal**. Justification: demo-safe, offline-safe, never breaks the loop.
- **Guardrails on the LLM output.** The review must be exactly `{reason, code_notes}`, and the whole proposal is re-validated against the metrics (the action must equal the deterministic gate's verdict). Justification: stop LLM drift from corrupting control fields.
- **Retune params step down with a floor + focus margin.** Threshold decreases each retune toward a floor aligned with Jack's schedule; `focus_labels` includes every class within `FOCUS_MARGIN` (0.05) of the weakest. Justification: each retune is a new attempt, and near-tied weak classes (common on real data) aren't missed by exact-equality.
- **Strict contract validation** — she checks `predictions_test.csv` columns against the contract and raises on mismatch. This is what caught the duplicate-`split` bug when the split column was first added.

### Manager (Jack)
- **The accuracy gate is pure rules, not the LLM.** retune-vs-proceed is decided by comparing accuracy to the target (0.60). An LLM (Llama-3.1) only writes the human-readable rationale. Justification: the decision that controls the whole loop must be deterministic and auditable.
- **The loop used to be open-loop.** A 2026-06-30 run did 5 iterations all stuck at 0.37, then hit the cap — because the decision only saw the current report and the retune request was constant every time. This is the problem the adaptive design fixes.
- **Convergence / early stop.** Once there are more than `patience` iterations, the best of the recent window is compared to the best of everything before it; if it didn't improve by at least `min_delta`, the loop is declared **plateaued** and proceeds. Justification: don't burn the whole iteration budget once accuracy has clearly stalled.
- **Adaptation via a retune schedule.** A fixed escalating schedule (threshold 0.45→0.20, max_length 128→256, boost 1.25→1.75); each retune picks the first entry **not already tried**. Justification: the old loop re-proposed the same defaults and got the same accuracy — the schedule guarantees every pass is a genuinely different attempt.
- **"Already tried" compares only shared keys.** Sabina omits params she doesn't set and Nadi fills them from the same defaults, so comparing full dicts would let an "identical" classifier slip through. Justification: avoid wasting an iteration regenerating the same code.
- **Override mechanism + iteration cap.** The Manager can accept or override Sabina's proposal (e.g. she says retune but convergence/cap forces proceed), and a hard `max_iterations` bounds the loop. Justification: the Manager owns the final call and the loop must always terminate.
- **Samples ~300 rows for explanation.** Enough for a meaningful qualitative review without over-spending on LLM calls.

### Explanation (Freddi)
- **Ollama `llama3.2` via a LangChain chain** (`prompt | ChatOllama | StrOutputParser`) — the exact pattern from the course's RAG exercise. Justification: reuse a known-good local-LLM pattern; no API key.
- **Option A — explains from the HEADLINE only, never sees the actual outcome.** `actual_label` is passed through for the human graders but never reaches the model. Justification: mirrors real prediction time (no lookahead) and keeps it to "explain your reasoning" — the model can't rationalise backwards from the answer.
- **Graceful fallback.** If Ollama is unreachable, a deterministic placeholder sentence is produced so a valid `explanations.csv` is *always* handed back. Justification: team rule "proceed on issue" — an infra hiccup must never crash the pipeline.
- **Low temperature (0.3), ~one sentence.** Justification: grounded, low-variance, concise explanations that are easy for humans to score.
- **First five columns pass through byte-for-byte; `manual_score` left blank.** Justification: contract fidelity; the blank column is filled by hand in the qualitative review.
- Feeds the **qualitative** evaluation axis: the team scores 30–50 explanations 1–5.

### Cross-cutting (the shared design philosophy)
- **"LLM narrates, rules decide."** The team's core pattern: every control decision (the accuracy gate, retune-vs-proceed, which params to try, the pass/fail metrics) is **deterministic code**; the LLM only writes the human-readable *rationale*. You can see it in the **Manager** (gate is rules, Llama writes the reason), the **Evaluator** (metrics/action deterministic, Ollama rewrites reason/code_notes), and now the **Classifier** too (LLM rewrites `classify()`, but a validation gauntlet decides whether to keep it or fall back to the template). Justification: this is how the system is "agentic" — genuine LLM reasoning in the loop — **without** letting a stochastic model corrupt reproducible, auditable control flow.
- **"Proceed on issue" / graceful degradation everywhere.** Freddi falls back to a placeholder sentence, Sabina falls back to the deterministic proposal, the Manager caps iterations. Justification: a single agent's infra failure (Ollama down, bad LLM output) must never crash the pipeline — a valid handoff file is always produced.
- **Every agent is a LangGraph behind one `.run()`** (`agents/base.py`, convention set 2026-06-26). Justification: a uniform interface so the Manager can invoke any agent identically, and each agent stays independently testable.
- **Contract-driven, mock-first.** Every agent builds against `mock_data/` before the upstream agent exists; column names/filenames are fixed by `data_contracts.md`. Justification: agents can be built in parallel, and bad data can't flow silently downstream.
- **Honest baseline = ~0.52 (all-neutral).** Any model must beat this to matter. The **0.60 target was set before anyone measured** whether the task supports it — a candidate for renegotiation with the prof.
- **The signal is genuinely thin.** Headline→next-day-direction correlation is ≈ 0 for sentiment, and fine-tuning only reached 0.46. Justification for framing: no agent cleverness can invent signal the data doesn't contain — so the deliverable is an honest, well-engineered *system*, not a magic accuracy number.

---

## 4. Status notes / caveats

- **Sabina's Evaluator is merged to `main`** (PR #30). The §3 rationale reflects that code.
- **Nadi's Classifier is merged to `main`** (PR #34) — split filter, fine-tuning, model-aware loading, and the guardrailed LLM code-gen are all in. §3's classifier bullets describe shipped code.
- **Pipeline wiring is Jack's to add.** For a *full pipeline* run to use the fine-tuned model or exclude COVID, `main.py`/`pipeline_graph.py` need two optional flags (`--model-dir`, `--dataset-end`) threaded to the classify/process nodes. Capabilities live in Nadi's + Aurora's agents; the run-switch is Jack's lane (drafted spec handed to him).
- **The train/test split was landed independently by Aurora** (Processing). The fine-tuning number in §3 (0.4631) is from *our own* clean reproduction run, on the COVID-excluded split.

---

## 5. Concepts glossary (plain English)

*No background assumed — quick definitions of the terms used above.*

### Model & training
- **FinBERT** — a language model that "reads" financial text (a finance-trained version of Google's BERT). We give it a headline; it outputs up / down / neutral.
- **Fine-tuning** — taking an already-trained model and training it *a bit more* on **our** labels, so it learns **our** task (price direction) instead of what it originally did (sentiment).
- **Epoch** — one full pass through all the training data. "3 epochs" = the model saw every training headline 3 times. More passes = more learning — but too many = it starts *memorizing* (see overfitting).
- **Batch / batch size** — the model learns from small groups at a time; batch size 16 = 16 headlines per step. Just a memory/speed setting.
- **Learning rate** — how big an adjustment the model makes at each step. Too big = unstable and forgets what it knew; too small = painfully slow. `2e-5` is the standard value for BERT.
- **Overfitting** — when the model *memorizes* the training examples instead of learning general patterns → looks great on training data, does badly on new data. We saw it: our model's score peaked after **1 epoch** and got *worse* after.
- **Checkpoint** — a saved snapshot of the model during training. We keep the **best** one (highest validation score), not the last.
- **Token / tokenization / max_length** — text is chopped into small pieces ("tokens") the model reads. `max_length 128` = keep up to 128 tokens per headline (plenty — headlines are short).
- **Logits → softmax → probabilities** — the model spits out raw scores (logits); **softmax** turns them into three probabilities that add up to 100% (up / down / neutral).
- **Argmax** — "take the biggest." The prediction = the class with the highest probability.
- **Confidence** — the probability of the chosen class (the highest of the three). With 3 classes the floor is ~33%, so ~40% means "barely sure."
- **Class weights / class imbalance** — our data is mostly "neutral," so we weight the rarer up/down classes more during training — otherwise the model just learns to always say "neutral."

### Data & scoring
- **Train / validation / test** — *train* = what the model learns from; *validation* = a held-out slice to check progress *during* training; *test* = final unseen data to measure real performance.
- **Data leakage** — accidentally letting the model peek at information it shouldn't (e.g. the future), which fakes good results. We split by **date** to prevent it.
- **Accuracy** — the % of predictions that are correct.
- **Per-class accuracy** — accuracy measured separately for up, down, and neutral — reveals if the model is quietly ignoring a class.
- **Balanced accuracy** — the average of the three per-class accuracies. Unlike plain accuracy, it **punishes** a model that just predicts the majority class.
- **Baseline** — the score of a dumb strategy you have to beat. Our "all-neutral baseline" (~0.52) = always guess neutral, which is right whenever the true answer *is* neutral (the majority class).
- **Threshold** — a confidence cutoff: only commit to up/down if confident enough, otherwise fall back to neutral.

### Pipeline / system
- **Agent** — one component with one job that hands a file to the next (we have 5).
- **LangGraph** — the framework that wires the agents together into the loop.
- **Retune loop** — the cycle where the classifier is re-run with tweaked settings to try to do better, until the Manager says stop.
- **LLM / Ollama** — a large language model (like ChatGPT) run **locally** on our machine via Ollama; it writes the plain-English text (Freddi's explanations, Sabina's reasoning).
