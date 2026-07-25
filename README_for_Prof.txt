===============================================================================
NLP LAB - PREDICTING STOCK MOVEMENT FROM FINANCIAL NEWS HEADLINES
Team 2: Aurora, Freddi, Jack, Nadi, Sabina
===============================================================================

Plain-text quick reference. 
The full documentation is in README.md (in VS Code, open it and press Cmd+Shift+V / Ctrl+Shift+V to render it).


-------------------------------------------------------------------------------
1. SEE THE RESULTS WITHOUT RUNNING ANYTHING
-------------------------------------------------------------------------------

Open:   outputs/final_report.json

This is the real output of a completed run. Headline numbers:

    test accuracy       0.50      (majority-class baseline: 0.516)
    balanced accuracy   0.39      (chance baseline: 0.333)
    per-class recall    up 0.36 / down 0.05 / neutral 0.76
    test set            2,196 headlines
    loop iterations     5

Every test prediction with its generated explanation is in:
        outputs/final_results.csv


-------------------------------------------------------------------------------
2. RUN THE PIPELINE
-------------------------------------------------------------------------------

Requires Python 3.13. Recommended: uv (https://docs.astral.sh/uv/)

    uv sync
    uv run main.py --no-ollama --dataset-end 2019-12-31

Without uv (uv is recommended though - more accurate versions used!!):

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py --no-ollama --dataset-end 2019-12-31

Notes:
  --dataset-end 2019-12-31   Keeps the COVID crash out of the test set.
                             ALL REPORTED RESULTS USE THIS FLAG.
  --no-ollama                Skips the local LLM used for explanations, so no
                             extra setup is needed.

The data ships with the submission (data/fnspid_raw.csv and the price cache),
so no yfinance download is needed. FinBERT itself (~440 MB) downloads from
Hugging Face on the first run, so that run needs internet.


-------------------------------------------------------------------------------
3. RUN THE TESTS
-------------------------------------------------------------------------------

    uv run python -m pytest tests/ -q

Expected: 72 passed.


-------------------------------------------------------------------------------
4. VIEW THE DASHBOARD (shown in our presentation)
-------------------------------------------------------------------------------

It is a marimo app, so "python dashboard.py" will NOT open it. Use:

    uv run marimo run dashboard.py


-------------------------------------------------------------------------------
5. FOLDER STRUCTURE
-------------------------------------------------------------------------------

main.py                    entry point: parses flags, runs the pipeline
dashboard.py               marimo dashboard
README.md                  full documentation (render with Cmd+Shift+V)
README_for_Prof.txt        this file

agents/                    one module per agent, one owner each
  pipeline_graph.py          THE ORCHESTRATION - all five agents wired into
                             one LangGraph, with the retune loop as a cycle
  base.py                    shared Agent base class
  state.py                   the shared state passed between nodes
  contracts.py               one definition of every handoff file format
  env.py                     .env loading
  aurora_processing.py       1. Processing agent  (Aurora)
  nadi_classifier.py         2. Classifier agent  (Nadi)
  sabina_evaluator.py        3. Evaluator agent   (Sabina)
  jack_manager.py            4. Manager agent     (Jack)
  freddi_explanation.py      5. Explanation agent (Freddi)
  finetune_finbert.py        standalone fine-tuning script (run once, offline)

data/
  fnspid_raw.csv             raw news headlines
  price_cache.pkl            cached prices, so no yfinance download is needed

outputs/                   committed results of a real end-to-end run
  final_report.json          headline metrics       <- START HERE
  final_results.csv          every prediction + explanation
  predictions_test.csv       raw classifier output
  evaluation_report.json     metrics + retune proposal
  explanations.csv           per-prediction justifications
  decision.json              the Manager's decision log

docs/                      design docs and the EDA notebook
  architecture.md            system design and agent roles
  data_contracts.md          exact columns/types of every handoff file
  processing_experiments.ipynb  EDA: how the +/-1% label band was chosen
  retune_loop.md             how the feedback loop adapts
  finetune_runs.md           fine-tuning experiments and the overfitting finding
  metric_experiment.md       gate-metric / decision-rule A/B/C test
  collaborating.md           how we split the work across five people
  pipeline_graph.png         the compiled LangGraph, rendered

experiment_finetuning_in_loop/
                           NOT part of the architecture we presented. A later
                           experiment that moves fine-tuning inside the retune
                           loop, kept separate so the submitted pipeline stays
                           as it was during the project. It has its own README
                           and writes its own outputs; nothing in agents/ or
                           main.py is changed by it.

mock_data/                 small valid sample of every handoff file (used by tests)
tests/                     pytest suite


-------------------------------------------------------------------------------
6. THE PIPELINE IN ONE PARAGRAPH
-------------------------------------------------------------------------------

Aurora joins news headlines to stock prices and labels each headline up, down,
or neutral based on the next trading day's move. Nadi runs FinBERT to predict
that label. Sabina scores the predictions. Jack compares the score to a target
and either sends Nadi back to retry with new settings, or proceeds. Freddi then
writes a one-sentence explanation for each prediction. The whole thing is one
compiled LangGraph, and the retune step is a real cycle in that graph.
