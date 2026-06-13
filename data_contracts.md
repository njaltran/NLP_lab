# Data Contracts

Agreed input/output formats between all agents. Column names are fixed; !!do not change without telling the whole team!!

**Primary data source:** Yahoo Finance, loaded via dlt into DuckDB (see `yahoo_finance_dlt/yahoo_finance_pipeline.py`). Two tables: `news` (headlines) and `prices` (daily OHLCV). The Processing Agent joins them to build the training/test set.

**Note on article text:** Yahoo Finance gives us a headline (`title`) and a short raw `summary` per article — no full body. The Processing Agent (Aurora) runs transformer-based **summarization** over the available text to produce a clean `article_summary`, and keeps `article_title` (the headline). FinBERT then classifies on this text. Headlines alone classify well; the generated summary gives a richer, normalised text input.

## Source — Yahoo Finance (dlt → DuckDB)

DuckDB dataset `market_data`. Tickers configurable (default AAPL, MSFT, GOOG). Types below are the DuckDB columns dlt produces. yfinance gives only the latest ~10 news articles per ticker (no deep history) — news coverage grows as the pipeline runs over time.

Every table also has dlt system columns `_dlt_load_id` (string) and `_dlt_id` (string).

### Table `news`

Primary key `(ticker, id)`, write disposition `merge`, incremental on `pub_date`.

| Column | Type | Example | Notes |
|---|---|---|---|
| ticker | VARCHAR | AAPL | source ticker symbol |
| id | VARCHAR | 0606e28d-4eda-... | yahoo article id — becomes `article_id` downstream |
| title | VARCHAR | Apple beats earnings... | headline — becomes `article_title` downstream |
| summary | VARCHAR | An Indian pollution... | article summary, may be empty |
| content_type | VARCHAR | STORY | yahoo content type |
| pub_date | TIMESTAMP WITH TIME ZONE | 2026-06-13T10:33:55+02:00 | publish time; tz-aware → `date` downstream |
| url | VARCHAR | https://... | canonical article url |
| publisher | VARCHAR | Reuters | provider display name |

> Same article can appear under 2 tickers → 2 rows (PK includes ticker).

### Table `prices`

Daily OHLCV history (default 1y). Primary key `(ticker, date)`, write disposition `merge`, incremental on `date`. Used to compute `price_t`, `price_t1`, `pct_change`, `label`.

| Column | Type | Example | Notes |
|---|---|---|---|
| ticker | VARCHAR | AAPL | source ticker symbol |
| date | TIMESTAMP WITH TIME ZONE | 2026-06-12T06:00:00+02:00 | bar timestamp; tz-aware |
| open | DOUBLE | 290.10 | open price |
| high | DOUBLE | 293.50 | session high |
| low | DOUBLE | 288.20 | session low |
| close | DOUBLE | 291.13 | close price — source of `price_t` / `price_t1` |
| volume | BIGINT | 38742100 | shares traded |
| dividends | DOUBLE | 0.0 | dividend on the day |
| stock_splits | DOUBLE | 0.0 | split ratio on the day |

## Handoff 1 — Processing Agent (Aurora) → Classifier Agent (Nadi)

**Filename:** `processed_data.csv`

Built by joining `news` to `prices` on `ticker` + publication day, plus a summarization step. For each article: take `close` on the publication trading day as `price_t` and `close` on the next trading day as `price_t1`; derive `pct_change` and `label`; generate `article_summary` from the article text.

| Column | Type | Example | Notes |
|---|---|---|---|
| article_id | string | 0606e28d-4eda-... | from Yahoo `news.id` |
| date | string | 2026-06-12 | from `news.pub_date`, date part only, format YYYY-MM-DD |
| ticker | string | AAPL | from `news.ticker` |
| article_title | string | Apple beats earnings... | from `news.title`, no HTML or special characters |
| article_summary | string | Apple reported record... | summarization output by Aurora over `news.title` + `news.summary`; may be empty if no source text |
| price_t | float | 291.13 | `prices.close` on publication trading day |
| price_t1 | float | 295.63 | `prices.close` on next trading day |
| pct_change | float | 1.55 | percentage change T to T+1 |
| label | string | up | derived from pct_change — allowed values: up, down, neutral |

> Articles with no matching trading day in `prices` (e.g. weekend/holiday with no next session yet) are dropped by Aurora.

## Handoff 2 — Classifier Agent (Nadi) → Evaluator Agent (Sabina)

**Filename:** `predictions_test.csv`

All columns from Handoff 1, plus:

| Column | Type | Example | Notes |
|---|---|---|---|
| predicted_label | string | up | allowed values: up, down, neutral |
| confidence | float | 0.87 | score between 0 and 1 |
| split | string | test | all rows in this file must be test rows only |

## Handoff 3 — Evaluator Agent (Sabina) → Manager Agent (Jack)

**Filename:** `evaluation_report.json`

| Field | Type | Example | Notes |
|---|---|---|---|
| accuracy | float | 0.63 | overall accuracy on test set |
| below_threshold | boolean | false | true if accuracy is below 0.60 |
| class_accuracy | object | {"up": 0.71, "down": 0.58, "neutral": 0.61} | accuracy per label |
| misclassified_count | integer | 148 | total number of wrong predictions |
| misclassified_ids | list | ["0606e28d-4eda-...", ...] | article_ids of wrong predictions |

## Handoff 3b — Manager Agent (Jack) → Classifier Agent (Nadi) — retune loop

**Filename:** `retune_request.json`

Sent only when `below_threshold = true` in Handoff 3. Tells Nadi to re-run with adjustments. Closes the iterative-improvement loop (Sabina → Jack → Nadi). When accuracy clears the threshold, no `retune_request.json` is written and the flow proceeds to Handoff 4.

| Field | Type | Example | Notes |
|---|---|---|---|
| iteration | integer | 2 | loop counter, starts at 1 |
| reason | string | accuracy 0.54 below target 0.60 | human-readable trigger |
| current_accuracy | float | 0.54 | from the report that triggered the loop |
| target_accuracy | float | 0.60 | threshold to clear |
| focus_labels | list | ["down", "neutral"] | classes with lowest per-class accuracy to prioritise |
| misclassified_ids | list | ["0606e28d-4eda-...", ...] | rows to inspect / reweight |
| suggested_params | object | {"threshold": 0.5, "max_length": 128} | hyperparameters Nadi should try next |

## Handoff 4 — Manager Agent (Jack) → Explanation Agent (Freddi)

**Filename:** `sample_for_explanation.csv`

| Column | Type | Example | Notes |
|---|---|---|---|
| article_id | string | 0606e28d-4eda-... | to match back to original data |
| article_title | string | Apple beats earnings... | headline from Processing Agent |
| predicted_label | string | up | final prediction after loop converges |
| actual_label | string | up | ground truth label from Processing Agent |
| confidence | float | 0.87 | model confidence score |

## Handoff 5 — Explanation Agent (Freddi) → Manager Agent (Jack)

**Filename:** `explanations.csv`

Same columns as Handoff 4, plus:

| Column | Type | Example | Notes |
|---|---|---|---|
| explanation | string | The headline references... | generated by Ollama |
| manual_score | integer | 4 | 1–5, only filled for the 30–50 manually reviewed rows, leave blank otherwise |


## General Rules

- Source of truth is the DuckDB dataset `market_data` (Yahoo Finance via dlt); all CSV handoffs derive from it
- All CSV files use UTF-8 encoding
- All CSV files use comma as separator
- Null values written as empty string, not "NULL" or "NaN"
- Label column only ever contains: up, down, neutral (lowercase)
- Date format is always YYYY-MM-DD (tz-aware DuckDB timestamps are truncated to the date part)
- Do not rename columns without notifying the full team
