# Agent Architecture

Agent-based system for **stock price prediction from financial news**. A classification model (FinBERT) predicts next-day move (up/down/neutral) from headlines; agents orchestrate, evaluate, and **iteratively improve** performance, then produce a **text justification** for each prediction. Evaluation = prediction accuracy on the test set + manual review of explanations.

Data contracts in [`data_contracts.md`](./data_contracts.md).

```mermaid
flowchart TD
    FNSPID["FNSPID dataset<br/>(Zihan1004/FNSPID on HuggingFace)<br/>news headlines, 2009–2023"] -->|streaming, sampled by ticker| AURORA
    YF["Yahoo Finance<br/>(yfinance)"] -->|closing prices T and T+1| AURORA

    subgraph AGENTS["Agent framework"]
        JACK{{"Manager Agent — Jack<br/>orchestrates, threshold gate"}}
        AURORA["Processing Agent — Aurora<br/>join headlines↔prices, label move"]
        NADI["Classifier Agent — Nadi<br/>FinBERT → up/down/neutral"]
        SABINA["Evaluator Agent — Sabina<br/>accuracy, per-class metrics"]
        FREDDI["Explanation Agent — Freddi<br/>Ollama text justification"]
    end

    AURORA -->|"processed_data.csv"| NADI
    NADI -->|"predictions_test.csv"| SABINA
    SABINA -->|"evaluation_report.json"| JACK
    JACK -->|"accuracy < 0.60<br/>retune & re-run (iterate)"| NADI
    JACK -->|"accuracy OK<br/>sample_for_explanation.csv"| FREDDI
    FREDDI -->|"explanations.csv"| JACK
    JACK -->|"30–50 rows"| HUMAN(["Manual evaluation<br/>(team scores 1–5)"])
```

## Data sources

| Source | Used for | How |
|---|---|---|
| FNSPID (`Zihan1004/FNSPID`) | News headlines | HuggingFace streaming, sampled by target ticker list (~500 articles per ticker) |
| yfinance | Closing prices T and T+1 | `Ticker.history()` with a small date window per article; weekends/holidays handled automatically |

FNSPID's full article body is unavailable in the HuggingFace version — `article_title` (headline) is used as the text input. FinBERT is designed for short financial text and performs well on headlines.

## Agents

| Agent | Owner | Role | Input | Output |
|---|---|---|---|---|
| Manager | Jack | Orchestrate loop, apply accuracy threshold, sample for explanation | `evaluation_report.json` | gate decision + `sample_for_explanation.csv` |
| Processing | Aurora | Join FNSPID headlines to yfinance prices on date + ticker, derive next-day label | FNSPID + yfinance | `processed_data.csv` |
| Classifier | Nadi | FinBERT sentiment → up/down/neutral | `processed_data.csv` | `predictions_test.csv` |
| Evaluator | Sabina | Accuracy and per-class metrics on test split | `predictions_test.csv` | `evaluation_report.json` |
| Explanation | Freddi | Ollama-generated justification per prediction (~300 rows) | `sample_for_explanation.csv` | `explanations.csv` |

## Iterative improvement loop

1. Aurora joins FNSPID headlines to yfinance closing prices on `date` + `ticker`, calculates percentage change, assigns labels (>+1% = up, <-1% = down, in between = neutral), outputs `processed_data.csv`.
2. Nadi classifies the test split with FinBERT.
3. Sabina calculates accuracy and per-class metrics, outputs `evaluation_report.json`.
4. Jack checks the threshold — accuracy < 0.60 → loop back to Nadi to retune and re-run; otherwise proceed.
5. Once cleared, Jack samples ~300 rows and sends to Freddi for text justifications.
6. Team manually evaluates 30–50 explanations (score 1–5) — the second evaluation axis alongside accuracy.

## Evaluation (two axes)

- **Quantitative:** prediction accuracy on the held-out test set — Sabina. Accuracy is used because this is a 3-class classification task (up/down/neutral), not a continuous numerical prediction.
- **Qualitative:** manual review of 30–50 generated explanations, scored 1–5 — team, via Freddi's output.
