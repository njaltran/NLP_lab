# NLP_lab

## Agent-based stock prediction from financial news

This repository now includes a minimal multi-agent framework in
`stock_prediction_agents.py` with:

- `ManagerAgent` for iterative model selection and orchestration
- `ProcessingAgent` for text preprocessing
- `ClassifierAgent` for news-to-market-direction classification
- `EvaluatorAgent` for test accuracy and manual explanation review set generation

### Run

```bash
python stock_prediction_agents.py \
  --dataset /absolute/path/to/financial_news.csv \
  --manual-eval-output /absolute/path/manual_explanation_eval.csv \
  --example-text "Company beats earnings expectations"
```

Output includes validation/test accuracy and a CSV file that can be manually scored for explanation quality (`manual_score_1_to_5` column).

### Tests

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```
