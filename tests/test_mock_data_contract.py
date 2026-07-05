import csv
from agents.contracts import PREDICTION_COLUMNS, EXPLANATION_SAMPLE_COLUMNS, LABELS

def _cols(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))

def test_predictions_mock_matches_contract():
    assert _cols("mock_data/predictions_test.csv") == PREDICTION_COLUMNS

def test_sample_mock_matches_contract():
    assert _cols("mock_data/sample_for_explanation.csv") == EXPLANATION_SAMPLE_COLUMNS

def test_processed_mock_labels_binary():
    with open("mock_data/processed_data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["label"] for r in rows} <= set(LABELS)
