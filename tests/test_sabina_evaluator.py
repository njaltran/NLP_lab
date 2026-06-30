"""Tests for Sabina's Evaluator agent.

They use simple assert statements so the expectations read like Diana's small
classroom examples, but they are also pytest-compatible when pytest is present.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.sabina_evaluator as se


PREDICTIONS = "mock_data/predictions_test.csv"
CLASSIFIER = "mock_data/classifier.py"


def _load_mock_rows():
    with open(PREDICTIONS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_report_matches_mock_data_contract():
    """Given mock predictions, Sabina should produce Jack's expected report."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    report = se.build_report(rows, code)

    assert report["accuracy"] == 0.67
    assert report["below_threshold"] is False
    assert report["class_accuracy"] == {"up": 0.80, "down": 0.75, "neutral": 0.33}
    assert report["misclassified_count"] == 4
    assert report["misclassified_ids"] == [
        "FNSPID_00006",
        "FNSPID_00010",
        "FNSPID_00011",
        "FNSPID_00012",
    ]
    assert report["proposal"] == {
        "recommended_action": "proceed",
        "reason": (
            "accuracy 0.67 clears the 0.60 target; neutral class is weakest "
            "(0.33) but the iteration budget favours proceeding"
        ),
        "focus_labels": ["neutral"],
        "suggested_params": {},
        "code_notes": (
            "threshold hardcoded at 0.5 in classifier.py; the neutral band "
            "(+/-1%) may be too narrow for the neutral class"
        ),
    }


def test_report_written_to_evaluation_report_json():
    """Sabina should write exactly one JSON contract output for Jack."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)
    report = se.build_report(rows, code)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "evaluation_report.json"
        se.write_report({"report": report, "output_path": str(out)})

        written = json.loads(out.read_text(encoding="utf-8"))

    assert written == report


def test_low_accuracy_recommends_retune():
    """If accuracy is below 0.60, Sabina should recommend retuning only."""
    rows = _load_mock_rows()
    for row in rows:
        row["predicted_label"] = "neutral"
        row["confidence"] = "0.90"
        row["prob_up"] = "0.05"
        row["prob_down"] = "0.05"
        row["prob_neutral"] = "0.90"

    report = se.build_report(rows, "THRESHOLD = 0.5\nMAX_LENGTH = 128\n")

    assert report["accuracy"] < 0.60
    assert report["below_threshold"] is True
    assert report["proposal"]["recommended_action"] == "retune"
    assert report["proposal"]["suggested_params"] == {"threshold": 0.5, "max_length": 128}


def test_validation_rejects_non_test_rows():
    """Sabina should reject rows that are not from the test split."""
    rows = _load_mock_rows()
    rows[0]["split"] = "train"

    try:
        se.validate_predictions(rows)
    except ValueError as error:
        assert "split must be test" in str(error)
    else:
        assert False, "Expected validate_predictions to reject train split rows"


if __name__ == "__main__":
    test_build_report_matches_mock_data_contract()
    test_report_written_to_evaluation_report_json()
    test_low_accuracy_recommends_retune()
    test_validation_rejects_non_test_rows()
    print("Sabina evaluator tests passed")
