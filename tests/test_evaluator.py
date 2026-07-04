"""Tests for the evaluator agent.

They use simple assert statements and are also pytest-compatible when pytest is
present.
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
    """Given mock predictions, the report should match the contract."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    report = se.build_report(rows, code)

    assert report["accuracy"] == 0.60
    assert report["below_threshold"] is False
    assert report["eval_split"] == "test"
    assert report["class_accuracy"] == {"up": 0.67, "down": 0.75, "neutral": 0.33}
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
            "accuracy 0.60 clears the 0.60 target; neutral class is weakest "
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
    """The evaluator should write exactly one JSON contract output."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)
    report = se.build_report(rows, code)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "evaluation_report.json"
        se.write_report({"report": report, "output_path": str(out)})

        written = json.loads(out.read_text(encoding="utf-8"))

    assert written == report


def test_low_accuracy_recommends_retune():
    """If accuracy is below 0.60, the proposal should recommend retuning."""
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
    assert report["proposal"]["suggested_params"] == {
        "threshold": 0.45,
        "max_length": 128,
    }


def test_retune_threshold_steps_down_from_current():
    """Each retune should lower the classifier threshold by one deterministic step."""
    rows = _load_mock_rows()
    for row in rows:
        row["predicted_label"] = "neutral"
        row["confidence"] = "0.90"
        row["prob_up"] = "0.05"
        row["prob_down"] = "0.05"
        row["prob_neutral"] = "0.90"

    report = se.build_report(rows, "THRESHOLD = 0.45\nMAX_LENGTH = 128\n")

    assert report["proposal"]["suggested_params"]["threshold"] == 0.40


def test_retune_threshold_floors():
    """The threshold step-down should stop at the manager-aligned floor."""
    rows = _load_mock_rows()
    for row in rows:
        row["predicted_label"] = "neutral"
        row["confidence"] = "0.90"
        row["prob_up"] = "0.05"
        row["prob_down"] = "0.05"
        row["prob_neutral"] = "0.90"

    report = se.build_report(rows, "THRESHOLD = 0.20\nMAX_LENGTH = 128\n")

    assert report["proposal"]["suggested_params"]["threshold"] == 0.20


def test_focus_labels_include_near_weakest_classes():
    """The proposal should focus all labels within FOCUS_MARGIN of the weakest."""
    assert se._weakest_labels({"up": 0.30, "down": 0.28, "neutral": 0.45}) == [
        "up",
        "down",
    ]


def test_classifier_summary_uses_generated_model_constant():
    """Nadi's generated classifier exposes MODEL/MODEL_DIR, not MODEL_NAME."""
    summary = se._classifier_summary_for_prompt(
        'MODEL = "ProsusAI/finbert"\nMODEL_DIR = "outputs/model"\n'
    )

    assert summary["model"] == '"ProsusAI/finbert"'
    assert summary["model_dir"] == '"outputs/model"'


def test_validation_rejects_train_rows():
    """Held-out rows are val or test; train rows must be rejected."""
    rows = _load_mock_rows()
    rows[0]["split"] = "train"

    try:
        se.validate_predictions(rows)
    except ValueError as error:
        assert "split must be val or test" in str(error)
    else:
        assert False, "Expected validate_predictions to reject train split rows"


def test_build_report_scores_requested_split_only():
    """eval_split=val must score ONLY the val rows — the loop's own scoring
    set — leaving the test rows untouched until the final report."""
    rows = _load_mock_rows()
    code = se._read_code(CLASSIFIER)
    val_rows = [r for r in rows if r["split"] == "val"]
    assert val_rows, "mock predictions must ship val rows"

    report = se.build_report(rows, code, eval_split="val")

    assert report["eval_split"] == "val"
    scored = report["misclassified_count"] + round(
        report["accuracy"] * len(val_rows))
    assert scored == len(val_rows) or report["accuracy"] in (0.0, 1.0)
    assert set(report["misclassified_ids"]) <= {r["article_id"] for r in val_rows}


def test_build_report_raises_when_split_absent():
    """Silently scoring the wrong split would defeat the val/test separation —
    a missing split must be a hard error."""
    rows = [dict(r, split="test") for r in _load_mock_rows()]
    code = se._read_code(CLASSIFIER)

    try:
        se.build_report(rows, code, eval_split="val")
    except ValueError as error:
        assert "no split=val rows" in str(error)
    else:
        assert False, "Expected build_report to reject missing val rows"


def test_code_notes_flag_class_collapse():
    """A class with near-zero recall must be called out — aggregate accuracy
    alone hides a classifier that predicts one class for everything."""
    class_accuracy = {"up": 0.97, "down": 0.0, "neutral": 0.04}
    notes = se.review_classifier_code("THRESHOLD = 0.2", class_accuracy)
    assert "class collapse" in notes
    assert "down" in notes and "neutral" in notes


def test_llm_review_can_supply_valid_judgment_text():
    """A valid LLM review should update reason/code_notes only."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    def fake_llm(prompt):
        assert "misclassified_ids" not in prompt
        assert "classifier_summary" in prompt
        assert "misclassified_sample" in prompt
        assert "headline" in prompt
        assert "true_label" in prompt
        assert "predicted_label" in prompt
        assert "prob_up" in prompt
        assert "prob_down" in prompt
        assert "prob_neutral" in prompt
        return json.dumps({
            "reason": "accuracy clears the target; neutral remains weakest",
            "code_notes": "threshold is fixed at 0.5 and neutral remains weak",
        })

    report = se.build_report(rows, code, llm_fn=fake_llm)

    assert (
        report["proposal"]["reason"]
        == "accuracy clears the target; neutral remains weakest"
    )
    assert (
        report["proposal"]["code_notes"]
        == "threshold is fixed at 0.5 and neutral remains weak"
    )
    assert report["proposal"]["recommended_action"] == "proceed"
    assert report["proposal"]["focus_labels"] == ["neutral"]
    assert report["proposal"]["suggested_params"] == {}


def test_llm_review_cannot_change_control_fields():
    """If the LLM returns proposal fields, the review should be ignored."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    def bad_llm(prompt):
        return json.dumps({
            "recommended_action": "retune",
            "reason": "I feel like retuning anyway",
            "focus_labels": ["neutral"],
            "suggested_params": {"threshold": 0.5},
            "code_notes": "try to override the gate",
        })

    report = se.build_report(rows, code, llm_fn=bad_llm)

    assert report["below_threshold"] is False
    assert report["proposal"]["recommended_action"] == "proceed"
    assert report["proposal"]["suggested_params"] == {}
    assert report["proposal"]["reason"].startswith("accuracy 0.60 clears")


def test_fenced_json_review_can_be_parsed_and_applied():
    """LLM JSON wrapped in markdown fences should still be accepted."""
    rows = se._read_predictions(PREDICTIONS)
    metrics = se.compute_metrics(rows)
    base = se.make_base_proposal(metrics, se._read_code(CLASSIFIER), "static notes")
    text = """```json
{
  "reason": "accuracy clears the target",
  "code_notes": "threshold hardcoded"
}
```"""

    proposal = se.apply_llm_review(
        metrics,
        "THRESHOLD = 0.5\n",
        base,
        se._misclassified_sample(rows),
        llm_fn=lambda _: text,
    )

    assert proposal["recommended_action"] == "proceed"
    assert proposal["focus_labels"] == ["neutral"]
    assert proposal["suggested_params"] == {}
    assert proposal["reason"] == "accuracy clears the target"


def test_invalid_llm_json_falls_back_to_base_proposal():
    """Malformed LLM text should not stop evaluation_report.json from existing."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    report = se.build_report(rows, code, llm_fn=lambda _: "Here is my answer: proceed")

    assert report["proposal"]["recommended_action"] == "proceed"
    assert report["proposal"]["reason"].startswith("accuracy 0.60 clears")


def test_prompt_excludes_misclassified_ids_and_full_source():
    """The LLM prompt should stay small while showing failure patterns."""
    rows = se._read_predictions(PREDICTIONS)
    metrics = se.compute_metrics(rows)
    code = "THRESHOLD = 0.5\nMAX_LENGTH = 128\n# lots of generated source\n"
    base = se.make_base_proposal(metrics, code, "static notes")

    failure_sample = se._misclassified_sample(rows)
    prompt = se._build_llm_prompt(metrics, code, base, failure_sample)

    assert "misclassified_ids" not in prompt
    assert "FNSPID_00006" not in prompt
    assert "misclassified_sample" in prompt
    assert "headline" in prompt
    assert "true_label" in prompt
    assert "predicted_label" in prompt
    assert "classifier_summary" in prompt
    assert "# lots of generated source" not in prompt


def test_misclassified_sample_is_limited_and_excludes_ids():
    """The LLM gets representative failures, not the full opaque id list."""
    rows = _load_mock_rows() * 10
    for row in rows:
        row["predicted_label"] = "neutral"

    sample = se._misclassified_sample(rows)

    assert len(sample) == se.MISCLASSIFIED_SAMPLE_SIZE
    assert set(sample[0]) == {
        "headline",
        "true_label",
        "predicted_label",
        "confidence",
        "prob_up",
        "prob_down",
        "prob_neutral",
    }
    assert "article_id" not in sample[0]


def test_llm_network_failure_falls_back_to_base_proposal():
    """A dead Ollama (OSError) must not stop evaluation_report.json."""
    rows = se._read_predictions(PREDICTIONS)
    code = se._read_code(CLASSIFIER)

    def dead_ollama(prompt):
        raise OSError("connection refused")

    report = se.build_report(rows, code, llm_fn=dead_ollama)

    assert report["proposal"]["recommended_action"] == "proceed"
    assert report["proposal"]["reason"].startswith("accuracy 0.60 clears")


def test_llm_review_applies_on_retune_without_touching_params():
    """On the retune path the LLM updates prose only; params survive the merge."""
    rows = _load_mock_rows()
    for row in rows:
        row["predicted_label"] = "neutral"
        row["confidence"] = "0.90"
        row["prob_up"] = "0.05"
        row["prob_down"] = "0.05"
        row["prob_neutral"] = "0.90"

    def fake_llm(prompt):
        return json.dumps({
            "reason": "accuracy misses the target; up and down carry no signal",
            "code_notes": "threshold still hardcoded",
        })

    report = se.build_report(rows, "THRESHOLD = 0.5\nMAX_LENGTH = 128\n", llm_fn=fake_llm)

    assert report["below_threshold"] is True
    assert report["proposal"]["recommended_action"] == "retune"
    assert report["proposal"]["suggested_params"] == {
        "threshold": 0.45,
        "max_length": 128,
    }
    assert (
        report["proposal"]["reason"]
        == "accuracy misses the target; up and down carry no signal"
    )


if __name__ == "__main__":
    test_build_report_matches_mock_data_contract()
    test_report_written_to_evaluation_report_json()
    test_low_accuracy_recommends_retune()
    test_validation_rejects_non_test_rows()
    test_llm_review_can_supply_valid_judgment_text()
    test_llm_review_cannot_change_control_fields()
    test_fenced_json_review_can_be_parsed_and_applied()
    test_invalid_llm_json_falls_back_to_base_proposal()
    test_prompt_excludes_misclassified_ids_and_full_source()
    test_misclassified_sample_is_limited_and_excludes_ids()
    test_llm_network_failure_falls_back_to_base_proposal()
    test_llm_review_applies_on_retune_without_touching_params()
    print("Evaluator tests passed")
