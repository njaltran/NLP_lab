"""Evaluator Agent (Sabina) — score Nadi's predictions and propose next step.

Reads Nadi's `predictions_test.csv` and generated `classifier.py`, computes
classification metrics, reviews the code statically, and writes Jack's input:
`evaluation_report.json`.

See docs/data_contracts.md (Handoff 3).
"""

import csv
import json
import os
import re
from typing import TypedDict

try:
    from agents.base import Agent
except ModuleNotFoundError:
    from base import Agent

OUTPUT_DIR = "outputs"
TARGET_ACCURACY = 0.60
LABELS = ("up", "down", "neutral")
PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_up", "prob_down", "prob_neutral", "split",
]


class EvaluatorState(TypedDict, total=False):
    predictions_path: str
    classifier_code_path: str
    output_path: str
    predictions: list[dict]
    code_text: str
    code_notes: str
    report: dict


def _read_predictions(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if reader.fieldnames != PREDICTION_COLUMNS:
        raise ValueError(
            "predictions_test.csv columns do not match data contract: "
            f"{reader.fieldnames}"
        )
    if not rows:
        raise ValueError("predictions_test.csv must contain at least one test row")
    return rows


def _read_code(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def validate_predictions(rows: list[dict]) -> None:
    """Check the parts of Handoff 2 Sabina relies on before scoring."""
    for row in rows:
        article_id = row.get("article_id", "")
        label = row.get("label", "")
        predicted = row.get("predicted_label", "")
        if row.get("split") != "test":
            raise ValueError(f"{article_id}: split must be test")
        if label not in LABELS:
            raise ValueError(f"{article_id}: invalid label {label!r}")
        if predicted not in LABELS:
            raise ValueError(f"{article_id}: invalid predicted_label {predicted!r}")

        probs = [float(row[f"prob_{label_name}"]) for label_name in LABELS]
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError(f"{article_id}: confidence must be between 0 and 1")
        if any(prob < 0 or prob > 1 for prob in probs):
            raise ValueError(f"{article_id}: probabilities must be between 0 and 1")
        if abs(sum(probs) - 1.0) > 0.02:
            raise ValueError(f"{article_id}: prob_* columns must sum to about 1")
        if abs(confidence - max(probs)) > 0.02:
            raise ValueError(f"{article_id}: confidence must equal max prob_*")


def compute_metrics(rows: list[dict]) -> dict:
    """Compute overall and per-class accuracy, like Diana's classification checks."""
    total = len(rows)
    wrong = [row for row in rows if row["label"] != row["predicted_label"]]
    class_accuracy = {}

    for label_name in LABELS:
        class_rows = [row for row in rows if row["label"] == label_name]
        correct = sum(row["predicted_label"] == label_name for row in class_rows)
        class_accuracy[label_name] = (
            round(correct / len(class_rows), 2) if class_rows else 0.0
        )

    accuracy = round((total - len(wrong)) / total, 2)
    return {
        "accuracy": accuracy,
        "below_threshold": accuracy < TARGET_ACCURACY,
        "class_accuracy": class_accuracy,
        "misclassified_count": len(wrong),
        "misclassified_ids": [row["article_id"] for row in wrong],
    }


def _find_assignment(code_text: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^\n#]+)", code_text, re.M)
    return match.group(1).strip() if match else None


def review_classifier_code(code_text: str, class_accuracy: dict) -> str:
    """Return concise static observations from Nadi's generated classifier.py."""
    notes = []
    threshold = _find_assignment(code_text, "THRESHOLD")
    if threshold is not None:
        notes.append(f"threshold hardcoded at {threshold} in classifier.py")

    weakest_label = min(class_accuracy, key=class_accuracy.get)
    if weakest_label == "neutral":
        notes.append("the neutral band (+/-1%) may be too narrow for the neutral class")

    return "; ".join(notes)


def make_proposal(metrics: dict, code_notes: str) -> dict:
    weakest_score = min(metrics["class_accuracy"].values())
    focus_labels = [
        label_name
        for label_name, score in metrics["class_accuracy"].items()
        if score == weakest_score
    ]

    if metrics["below_threshold"]:
        reason = (
            f"accuracy {metrics['accuracy']:.2f} below target "
            f"{TARGET_ACCURACY:.2f}; {', '.join(focus_labels)} class weakest"
        )
        return {
            "recommended_action": "retune",
            "reason": reason,
            "focus_labels": focus_labels,
            "suggested_params": {"threshold": 0.5, "max_length": 128},
            "code_notes": code_notes,
        }

    reason = (
        f"accuracy {metrics['accuracy']:.2f} clears the {TARGET_ACCURACY:.2f} "
        f"target; {', '.join(focus_labels)} class is weakest "
        f"({weakest_score:.2f}) but the iteration budget favours proceeding"
    )
    return {
        "recommended_action": "proceed",
        "reason": reason,
        "focus_labels": focus_labels,
        "suggested_params": {},
        "code_notes": code_notes,
    }


def build_report(rows: list[dict], code_text: str) -> dict:
    validate_predictions(rows)
    metrics = compute_metrics(rows)
    code_notes = review_classifier_code(code_text, metrics["class_accuracy"])
    return {**metrics, "proposal": make_proposal(metrics, code_notes)}


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_inputs(state: EvaluatorState) -> dict:
    return {
        "predictions": _read_predictions(state["predictions_path"]),
        "code_text": _read_code(state["classifier_code_path"]),
    }


def evaluate(state: EvaluatorState) -> dict:
    return {"report": build_report(state["predictions"], state["code_text"])}


def write_report(state: EvaluatorState) -> dict:
    output_path = state.get("output_path") or os.path.join(OUTPUT_DIR, "evaluation_report.json")
    _write_json(output_path, state["report"])
    return {"output_path": output_path}


def build_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(EvaluatorState)
    builder.add_node("load_inputs", load_inputs)
    builder.add_node("evaluate", evaluate)
    builder.add_node("write_report", write_report)
    builder.add_edge(START, "load_inputs")
    builder.add_edge("load_inputs", "evaluate")
    builder.add_edge("evaluate", "write_report")
    builder.add_edge("write_report", END)
    return builder.compile(checkpointer=checkpointer)


class EvaluatorAgent(Agent):
    """Sabina's evaluator behind the shared `.run()` interface."""

    def __init__(self, *, output_dir=OUTPUT_DIR, checkpointer=None, thread_id="evaluator"):
        self._output_dir = output_dir
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, predictions: str, classifier_code: str) -> dict:
        output_path = os.path.join(self._output_dir, "evaluation_report.json")
        return self._invoke({
            "predictions_path": predictions,
            "classifier_code_path": classifier_code,
            "output_path": output_path,
        })


if __name__ == "__main__":
    agent = EvaluatorAgent()
    state = agent.run(
        predictions="mock_data/predictions_test.csv",
        classifier_code="mock_data/classifier.py",
    )
    print("Output file:", state["output_path"])
