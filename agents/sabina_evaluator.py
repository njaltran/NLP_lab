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
THRESHOLD_STEP = 0.05    # lower the gate this much per retune so the loop explores
THRESHOLD_FLOOR = 0.20   # stop here — matches jack_manager.py's _RETUNE_SCHEDULE floor
DEFAULT_THRESHOLD = 0.5  # assume when classifier.py has no parseable THRESHOLD
FOCUS_MARGIN = 0.05      # also flag classes within this much of the weakest score
USE_OLLAMA = os.environ.get("EVALUATOR_USE_OLLAMA", "").lower() in {"1", "true", "yes"}
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("EVALUATOR_OLLAMA_MODEL", "llama3.2")
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

    weakest_labels = _weakest_labels(class_accuracy)
    if "neutral" in weakest_labels:
        notes.append("the neutral band (+/-1%) may be too narrow for the neutral class")

    return "; ".join(notes)


def _classifier_summary_for_prompt(code_text: str) -> dict:
    """Extract the classifier settings the LLM needs without executing code."""
    return {
        "model": _find_assignment(code_text, "MODEL"),
        "model_dir": _find_assignment(code_text, "MODEL_DIR"),
        "threshold": _find_assignment(code_text, "THRESHOLD"),
        "max_length": _find_assignment(code_text, "MAX_LENGTH"),
    }


def _weakest_labels(class_accuracy: dict) -> list[str]:
    weakest_score = min(class_accuracy.values())
    return [
        label_name
        for label_name, score in class_accuracy.items()
        if score <= weakest_score + FOCUS_MARGIN
    ]


def _next_threshold(code_text: str) -> float:
    """Read the classifier's current THRESHOLD and step it down (floored), so
    each retune proposes a gate Nadi hasn't run yet instead of a fixed 0.5."""
    current = _find_assignment(code_text, "THRESHOLD")
    try:
        current_val = float(current)
    except (TypeError, ValueError):
        current_val = DEFAULT_THRESHOLD
    return round(max(THRESHOLD_FLOOR, current_val - THRESHOLD_STEP), 2)


def _fallback_reason(metrics: dict, focus_labels: list[str]) -> str:
    weakest_score = min(metrics["class_accuracy"].values())
    if metrics["below_threshold"]:
        return (
            f"accuracy {metrics['accuracy']:.2f} below target "
            f"{TARGET_ACCURACY:.2f}; {', '.join(focus_labels)} class weakest"
        )
    return (
        f"accuracy {metrics['accuracy']:.2f} clears the {TARGET_ACCURACY:.2f} "
        f"target; {', '.join(focus_labels)} class is weakest "
        f"({weakest_score:.2f}) but the iteration budget favours proceeding"
    )


def _ollama_text_fields(metrics: dict, code_text: str, fallback: dict) -> dict:
    """Let Ollama rewrite only human-readable proposal text, never decisions."""
    if not USE_OLLAMA:
        return fallback

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        prompt = ChatPromptTemplate.from_template(
            """
You are Sabina's evaluator agent.

The deterministic evaluator has already decided the proposal's action,
focus_labels, and suggested_params. You must not change them.

Return ONLY a JSON object with these string fields:
- "reason": a short explanation grounded in the metrics
- "code_notes": concise static observations about classifier.py

classifier summary (JSON):
{classifier_summary}

metrics and deterministic proposal (JSON):
{proposal_context}
"""
        )
        llm = ChatOllama(
            model=DEFAULT_OLLAMA_MODEL,
            base_url=DEFAULT_OLLAMA_URL,
            temperature=0,
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "classifier_summary": json.dumps(_classifier_summary_for_prompt(code_text), indent=2),
            "proposal_context": json.dumps({**metrics, "proposal": fallback}, indent=2),
        })
        parsed = json.loads(response)
        return {
            "reason": str(parsed.get("reason") or fallback["reason"]),
            "code_notes": str(parsed.get("code_notes") or fallback["code_notes"]),
        }
    except Exception as exc:
        return {
            "reason": fallback["reason"],
            "code_notes": f"{fallback['code_notes']} (LLM skipped: {exc})",
        }


def make_proposal(metrics: dict, code_notes: str, code_text: str = "") -> dict:
    focus_labels = _weakest_labels(metrics["class_accuracy"])
    text_fields = {
        "reason": _fallback_reason(metrics, focus_labels),
        "code_notes": code_notes,
    }
    text_fields = _ollama_text_fields(metrics, code_text, text_fields)

    if metrics["below_threshold"]:
        return {
            "recommended_action": "retune",
            "reason": text_fields["reason"],
            "focus_labels": focus_labels,
            "suggested_params": {"threshold": _next_threshold(code_text), "max_length": 128},
            "code_notes": text_fields["code_notes"],
        }

    return {
        "recommended_action": "proceed",
        "reason": text_fields["reason"],
        "focus_labels": focus_labels,
        "suggested_params": {},
        "code_notes": text_fields["code_notes"],
    }


def build_report(rows: list[dict], code_text: str) -> dict:
    validate_predictions(rows)
    metrics = compute_metrics(rows)
    code_notes = review_classifier_code(code_text, metrics["class_accuracy"])
    return {**metrics, "proposal": make_proposal(metrics, code_notes, code_text)}


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
    output_path = state.get("output_path") or os.path.join(
        OUTPUT_DIR, "evaluation_report.json"
    )
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
        return self._invoke(
            {
                "predictions_path": predictions,
                "classifier_code_path": classifier_code,
                "output_path": output_path,
            }
        )


if __name__ == "__main__":
    agent = EvaluatorAgent()
    state = agent.run(
        predictions="mock_data/predictions_test.csv",
        classifier_code="mock_data/classifier.py",
    )
    print("Output file:", state["output_path"])
