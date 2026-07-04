"""Evaluator Agent
Owner: Sabina Rudolph

Reads the Classifier Agent's `predictions_test.csv` and generated
`classifier.py`, computes classification metrics deterministically, asks an LLM
to review the classifier code and metrics, and writes the Manager Agent's input:
`evaluation_report.json`.

Scoring is split-aware: retune cycles are scored on the `val` rows so the loop
cannot overfit the `test` rows, which are scored exactly once for the final
report (`eval_split` parameter, recorded in the report).

Exports
-------
EvaluatorAgent   Agent subclass — callers do EvaluatorAgent().run()
build_report     pure report builder used by the graph and tests

Usage (standalone test):
    python agents/sabina_evaluator.py

See docs/data_contracts.md, Handoff 3.
"""

import csv
import json
import os
import re
import sys
import urllib.request
from typing import Callable, TypedDict

try:
    from agents.base import Agent
except ModuleNotFoundError:
    from base import Agent

OUTPUT_DIR = "outputs"
TARGET_ACCURACY = 0.60
PROBABILITY_TOLERANCE = 0.02
DEFAULT_RETUNE_THRESHOLD = 0.50
THRESHOLD_STEP = 0.05
THRESHOLD_FLOOR = 0.20
DEFAULT_MAX_LENGTH = 128
FOCUS_MARGIN = 0.05
CLASS_COLLAPSE_FLOOR = 0.05   # per-class recall below this = collapse, flagged in code_notes
# Read once at import time so tests and demos get stable evaluator behavior.
USE_OLLAMA = os.getenv("EVALUATOR_USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("EVALUATOR_OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT_SECONDS = 30
LLM_TEMPERATURE = 0.2
MISCLASSIFIED_SAMPLE_SIZE = 25
LABELS = ("up", "down", "neutral")
PROPOSAL_FIELDS = {
    "recommended_action", "reason", "focus_labels", "suggested_params", "code_notes",
}
PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_up", "prob_down", "prob_neutral", "split",
]


class EvaluatorState(TypedDict, total=False):
    """State passed through the evaluator LangGraph.

    The graph starts with file paths, then adds the loaded predictions,
    classifier source text, and final report before writing JSON to disk.
    """

    predictions_path: str
    classifier_code_path: str
    output_path: str
    eval_split: str
    predictions: list[dict]
    code_text: str
    code_notes: str
    report: dict


def _read_predictions(path: str) -> list[dict]:
    """Read classifier predictions and enforce the exact Handoff 2 columns."""
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
    """Read the generated classifier.py as text for static review."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def validate_predictions(rows: list[dict]) -> None:
    """Check the Handoff 2 fields needed before scoring."""
    for row in rows:
        article_id = row.get("article_id", "")
        label = row.get("label", "")
        predicted = row.get("predicted_label", "")
        if row.get("split") not in ("val", "test"):
            raise ValueError(f"{article_id}: split must be val or test")
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
        if abs(sum(probs) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(f"{article_id}: prob_* columns must sum to about 1")
        if abs(confidence - max(probs)) > PROBABILITY_TOLERANCE:
            raise ValueError(f"{article_id}: confidence must equal max prob_*")


def compute_metrics(rows: list[dict]) -> dict:
    """Compute overall and per-class classification accuracy."""
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
    """Return the right-hand side of a simple `NAME = value` assignment."""
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^\n#]+)", code_text, re.M)
    return match.group(1).strip() if match else None


def _warn_unparseable(name: str, value: str, default: float | int) -> None:
    # A silent default here would make every retune re-propose the same
    # params, so the stalled loop must be visible in the run output.
    print(
        f"[sabina] could not parse {name}={value!r} in classifier.py; "
        f"assuming {default}",
        file=sys.stderr,
    )


def _float_assignment(code_text: str, name: str, default: float) -> float:
    """Read a numeric assignment from classifier.py, falling back if missing."""
    value = _find_assignment(code_text, name)
    if value is None:
        return default
    try:
        return float(value.strip("\"'"))
    except ValueError:
        _warn_unparseable(name, value, default)
        return default


def _int_assignment(code_text: str, name: str, default: int) -> int:
    """Read an integer assignment from classifier.py, falling back if missing."""
    value = _find_assignment(code_text, name)
    if value is None:
        return default
    try:
        return int(float(value.strip("\"'")))
    except ValueError:
        _warn_unparseable(name, value, default)
        return default


def _weakest_labels(class_accuracy: dict) -> list[str]:
    """Return labels within FOCUS_MARGIN of the weakest class accuracy."""
    weakest_score = min(class_accuracy.values())
    return [
        label_name
        for label_name, score in class_accuracy.items()
        if score <= weakest_score + FOCUS_MARGIN
    ]


def review_classifier_code(code_text: str, class_accuracy: dict) -> str:
    """Return concise static observations from the generated classifier.py."""
    notes = []
    threshold = _find_assignment(code_text, "THRESHOLD")
    if threshold is not None:
        notes.append(f"threshold hardcoded at {threshold} in classifier.py")

    # A class with (near-)zero recall means the classifier has collapsed onto
    # the other classes — the aggregate accuracy then mostly reflects the label
    # mix, not real signal. Say so explicitly; it changes how a retune should go.
    collapsed = [name for name, score in class_accuracy.items()
                 if score < CLASS_COLLAPSE_FLOOR]
    if collapsed:
        notes.append(
            f"class collapse: {', '.join(collapsed)} recall near zero — "
            "aggregate accuracy mostly reflects the majority class share, not signal"
        )

    weakest_labels = _weakest_labels(class_accuracy)
    if "neutral" in weakest_labels:
        notes.append("the neutral band (+/-1%) may be too narrow for the neutral class")

    return "; ".join(notes)


def _suggest_retune_params(code_text: str) -> dict:
    """Suggest the next deterministic retune step without asking the LLM."""
    current_threshold = _float_assignment(
        code_text, "THRESHOLD", DEFAULT_RETUNE_THRESHOLD
    )
    next_threshold = max(THRESHOLD_FLOOR, current_threshold - THRESHOLD_STEP)
    max_length = _int_assignment(code_text, "MAX_LENGTH", DEFAULT_MAX_LENGTH)
    return {
        "threshold": round(next_threshold, 2),
        "max_length": max_length,
    }


def make_base_proposal(metrics: dict, code_text: str, code_notes: str) -> dict:
    """Create the contract proposal with deterministic action and params."""
    weakest_score = min(metrics["class_accuracy"].values())
    focus_labels = _weakest_labels(metrics["class_accuracy"])

    if metrics["below_threshold"]:
        reason = (
            f"accuracy {metrics['accuracy']:.2f} below target "
            f"{TARGET_ACCURACY:.2f}; {', '.join(focus_labels)} class weakest"
        )
        return {
            "recommended_action": "retune",
            "reason": reason,
            "focus_labels": focus_labels,
            "suggested_params": _suggest_retune_params(code_text),
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


def validate_proposal(proposal: dict, metrics: dict) -> dict:
    """Validate the proposal before it can enter evaluation_report.json."""
    if not isinstance(proposal, dict):
        raise ValueError("LLM proposal must be a JSON object")

    missing = PROPOSAL_FIELDS - set(proposal)
    extra = set(proposal) - PROPOSAL_FIELDS
    if missing:
        raise ValueError(f"LLM proposal missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"LLM proposal has unexpected fields: {sorted(extra)}")

    action = proposal["recommended_action"]
    if action not in {"retune", "proceed"}:
        raise ValueError("recommended_action must be retune or proceed")

    expected_action = "retune" if metrics["below_threshold"] else "proceed"
    if action != expected_action:
        raise ValueError(
            "recommended_action conflicts with deterministic threshold gate: "
            f"expected {expected_action}, got {action}"
        )

    focus_labels = proposal["focus_labels"]
    if not isinstance(focus_labels, list) or not focus_labels:
        raise ValueError("focus_labels must be a non-empty list")
    invalid_labels = [label for label in focus_labels if label not in LABELS]
    if invalid_labels:
        raise ValueError(f"focus_labels contains invalid labels: {invalid_labels}")

    if not isinstance(proposal["suggested_params"], dict):
        raise ValueError("suggested_params must be an object")
    if action == "proceed" and proposal["suggested_params"] != {}:
        raise ValueError("proceed proposals must not carry suggested_params")
    if action == "retune":
        params = proposal["suggested_params"]
        if not {"threshold", "max_length"} <= set(params):
            raise ValueError("retune proposals need threshold and max_length")
        if not isinstance(params["threshold"], (int, float)):
            raise ValueError("threshold must be numeric")
        if not isinstance(params["max_length"], int):
            raise ValueError("max_length must be an integer")
    if not isinstance(proposal["reason"], str) or not proposal["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    if not isinstance(proposal["code_notes"], str):
        raise ValueError("code_notes must be a string")

    return proposal


def _extract_json_object(text: str) -> dict:
    """Parse plain JSON or a fenced JSON object returned by an LLM."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _prompt_metrics(metrics: dict) -> dict:
    """Small metrics view for the LLM; opaque row ids stay out of the prompt."""
    return {
        "accuracy": metrics["accuracy"],
        "below_threshold": metrics["below_threshold"],
        "class_accuracy": metrics["class_accuracy"],
        "misclassified_count": metrics["misclassified_count"],
    }


def _misclassified_sample(
    rows: list[dict],
    limit: int = MISCLASSIFIED_SAMPLE_SIZE,
) -> list[dict]:
    """Small failure sample for LLM pattern analysis; row ids stay out."""
    sample = []
    for row in rows:
        if row["label"] == row["predicted_label"]:
            continue
        sample.append({
            "headline": row["article_title"],
            "true_label": row["label"],
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "prob_up": float(row["prob_up"]),
            "prob_down": float(row["prob_down"]),
            "prob_neutral": float(row["prob_neutral"]),
        })
        if len(sample) >= limit:
            break
    return sample


def _classifier_summary_for_prompt(code_text: str) -> dict:
    """Summarise source signals instead of sending the whole generated file."""
    model = _find_assignment(code_text, "MODEL")
    return {
        "threshold": _find_assignment(code_text, "THRESHOLD"),
        "max_length": _find_assignment(code_text, "MAX_LENGTH"),
        "model": model,
        "model_dir": _find_assignment(code_text, "MODEL_DIR"),
        "uses_finbert": "finbert" in code_text.lower(),
        "maps_sentiment_to_label": "predicted_label" in code_text,
    }


def _build_llm_prompt(
    metrics: dict,
    code_text: str,
    base_proposal: dict,
    failure_sample: list[dict],
) -> str:
    """Prompt the LLM for judgement text only; control fields are deterministic."""
    payload = {
        "eval_split": metrics.get("eval_split", "test"),
        "metrics": _prompt_metrics(metrics),
        "classifier_summary": _classifier_summary_for_prompt(code_text),
        "misclassified_sample": failure_sample,
        "deterministic_proposal": base_proposal,
    }
    return (
        "You are the evaluator agent in a multi-agent stock-move "
        "prediction pipeline.\n\n"
        "Review the deterministic metrics and classifier summary. The action, "
        "focus labels, and suggested params are already fixed by code and must "
        "not be changed by you.\n\n"
        "Return ONLY valid JSON with exactly these two string fields:\n"
        '{"reason": "...", "code_notes": "..."}\n\n'
        "Ground your reason in accuracy, target, weakest labels, and any useful "
        "classifier observation. Use the misclassified sample to describe the "
        "failure pattern when possible. Do not include markdown or extra keys.\n\n"
        "Judge per-class accuracy, not just the aggregate: a classifier that "
        "predicts one class for almost everything can score near that class's "
        "share of the data while learning nothing — call that out in code_notes "
        "if you see it. The eval_split field says which held-out split these "
        "metrics come from; retunes are scored on val so the test split stays "
        "unseen until the final report. Prefer observations that improve "
        "balance across classes over ones that chase the aggregate number.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}"
    )


def _validate_llm_review(review: dict) -> dict:
    """Accept only the fields the LLM is allowed to write."""
    if not isinstance(review, dict):
        raise ValueError("LLM review must be a JSON object")
    if set(review) != {"reason", "code_notes"}:
        raise ValueError("LLM review may only contain reason and code_notes")
    if not isinstance(review["reason"], str) or not review["reason"].strip():
        raise ValueError("LLM reason must be a non-empty string")
    if not isinstance(review["code_notes"], str):
        raise ValueError("LLM code_notes must be a string")
    return {
        "reason": review["reason"].strip(),
        "code_notes": review["code_notes"].strip(),
    }


def _ollama_generate(prompt: str) -> str:
    """Call local Ollama when explicitly enabled; no API key is needed."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE},
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("response", "")


def apply_llm_review(
    metrics: dict,
    code_text: str,
    base_proposal: dict,
    failure_sample: list[dict],
    llm_fn: Callable[[str], str] | None = None,
) -> dict:
    """Let an LLM improve reason/code_notes without changing control fields."""
    if llm_fn is None and not USE_OLLAMA:
        return base_proposal

    prompt = _build_llm_prompt(metrics, code_text, base_proposal, failure_sample)
    try:
        response = (llm_fn or _ollama_generate)(prompt)
        review = _validate_llm_review(_extract_json_object(response))
    except (
        ValueError,
        json.JSONDecodeError,
        TimeoutError,
        OSError,
    ) as error:
        print(f"[sabina] LLM review failed ({error}); using deterministic fallback.")
        return base_proposal

    proposal = {
        **base_proposal,
        "reason": review["reason"],
        "code_notes": review["code_notes"],
    }
    return validate_proposal(proposal, metrics)


def _select_split(rows: list[dict], eval_split: str) -> list[dict]:
    """Return only the rows belonging to `eval_split` ("val" or "test").

    The retune loop scores itself on the val rows so the test rows stay unseen
    until the final report — repeatedly tuning against the test set would
    overfit it and inflate the final accuracy. Missing rows are a hard error:
    silently scoring the wrong split would defeat the whole point.
    """
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    selected = [row for row in rows if row.get("split") == eval_split]
    if not selected:
        raise ValueError(
            f"no split={eval_split} rows in predictions — regenerate "
            "processed_data.csv with the Processing agent (train/val/test split)"
        )
    return selected


def build_report(
    rows: list[dict],
    code_text: str,
    llm_fn: Callable[[str], str] | None = None,
    eval_split: str = "test",
) -> dict:
    """Build the complete Handoff 3 `evaluation_report.json` object, scored on
    the `eval_split` rows only ("val" during retune cycles, "test" for the
    final report).

    Metrics, action, focus labels, and suggested params are deterministic.
    The optional LLM can only improve `reason` and `code_notes`.
    """
    rows = _select_split(rows, eval_split)
    validate_predictions(rows)
    metrics = {**compute_metrics(rows), "eval_split": eval_split}
    code_notes = review_classifier_code(code_text, metrics["class_accuracy"])
    base_proposal = validate_proposal(
        make_base_proposal(metrics, code_text, code_notes),
        metrics,
    )
    failure_sample = _misclassified_sample(rows)
    proposal = apply_llm_review(
        metrics,
        code_text,
        base_proposal,
        failure_sample,
        llm_fn=llm_fn,
    )
    return {**metrics, "proposal": proposal}


def _write_json(path: str, obj: dict) -> None:
    """Write UTF-8 JSON and create the output folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_inputs(state: EvaluatorState) -> dict:
    """LangGraph node — load classifier predictions and generated code."""
    return {
        "predictions": _read_predictions(state["predictions_path"]),
        "code_text": _read_code(state["classifier_code_path"]),
    }


def evaluate(state: EvaluatorState) -> dict:
    """LangGraph node — compute metrics and build the evaluator report."""
    return {"report": build_report(state["predictions"], state["code_text"],
                                   eval_split=state.get("eval_split", "test"))}


def write_report(state: EvaluatorState) -> dict:
    """LangGraph node — write `evaluation_report.json` for the Manager Agent."""
    output_path = state.get("output_path") or os.path.join(
        OUTPUT_DIR,
        "evaluation_report.json",
    )
    _write_json(output_path, state["report"])
    return {"output_path": output_path}


def build_graph(checkpointer):
    """Compile the evaluator's three-step LangGraph: load, evaluate, write."""
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
    """Evaluator behind the shared `.run()` interface."""

    def __init__(self, *, output_dir=OUTPUT_DIR, checkpointer=None, thread_id="evaluator"):
        self._output_dir = output_dir
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, predictions: str, classifier_code: str, eval_split: str = "test") -> dict:
        """`eval_split` picks which held-out rows to score: "val" during retune
        cycles (so the loop can't overfit the test set), "test" for the final
        report only."""
        output_path = os.path.join(self._output_dir, "evaluation_report.json")
        return self._invoke({
            "predictions_path": predictions,
            "classifier_code_path": classifier_code,
            "output_path": output_path,
            "eval_split": eval_split,
        })


if __name__ == "__main__":
    agent = EvaluatorAgent()
    state = agent.run(
        predictions="mock_data/predictions_test.csv",
        classifier_code="mock_data/classifier.py",
    )
    print("Output file:", state["output_path"])
