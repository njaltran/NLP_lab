"""Sabina's Evaluator Agent.

This agent runs after the Classifier Agent and before the Manager Agent. It does
not train the model again and it does not create new predictions. Its main task
is to check the classifier output, calculate evaluation metrics, and write the
report that is passed to the Manager Agent.

Inputs
------
predictions_test.csv
    The prediction file from the Classifier Agent. It contains the true label,
    predicted label, confidence score, class probabilities, and split for each
    row.

classifier.py
    The generated classifier code. The Evaluator reads this file as text so it
    can add short notes about the classifier setup, for example the threshold
    used for prediction.

Output
------
evaluation_report.json
    The report written by the Evaluator Agent. It contains the metrics and my
    recommendation (`retune` or `proceed`). The Manager Agent uses this report
    in the next step, but the final pipeline decision is still made there.

The metric calculation and recommendation are rule-based. If the optional LLM is
enabled, it is only used to make the explanation text clearer. It is not allowed
to change the metrics, action, focus labels, or suggested parameters.

The Evaluator can score different data splits. During retuning, it should score
the `val` rows. For the final report, it should score the `test` rows. This
keeps the test data separate until the final evaluation.

Exports
-------
EvaluatorAgent
    Agent class used by the pipeline through `EvaluatorAgent().run()`.

build_report
    Pure report-building function used by the graph and by tests.

Usage
-----
Run this file directly for a small standalone test:

    python agents/sabina_evaluator.py

See also: docs/data_contracts.md, Handoff 3.
"""

import json
import os
import re
import sys
import urllib.request
from typing import Callable, TypedDict

try:
    from agents.base import Agent
    from agents.contracts import (
        LABELS,
        PREDICTION_COLUMNS,
        read_prediction_rows,
        validate_prediction_rows,
    )
except ModuleNotFoundError:
    from base import Agent
    from contracts import (
        LABELS,
        PREDICTION_COLUMNS,
        read_prediction_rows,
        validate_prediction_rows,
    )

OUTPUT_DIR = "outputs"
TARGET_ACCURACY = 0.60
DEFAULT_RETUNE_THRESHOLD = 0.50
THRESHOLD_STEP = 0.05
THRESHOLD_FLOOR = 0.20
DEFAULT_MAX_LENGTH = 128
FOCUS_MARGIN = 0.05
CLASS_COLLAPSE_FLOOR = 0.05   # per-class recall below this is treated as class collapse

USE_OLLAMA = os.getenv("EVALUATOR_USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("EVALUATOR_OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT_SECONDS = 30
LLM_TEMPERATURE = 0.2
MISCLASSIFIED_SAMPLE_SIZE = 25
PROPOSAL_FIELDS = {
    "recommended_action", "reason", "focus_labels", "suggested_params", "code_notes",
}


class EvaluatorState(TypedDict, total=False):
    """State passed through the Evaluator's internal LangGraph.

    The graph starts with the input file paths. `load_inputs` adds the loaded
    prediction rows and classifier source text. `evaluate` builds the report.
    `write_report` saves the report as JSON.
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
    """Read the prediction CSV from the Classifier Agent."""
    return read_prediction_rows(path)


def _read_code(path: str) -> str:
    """Read the generated classifier code as text.

    The Evaluator does not execute the classifier again. It only reads the code
    so it can add simple notes to the final report.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def validate_predictions(rows: list[dict]) -> None:
    """Check that the prediction rows match the expected input format.

    This checks required columns, valid labels, probability values, confidence,
    and split. If this validation fails, the evaluation report should not be
    trusted.
    """
    validate_prediction_rows(rows)


def compute_metrics(rows: list[dict]) -> dict:
    """Calculate the metric section of `evaluation_report.json`.

    The report contains overall accuracy, class accuracy, class support,
    misclassified count, and the ids of misclassified articles. Class support is
    included so that a missing class is not confused with model collapse.
    """
    total = len(rows)
    wrong = [row for row in rows if row["label"] != row["predicted_label"]]
    class_accuracy = {}
    class_support = {}

    for label_name in LABELS:
        class_rows = [row for row in rows if row["label"] == label_name]
        correct = sum(row["predicted_label"] == label_name for row in class_rows)
        class_support[label_name] = len(class_rows)
        class_accuracy[label_name] = (
            round(correct / len(class_rows), 2) if class_rows else 0.0
        )

    accuracy = round((total - len(wrong)) / total, 2)
    return {
        "accuracy": accuracy,
        "below_threshold": accuracy < TARGET_ACCURACY,
        "class_accuracy": class_accuracy,
        "class_support": class_support,
        "misclassified_count": len(wrong),
        "misclassified_ids": [row["article_id"] for row in wrong],
    }


def _find_assignment(code_text: str, name: str) -> str | None:
    """Find the value of a simple `NAME = value` assignment in classifier.py."""
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^\n#]+)", code_text, re.M)
    return match.group(1).strip() if match else None


def _warn_unparseable(name: str, value: str, default: float | int) -> None:
    """Print a warning when a classifier setting cannot be parsed."""
    print(
        f"[sabina] could not parse {name}={value!r} in classifier.py; "
        f"assuming {default}",
        file=sys.stderr,
    )


def _float_assignment(code_text: str, name: str, default: float) -> float:
    """Read a numeric setting from classifier.py, or use the default value."""
    value = _find_assignment(code_text, name)
    if value is None:
        return default
    try:
        return float(value.strip("\"'"))
    except ValueError:
        _warn_unparseable(name, value, default)
        return default


def _int_assignment(code_text: str, name: str, default: int) -> int:
    """Read an integer setting from classifier.py, or use the default value."""
    value = _find_assignment(code_text, name)
    if value is None:
        return default
    try:
        return int(float(value.strip("\"'")))
    except ValueError:
        _warn_unparseable(name, value, default)
        return default


def _weakest_labels(class_accuracy: dict, class_support: dict | None = None) -> list[str]:
    """Return the labels with the weakest class accuracy."""
    supported = {
        label_name: score
        for label_name, score in class_accuracy.items()
        if class_support is None or class_support.get(label_name, 0) > 0
    }
    if not supported:
        supported = class_accuracy
    weakest_score = min(supported.values())
    return [
        label_name
        for label_name, score in supported.items()
        if score <= weakest_score + FOCUS_MARGIN
    ]


def review_classifier_code(code_text: str, class_accuracy: dict) -> str:
    """Keep the older test interface working when only class accuracy is passed."""
    return review_classifier_metrics(code_text, class_accuracy)


def review_classifier_metrics(
    code_text: str,
    class_accuracy: dict,
    class_support: dict | None = None,
) -> str:
    """Create short notes about the classifier and the metric results.

    This is not a full code review. It only checks simple points that are useful
    for the Manager Agent, such as the threshold, weak labels, missing classes,
    and possible class collapse.
    """
    notes = []
    threshold = _find_assignment(code_text, "THRESHOLD")
    if threshold is not None:
        notes.append(f"threshold hardcoded at {threshold} in classifier.py")

    collapsed = [
        name for name, score in class_accuracy.items()
        if (class_support is None or class_support.get(name, 0) > 0)
        and score < CLASS_COLLAPSE_FLOOR
    ]
    if collapsed:
        notes.append(
            f"class collapse: {', '.join(collapsed)} recall near zero — "
            "aggregate accuracy mostly reflects the majority class share, not signal"
        )

    missing = [
        name for name in class_accuracy
        if class_support is not None and class_support.get(name, 0) == 0
    ]
    if missing:
        notes.append(
            f"no {', '.join(missing)} rows in this eval split — support is zero, "
            "so this is not evidence of class collapse"
        )

    weakest_labels = _weakest_labels(class_accuracy, class_support)
    if "neutral" in weakest_labels:
        notes.append("the neutral band (+/-1%) may be too narrow for the neutral class")

    return "; ".join(notes)


def _suggest_retune_params(code_text: str) -> dict:
    """Suggest the next retune parameters using fixed rules."""
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
    """Create the Evaluator's rule-based recommendation.

    If accuracy is below the target, the recommendation is `retune`. If the
    target is reached, the recommendation is `proceed`. The optional LLM is not
    allowed to change this action.
    """
    focus_labels = _weakest_labels(
        metrics["class_accuracy"],
        metrics.get("class_support"),
    )
    weakest_score = min(metrics["class_accuracy"][label] for label in focus_labels)

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
    """Validate the recommendation before it is written into the report.

    This makes sure that the handoff to the Manager Agent has the expected
    fields and that the recommendation follows the fixed threshold rule.
    """
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
    """Select the metric fields that are useful for optional LLM wording."""
    return {
        "accuracy": metrics["accuracy"],
        "below_threshold": metrics["below_threshold"],
        "class_accuracy": metrics["class_accuracy"],
        "class_support": metrics["class_support"],
        "misclassified_count": metrics["misclassified_count"],
    }


def _misclassified_sample(
    rows: list[dict],
    limit: int = MISCLASSIFIED_SAMPLE_SIZE,
) -> list[dict]:
    """Collect a small sample of wrong predictions for the optional LLM.

    Article ids are left out because the LLM only needs the failure pattern, not
    the full list of report identifiers.
    """
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
    """Summarise the classifier code instead of sending the full source text."""
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
    """Build the optional LLM prompt for `reason` and `code_notes` only."""
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
        "Judge per-class accuracy together with class_support, not just the "
        "aggregate: a classifier that predicts one class for almost everything "
        "can score near that class's share of the data while learning nothing. "
        "A class with support 0 is absent from this split, not collapsed. The "
        "eval_split field says which held-out split these metrics come from; "
        "retunes are scored on val so the test split stays unseen until the "
        "final report. Prefer observations that improve balance across classes "
        "over ones that chase the aggregate number.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}"
    )


def _validate_llm_review(review: dict) -> dict:
    """Accept only the two text fields that the LLM is allowed to write."""
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
    """Call local Ollama when it is explicitly enabled."""
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
    """Let the optional LLM improve the explanation text.

    The LLM is not allowed to change the recommendation or the metrics. If the
    LLM is disabled, unavailable, or returns invalid JSON, the Evaluator keeps
    the rule-based proposal so the pipeline can still continue.
    """
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
    """Return only the rows from the requested evaluation split.

    Retune cycles should use `val`, while the final report should use `test`.
    This keeps the test split from being reused during tuning.
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
    """Build the complete Evaluator report for the Manager Agent.

    Parameters
    ----------
    rows:
        Prediction rows from the Classifier Agent.
    code_text:
        Generated classifier source code.
    llm_fn:
        Optional function used for LLM review in tests or demos.
    eval_split:
        Split to score: `val` during retuning and `test` for final evaluation.

    Returns
    -------
    dict
        Full content of `evaluation_report.json`.
    """
    rows = _select_split(rows, eval_split)
    validate_predictions(rows)
    metrics = {**compute_metrics(rows), "eval_split": eval_split}
    code_notes = review_classifier_metrics(
        code_text,
        metrics["class_accuracy"],
        metrics["class_support"],
    )
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
    """Write a JSON file and create the output folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_inputs(state: EvaluatorState) -> dict:
    """LangGraph node 1: load the prediction CSV and classifier code."""
    return {
        "predictions": _read_predictions(state["predictions_path"]),
        "code_text": _read_code(state["classifier_code_path"]),
    }


def evaluate(state: EvaluatorState) -> dict:
    """LangGraph node 2: validate the input and build the report."""
    return {"report": build_report(state["predictions"], state["code_text"],
                                   eval_split=state["eval_split"])}


def write_report(state: EvaluatorState) -> dict:
    """LangGraph node 3: write `evaluation_report.json` for the Manager."""
    output_path = state.get("output_path") or os.path.join(
        OUTPUT_DIR,
        "evaluation_report.json",
    )
    _write_json(output_path, state["report"])
    return {"output_path": output_path}


def build_graph(checkpointer):
    """Compile the Evaluator Agent's three-step LangGraph.

    The Evaluator itself is a simple linear graph: load inputs, evaluate, and
    write the report. The larger retuning loop is handled by the full pipeline
    and the Manager Agent.
    """
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
    """Sabina's Evaluator Agent using the shared `.run()` interface."""

    def __init__(self, *, output_dir=OUTPUT_DIR, checkpointer=None, thread_id="evaluator"):
        self._output_dir = output_dir
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, predictions: str, classifier_code: str, eval_split: str = "test") -> dict:
        """Run the Evaluator Agent.

        Parameters
        ----------
        predictions:
            Path to the Classifier Agent's prediction CSV.
        classifier_code:
            Path to the generated `classifier.py`.
        eval_split:
            Split to score: `val` for retuning or `test` for final evaluation.

        Returns
        -------
        dict
            Final LangGraph state, including the path to `evaluation_report.json`.
        """
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
