"""Classifier Agent (Nadi) — generates the classifier code and executes it.

Reads processed_data.csv and optionally retune_request.json, generates the
classifier script classifier.py (ProsusAI/finbert), runs it to generate
predictions_test.csv, and outputs both.

See docs/data_contracts.md (Handoff 2).
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from typing import TypedDict

# Allow running as a package or direct script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from agents.base import Agent
    from agents.state import PipelineState
except ModuleNotFoundError:
    from base import Agent
    from state import PipelineState

OUTPUT_DIR = "outputs"

# --- Optional "agentic" code generation via a local LLM (Ollama) ---------------
# By default the classifier is generated from a fixed template (fully
# deterministic). When CLASSIFIER_USE_OLLAMA=true, we also ask a local LLM to
# ADAPT the classify() logic to the evaluator's feedback — but only keep its
# version if it passes a strict validation gauntlet, otherwise we fall back to
# the template. This mirrors how Sabina and Freddi use the LLM: the model may
# write code, but rules decide whether that code is allowed to run.
USE_OLLAMA = os.getenv("CLASSIFIER_USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("CLASSIFIER_OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = 120   # code generation is slower than a one-line answer
OLLAMA_MAX_TOKENS = 500        # a classify() function is short; cap it to bound time

# The exact columns the generated classifier must produce on the mock input
# (Handoff 2). Used to validate LLM-written code before we trust it.
EXPECTED_PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_up", "prob_down", "prob_neutral", "split",
]
MOCK_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mock_data", "processed_data.csv")

CLASSIFIER_TEMPLATE = """
import csv
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "ProsusAI/finbert"
MODEL_DIR = {model_dir}   # folder with our fine-tuned weights, or None to use pretrained FinBERT
MAX_LENGTH = {max_length}
THRESHOLD = {threshold}
FOCUS_LABELS = {focus_labels}
BOOST_FACTOR = {boost_factor}
SENTIMENT_TO_LABEL = {{"positive": "up", "negative": "down", "neutral": "neutral"}}

# Authenticate to the HF Hub when a token is in the env (higher rate limits,
# faster downloads); fall back to unauthenticated access when it is absent.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if MODEL_DIR and os.path.isdir(MODEL_DIR):
    # We have a fine-tuned model. Its head already outputs up/down/neutral, so
    # the label mapping comes straight from the model config (no sentiment step).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    ID2OURS = {{i: model.config.id2label[i].lower() for i in model.config.id2label}}
    print("[classifier] using fine-tuned model:", MODEL_DIR)
else:
    # No fine-tuned model: use pretrained FinBERT and translate its sentiment
    # (positive/negative/neutral) into a move direction (up/down/neutral).
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, token=HF_TOKEN)
    ID2OURS = {{i: SENTIMENT_TO_LABEL[model.config.id2label[i].lower()] for i in model.config.id2label}}
    print("[classifier] using pretrained FinBERT (no fine-tuned model found)")
model.eval()

def classify(title: str) -> dict:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    
    by_label = {{ID2OURS[i]: float(p) for i, p in enumerate(probs)}}
    
    # Apply class boost to focus labels if specified
    for fl in FOCUS_LABELS:
        if fl in by_label:
            by_label[fl] *= BOOST_FACTOR
            
    # Normalize probabilities after boosting
    total_prob = sum(by_label.values())
    if total_prob > 0:
        by_label = {{k: round(v / total_prob, 4) for k, v in by_label.items()}}
        
    top_label = max(by_label, key=by_label.get)
    top_prob = by_label[top_label]
    
    predicted = top_label if top_prob >= THRESHOLD else "neutral"
    
    return {{
        "predicted_label": predicted,
        "confidence": top_prob,
        "prob_up": by_label["up"],
        "prob_down": by_label["down"],
        "prob_neutral": by_label["neutral"],
    }}

def main(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {{src}}")
        
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        raise ValueError("Source file is empty")
        
    # Predict only the held-out rows (val + test); the train rows were used to
    # fit the model, so scoring them would leak. The evaluator scores val rows
    # during retune cycles and test rows only for the final report.
    # No split column -> treat all rows as test.
    if "split" in rows[0]:
        rows = [r for r in rows if r["split"] != "train"]
        if not rows:
            raise ValueError("No val/test rows in input")

    out_cols = [c for c in rows[0].keys() if c != "split"] + [
        "predicted_label", "confidence", "prob_up", "prob_down", "prob_neutral", "split"
    ]

    # Predict each held-out row, keeping its original split value.
    for row in rows:
        pred_data = classify(row["article_title"])
        row.update(pred_data)
        row.setdefault("split", "test")
        
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    src_file = sys.argv[1] if len(sys.argv) > 1 else "processed_data.csv"
    dst_file = sys.argv[2] if len(sys.argv) > 2 else "predictions_test.csv"
    main(src_file, dst_file)
"""

def _ollama_generate(prompt: str) -> str:
    """Call the local Ollama server and return its text answer. Same simple
    urllib approach Sabina's evaluator uses — no API key, no extra package."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": OLLAMA_MAX_TOKENS},
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        answer = json.loads(response.read().decode("utf-8"))
    return answer.get("response", "")


def _build_llm_prompt(focus_labels, code_notes) -> str:
    """Ask the LLM to rewrite ONLY the classify() function, adapting to the
    evaluator's feedback. We tell it exactly which globals it may use and what
    the function must return, so the output plugs straight into the template."""
    return (
        "You are the classifier agent in a stock-move prediction pipeline.\n"
        "Rewrite ONLY the Python function `classify(title)` to improve accuracy,\n"
        "responding to this feedback from the evaluator.\n\n"
        f"Weakest classes to focus on: {focus_labels}\n"
        f"Evaluator notes: {code_notes or 'none'}\n\n"
        "Rules:\n"
        "- Return ONLY the function, starting with `def classify(title):`.\n"
        "- It must return a dict with exactly these keys: predicted_label,\n"
        "  confidence, prob_up, prob_down, prob_neutral.\n"
        "- predicted_label must be one of: up, down, neutral.\n"
        "- You may use these already-defined globals: tokenizer, model, torch,\n"
        "  ID2OURS, THRESHOLD, MAX_LENGTH, FOCUS_LABELS, BOOST_FACTOR.\n"
        "- No imports, no comments outside the function, no markdown."
    )


def _extract_code(text: str) -> str:
    """Strip ```python fences if the LLM wrapped its answer in them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("python"):
            text = text[len("python"):]
    return text.strip()


def _swap_classify(default_code: str, new_classify: str) -> str:
    """Replace the classify() function in the generated code with the LLM's
    version, keeping everything else (imports, model loading, main()) intact."""
    start = default_code.index("def classify(")
    end = default_code.index("def main(", start)
    return default_code[:start] + new_classify.strip() + "\n\n\n" + default_code[end:]


def _looks_like_valid_classify(code: str) -> bool:
    """Cheap static checks (no model needed): the whole file parses, and it
    defines a function literally named `classify`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "classify"
               for node in ast.walk(tree))


def _runs_on_mock(code: str) -> bool:
    """The strong guardrail: actually run the candidate classifier on the mock
    data and confirm it produces the exact contract columns AND sane values.
    Any error, wrong columns, bad label, or non-numeric confidence means we
    reject the LLM's code."""
    import csv as _csv

    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "candidate.py")
        out_csv = os.path.join(tmp, "out.csv")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            subprocess.run([sys.executable, script, MOCK_DATA, out_csv],
                           check=True, timeout=OLLAMA_TIMEOUT_SECONDS * 4,
                           capture_output=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        if not os.path.exists(out_csv):
            return False

        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if reader.fieldnames != EXPECTED_PREDICTION_COLUMNS:
                return False
            rows = list(reader)

    if not rows:
        return False
    for row in rows:
        if row["predicted_label"] not in ("up", "down", "neutral"):
            return False
        try:
            float(row["confidence"])
        except (TypeError, ValueError):
            return False
    return True


def try_llm_classifier(default_code, retune_req, llm_fn=None):
    """Try to get an LLM-adapted classifier. Returns the adapted code if it
    passes every guardrail, otherwise None (caller falls back to the template).

    llm_fn lets tests inject a fake LLM; in normal use it is the real Ollama call."""
    call_llm = llm_fn or _ollama_generate
    focus_labels = retune_req.get("focus_labels", [])
    code_notes = retune_req.get("code_notes") or retune_req.get("reason")

    try:
        answer = call_llm(_build_llm_prompt(focus_labels, code_notes))
    except Exception as error:  # network down, timeout, bad server, ...
        print(f"[nadi] LLM call failed ({error}); keeping the template classifier")
        return None

    new_classify = _extract_code(answer)
    candidate = _swap_classify(default_code, new_classify)

    # Gauntlet: cheap static check first, then the real run-on-mock check.
    if not _looks_like_valid_classify(candidate):
        print("[nadi] LLM code did not parse / had no classify(); using template")
        return None
    if not _runs_on_mock(candidate):
        print("[nadi] LLM code failed the mock run / contract check; using template")
        return None
    return candidate


def generate_code(state: PipelineState) -> dict:
    """LangGraph node to read parameters from state/retune request and write classifier.py."""
    threshold = 0.5
    max_length = 128
    focus_labels = []
    boost_factor = 1.25

    # Read from retune request if it exists in state
    retune_req = state.get("retune_request")
    if retune_req:
        suggested = retune_req.get("suggested_params", {})
        threshold = suggested.get("threshold", threshold)
        max_length = suggested.get("max_length", max_length)
        boost_factor = suggested.get("boost_factor", boost_factor)
        focus_labels = retune_req.get("focus_labels", focus_labels)

    code_path = state.get("classifier_code_path") or os.path.join(OUTPUT_DIR, "classifier.py")
    os.makedirs(os.path.dirname(code_path) or ".", exist_ok=True)

    # Use fine-tuned weights only when the caller explicitly asks for them via
    # state["model_dir"] AND that folder exists. This is opt-in on purpose: a
    # stray folder should never silently swap the model out from under a run.
    model_dir = state.get("model_dir")
    if model_dir and not os.path.isdir(model_dir):
        model_dir = None

    formatted_code = CLASSIFIER_TEMPLATE.format(
        model_dir=repr(model_dir),
        threshold=threshold,
        max_length=max_length,
        focus_labels=repr(focus_labels),
        boost_factor=boost_factor
    )

    # Optional "agentic" step: on a retune, let the LLM adapt the classify()
    # logic to the evaluator's feedback. We only keep its version if it passes
    # validation; otherwise formatted_code stays the deterministic template.
    llm_fn = state.get("llm_fn")
    if retune_req and (llm_fn or USE_OLLAMA):
        adapted_code = try_llm_classifier(formatted_code, retune_req, llm_fn)
        if adapted_code:
            formatted_code = adapted_code
            print("[nadi] using LLM-adapted classifier (passed validation)")

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    print(f"[nadi] Generated classifier code at: {code_path}")

    # Keep a per-iteration copy so past retune attempts aren't lost when
    # classifier.py (the single contract file Sabina reads) gets overwritten.
    # iteration 0 = first pass before any retune; N = the Nth retune's code.
    iteration = retune_req.get("iteration", 0) if retune_req else 0
    history_dir = os.path.join(os.path.dirname(code_path) or ".", "classifier_history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"classifier_iter{iteration}.py")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    metadata = {
        "model_name": model_dir or "ProsusAI/finbert",
        "fine_tuning_params": {
            "threshold": threshold,
            "max_length": max_length,
            "focus_labels": focus_labels,
            "boost_factor": boost_factor
        }
    }

    return {
        "classifier_code_path": code_path,
        "classifier_history_path": history_path,
        "classifier_metadata": metadata
    }

def run_classifier(state: PipelineState) -> dict:
    """LangGraph node to run the generated classifier.py on processed_data.csv."""
    code_path = state["classifier_code_path"]
    data_path = state.get("processed_data_path") or "mock_data/processed_data.csv"
    pred_path = state.get("predictions_path") or os.path.join(OUTPUT_DIR, "predictions_test.csv")

    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)

    print(f"[nadi] Running classifier: {code_path} on {data_path} -> {pred_path}")
    
    # Run generated file as a subprocess
    subprocess.run([sys.executable, code_path, data_path, pred_path], check=True)

    print(f"[nadi] Predictions saved to: {pred_path}")

    return {
        "predictions_path": pred_path
    }

def build_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(PipelineState)
    builder.add_node("generate_code", generate_code)
    builder.add_node("run_classifier", run_classifier)
    builder.add_edge(START, "generate_code")
    builder.add_edge("generate_code", "run_classifier")
    builder.add_edge("run_classifier", END)
    return builder.compile(checkpointer=checkpointer)

class ClassifierAgent(Agent):
    """Classifier Agent (Nadi) behind the shared `.run()` interface."""

    def __init__(self, *, checkpointer=None, thread_id="classifier"):
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, processed_data: str, classifier_code: str, predictions: str,
            retune_request: str | None = None, model_dir: str | None = None) -> dict:
        """Runs the classifier generation and prediction step.

        Args:
            processed_data: Path to input processed_data.csv
            classifier_code: Output path for classifier.py
            predictions: Output path for predictions_test.csv
            retune_request: Optional path to input retune_request.json
            model_dir: Optional path to fine-tuned weights (outputs/finbert_finetuned).
                       When set and the folder exists, the classifier loads it instead
                       of pretrained FinBERT.
        """
        state = {
            "processed_data_path": os.path.abspath(processed_data),
            "classifier_code_path": os.path.abspath(classifier_code),
            "predictions_path": os.path.abspath(predictions),
        }
        if model_dir:
            state["model_dir"] = os.path.abspath(model_dir)
        if retune_request is not None and os.path.exists(retune_request):
            try:
                with open(retune_request, "r", encoding="utf-8") as f:
                    state["retune_request"] = json.load(f)
            except Exception as e:
                print(f"[nadi] Warning: Failed to parse retune_request file: {e}")
            
        return self._invoke(state)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Classifier Agent (Nadi).")
    parser.add_argument("--processed-data", default="mock_data/processed_data.csv", help="Input CSV path")
    parser.add_argument("--classifier-code", default="outputs/classifier.py", help="Output Python script path")
    parser.add_argument("--predictions", default="outputs/predictions_test.csv", help="Output predictions path")
    parser.add_argument("--retune-request", default=None, help="Input retune request JSON path")
    parser.add_argument("--model-dir", default=None,
                        help="Optional fine-tuned weights folder (e.g. outputs/finbert_finetuned)")

    args = parser.parse_args()

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=args.processed_data,
        classifier_code=args.classifier_code,
        predictions=args.predictions,
        retune_request=args.retune_request,
        model_dir=args.model_dir,
    )
    print("\nClassifier completed. Predictions at:", res["predictions_path"])
