"""Classifier Agent (Nadi) — generates the classifier code and executes it.

Reads processed_data.csv and optionally retune_request.json, generates the
classifier script classifier.py (ProsusAI/finbert), runs it to generate
predictions_test.csv, and outputs both.

See docs/data_contracts.md (Handoff 2).
"""

import json
import os
import subprocess
import sys
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

CLASSIFIER_TEMPLATE = """\"\"\"Generated Classifier Script.
Runs FinBERT inference on processed_data.csv and writes predictions_test.csv.
\"\"\"

import csv
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "ProsusAI/finbert"
MAX_LENGTH = {max_length}
THRESHOLD = {threshold}
FOCUS_LABELS = {focus_labels}
SENTIMENT_TO_LABEL = {{"positive": "up", "negative": "down", "neutral": "neutral"}}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

ID2OURS = {{i: SENTIMENT_TO_LABEL[model.config.id2label[i].lower()] for i in model.config.id2label}}

def classify(title: str) -> dict:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    
    by_label = {{ID2OURS[i]: float(p) for i, p in enumerate(probs)}}
    
    # Apply class boost to focus labels if specified
    for fl in FOCUS_LABELS:
        if fl in by_label:
            by_label[fl] *= 1.25
            
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
        
    out_cols = list(rows[0].keys()) + [
        "predicted_label", "confidence", "prob_up", "prob_down", "prob_neutral", "split"
    ]
    
    # Process all rows and mark them as test split
    for row in rows:
        pred_data = classify(row["article_title"])
        row.update(pred_data)
        row["split"] = "test"
        
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

def generate_code(state: PipelineState) -> dict:
    """LangGraph node to read parameters from state/retune request and write classifier.py."""
    threshold = 0.5
    max_length = 128
    focus_labels = []

    # Read from retune request if it exists in state
    retune_req = state.get("retune_request")
    if retune_req and isinstance(retune_req, dict):
        suggested = retune_req.get("suggested_params", {})
        threshold = suggested.get("threshold", threshold)
        max_length = suggested.get("max_length", max_length)
        focus_labels = retune_req.get("focus_labels", focus_labels)

    code_path = state.get("classifier_code_path") or os.path.join(OUTPUT_DIR, "classifier.py")
    os.makedirs(os.path.dirname(code_path) or ".", exist_ok=True)

    formatted_code = CLASSIFIER_TEMPLATE.format(
        threshold=threshold,
        max_length=max_length,
        focus_labels=repr(focus_labels)
    )

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    print(f"[nadi] Generated classifier code at: {code_path}")

    metadata = {
        "model_name": "ProsusAI/finbert",
        "fine_tuning_params": {
            "threshold": threshold,
            "max_length": max_length,
            "focus_labels": focus_labels
        }
    }

    return {
        "classifier_code_path": code_path,
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

    def run(self, processed_data: str, classifier_code: str, predictions: str, retune_request: str | None = None) -> dict:
        """Runs the classifier generation and prediction step.
        
        Args:
            processed_data: Path to input processed_data.csv
            classifier_code: Output path for classifier.py
            predictions: Output path for predictions_test.csv
            retune_request: Optional path to input retune_request.json
        """
        state = {
            "processed_data_path": os.path.abspath(processed_data),
            "classifier_code_path": os.path.abspath(classifier_code),
            "predictions_path": os.path.abspath(predictions),
        }
        if retune_request is not None:
            if not os.path.exists(retune_request):
                print(
                    f"[nadi] WARNING: retune_request file path was provided but the file does not exist: {retune_request}",
                    file=sys.stderr
                )
                state["retune_request"] = "no retune applied"
            else:
                try:
                    with open(retune_request, "r", encoding="utf-8") as f:
                        state["retune_request"] = json.load(f)
                except Exception as e:
                    print(
                        f"[nadi] ERROR: Failed to parse retune_request file {retune_request}: {e}",
                        file=sys.stderr
                    )
                    raise e
            
        return self._invoke(state)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Classifier Agent (Nadi).")
    parser.add_argument("--processed-data", default="mock_data/processed_data.csv", help="Input CSV path")
    parser.add_argument("--classifier-code", default="outputs/classifier.py", help="Output Python script path")
    parser.add_argument("--predictions", default="outputs/predictions_test.csv", help="Output predictions path")
    parser.add_argument("--retune-request", default=None, help="Input retune request JSON path")
    
    args = parser.parse_args()

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=args.processed_data,
        classifier_code=args.classifier_code,
        predictions=args.predictions,
        retune_request=args.retune_request
    )
    print("\nClassifier completed. Predictions at:", res["predictions_path"])
