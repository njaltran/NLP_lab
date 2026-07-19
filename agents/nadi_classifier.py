"""Classifier Agent (Nadi) — generates the classifier code and executes it.

Reads processed_data.csv and optionally retune_request.json, generates the
classifier script classifier.py (ProsusAI/finbert), runs it to generate
predictions_test.csv, and outputs both.

See docs/data_contracts.md (Handoff 2).
"""

import inspect
import json
import os
import subprocess
import sys
from typing import TypedDict

# Allow running as a package or direct script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from agents.base import Agent
    from agents.contracts import combine_binary_probs
    from agents.finbert_finetuner import HEADS, train_finbert
    from agents.state import PipelineState
except ModuleNotFoundError:
    from base import Agent
    from contracts import combine_binary_probs
    from finbert_finetuner import HEADS, train_finbert
    from state import PipelineState

# The generated classifier.py runs as a standalone subprocess with only its own
# directory on sys.path — it cannot import agents.contracts. Its source is
# injected into the template verbatim instead, so the trainer's validation math
# (agents/finbert_finetuner.py) and the classifier's inference math are
# provably the same function, not two hand-kept-in-sync copies.
_COMBINE_BINARY_PROBS_SOURCE = inspect.getsource(combine_binary_probs)

OUTPUT_DIR = "outputs"

# Shared between training and inference: fine_tune() trains each head at this
# length and CLASSIFIER_TEMPLATE tokenizes at the same length at inference.
# Pinned, not retune-tunable (ADR 0003) -- a mismatch here would train one
# truncation and infer at another.
MAX_LENGTH = 128

# Epochs per fine-tune call, pinned like MAX_LENGTH (ADR 0001 Q11) -- not a
# retune-tunable knob. _train_epochs() (finbert_finetuner.py) already keeps
# only the best-val-accuracy checkpoint within a call, so more epochs per
# round costs time, not quality; 3 matches the manual CLI's existing default
# (agents/finetune_finbert.py --epochs 3) rather than the trainer's own
# single-epoch default, which was too small a per-round increment -- it risked
# the convergence check mistaking under-training for a plateau.
EPOCHS_PER_ROUND = 3

# --- Per-head fine-tuning hyperparameter schedule (ADR 0001) -------------------
# Nadi, not Manager, picks these now: given one directional head's own prior
# retune attempts, choose the next learning rate and focus weight. Values
# mirror the schedule this pipeline already validated for a single 3-class
# head — halving on regression, stepping the focus weight up otherwise.
TRAINING_LR_START = 5e-6
TRAINING_LR_FLOOR = 2e-6
TRAINING_LR_BACKOFF = 0.5
TRAINING_MULTIPLIER_STEP = 0.25
TRAINING_MULTIPLIER_MIN = 1.0
TRAINING_MULTIPLIER_MAX = 2.0


def next_head_training_params(history: list[dict], *, collapsed: bool) -> dict:
    """Pick the next learning_rate/focus_weight_multiplier for ONE directional
    head, from that head's own prior attempts only — the other head's history
    never enters this, since each head is tuned independently (ADR 0002).

    First attempt: start at TRAINING_LR_START; the focus weight starts higher
    when this retune is fixing a collapse, since the head needs a stronger
    push toward its own class right away.

    Later attempts: if the last attempt for this head regressed, back the
    focus weight off one step and halve the learning rate (floored);
    otherwise hold the learning rate and push the focus weight one step
    further. Only the most recent attempt matters — the schedule reacts to
    what just happened, not the full history.
    """
    if not history:
        return {
            "learning_rate": TRAINING_LR_START,
            "focus_weight_multiplier": 1.5 if collapsed else 1.25,
        }
    last = history[-1]
    if last["regressed"]:
        return {
            "learning_rate": max(TRAINING_LR_FLOOR, last["learning_rate"] * TRAINING_LR_BACKOFF),
            "focus_weight_multiplier": max(
                TRAINING_MULTIPLIER_MIN,
                last["focus_weight_multiplier"] - TRAINING_MULTIPLIER_STEP,
            ),
        }
    return {
        "learning_rate": last["learning_rate"],
        "focus_weight_multiplier": min(
            TRAINING_MULTIPLIER_MAX,
            last["focus_weight_multiplier"] + TRAINING_MULTIPLIER_STEP,
        ),
    }


def _heads_needing_training(requested: list[str], already_trained: set[str]) -> set[str]:
    """Which head(s) train this pass. Before either head has a checkpoint, the
    baseline pass trains both regardless of what was requested (Nadi cannot
    classify with only one head published). Once both exist, only the head(s)
    Manager actually flagged retrain."""
    if not already_trained:
        return set(HEADS)
    return set(requested)


def fine_tune(state: PipelineState) -> dict:
    """LangGraph node: train whichever directional head(s) need it this pass
    (ADR 0001). A no-op until the first retune — cold start (iteration 0, no
    retune_request at all) stays on pretrained FinBERT, never fine-tuning
    just because no checkpoint exists yet. `heads_to_retrain` is only ever
    set by `ClassifierAgent.run()` when a retune_request was parsed, so its
    absence from state is exactly the cold-start signal.

    Heads not in the returned dict are left untouched — LangGraph merges only
    the keys a node returns, so an unflagged head's checkpoint and history
    survive unchanged."""
    if "heads_to_retrain" not in state:
        return {}
    already_trained = {head for head in HEADS if state.get(f"{head}_model_dir")}
    heads = _heads_needing_training(state.get("heads_to_retrain") or [], already_trained)
    if not heads:
        return {}

    base_dir = state.get("finetuned_base_dir") or os.path.join(OUTPUT_DIR, "finbert_finetuned")
    collapsed = set(state.get("collapsed_heads") or [])
    out: dict = {}
    for head in sorted(heads):
        history = state.get(f"{head}_training_history", [])
        params = next_head_training_params(history, collapsed=head in collapsed)
        result = train_finbert(
            data_path=state["processed_data_path"],
            out_dir=os.path.join(base_dir, head),
            head=head,
            learning_rate=params["learning_rate"],
            focus_weight_multiplier=params["focus_weight_multiplier"],
            parent_model_dir=state.get(f"{head}_model_dir"),
            max_length=MAX_LENGTH,
            epochs=state.get("epochs", EPOCHS_PER_ROUND),
        )
        own_accuracy = result["report"]["val_class_accuracy"].get(head, 0.0)
        previous_accuracy = history[-1].get("val_class_accuracy") if history else None
        regressed = previous_accuracy is not None and own_accuracy < previous_accuracy
        out[f"{head}_model_dir"] = result["model_dir"]
        out[f"{head}_training_history"] = [
            {**params, "val_class_accuracy": own_accuracy, "regressed": regressed}
        ]
    return out


CLASSIFIER_TEMPLATE = """
import csv
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "ProsusAI/finbert"
UP_MODEL_DIR = {up_model_dir}     # up-head checkpoint, or None to use pretrained FinBERT
DOWN_MODEL_DIR = {down_model_dir} # down-head checkpoint, or None to use pretrained FinBERT
MAX_LENGTH = {max_length}
THRESHOLD = {threshold}
SENTIMENT_TO_LABEL = {{"positive": "up", "negative": "down", "neutral": "neutral"}}

# Authenticate to the HF Hub when a token is in the env (higher rate limits,
# faster downloads); fall back to unauthenticated access when it is absent.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

{combine_binary_probs_source}

if UP_MODEL_DIR and DOWN_MODEL_DIR and os.path.isdir(UP_MODEL_DIR) and os.path.isdir(DOWN_MODEL_DIR):
    # Both directional heads are fine-tuned: combine their positive-class
    # probabilities into a three-way distribution (ADR 0002).
    tokenizer = AutoTokenizer.from_pretrained(UP_MODEL_DIR)
    up_model = AutoModelForSequenceClassification.from_pretrained(UP_MODEL_DIR)
    down_model = AutoModelForSequenceClassification.from_pretrained(DOWN_MODEL_DIR)
    up_model.eval()
    down_model.eval()
    print("[classifier] using fine-tuned heads:", UP_MODEL_DIR, DOWN_MODEL_DIR)

    def classify(title: str) -> dict:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            p_up = torch.softmax(up_model(**inputs).logits, dim=1)[0][1].item()
            p_down = torch.softmax(down_model(**inputs).logits, dim=1)[0][1].item()
        by_label = combine_binary_probs(p_up, p_down)

        top_label = max(by_label, key=by_label.get)
        top_prob = by_label[top_label]
        predicted = top_label if top_prob >= THRESHOLD else "neutral"

        return {{
            "predicted_label": predicted,
            "confidence": round(top_prob, 4),
            "prob_up": round(by_label["up"], 4),
            "prob_down": round(by_label["down"], 4),
            "prob_neutral": round(by_label["neutral"], 4),
        }}
else:
    # No fine-tuned heads yet: use pretrained FinBERT and translate its
    # sentiment (positive/negative/neutral) into a move direction.
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, token=HF_TOKEN)
    ID2OURS = {{i: SENTIMENT_TO_LABEL[model.config.id2label[i].lower()] for i in model.config.id2label}}
    model.eval()
    print("[classifier] using pretrained FinBERT (no fine-tuned heads found)")

    def classify(title: str) -> dict:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        by_label = {{ID2OURS[i]: float(p) for i, p in enumerate(probs)}}

        top_label = max(by_label, key=by_label.get)
        top_prob = by_label[top_label]
        predicted = top_label if top_prob >= THRESHOLD else "neutral"

        return {{
            "predicted_label": predicted,
            "confidence": round(top_prob, 4),
            "prob_up": round(by_label["up"], 4),
            "prob_down": round(by_label["down"], 4),
            "prob_neutral": round(by_label["neutral"], 4),
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
    if "split" not in rows[0]:
        raise ValueError("Input has no split column (required by Handoff 1)")
    rows = [r for r in rows if r["split"] != "train"]
    if not rows:
        raise ValueError("No val/test rows in input")

    out_cols = [c for c in rows[0].keys() if c != "split"] + [
        "predicted_label", "confidence", "prob_up", "prob_down", "prob_neutral", "split"
    ]

    # Predict each held-out row; its split value passes through untouched.
    for row in rows:
        pred_data = classify(row["article_title"])
        row.update(pred_data)

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
    """LangGraph node: write classifier.py. Two-head fine-tuned template when
    both directional heads have published checkpoints (fine_tune ran, ADR
    0001); pretrained-sentiment fallback otherwise (cold start, ADR 0001 Q12).

    THRESHOLD/MAX_LENGTH are fixed constants, not retune-tunable — every
    retune is a fine-tune pass now (ADR 0003), so there is no more inference-
    time knob to nudge; the weights are the only thing retuning changes.
    """
    threshold = 0.5

    code_path = state.get("classifier_code_path") or os.path.join(OUTPUT_DIR, "classifier.py")
    os.makedirs(os.path.dirname(code_path) or ".", exist_ok=True)

    # Both heads required, or neither is used — a lone head (mid-bootstrap,
    # or a stale directory) would silently combine a real signal with zero,
    # which is worse than the plain pretrained fallback.
    up_model_dir = state.get("up_model_dir")
    down_model_dir = state.get("down_model_dir")
    if not (up_model_dir and down_model_dir
            and os.path.isdir(up_model_dir) and os.path.isdir(down_model_dir)):
        up_model_dir = down_model_dir = None

    formatted_code = CLASSIFIER_TEMPLATE.format(
        up_model_dir=repr(up_model_dir),
        down_model_dir=repr(down_model_dir),
        threshold=threshold,
        max_length=MAX_LENGTH,
        combine_binary_probs_source=_COMBINE_BINARY_PROBS_SOURCE,
    )

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    print(f"[nadi] Generated classifier code at: {code_path}")

    # Keep a per-iteration copy so past retune attempts aren't lost when
    # classifier.py (the single contract file Sabina reads) gets overwritten.
    # iteration 0 = first pass before any retune; N = the Nth retune's code.
    retune_req = state.get("retune_request")
    iteration = retune_req.get("iteration", 0) if retune_req else 0
    history_dir = os.path.join(os.path.dirname(code_path) or ".", "classifier_history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"classifier_iter{iteration}.py")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    metadata = {
        "model_name": (f"{up_model_dir}, {down_model_dir}"
                       if up_model_dir and down_model_dir else "ProsusAI/finbert"),
        "fine_tuning_params": {"threshold": threshold, "max_length": MAX_LENGTH},
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
    builder.add_node("fine_tune", fine_tune)
    builder.add_node("generate_code", generate_code)
    builder.add_node("run_classifier", run_classifier)
    builder.add_edge(START, "fine_tune")
    builder.add_edge("fine_tune", "generate_code")
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
            retune_request: str | None = None,
            finetuned_base_dir: str = os.path.join(OUTPUT_DIR, "finbert_finetuned"),
            epochs: int = EPOCHS_PER_ROUND) -> dict:
        """Runs the fine-tuning (ADR 0001), classifier generation, and
        prediction steps.

        Args:
            processed_data: Path to input processed_data.csv
            classifier_code: Output path for classifier.py
            predictions: Output path for predictions_test.csv
            retune_request: Optional path to input retune_request.json. Its
                `heads_to_retrain` / `collapsed_heads` fields (Manager's
                signal, Task 8) drive which directional head(s) fine_tune()
                retrains this pass.
            finetuned_base_dir: Base directory the two directional heads publish
                       their checkpoints under (<base>/up, <base>/down). Nadi
                       reads its own prior checkpoints from here across retune
                       iterations — there is no longer a caller-supplied
                       single `model_dir` opt-in.
            epochs: Epochs per fine-tune round for every head trained this
                       pass. Still a fixed value applied uniformly (ADR 0001
                       Q11) — not something Manager or a retune schedule picks
                       per iteration, just an override point for manual runs.
        """
        state = {
            "processed_data_path": os.path.abspath(processed_data),
            "classifier_code_path": os.path.abspath(classifier_code),
            "predictions_path": os.path.abspath(predictions),
            "finetuned_base_dir": os.path.abspath(finetuned_base_dir),
            "epochs": epochs,
        }
        if retune_request is not None and os.path.exists(retune_request):
            try:
                with open(retune_request, "r", encoding="utf-8") as f:
                    retune_req = json.load(f)
                state["retune_request"] = retune_req
                state["heads_to_retrain"] = retune_req.get("heads_to_retrain", [])
                state["collapsed_heads"] = retune_req.get("collapsed_heads", [])
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
    parser.add_argument("--finetuned-base-dir", default=os.path.join(OUTPUT_DIR, "finbert_finetuned"),
                        help="Base dir the up/down heads publish checkpoints under")
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_ROUND,
                        help="Epochs per fine-tune round for every head trained this pass")

    args = parser.parse_args()

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=args.processed_data,
        classifier_code=args.classifier_code,
        predictions=args.predictions,
        retune_request=args.retune_request,
        finetuned_base_dir=args.finetuned_base_dir,
        epochs=args.epochs,
    )
    print("\nClassifier completed. Predictions at:", res["predictions_path"])
