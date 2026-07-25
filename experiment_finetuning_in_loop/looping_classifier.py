"""Classifier Agent variant that fine-tunes on every retune.

The submitted Classifier Agent trains once, offline, and each retune only
adjusts inference settings (`threshold`, `boost_factor`) on top of fixed
weights. This variant adds one node in front of the agent's graph, so a retune
actually trains the model on the class the evaluator flagged:

    submitted:  START -> generate_code -> run_classifier -> END
    here:       START -> fine_tune -> generate_code -> run_classifier -> END

Nothing in `agents/` is modified. `LoopFineTuningClassifier` subclasses the real
`ClassifierAgent` and overrides `build_graph`, so the two versions can be
compared directly and the submitted pipeline is untouched.

Exports
-------
LoopFineTuningClassifier   drop-in replacement for ClassifierAgent
fine_tune                  the added LangGraph node

See experiment_finetuning_in_loop/README.md.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nadi_classifier import ClassifierAgent, generate_code, run_classifier
from agents.state import PipelineState

from train_head import train_head

# Checkpoints live under outputs/ like every other run artifact, but in their own
# folder so an experiment run can never overwrite the submitted pipeline's
# offline weights in outputs/finbert_finetuned/.
MODEL_DIR = os.path.join("outputs", "finbert_loop_finetuned")

# Every round's report, appended, so the experiment can be read without
# re-running it.
HISTORY_PATH = os.path.join("outputs", "loop_finetune_history.json")


def _append_history(entry: dict) -> None:
    """Append one round's report to HISTORY_PATH, creating it if needed."""
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def fine_tune(state: PipelineState) -> dict:
    """LangGraph node — train one round before the classifier runs.

    Skipped on the first pass, when there is no retune request yet: that pass
    classifies with pretrained FinBERT and gives the Manager a baseline to
    compare the trained rounds against. Every later pass continues from the
    previous round's checkpoint.

    The retune request is used only as the signal that a retune is happening —
    its contents do not change how the round trains. This experiment isolates
    one variable, when training happens, so the training objective is the same
    every round.
    """
    retune_request = state.get("retune_request")
    if not retune_request:
        print("[loop-finetune] first pass — pretrained FinBERT, no training yet")
        return {}

    # Continue from the previous round if one exists; the first trained round
    # starts from pretrained.
    parent = MODEL_DIR if os.path.isdir(MODEL_DIR) else None

    report = train_head(
        data_path=state["processed_data_path"],
        out_dir=MODEL_DIR,
        parent_model_dir=parent,
    )
    _append_history(report)

    # Hand the checkpoint to `generate_code`, which writes it into the generated
    # classifier.py as MODEL_DIR.
    return {"model_dir": os.path.abspath(MODEL_DIR)}


def build_graph(checkpointer):
    """The submitted classifier's graph with `fine_tune` in front."""
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


class LoopFineTuningClassifier(ClassifierAgent):
    """Classifier Agent that trains on every retune.

    Inherits `run()` from the real agent, so the pipeline calls it exactly the
    same way — only the graph differs.
    """

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)
