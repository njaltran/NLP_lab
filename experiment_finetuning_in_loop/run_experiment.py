"""Run the pipeline with fine-tuning inside the retune loop.

Same five agents, same graph, same contract files as `main.py` — the only
difference is that the Classifier Agent is `LoopFineTuningClassifier`, which
trains before it classifies. `agents/pipeline_graph.py` already accepts an
`Agents` bundle, so swapping one agent needs no changes to the submitted code.

Run (from the repo root):

    uv run experiment_finetuning_in_loop/run_experiment.py --dataset-end 2019-12-31

This is slow: every retune is a training round, not a fast inference pass.
Expect roughly 5-15 minutes per iteration on a laptop GPU.

See experiment_finetuning_in_loop/README.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.aurora_processing import ProcessingAgent
from agents.env import load_dotenv
from agents.freddi_explanation import ExplanationAgent
from agents.jack_manager import ManagerAgent
from agents.pipeline_graph import (
    EXPL,
    OUT,
    PREDS,
    RECURSION_LIMIT,
    Agents,
    build_pipeline,
)
from agents.sabina_evaluator import EvaluatorAgent

from looping_classifier import MODEL_DIR, LoopFineTuningClassifier


def main():
    """Parse flags, build the pipeline with the looping classifier, run it once."""
    p = argparse.ArgumentParser(
        description="Run the pipeline with fine-tuning inside the retune loop.")
    p.add_argument("--threshold", type=float, default=0.01, help="Aurora label band (+/-, decimal)")
    p.add_argument("--target-accuracy", type=float, default=0.60, help="Manager accuracy gate")
    p.add_argument("--max-iterations", type=int, default=3,
                   help="retune cap. Lower than main.py's 5: every iteration trains")
    p.add_argument("--patience", type=int, default=2, help="iterations watched for convergence")
    p.add_argument("--min-delta", type=float, default=0.01, help="accuracy gain that counts as progress")
    p.add_argument("--sample-size", type=int, default=300, help="rows sampled for explanation")
    p.add_argument("--no-ollama", action="store_true", help="force Freddi's offline fallback")
    p.add_argument("--dataset-end", default=None, metavar="YYYY-MM-DD",
                   help="drop rows after this date before splitting (use 2019-12-31 "
                        "to match the submitted run)")
    args = p.parse_args()

    load_dotenv()  # make .env secrets (e.g. HF_TOKEN) visible to the agents

    # The submitted bundle, with one agent replaced.
    agents = Agents(
        aurora=ProcessingAgent(),
        nadi=LoopFineTuningClassifier(),
        sabina=EvaluatorAgent(output_dir=OUT),
        manager=ManagerAgent(predictions_path=PREDS,
                             target_accuracy=args.target_accuracy,
                             max_iterations=args.max_iterations,
                             patience=args.patience,
                             min_delta=args.min_delta,
                             sample_size=args.sample_size),
        freddi=ExplanationAgent(use_ollama=not args.no_ollama, output_path=EXPL),
    )

    from langgraph.checkpoint.memory import MemorySaver

    graph = build_pipeline(agents, threshold=args.threshold,
                           dataset_end=args.dataset_end,
                           checkpointer=MemorySaver())
    final = graph.invoke(
        {"retune_request_path": None},
        {"configurable": {"thread_id": "experiment"}, "recursion_limit": RECURSION_LIMIT},
    )

    print(f"\n[experiment] done -- {final['final_action']} at iteration "
          f"{final['iteration']}. Outputs in {OUT}/, checkpoints in {MODEL_DIR}/")


if __name__ == "__main__":
    main()
