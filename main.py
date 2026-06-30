"""Pipeline orchestrator (Jack).

Drives the five agents end-to-end: Aurora builds the labelled dataset, then the
Nadi -> Sabina -> Manager retune loop runs until the Manager's accuracy gate
clears (or the iteration cap forces it), after which Freddi explains the sampled
rows and the Manager writes the final outputs.

Pure coordination: it owns no agent logic, only sequences each agent's `.run()`
and hands the contract files between them. Run with `uv run main.py`.
"""

import argparse
import os

from agents.aurora_processing import ProcessingAgent
from agents.nadi_classifier import ClassifierAgent
from agents.sabina_evaluator import EvaluatorAgent
from agents.freddi_explanation import ExplanationAgent
from agents.jack_manager import ManagerAgent, OUTPUT_DIR as OUT

# Contract files exchanged between agents. All live under the Manager's OUTPUT_DIR
# except processed_data.csv, whose path Aurora returns.
CODE = os.path.join(OUT, "classifier.py")
PREDS = os.path.join(OUT, "predictions_test.csv")
EVAL = os.path.join(OUT, "evaluation_report.json")
SAMPLE = os.path.join(OUT, "sample_for_explanation.csv")
RETUNE = os.path.join(OUT, "retune_request.json")
EXPL = os.path.join(OUT, "explanations.csv")


def classify_and_evaluate(nadi, sabina, processed, retune):
    """Run one Nadi -> Sabina pass: (re)generate and run the classifier, applying
    `retune` if given, then score the predictions. Writes CODE, PREDS and EVAL."""
    nadi.run(processed_data=processed, classifier_code=CODE,
             predictions=PREDS, retune_request=retune)
    sabina.run(predictions=PREDS, classifier_code=CODE)


def run_pipeline(threshold, target_accuracy, max_iterations, sample_size, use_ollama):
    """Drive the whole pipeline once and return the Manager's final state.

    Aurora builds processed_data, then the retune loop (Nadi -> Sabina -> Manager
    gate) repeats until the Manager proceeds; Freddi then explains the sampled
    rows and the Manager finalizes. The Manager instance is reused so its
    checkpointer carries the iteration counter across the loop, and it is given
    the real PREDS path so it never falls back to the mock default.
    """
    aurora = ProcessingAgent()
    nadi = ClassifierAgent()
    sabina = EvaluatorAgent(output_dir=OUT)
    freddi = ExplanationAgent(use_ollama=use_ollama, output_path=EXPL)
    manager = ManagerAgent(predictions_path=PREDS, target_accuracy=target_accuracy,
                           max_iterations=max_iterations, sample_size=sample_size)

    processed = aurora.run(threshold=threshold)["processed_data_path"]

    retune = None  # first pass: no request; later passes feed the Manager's RETUNE back
    for _ in range(max_iterations):
        classify_and_evaluate(nadi, sabina, processed, retune)
        if manager.run(evaluation_report=EVAL)["final_action"] == "proceed":
            break
        retune = RETUNE

    freddi.run(sample_for_explanation=SAMPLE, output=EXPL)
    return manager.run(evaluation_report=EVAL, explanations=EXPL)


def main():
    """Parse CLI flags, run the pipeline, and print where the outputs landed."""
    p = argparse.ArgumentParser(description="Run the full stock-move prediction pipeline.")
    p.add_argument("--threshold", type=float, default=0.01, help="Aurora label band (+/-, decimal)")
    p.add_argument("--target-accuracy", type=float, default=0.60, help="Manager accuracy gate")
    p.add_argument("--max-iterations", type=int, default=5, help="retune cap before forced proceed")
    p.add_argument("--sample-size", type=int, default=300, help="rows sampled for explanation")
    p.add_argument("--no-ollama", action="store_true", help="force Freddi's offline fallback")
    args = p.parse_args()

    final = run_pipeline(args.threshold, args.target_accuracy, args.max_iterations,
                         args.sample_size, use_ollama=not args.no_ollama)
    print(f"\n[orchestrator] done -- {final['final_action']} at iteration "
          f"{final['iteration']}. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
