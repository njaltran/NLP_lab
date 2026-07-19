"""Pipeline entry point (Jack).

Thin CLI wrapper around the unified pipeline graph in
`agents/pipeline_graph.py`, which drives all five agents end to end with the
retune loop expressed as a real LangGraph cycle. This file only loads local
secrets and parses flags — the orchestration lives in the graph. Run:

    uv run main.py
"""

import argparse
import json
import os

from agents.console import panel
from agents.env import load_dotenv
from agents.pipeline_graph import run, OUT


def _print_header(args) -> None:
    """What this run is about to do, printed before the first slow step (training)."""
    rows = [
        ("gate", f"accuracy ≥ {args.target_accuracy:.2f} · max {args.max_iterations}"
                 f" iterations · patience {args.patience}"),
        ("label band", f"±{args.threshold:.1%} next-day close → up / down / neutral"),
        ("fine-tuning", f"{args.epochs} epoch{'s' if args.epochs != 1 else ''} per round"
                        if args.epochs is not None else "pinned default epochs per round"),
    ]
    if args.dataset_end:
        rows.append(("dataset end", f"{args.dataset_end} (later rows dropped)"))
    rows.append(("explanations", "offline fallback" if args.no_ollama else "Ollama"))
    print()
    panel("stock-move prediction · FinBERT + feedback retune loop", rows)
    print()


def _load(name: str) -> dict:
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_summary(final: dict, target: float) -> None:
    """The numbers someone actually wants after a run: what the model scored on
    rows it never saw, and how the loop got there. Read back from the contract
    files rather than from the returned state, so what is printed is what was
    actually written."""
    report, decision = _load("final_report.json"), _load("decision.json")
    rows = []

    accuracy = report.get("final_accuracy")
    if accuracy is not None:
        verdict = (f"[green]target {target:.2f} met[/]" if accuracy >= target
                   else f"[red]below target {target:.2f}[/]")
        rows.append(("test accuracy", f"[bold]{accuracy:.3f}[/]  {verdict}"))

    per_class = report.get("class_accuracy", {})
    support = report.get("class_support", {})
    if per_class:
        rows.append(("recall", "  ".join(
            f"{label} {score:.2f}" + (f"[dim] (n={support[label]})[/]" if label in support else "")
            for label, score in per_class.items())))

    trend = decision.get("accuracy_history", [])
    if trend:
        marks = " → ".join(f"{score:.2f}" for score in trend)
        rows.append(("accuracy trend", f"{marks}  [dim](validation)[/]"))

    rows.append(("explanations", f"{report.get('explanations_generated', 0)}"
                                 f"/{report.get('test_set_size', 0)} test rows"))
    rows.append(("outputs", f"{OUT}/final_results.csv · final_report.json"))

    print()
    panel(f"complete · {final['final_action']} at iteration {final['iteration']}", rows)
    print()


def main():
    """Parse CLI flags, run the pipeline graph, and print where the outputs landed."""
    p = argparse.ArgumentParser(description="Run the full stock-move prediction pipeline.")
    p.add_argument("--threshold", type=float, default=0.01, help="Aurora label band (+/-, decimal)")
    p.add_argument("--target-accuracy", type=float, default=0.60, help="Manager accuracy gate")
    p.add_argument("--max-iterations", type=int, default=5, help="retune cap before forced proceed")
    p.add_argument("--patience", type=int, default=2, help="iterations watched for convergence")
    p.add_argument("--min-delta", type=float, default=0.01, help="accuracy gain that counts as progress")
    p.add_argument("--sample-size", type=int, default=300, help="rows sampled for explanation")
    p.add_argument("--no-ollama", action="store_true", help="force Freddi's offline fallback")
    p.add_argument("--data-dir", default=None, help="override dir holding fnspid_raw.csv")
    p.add_argument("--dataset-end", default=None, metavar="YYYY-MM-DD",
                   help="drop rows after this date before splitting (e.g. 2019-12-31 "
                        "keeps the COVID regime out of the test window, matching the "
                        "fine-tuned model's training data)")
    p.add_argument("--epochs", type=int, default=None,
                   help="epochs per fine-tune round for every head trained that pass "
                        "(default: Nadi's pinned EPOCHS_PER_ROUND)")
    args = p.parse_args()

    load_dotenv()  # make .env secrets (e.g. HF_TOKEN) visible to the agents
    _print_header(args)
    final = run(threshold=args.threshold, target_accuracy=args.target_accuracy,
                max_iterations=args.max_iterations, patience=args.patience,
                min_delta=args.min_delta, sample_size=args.sample_size,
                use_ollama=not args.no_ollama, data_dir=args.data_dir,
                dataset_end=args.dataset_end, epochs=args.epochs)
    _print_summary(final, args.target_accuracy)


if __name__ == "__main__":
    main()
