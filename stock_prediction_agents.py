"""Public entrypoint for agent-based stock prediction from financial news text."""

from __future__ import annotations

import argparse

from helpers import (
    ClassifierAgent,
    EvaluatorAgent,
    ManagerAgent,
    NewsSample,
    ProcessingAgent,
    load_financial_news_csv,
    write_manual_eval_csv,
)


def _write_manual_eval_csv(rows, output_path: str) -> None:
    """Backward-compatible wrapper for manual-evaluation CSV writing."""
    write_manual_eval_csv(rows, output_path)


def main() -> None:
    """CLI entrypoint for iterative training and explainable prediction demo."""
    parser = argparse.ArgumentParser(description="Agent-based stock prediction from financial news")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--manual-eval-output",
        default="manual_explanation_eval.csv",
        help="Output CSV for manual explanation scoring",
    )
    parser.add_argument("--example-text", default="", help="Optional text for one prediction demo")
    args = parser.parse_args()

    samples = load_financial_news_csv(args.dataset)
    manager = ManagerAgent()
    metrics = manager.run_iterative_training(samples)
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    manual_rows = manager.generate_manual_evaluation_rows(samples, limit=30)
    _write_manual_eval_csv(manual_rows, args.manual_eval_output)
    print(f"Manual explanation file: {args.manual_eval_output} ({len(manual_rows)} rows)")
    if args.example_text:
        output = manager.predict_with_justification(args.example_text)
        print(f"Prediction: {output['prediction']}")
        print(f"Explanation: {output['explanation']}")


__all__ = [
    "ClassifierAgent",
    "EvaluatorAgent",
    "ManagerAgent",
    "NewsSample",
    "ProcessingAgent",
    "load_financial_news_csv",
]


if __name__ == "__main__":
    main()
