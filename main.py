"""Pipeline entry point (Jack).

Thin CLI wrapper around the unified pipeline graph in
`agents/pipeline_graph.py`, which drives all five agents end to end with the
retune loop expressed as a real LangGraph cycle. This file only loads local
secrets and parses flags — the orchestration lives in the graph. Run:

    uv run main.py
"""

import argparse

from agents.env import load_dotenv
from agents.pipeline_graph import run, OUT


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
    p.add_argument("--model-dir", default=None,
                   help="fine-tuned weights folder for Nadi (e.g. outputs/finbert_finetuned); "
                        "omitted = pretrained FinBERT")
    args = p.parse_args()

    load_dotenv()  # make .env secrets (e.g. HF_TOKEN) visible to the agents
    final = run(threshold=args.threshold, target_accuracy=args.target_accuracy,
                max_iterations=args.max_iterations, patience=args.patience,
                min_delta=args.min_delta, sample_size=args.sample_size,
                use_ollama=not args.no_ollama, data_dir=args.data_dir,
                model_dir=args.model_dir)
    print(f"\n[main] done -- {final['final_action']} at iteration "
          f"{final['iteration']}. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
