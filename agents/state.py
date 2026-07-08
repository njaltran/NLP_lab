"""
Shared pipeline state for the LangGraph graph.

Each field maps to a handoff between agents; agents/contracts.py holds the
column-level contracts.
All paths are absolute strings pointing to files the agents write to disk.
"""

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Runtime config ───────────────────────────────────────────────────────
    threshold: float        # price-change boundary (default 0.01 = ±1%)
    data_dir: str | None    # repo root; agents resolve file paths from here
    dataset_end: str | None  # optional YYYY-MM-DD; Processing drops later rows
                             # before the train/test split (e.g. "2019-12-31"
                             # keeps COVID out of the test window)
    loop_iteration: int     # retune loop counter, starts at 1

    # ── Handoff 1: Processing → Classifier ──────────────────────────────────
    # Handoff 1
    processed_data_path: str   # absolute path to processed_data.csv

    # ── Handoff 2: Classifier → Evaluator ───────────────────────────────────
    # Handoff 2
    # Prof note: Sabina gets code + results, not only predictions CSV
    model_dir: str | None       # optional fine-tuned weights folder; when set and it
                                # exists, the classifier loads it instead of pretrained
    llm_fn: object              # optional injected LLM callable used by the classifier's
                                # agentic code-gen (tests pass a fake; None = real Ollama)
    predictions_path: str       # absolute path to predictions_test.csv
    classifier_code_path: str   # absolute path to classifier.py (the generated script)
    classifier_history_path: str  # absolute path to this iteration's archived copy
                                   # (classifier_history/classifier_iterN.py) — classifier.py
                                   # itself gets overwritten every retune
    classifier_metadata: dict   # model_name, fine_tuning_params, confidence_distribution

    # ── Handoff 3: Evaluator → Manager ──────────────────────────────────────
    # Handoff 3
    evaluation_report: dict     # accuracy, below_threshold, class_accuracy, class_support,
                                # misclassified_count, misclassified_ids, proposal

    # ── Handoff 3b: Manager decision ────────────────────────────────────────
    # Handoff 3b
    decision: dict              # iteration, decision, final_action, based_on_proposal,
                                # overrides, notes  →  written to decision.json
    retune_request: dict        # written to retune_request.json only when final_action=retune
    final_action: str           # "retune" or "proceed"

    # ── Handoff 4: Manager → Explanation ────────────────────────────────────
    # Handoff 4
    sample_for_explanation_path: str   # absolute path to sample_for_explanation.csv

    # ── Handoff 5: Explanation → Manager ────────────────────────────────────
    # Handoff 5
    explanations_path: str      # absolute path to explanations.csv

    # ── Handoff 6: Final output ──────────────────────────────────────────────
    # Handoff 6
    final_results_path: str     # absolute path to final_results.csv
    final_report: dict          # final_accuracy, loop_iterations, class_accuracy, class_support,
                                # test_set_size, explanations_generated, manually_scored
