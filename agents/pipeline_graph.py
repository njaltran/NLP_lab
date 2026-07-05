"""Unified pipeline graph (Increment 2) — the retune loop as a real LangGraph cycle.

`main.py` originally drove the five agents with a plain Python `for`-loop. This
module expresses the *same* flow as ONE compiled LangGraph whose retune loop is a
genuine graph **cycle** (`gate → classify → evaluate → gate`) rather than Python
control flow. That is the piece the earlier design was missing: state-carrying
cycles are LangGraph's headline feature.

## Design choice: compose, don't flatten

Each agent is already its own small graph with its own internal state keys (and
some of those keys collide across agents — e.g. Sabina and Freddi both use
`output_path`). Flattening all five into one node-set would force those keys to
share one namespace and would delete the per-agent test isolation the team built
on purpose. So instead each **node here invokes an agent** (its `.run()` / sub-graph)
and the unified graph only carries the loop's control state — file paths, the loop
counter, and the gate's verdict. The agents stay black boxes; nothing about them
changes. The retune loop becomes a real graph edge; that was the goal.

## Graph shape

    START → process → classify → evaluate → gate ─(retune)→ classify   [CYCLE]
                                              └(proceed)→ select_best → evaluate_test → explain → finalize → END

`evaluate` snapshots each new-best iteration's artifacts into `outputs/best/`;
`select_best` restores that snapshot (and redraws the explanation sample) when the
loop's LAST iteration wasn't its best — accuracy can regress across retunes, and
without this the pipeline would finalize on the regressed predictions.

`gate` is the Manager: calling it writes either `retune_request.json` (retune) or
`sample_for_explanation.csv` (proceed). Its own checkpointer carries the iteration
counter, accuracy history, and adaptive-param state across cycle passes (Increment
1), so convergence/adaptation work exactly as when driven by the Python loop.

## Testing

`build_pipeline(agents=...)` takes an optional `Agents` bundle so tests can inject
lightweight fakes for the network/GPU-bound agents (Aurora, Nadi) and exercise the
real cycle + Manager gate offline. `Agents.build()` constructs the real ones.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

from typing_extensions import TypedDict

try:
    from agents.aurora_processing import ProcessingAgent
    from agents.nadi_classifier import ClassifierAgent
    from agents.sabina_evaluator import EvaluatorAgent
    from agents.freddi_explanation import ExplanationAgent
    from agents.jack_manager import ManagerAgent, report_score, write_sample, OUTPUT_DIR as OUT
except ModuleNotFoundError:  # running as a bare script with agents/ on sys.path
    from aurora_processing import ProcessingAgent
    from nadi_classifier import ClassifierAgent
    from sabina_evaluator import EvaluatorAgent
    from freddi_explanation import ExplanationAgent
    from jack_manager import ManagerAgent, report_score, write_sample, OUTPUT_DIR as OUT

# Contract files exchanged between the nodes. All live under the Manager's
# OUTPUT_DIR except processed_data.csv, whose path Aurora returns at runtime.
CODE = os.path.join(OUT, "classifier.py")
PREDS = os.path.join(OUT, "predictions_test.csv")
EVAL = os.path.join(OUT, "evaluation_report.json")
SAMPLE = os.path.join(OUT, "sample_for_explanation.csv")
RETUNE = os.path.join(OUT, "retune_request.json")
EXPL = os.path.join(OUT, "explanations.csv")

# Snapshot dir for the best iteration's artifacts (internal to the loop, not a
# contract handoff). Holds copies of CODE/PREDS/EVAL from the highest-scoring pass.
BEST_DIR = os.path.join(OUT, "best")

# Nadi's internal per-iteration code snapshots (name mirrors nadi_classifier.py —
# keep the two in sync).
HISTORY_DIR = os.path.join(OUT, "classifier_history")

# A retune cycle is 3 nodes (classify → evaluate → gate); with the process/explain/
# finalize tail, ~5 iterations stays well under this. LangGraph aborts a runaway
# cycle at recursion_limit, so we set headroom rather than rely on the default 25.
RECURSION_LIMIT = 60


def clean_outputs() -> None:
    """Remove the previous run's loop artifacts from OUTPUT_DIR before a fresh
    run. Stale files actively mislead a new run: an old evaluation_report.json
    carries code_notes that can trigger Nadi's LLM rewrite on iteration 0, and a
    stale best/ snapshot could win select_best over the new run's iterations.
    Non-loop artifacts (fine-tuned weights, finetune_report.json) are kept."""
    loop_files = [CODE, PREDS, EVAL, SAMPLE, RETUNE, EXPL,
                  os.path.join(OUT, "decision.json"),
                  os.path.join(OUT, "final_results.csv"),
                  os.path.join(OUT, "final_report.json")]
    for path in loop_files:
        if os.path.exists(path):
            os.remove(path)
    for directory in (BEST_DIR, HISTORY_DIR):
        shutil.rmtree(directory, ignore_errors=True)


class PipelineState(TypedDict, total=False):
    """Control state carried around the unified graph. Deliberately small — the
    per-row data lives in the contract CSV/JSON files the agents read and write,
    not in here. Only what the *graph* needs to route and cycle lives in state."""

    processed_data_path: str        # set by `process`, read by `classify`
    retune_request_path: str | None  # None on the first pass; RETUNE on cycle passes
    final_action: str               # the gate's verdict: "retune" | "proceed"
    iteration: int                  # Manager's loop counter (for reporting)
    last_score: float               # collapse-penalized accuracy of the latest evaluate pass
    best_score: float               # best score seen across cycle passes (see report_score)


@dataclass
class Agents:
    """The five agents the graph drives. Bundled so tests can inject fakes for the
    network/GPU-bound ones while using the real, offline-capable others."""

    aurora: object
    nadi: object
    sabina: object
    manager: object
    freddi: object

    @classmethod
    def build(cls, *, target_accuracy=0.516, max_iterations=5, patience=2,
              min_delta=0.01, sample_size=300, use_ollama=True) -> "Agents":
        """Construct the real agents wired with the run's config. The Manager gets
        the real PREDS path so it never falls back to the mock default."""
        return cls(
            aurora=ProcessingAgent(),
            nadi=ClassifierAgent(),
            sabina=EvaluatorAgent(output_dir=OUT),
            manager=ManagerAgent(predictions_path=PREDS, target_accuracy=target_accuracy,
                                 max_iterations=max_iterations, patience=patience,
                                 min_delta=min_delta, sample_size=sample_size),
            freddi=ExplanationAgent(use_ollama=use_ollama, output_path=EXPL),
        )


def build_pipeline(agents: Agents, *, threshold=0.01, data_dir=None, model_dir=None,
                   dataset_end=None, sample_size=300, checkpointer=None):
    """Compile the unified pipeline graph. `agents` supplies the five agents (real
    or fake); `threshold` is Aurora's labelling band; `data_dir` overrides where
    Aurora reads fnspid_raw.csv (defaults to the repo `data/`); `model_dir`, when
    set, is handed to Nadi so the classifier loads fine-tuned weights instead of
    pretrained FinBERT (opt-in — omitted means today's behaviour); `dataset_end`
    (YYYY-MM-DD) drops later rows before Aurora's train/test split; `sample_size`
    is how many rows `select_best` redraws for the explanation sample. The node
    functions close over these, so no non-serialisable objects live in the graph
    state.
    """
    from langgraph.graph import StateGraph, START, END

    def process(state: PipelineState) -> dict:
        """Aurora: build the labelled dataset. Runs once, before the loop.
        Cleanup runs only after she succeeds (see docs/retune_loop.md)."""
        extra = {"data_dir": data_dir} if data_dir else {}
        if dataset_end:
            extra["dataset_end"] = dataset_end
        processed = agents.aurora.run(threshold=threshold, **extra)["processed_data_path"]
        clean_outputs()
        return {"processed_data_path": processed}

    def classify(state: PipelineState) -> dict:
        """Nadi: (re)generate and run the classifier. On cycle passes,
        `retune_request_path` points at the Manager's latest retune request so the
        params escalate; on the first pass it is None (fresh classifier)."""
        extra = {"model_dir": model_dir} if model_dir else {}
        agents.nadi.run(processed_data=state["processed_data_path"],
                        classifier_code=CODE, predictions=PREDS,
                        retune_request=state.get("retune_request_path"), **extra)
        return {}

    def evaluate(state: PipelineState) -> dict:
        """Sabina: score the predictions on the VAL split — retune decisions must
        never see the test rows, or the loop tunes against (overfits) the final
        measurement. Also snapshots this pass's artifacts into BEST_DIR whenever
        its score (accuracy, collapse-penalized by `report_score` — same rule as
        the gate's floor) sets a new best, so `select_best` can restore them if
        later retunes regress."""
        agents.sabina.run(predictions=PREDS, classifier_code=CODE, eval_split="val")
        with open(EVAL, encoding="utf-8") as f:
            score = report_score(json.load(f))
        out = {"last_score": score}
        if score > state.get("best_score", float("-inf")):
            os.makedirs(BEST_DIR, exist_ok=True)
            # PREDS + CODE only: the finals' report comes from `evaluate_test`
            # (test split), so snapshotting the val report would be dead weight.
            for path in (PREDS, CODE):
                shutil.copy2(path, os.path.join(BEST_DIR, os.path.basename(path)))
            out["best_score"] = score
        return out

    def gate(state: PipelineState) -> dict:
        """Manager gate. Invoking it applies the accuracy gate (with convergence +
        adaptive retune) and, as a side effect, writes EITHER retune_request.json
        (retune) OR sample_for_explanation.csv (proceed). We surface its verdict so
        the router can branch, and expose RETUNE as the next classify input when
        retuning."""
        st = agents.manager.run(evaluation_report=EVAL)
        out = {"final_action": st["final_action"], "iteration": st["iteration"]}
        if st["final_action"] == "retune":
            out["retune_request_path"] = RETUNE  # fed back into `classify` on the cycle
        return out

    def select_best(state: PipelineState) -> dict:
        """The gate proceeded with the LAST iteration's artifacts, which are not
        necessarily the best ones (accuracy can regress across retunes). If an
        earlier pass scored higher, restore its snapshot over the canonical paths
        and redraw the explanation sample from the restored predictions, so
        explain/finalize run on the best iteration. Gate semantics are untouched:
        the Manager already made its verdict from the last iteration's report."""
        best, last = state.get("best_score", float("-inf")), state.get("last_score", float("-inf"))
        if best <= last:
            return {}
        for path in (PREDS, CODE):
            shutil.copy2(os.path.join(BEST_DIR, os.path.basename(path)), path)
        write_sample(PREDS, sample_size)
        print(f"[pipeline] last iteration regressed (score {last:.2f}) — restored best "
              f"iteration's artifacts (score {best:.2f}) for explanation + finals")
        return {}

    def evaluate_test(state: PipelineState) -> dict:
        """Sabina one last time, now on the TEST split: the loop tuned and
        selected on val, so the test rows are scored exactly once, here — an
        honest held-out number. Overwrites EVAL, which finalize reads for
        final_report.json (the finalize entry point does not re-run the gate)."""
        agents.sabina.run(predictions=PREDS, classifier_code=CODE, eval_split="test")
        return {}

    def explain(state: PipelineState) -> dict:
        """Freddi: justify each sampled prediction into explanations.csv."""
        agents.freddi.run(sample_for_explanation=SAMPLE, output=EXPL)
        return {}

    def finalize(state: PipelineState) -> dict:
        """Manager again, now with explanations present → writes final_results.csv
        and final_report.json. Not a new iteration (the gate already proceeded)."""
        st = agents.manager.run(evaluation_report=EVAL, explanations=EXPL)
        return {"iteration": st["iteration"]}

    def route(state: PipelineState) -> str:
        """The cycle's branch point: loop back to `classify` on retune, else move
        on to the explanation stage."""
        return "retune" if state["final_action"] == "retune" else "proceed"

    b = StateGraph(PipelineState)
    for name, fn in [("process", process), ("classify", classify), ("evaluate", evaluate),
                     ("gate", gate), ("select_best", select_best),
                     ("evaluate_test", evaluate_test), ("explain", explain),
                     ("finalize", finalize)]:
        b.add_node(name, fn)
    b.add_edge(START, "process")
    b.add_edge("process", "classify")
    b.add_edge("classify", "evaluate")
    b.add_edge("evaluate", "gate")
    b.add_conditional_edges("gate", route, {"retune": "classify", "proceed": "select_best"})
    b.add_edge("select_best", "evaluate_test")
    b.add_edge("evaluate_test", "explain")
    b.add_edge("explain", "finalize")
    b.add_edge("finalize", END)
    return b.compile(checkpointer=checkpointer)


def run(*, threshold=0.01, target_accuracy=0.516, max_iterations=5, patience=2,
        min_delta=0.01, sample_size=300, use_ollama=True, data_dir=None,
        model_dir=None, dataset_end=None) -> dict:
    """Build the pipeline with real agents and run it once end to end, returning the
    final graph state. This is the entry point `main.py` calls."""
    from langgraph.checkpoint.memory import MemorySaver

    agents = Agents.build(target_accuracy=target_accuracy, max_iterations=max_iterations,
                          patience=patience, min_delta=min_delta, sample_size=sample_size,
                          use_ollama=use_ollama)
    graph = build_pipeline(agents, threshold=threshold, data_dir=data_dir,
                           model_dir=model_dir, dataset_end=dataset_end,
                           sample_size=sample_size, checkpointer=MemorySaver())
    return graph.invoke(
        {"retune_request_path": None},
        {"configurable": {"thread_id": "pipeline"}, "recursion_limit": RECURSION_LIMIT},
    )
