"""Manager Agent (Jack) — orchestration loop + threshold gate + final output.

Reads Sabina's evaluation_report.json, decides retune-vs-proceed via a
deterministic accuracy gate, and (later steps) writes the retune request,
explanation sample, and final outputs. LLM (Llama via HF) only writes the
human-readable rationale — the gate itself is pure rules.

See docs/data_contracts.md (Handoffs 3, 3b, 4, 6).
"""

import json
import operator
import os
from typing import Annotated, TypedDict

OUTPUT_DIR = "outputs"


class ManagerState(TypedDict):
    """Shared state threaded through the LangGraph. Nodes return partial
    updates to these keys; LangGraph merges them in (latest-wins), except
    `decision_log`, which appends via its reducer."""

    # --- inputs / config (set once at start) ---
    evaluation_report: dict      # parsed Sabina report for this iteration
    target_accuracy: float       # gate threshold, 0.60 per contract
    max_iterations: int          # iteration cap before forced proceed

    # --- loop progress (plain fields → merged, latest wins) ---
    iteration: int               # loop counter, starts at 1
    accuracy: float              # accuracy from the current report
    final_action: str            # "retune" | "proceed"
    decision: str                # "accept" | "override"
    overrides: dict              # fields Jack changed; {} if accept
    notes: str                   # rationale (filled by the LLM node, step 5)

    # --- running history (reducer field → appended every iteration) ---
    decision_log: Annotated[list, operator.add]

    # --- I/O config (paths to contract files; optional, read via .get) ---
    predictions_path: str        # Nadi's predictions_test.csv (to sample / finalize)
    explanations_path: str       # Freddi's explanations.csv (present → finalize)
    sample_size: int             # rows for sample_for_explanation.csv (~300)


def _write_json(name: str, obj: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_decision(state: "ManagerState") -> None:
    """decision.json — Jack's record, written every iteration (Handoff 3b)."""
    report = state["evaluation_report"]
    _write_json("decision.json", {
        "iteration": state["iteration"],
        "decision": state["decision"],
        "final_action": state["final_action"],
        "based_on_proposal": report.get("proposal", {}),
        "overrides": state.get("overrides", {}),
        "notes": state["notes"],
    })


def decide(state: ManagerState) -> dict:
    """Threshold gate (deterministic). Returns only the keys it changed."""
    report = state["evaluation_report"]
    accuracy = report["accuracy"]
    # count retune cycles only: once we've proceeded, the finalize pass is not
    # a new iteration, so don't bump the counter past convergence.
    iteration = state.get("iteration", 0)
    if state.get("final_action") != "proceed":
        iteration += 1

    cleared = accuracy >= state["target_accuracy"]
    cap_hit = iteration >= state["max_iterations"]
    final_action = "proceed" if (cleared or cap_hit) else "retune"

    if cleared:
        why = "cleared target"
    elif cap_hit:
        why = "cap hit, forcing proceed"
    else:
        why = "below target, retuning"
    note = (f"iteration {iteration}: accuracy {accuracy:.2f} vs target "
            f"{state['target_accuracy']:.2f} — {why}")

    # Override = the gate's action disagrees with Sabina's recommendation
    # (e.g. she says retune but the cap forces proceed). Else accept.
    recommended = report.get("proposal", {}).get("recommended_action")
    if recommended is not None and recommended != final_action:
        decision, overrides = "override", {"final_action": final_action}
    else:
        decision, overrides = "accept", {}

    return {
        "iteration": iteration,     # saved → next invocation resumes from here
        "accuracy": accuracy,
        "final_action": final_action,
        "decision": decision,
        "overrides": overrides,
        "notes": note,              # plain field → overwrites
        "decision_log": [note],     # reducer field → appended
    }


def route_after_decide(state: ManagerState) -> str:
    """Router: retune, or — on proceed — finalize if Freddi's explanations are
    back, else sample and wait. Reads decisions the nodes already made."""
    if state["final_action"] == "retune":
        return "retune"
    return "finalize" if state.get("explanations_path") else "sample"


def write_retune(state: ManagerState) -> dict:
    """retune_request.json (Handoff 3b) — approved proposal for Nadi. Terminal:
    Manager hands off and waits for Nadi+Sabina to produce the next report."""
    report = state["evaluation_report"]
    proposal = report.get("proposal", {})
    overrides = state.get("overrides", {})
    _write_decision(state)
    _write_json("retune_request.json", {
        "iteration": state["iteration"],
        "reason": proposal.get("reason", state["notes"]),
        "current_accuracy": state["accuracy"],
        "target_accuracy": state["target_accuracy"],
        "focus_labels": overrides.get("focus_labels", proposal.get("focus_labels", [])),
        "misclassified_ids": report.get("misclassified_ids", []),
        "suggested_params": {**proposal.get("suggested_params", {}),
                             **overrides.get("suggested_params", {})},
    })
    return {"decision_log": [f"iteration {state['iteration']}: wrote retune_request.json"]}


def proceed(state: ManagerState) -> dict:
    """sample_for_explanation.csv (Handoff 4) — drawn from predictions_test.csv.
    Terminal: hands off to Freddi and waits for explanations.csv."""
    import pandas as pd

    preds = pd.read_csv(state.get("predictions_path", "mock_data/predictions_test.csv"))
    sample = (preds[["article_id", "article_title", "predicted_label", "label",
                     "confidence", "prob_up", "prob_down", "prob_neutral"]]
              .rename(columns={"label": "actual_label"}))
    n = min(len(sample), state.get("sample_size", 300))
    _write_decision(state)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample.head(n).to_csv(os.path.join(OUTPUT_DIR, "sample_for_explanation.csv"), index=False)
    return {"decision_log": [f"iteration {state['iteration']}: wrote sample_for_explanation.csv ({n} rows)"]}


def finalize(state: ManagerState) -> dict:
    """final_results.csv + final_report.json (Handoff 6) — once explanations.csv
    is back from Freddi. Joins predictions to explanations and writes the finals."""
    import pandas as pd

    preds = pd.read_csv(state.get("predictions_path", "mock_data/predictions_test.csv"))
    expl = pd.read_csv(state["explanations_path"])[["article_id", "explanation", "manual_score"]]
    final = preds.merge(expl, on="article_id", how="left")[[
        "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
        "pct_change", "label", "predicted_label", "confidence", "explanation", "manual_score"]]
    final["explanation"] = final["explanation"].fillna("")          # null → empty string
    final["manual_score"] = final["manual_score"].astype("Int64")   # int, NA → empty

    report = state["evaluation_report"]
    _write_decision(state)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final.to_csv(os.path.join(OUTPUT_DIR, "final_results.csv"), index=False)
    _write_json("final_report.json", {
        "final_accuracy": report.get("accuracy"),
        "loop_iterations": state["iteration"],
        "class_accuracy": report.get("class_accuracy", {}),
        "test_set_size": int(len(preds)),
        "explanations_generated": int((final["explanation"] != "").sum()),
        "manually_scored": int(final["manual_score"].notna().sum()),
    })
    return {"decision_log": [f"iteration {state['iteration']}: wrote final_results.csv + final_report.json"]}


LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _llama_rationale(state: ManagerState) -> str:
    """Ask Llama for a short rationale. Falls back to the gate's deterministic
    note when HF_TOKEN is unset, so offline mock_data tests still run."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return state["notes"] + " (LLM skipped: no HF_TOKEN)"

    from huggingface_hub import InferenceClient

    proposal = state["evaluation_report"].get("proposal", {})
    client = InferenceClient(model=LLAMA_MODEL, token=token)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": (
                "You are the Manager of an ML pipeline. The retune/proceed decision "
                "is ALREADY made by a deterministic gate — do NOT change or dispute it. "
                "Write 2-3 factual sentences explaining it for a human log.")},
            {"role": "user", "content": (
                f"Decision: {state['final_action']} at iteration {state['iteration']}. "
                f"Test accuracy {state['accuracy']:.2f} vs target "
                f"{state['target_accuracy']:.2f}. Evaluator proposal: {proposal}.")},
        ],
        max_tokens=160,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def rationale(state: ManagerState) -> dict:
    """LLM node: writes `notes` ONLY. Never touches final_action (control flow)."""
    return {"notes": _llama_rationale(state)}


def build_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END

    b = StateGraph(ManagerState)
    b.add_node("decide", decide)
    b.add_node("rationale", rationale)
    b.add_node("write_retune", write_retune)
    b.add_node("proceed", proceed)
    b.add_node("finalize", finalize)
    b.add_edge(START, "decide")
    b.add_edge("decide", "rationale")       # gate first, then explain
    b.add_conditional_edges(
        "rationale", route_after_decide,
        {"retune": "write_retune", "sample": "proceed", "finalize": "finalize"},
    )
    b.add_edge("write_retune", END)
    b.add_edge("proceed", END)
    b.add_edge("finalize", END)
    return b.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langgraph.checkpoint.sqlite import SqliteSaver

    cfg = {"configurable": {"thread_id": "demo"}}
    base = {"target_accuracy": 0.60, "max_iterations": 5,
            "predictions_path": "mock_data/predictions_test.csv"}

    retune_report = {
        "accuracy": 0.54, "below_threshold": True,
        "class_accuracy": {"up": 0.60, "down": 0.40, "neutral": 0.30},
        "misclassified_ids": ["FNSPID_00006", "FNSPID_00010"],
        "proposal": {
            "recommended_action": "retune",
            "reason": "accuracy 0.54 below target 0.60; down/neutral weakest",
            "focus_labels": ["down", "neutral"],
            "suggested_params": {"threshold": 0.5, "max_length": 128},
            "code_notes": "threshold hardcoded at 0.5 in classifier.py"},
    }
    with open("mock_data/evaluation_report.json") as f:
        proceed_report = json.load(f)

    with SqliteSaver.from_conn_string("manager_state.sqlite") as cp:
        graph = build_graph(cp)
        r1 = graph.invoke({**base, "evaluation_report": retune_report}, cfg)
        print(f"R1 → it={r1['iteration']} action={r1['final_action']:7s} → retune branch")
        r2 = graph.invoke({**base, "evaluation_report": proceed_report}, cfg)
        print(f"R2 → it={r2['iteration']} action={r2['final_action']:7s} → sample branch")
        r3 = graph.invoke({**base, "evaluation_report": proceed_report,
                           "explanations_path": "mock_data/explanations.csv"}, cfg)
        print(f"R3 → it={r3['iteration']} action={r3['final_action']:7s} → finalize branch")
    print("\noutputs/ now holds:", sorted(os.listdir(OUTPUT_DIR)))
