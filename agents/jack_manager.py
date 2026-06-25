"""Manager Agent (Jack) — orchestration loop + threshold gate + final output.

Reads Sabina's evaluation_report.json, decides retune-vs-proceed via a
deterministic accuracy gate, and (later steps) writes the retune request,
explanation sample, and final outputs. LLM (Llama via HF) only writes the
human-readable rationale — the gate itself is pure rules.

See docs/data_contracts.md (Handoffs 3, 3b, 4, 6).
"""

import operator
from typing import Annotated, TypedDict


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


def decide(state: ManagerState) -> dict:
    """Threshold gate (deterministic). Returns only the keys it changed."""
    report = state["evaluation_report"]
    accuracy = report["accuracy"]
    iteration = state.get("iteration", 0) + 1   # resumes from restored state

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

    return {
        "iteration": iteration,     # saved → next invocation resumes from here
        "accuracy": accuracy,
        "final_action": final_action,
        "decision": "accept",       # LLM may flip to "override" in step 5
        "notes": note,              # plain field → overwrites
        "decision_log": [note],     # reducer field → appended
    }


def route_after_decide(state: ManagerState) -> str:
    """Router: dumb — just reports the action the gate already chose."""
    return state["final_action"]            # "retune" | "proceed"


def write_retune(state: ManagerState) -> dict:
    """STUB (real file-writing in Step 6). Terminal for this invocation."""
    return {"decision_log": [f"iteration {state['iteration']}: wrote retune_request.json (stub)"]}


def proceed(state: ManagerState) -> dict:
    """STUB (sample + final output in Steps 6-7)."""
    return {"decision_log": [f"iteration {state['iteration']}: proceeding to explanation (stub)"]}


def build_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END

    b = StateGraph(ManagerState)
    b.add_node("decide", decide)
    b.add_node("write_retune", write_retune)
    b.add_node("proceed", proceed)
    b.add_edge(START, "decide")
    b.add_conditional_edges(
        "decide", route_after_decide,
        {"retune": "write_retune", "proceed": "proceed"},
    )
    b.add_edge("write_retune", END)
    b.add_edge("proceed", END)
    return b.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langgraph.checkpoint.sqlite import SqliteSaver

    cfg = {"configurable": {"thread_id": "demo"}}
    base = {"target_accuracy": 0.60, "max_iterations": 5}
    with SqliteSaver.from_conn_string("manager_state.sqlite") as cp:
        graph = build_graph(cp)
        r1 = graph.invoke({**base, "evaluation_report": {"accuracy": 0.54}}, cfg)
        print(f"Round 1 → iteration={r1['iteration']} action={r1['final_action']}")
        r2 = graph.invoke({**base, "evaluation_report": {"accuracy": 0.67}}, cfg)
        print(f"Round 2 → iteration={r2['iteration']} action={r2['final_action']}")
        print("decision_log:")
        for line in r2["decision_log"]:
            print("  -", line)
