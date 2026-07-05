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

# Works both as a package import (`import agents.jack_manager` in tests) and as a
# direct script (`uv run agents/jack_manager.py`, where `agents/` is on sys.path).
try:
    from agents.base import Agent
    from agents.contracts import build_final_results, write_explanation_sample
except ModuleNotFoundError:
    from base import Agent
    from contracts import build_final_results, write_explanation_sample

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

    # --- convergence config (set once at start; see `decide`) ---
    patience: int                # how many recent iterations to watch for progress
    min_delta: float             # smallest accuracy gain that counts as "progress"
    min_class_accuracy: float    # per-class recall floor; below it = collapse, can't clear target

    # --- running history (reducer fields → APPENDED every iteration) ------------
    # A reducer field is merged with `operator.add` (list concatenation) instead of
    # being overwritten, so each node returns a *one-element list* and LangGraph
    # appends it. This is how the Manager remembers what happened in earlier
    # iterations — the raw material for the convergence and adaptation decisions.
    decision_log: Annotated[list, operator.add]      # human-readable note per step
    accuracy_history: Annotated[list, operator.add]  # one accuracy per iteration
    tried_params: Annotated[list, operator.add]      # retune params used per retune

    # --- I/O config (paths to contract files; optional, read via .get) ---
    predictions_path: str        # Nadi's predictions_test.csv (to sample / finalize)
    explanations_path: str       # Freddi's explanations.csv (present → finalize)
    sample_size: int             # rows for sample_for_explanation.csv (~300)


def _write_json(name: str, obj: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _test_rows(preds):
    """Keep only the TEST rows of a predictions frame. Predictions carry val +
    test rows; the val rows exist for the loop's own scoring, while Freddi's
    sample and the finals must be test-only. Loud on a contract violation —
    silently shipping mixed rows is the leak the val split exists to prevent."""
    if "split" not in preds.columns:
        raise ValueError("predictions file has no split column (Handoff 2)")
    test = preds[preds["split"] == "test"]
    if test.empty:
        raise ValueError("predictions file has no split=test rows")
    return test


def _write_decision(state: "ManagerState") -> None:
    """decision.json — Jack's record, written every iteration (Handoff 3b). The
    file is overwritten each iteration, so `accuracy_history` (cumulative through
    the current iteration) is the only place the per-iteration trend survives on
    disk once the loop moves on."""
    report = state["evaluation_report"]
    _write_json("decision.json", {
        "iteration": state["iteration"],
        "decision": state["decision"],
        "final_action": state["final_action"],
        "based_on_proposal": report.get("proposal", {}),
        "overrides": state.get("overrides", {}),
        "notes": state["notes"],
        "accuracy_history": state.get("accuracy_history", []),
    })


# Ordered escalation schedule for the retune loop. The first retune re-uses
# Sabina's proposed params as-is; every retune AFTER that walks down this list
# (skipping anything already tried) so each attempt is genuinely different rather
# than a repeat. The levers: lower `threshold` so fewer rows get forced to the
# `neutral` fallback class, and raise `boost_factor` so Nadi's focus-label boost
# (applied to whatever Sabina flagged as weakest) has more effect each retune.
# `max_length` was dropped as a knob: no headline in the dataset exceeds 128
# tokens (max 117, measured 2026-07-04), so raising it never changed anything.
# Tune these values here — nothing downstream is hardcoded to them.
_RETUNE_SCHEDULE = [
    {"threshold": 0.45, "boost_factor": 1.25},
    {"threshold": 0.40, "boost_factor": 1.35},
    {"threshold": 0.35, "boost_factor": 1.45},
    {"threshold": 0.30, "boost_factor": 1.55},
    {"threshold": 0.25, "boost_factor": 1.65},
    {"threshold": 0.20, "boost_factor": 1.75},
]

# Revert-and-perturb policy: when the last iteration's accuracy falls more than
# _REGRESSION_DELTA below the best iteration's, the escalation clearly overshot —
# go back to the params that produced the best iteration and move ONE knob one
# step, instead of marching further down the schedule (live case: 0.39 → 0.23
# after the last schedule entry boosted the wrong labels).
_REGRESSION_DELTA = 0.05
_THRESHOLD_STEP = 0.05
_BOOST_STEP = 0.10
_BOOST_MAX = 2.0   # threshold is clamped below; without a ceiling here, repeated
                   # perturbs would escalate the boost without bound


def _same(a: dict, b: dict) -> bool:
    """Param-set equality on shared keys only: Sabina's proposal omits params she
    doesn't set (e.g. `boost_factor`), and Nadi fills those from the same defaults
    the schedule starts at — so a candidate that matches a tried set on every
    shared key would regenerate an identical classifier."""
    shared = a.keys() & b.keys()
    return bool(shared) and all(a[k] == b[k] for k in shared)


def _next_params(tried: list, history: list = ()) -> dict:
    """Pick the next retune params.

    1. Revert-and-perturb: if the latest accuracy regressed more than
       _REGRESSION_DELTA below the best, perturb the best iteration's params.
    2. Otherwise: first schedule entry not already tried, or the last entry once
       the schedule is exhausted (the iteration cap still bounds the loop, so
       returning a repeat here is safe)."""
    if len(history) >= 2 and tried:
        best = max(range(len(history)), key=lambda i: history[i])
        if history[-1] < history[best] - _REGRESSION_DELTA:
            # Params per iteration: iteration 1 ran on Nadi's defaults, each
            # later one on the next `tried` entry.
            per_iteration = [{}] + list(tried)
            base = per_iteration[best] if best < len(per_iteration) else {}
            # Fill both knobs with Nadi's defaults so every candidate carries
            # full params: a partial dict (e.g. boost only) would make _same()
            # falsely match schedule entries that share just that one key.
            threshold = base.get("threshold", 0.5)
            boost = base.get("boost_factor", 1.25)
            base = {**base, "threshold": threshold, "boost_factor": boost}
            candidates = [
                {**base, "threshold": round(max(0.05, threshold - _THRESHOLD_STEP), 2)},
                {**base, "threshold": round(min(0.60, threshold + _THRESHOLD_STEP), 2)},
                {**base, "boost_factor": round(min(_BOOST_MAX, boost + _BOOST_STEP), 2)},
            ]
            for candidate in candidates:
                if not any(_same(candidate, t) for t in tried):
                    return candidate

    for params in _RETUNE_SCHEDULE:
        if not any(_same(params, t) for t in tried):
            return params
    return _RETUNE_SCHEDULE[-1]


def _is_collapsed(report: dict, floor: float) -> bool:
    """True when any class's recall sits below `floor` — the aggregate accuracy
    is then a degenerate win (e.g. everything predicted neutral).

    Note: Sabina reports 0.0 for a class absent from the test set, which reads
    as a collapse here — see the caveat in docs/retune_loop.md."""
    class_accuracy = report.get("class_accuracy", {})
    return bool(class_accuracy) and min(class_accuracy.values()) < floor


def report_score(report: dict, floor: float = 0.05) -> float:
    """Rank an iteration for best-snapshot purposes: plain accuracy, pushed below
    every healthy score when a class collapsed. One definition shared with the
    pipeline's best-iteration snapshot, mirroring the gate's collapse floor — so
    `select_best` can never restore a degenerate pass over a healthy one."""
    return report["accuracy"] - (1.0 if _is_collapsed(report, floor) else 0.0)


def _converged(history: list, patience: int, min_delta: float) -> bool:
    """True when retuning has stopped paying off, so we should proceed instead of
    burning the rest of the iteration budget.

    Rule: once we have more than `patience` accuracies, compare the best of the
    last `patience` iterations against the best of everything before them. If the
    recent window failed to beat the earlier best by at least `min_delta`, the
    loop has plateaued. `history` includes the current iteration's accuracy.
    """
    if len(history) <= patience:
        return False  # not enough history yet — let the loop keep exploring
    recent_best = max(history[-patience:])
    earlier_best = max(history[:-patience])
    return recent_best - earlier_best < min_delta


def decide(state: ManagerState) -> dict:
    """Threshold gate (deterministic). Decides retune vs. proceed, and when
    retuning, chooses the next hyperparameters from history. Returns only the
    keys it changed (LangGraph merges them into the running state).
    """
    report = state["evaluation_report"]
    accuracy = report["accuracy"]
    # Retune cycles only — the finalize pass routes straight to `finalize` and
    # never reaches this node (see `route_entry`).
    iteration = state.get("iteration", 0) + 1

    # Full accuracy trend INCLUDING this iteration — the input to convergence.
    history = state.get("accuracy_history", []) + [accuracy]
    patience = state.get("patience", 2)
    min_delta = state.get("min_delta", 0.01)

    # A collapsed class means the aggregate accuracy is a degenerate win — don't
    # let it clear the gate. Cap/convergence can still force proceed; select_best
    # then restores the best (collapse-penalized) earlier iteration anyway.
    collapsed = _is_collapsed(report, state.get("min_class_accuracy", 0.05))

    # Three independent reasons to stop retuning and move on.
    cleared = accuracy >= state["target_accuracy"] and not collapsed
    cap_hit = iteration >= state["max_iterations"]        # out of budget
    converged = _converged(history, patience, min_delta)  # progress has stalled
    final_action = "proceed" if (cleared or cap_hit or converged) else "retune"

    if cleared:
        why = "cleared target"
    elif converged:
        why = "converged (no improvement), proceeding"
    elif cap_hit:
        why = "cap hit, forcing proceed"
    elif collapsed and accuracy >= state["target_accuracy"]:
        why = "accuracy clears target but a class collapsed, retuning"
    else:
        why = "below target, retuning"
    note = (f"iteration {iteration}: accuracy {accuracy:.2f} vs target "
            f"{state['target_accuracy']:.2f} — {why}")

    out = {
        "iteration": iteration,          # saved → next invocation resumes from here
        "accuracy": accuracy,
        "final_action": final_action,
        "notes": note,                   # plain field → overwrites
        "decision_log": [note],          # reducer field → appended
        "accuracy_history": [accuracy],  # reducer field → appended
    }

    if final_action == "retune":
        # First retune trusts Sabina's proposal as-is (accept); every retune after
        # that adapts the params (override), because a repeat proposal has already
        # failed once. `tried_params` records what we actually used either way.
        # A proposal without suggested_params (Sabina recommended proceed but the
        # collapse floor forced a retune) has nothing to accept — take the
        # schedule instead of re-running Nadi's defaults.
        proposal = report.get("proposal", {})
        tried = state.get("tried_params", [])
        if not tried and proposal.get("suggested_params"):
            used = proposal["suggested_params"]
            out["decision"], out["overrides"] = "accept", {}
        else:
            used = _next_params(tried, history)
            out["decision"] = "override"
            out["overrides"] = {"suggested_params": used,
                                "focus_labels": proposal.get("focus_labels", [])}
        out["tried_params"] = [used]
    else:
        # Proceeding. Override only when the gate overrules Sabina's recommendation
        # (e.g. she says retune but the cap/convergence forces proceed); else accept.
        recommended = report.get("proposal", {}).get("recommended_action")
        if recommended is not None and recommended != final_action:
            out["decision"], out["overrides"] = "override", {"final_action": final_action}
        else:
            out["decision"], out["overrides"] = "accept", {}

    return out


def route_entry(state: ManagerState) -> str:
    """Entry router: explanations back from Freddi → finalize directly. The gate
    already proceeded on this report, so re-running decide would re-count the
    iteration, re-append its accuracy, and burn an LLM rationale on a decision
    that cannot change."""
    return "finalize" if state.get("explanations_path") else "decide"


def route_after_decide(state: ManagerState) -> str:
    """Router: retune, or sample for explanation on proceed. Reads the decision
    the gate already made."""
    return "retune" if state["final_action"] == "retune" else "sample"


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
        # Sabina's code observations, passed through unchanged — Nadi's optional
        # LLM code adaptation prompts with these (falls back to `reason` if empty).
        "code_notes": proposal.get("code_notes", ""),
    })
    return {"decision_log": [f"iteration {state['iteration']}: wrote retune_request.json"]}


def write_sample(predictions_path: str, sample_size: int = 300) -> int:
    """Write sample_for_explanation.csv (Handoff 4) from a predictions file.
    Shared by the proceed node and the pipeline's best-iteration restore, so the
    sample format has exactly one definition."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return write_explanation_sample(
        predictions_path,
        os.path.join(OUTPUT_DIR, "sample_for_explanation.csv"),
        sample_size,
    )


def proceed(state: ManagerState) -> dict:
    """sample_for_explanation.csv (Handoff 4) — drawn from predictions_test.csv.
    Terminal: hands off to Freddi and waits for explanations.csv."""
    _write_decision(state)
    n = write_sample(state.get("predictions_path", "mock_data/predictions_test.csv"),
                     state.get("sample_size", 300))
    return {"decision_log": [f"iteration {state['iteration']}: wrote sample_for_explanation.csv ({n} rows)"]}


def finalize(state: ManagerState) -> dict:
    """final_results.csv + final_report.json (Handoff 6) — once explanations.csv
    is back from Freddi. Joins predictions to explanations and writes the finals."""
    import pandas as pd

    preds = pd.read_csv(state.get("predictions_path", "mock_data/predictions_test.csv"))
    expl = pd.read_csv(state["explanations_path"])
    final = build_final_results(preds, expl)

    report = state["evaluation_report"]
    _write_decision(state)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final.to_csv(os.path.join(OUTPUT_DIR, "final_results.csv"), index=False)
    _write_json("final_report.json", {
        "final_accuracy": report.get("accuracy"),
        "loop_iterations": state["iteration"],
        "class_accuracy": report.get("class_accuracy", {}),
        "test_set_size": int(len(final)),
        "explanations_generated": int((final["explanation"] != "").sum()),
        "manually_scored": int(final["manual_score"].notna().sum()),
    })
    return {"decision_log": [f"iteration {state['iteration']}: wrote final_results.csv + final_report.json"]}


LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Engineered with the lecture's prompt-engineering patterns, mapped here so the
# notebook can walk them one by one (docs/jack_manager_guide.ipynb §12):
#   Persona                — line 1: role + tone.
#   Separator chain        — ### SECTION ### delimiters fence each instruction block
#                            off from the others and from the user's data.
#   CAPITALS               — the determinism boundary the model must never cross.
#   Chain of Thought       — ### REASONING ###: reason through fixed criteria BEFORE
#                            writing, kept SILENT so the audit log stays clean prose.
#   Few-shot               — ### EXAMPLES ###: three input→output pairs spanning the
#                            proceed / retune / forced-proceed cases.
#   Contrastive (good/bad) — ### GOOD vs BAD ###: one positive and one negative example
#                            pinning the two failure modes (disputing the gate; markdown).
#   Output spec            — plain prose, 2-3 sentences.
# Earlier, simpler drafts of this prompt (bare instruction → +persona → +one-shot) are
# preserved in the notebook (§12a) as the prompt-engineering evolution story.
RATIONALE_SYSTEM_PROMPT = """You are the MANAGER AGENT of an ML pipeline — a precise, \
factual orchestrator who explains decisions for a human audit log.

### CONTEXT ###
A retune-vs-proceed decision has ALREADY been made by a deterministic accuracy gate.
You did NOT make it and you CANNOT change it.

### REASONING — think silently, DO NOT print these steps ###
Before writing, reason through, in order:
1. Does the accuracy clear the target, or fall below it?
2. What did the evaluator recommend, and does it agree with the decision?
3. Did the iteration budget or a stalled trend force the outcome?
Let the rationale follow from that reasoning — but output only the rationale itself.

### YOUR TASK ###
Write a 2-3 sentence rationale explaining WHY the decision is reasonable, grounded in
the accuracy, the target, and the evaluator's proposal.

### CONSTRAINTS ###
- DO NOT dispute, second-guess, or suggest changing the decision.
- DO NOT print your reasoning steps — output ONLY the final rationale.
- Output PLAIN PROSE ONLY — no code, JSON, markdown, lists, or preamble.
- Be factual and concise. No marketing tone.

### EXAMPLES — few-shot, one per decision type ###
Input  — Decision: proceed at iteration 2. Accuracy 0.67 vs target 0.52. Proposal: proceed.
Output — Test accuracy of 0.67 clears the 0.52 target, so the gate proceeds to the \
explanation stage. The evaluator agreed; though the neutral class remains weakest, the \
iteration budget favours moving forward.

Input  — Decision: retune at iteration 1. Accuracy 0.44 vs target 0.52. Proposal: retune, \
focus down/neutral.
Output — At 0.44 the model sits below the 0.52 target, so the gate retunes. The \
evaluator flagged the down and neutral classes as weakest, and this first retune adopts \
its suggested parameters.

Input  — Decision: proceed at iteration 5. Accuracy 0.47 vs target 0.52. Proposal: retune.
Output — Accuracy of 0.47 still trails the 0.52 target, but iteration 5 is the budget \
ceiling, so the gate proceeds despite the evaluator's retune recommendation rather than \
spend a cycle unlikely to close the gap.

### GOOD vs BAD ###
GOOD — Accuracy 0.49 fell short of the 0.52 target and the trend had flattened, so the \
gate proceeded on convergence rather than burn another retune.
BAD  — The gate proceeded, but honestly it should have retuned once more to hit 0.52. \
**Decision:** proceed.   (WRONG: disputes the locked decision AND uses markdown.)
"""


def _llama_rationale(state: ManagerState) -> str:
    """Ask Llama for a short rationale. Falls back to the gate's deterministic
    note when HF_TOKEN is unset, so offline mock_data tests still run."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return state["notes"] + " (LLM skipped: no HF_TOKEN)"

    from huggingface_hub import InferenceClient

    proposal = state["evaluation_report"].get("proposal", {})
    # Any HF failure (rate limit, gated-model 403, timeout, malformed response)
    # must not abort the graph — the gate's decision still needs to be written.
    # Fall back to the deterministic note, mirroring the no-token branch above.
    try:
        client = InferenceClient(model=LLAMA_MODEL, token=token)
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": RATIONALE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Decision: {state['final_action']} at iteration {state['iteration']}. "
                    f"Test accuracy {state['accuracy']:.2f} vs target "
                    f"{state['target_accuracy']:.2f}. Evaluator proposal: {proposal}.")},
            ],
            max_tokens=160,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return state["notes"] + f" (LLM failed: {e})"


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
    b.add_conditional_edges(START, route_entry, {"decide": "decide", "finalize": "finalize"})
    b.add_edge("decide", "rationale")       # gate first, then explain
    b.add_conditional_edges(
        "rationale", route_after_decide,
        {"retune": "write_retune", "sample": "proceed"},
    )
    b.add_edge("write_retune", END)
    b.add_edge("proceed", END)
    b.add_edge("finalize", END)
    return b.compile(checkpointer=checkpointer)


class ManagerAgent(Agent):
    """Manager (Jack) behind the shared `.run()` interface. Construct once, then
    feed contract file paths in: pass `evaluation_report` to retune or sample,
    add `explanations` to finalize. Outputs land in `OUTPUT_DIR` per the contract.

        mgr = ManagerAgent()
        mgr.run(evaluation_report="outputs/evaluation_report.json")
        mgr.run(evaluation_report="outputs/evaluation_report.json",
                explanations="outputs/explanations.csv")
    """

    def __init__(self, *, target_accuracy=0.60, max_iterations=5,
                 patience=2, min_delta=0.01, min_class_accuracy=0.05,
                 predictions_path="mock_data/predictions_test.csv",
                 sample_size=300, checkpointer=None, thread_id="manager"):
        # Set once and merged into every run's state; the iteration counter and
        # the history reducers accumulate across runs via the checkpointer.
        # `patience`/`min_delta` tune early-stopping: proceed once accuracy hasn't
        # gained `min_delta` over the best of the last `patience` iterations.
        self._defaults = {
            "target_accuracy": target_accuracy,
            "max_iterations": max_iterations,
            "patience": patience,
            "min_delta": min_delta,
            "min_class_accuracy": min_class_accuracy,
            "predictions_path": predictions_path,
            "sample_size": sample_size,
        }
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, evaluation_report: str, explanations: str | None = None) -> dict:
        """`evaluation_report`: path to Sabina's report JSON. `explanations`:
        path to Freddi's explanations.csv — present means finalize, absent means
        retune-or-sample."""
        with open(evaluation_report, encoding="utf-8") as f:
            report = json.load(f)
        state = {**self._defaults, "evaluation_report": report}
        if explanations is not None:
            state["explanations_path"] = explanations
        return self._invoke(state)


if __name__ == "__main__":
    import tempfile

    # Drive the whole loop through the public ManagerAgent.run() file API. One
    # instance keeps iteration state across the three runs via its in-memory
    # checkpointer (no sqlite cleanup needed — fresh process, fresh state).
    mgr = ManagerAgent(thread_id="demo")

    proceed_path = "mock_data/evaluation_report.json"      # ships a proceed report
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
    # No retune report ships in mock_data/, so materialise one to a temp file —
    # the API takes a path, so the demo feeds it a path.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(retune_report, f)
        retune_path = f.name

    r1 = mgr.run(evaluation_report=retune_path)
    print(f"R1 → it={r1['iteration']} action={r1['final_action']:7s} → retune branch")
    r2 = mgr.run(evaluation_report=proceed_path)
    print(f"R2 → it={r2['iteration']} action={r2['final_action']:7s} → sample branch")
    r3 = mgr.run(evaluation_report=proceed_path, explanations="mock_data/explanations.csv")
    print(f"R3 → it={r3['iteration']} action={r3['final_action']:7s} → finalize branch")

    os.remove(retune_path)
    print("\noutputs/ now holds:", sorted(os.listdir(OUTPUT_DIR)))
