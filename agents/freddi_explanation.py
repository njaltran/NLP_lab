"""Explanation Agent (Freddi) — Handoff 5, as a LangGraph agent.

Reads `sample_for_explanation.csv` (from Jack, the Manager) and, for each row,
generates a plain-text justification of the model's prediction. The LLM call is a
LangChain chain (`prompt | ChatOllama | StrOutputParser`) — the pattern from the
RAG exercise (notebook 7). The whole thing is wrapped in the shared `Agent`
interface (`agents/base.py`) so the Manager can trigger it with `.run()`.

Contract: docs/data_contracts.md, Handoffs 4 (input) and 5 (output).

Input columns  : article_id, article_title, predicted_label, actual_label,
                 confidence, prob_up, prob_down, prob_neutral
Output columns : article_id, article_title, predicted_label, actual_label,
                 confidence, explanation, manual_score

Design notes:
  - **Option A** — the model explains the prediction from the HEADLINE only; it is
    never shown the actual next-day outcome (mirrors real prediction time and keeps
    this to the "explain your reasoning" task). `actual_label` is still passed
    through to the output for the human graders, but never reaches the model.
  - **Graceful fallback** — if Ollama is unreachable, a deterministic placeholder
    sentence is produced instead, so the graph never crashes and a valid
    `explanations.csv` is ALWAYS handed back to Jack (team rule: proceed on issue).
  - First five columns are passed through byte-for-byte; `prob_*` feed the prompt
    but are not written out; `manual_score` is left blank (scored by hand later).

Run `uv run agents/freddi_explanation.py --help` for CLI options.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from typing_extensions import TypedDict

# Works both as a package import (`import agents.freddi_explanation` in tests) and
# as a direct script (`uv run agents/freddi_explanation.py`, where `agents/` is on
# sys.path) — same dual-import trick the other agents use.
try:
    from agents.base import Agent
except ModuleNotFoundError:
    from base import Agent

# Columns read in (Handoff 4) and the exact columns written out (Handoff 5).
INPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "prob_up", "prob_down", "prob_neutral",
]
OUTPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "explanation", "manual_score",
]
# Passed through from the input untouched (Golden rule 1).
PASSTHROUGH_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label", "confidence",
]

DEFAULT_INPUT = "mock_data/sample_for_explanation.csv"
DEFAULT_OUTPUT = "explanations.csv"
DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 180.0       # generous so the first call survives a cold model load
DEFAULT_KEEP_ALIVE = "15m"    # keep the model resident between rows


class ExplanationState(TypedDict, total=False):
    """State threaded through the LangGraph: load_sample → explain → write_output."""

    # --- inputs / config ---
    sample_path: str       # path to sample_for_explanation.csv (Handoff 4)
    output_path: str       # where to write explanations.csv (Handoff 5)
    model: str             # Ollama model name
    base_url: str          # Ollama base URL
    timeout: float         # per-request timeout (cold-load tolerance)
    use_ollama: bool       # False → always use the offline fallback
    limit: int | None      # process only the first N rows (testing)

    # --- working data ---
    rows: list             # input rows (list[dict]) read from the sample
    out_rows: list         # output rows shaped for OUTPUT_COLUMNS
    ollama_used: bool      # True if the real LLM produced at least one explanation


# Structured prompt, engineered with the lecture's components (as in Jack's agent):
# Persona (line 1), ### SECTIONS ###, CAPITALS for hard constraints, plain-prose
# Output rule, and one EXAMPLE (one-shot). Option A is enforced here: the model is
# never told the actual outcome.
EXPLANATION_SYSTEM_PROMPT = """You are a FINANCIAL ANALYST who explains a stock-move \
model's prediction in one plain sentence for a human reviewer.

### CONTEXT ###
A classifier predicted a stock's next-day move (up / down / neutral) from a single news
headline. You are given the headline and that prediction. You do NOT know the actual
next-day outcome and you must NOT guess it.

### YOUR TASK ###
Write ONE clear sentence (max ~25 words) explaining why the headline could justify the
predicted move.

### CONSTRAINTS ###
- Use ONLY what the headline states. DO NOT invent facts, numbers, or events.
- DO NOT mention whether the prediction was right or wrong, or refer to the real outcome.
- Output PLAIN PROSE ONLY — no preamble, no markdown, no lists, no restating the task.

### EXAMPLE ###
Input  — Headline: "Retailer cuts profit outlook on weak demand". Predicted move: a downward next-day move.
Output — The lowered profit outlook points to weaker earnings, which could push the stock down the next day.
"""


def _user_message(row: dict) -> str:
    """The per-row user turn. Carries the headline, the predicted move (in words),
    and the probabilities (so the model can sense how close the call was) — but
    NOT the actual outcome."""
    move_phrase = {
        "up": "an upward next-day move",
        "down": "a downward next-day move",
        "neutral": "little or no next-day move",
    }.get((row.get("predicted_label") or "").strip(), "the predicted next-day move")
    return (
        f'Headline: "{row.get("article_title", "")}". '
        f"Predicted move: {move_phrase}. "
        f"Model confidence: {row.get('confidence', '')} "
        f"(probabilities — up: {row.get('prob_up', '')}, "
        f"down: {row.get('prob_down', '')}, neutral: {row.get('prob_neutral', '')})."
    )


def _build_chain(model: str, base_url: str, timeout: float):
    """Build the LangChain chain `prompt | ChatOllama | StrOutputParser` — the
    exercise-7 pattern. Imported lazily so the offline fallback path needs no
    LangChain install, and so a bad import surfaces as a clean fallback, not a crash.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages(
        [("system", EXPLANATION_SYSTEM_PROMPT), ("user", "{user_input}")]
    )
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.3,            # low: grounded, low-variance explanations
        num_predict=80,             # one sentence is plenty
        keep_alive=DEFAULT_KEEP_ALIVE,
        client_kwargs={"timeout": timeout},
    )
    return prompt | llm | StrOutputParser()


def fallback_explanation(row: dict) -> str:
    """Deterministic placeholder when Ollama is unavailable. Like the real prompt
    (Option A) it explains the prediction from the headline only and never references
    the actual outcome. Replaced automatically by real output when Ollama is reachable.
    """
    title = (row.get("article_title") or "the headline").strip()
    pred = (row.get("predicted_label") or "").strip()
    direction = {
        "up": "an upward next-day move",
        "down": "a downward next-day move",
        "neutral": "little or no next-day move",
    }.get(pred, "the predicted next-day move")
    return _clean(
        f'The headline "{title}" reads as a {pred or "neutral"} signal for the stock, '
        f"which is consistent with {direction}."
    )


def _clean(text: str) -> str:
    """Collapse whitespace/newlines so each explanation is a single CSV-safe line."""
    return " ".join(text.split()).strip()


# --- LangGraph nodes ------------------------------------------------------------

def load_sample(state: ExplanationState) -> dict:
    """Read sample_for_explanation.csv into the state; warn on missing columns."""
    path = state["sample_path"]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"[freddi] WARNING: {path} missing expected columns: {missing}",
                  file=sys.stderr)
        rows = list(reader)
    limit = state.get("limit")
    if limit is not None:
        rows = rows[:limit]
    print(f"[freddi] read {len(rows)} rows from {path}")
    return {"rows": rows}


def explain(state: ExplanationState) -> dict:
    """Generate an explanation per row via the LangChain chain, falling back to the
    deterministic placeholder on any failure so a valid output is always produced."""
    rows = state.get("rows", [])
    chain = None
    if state.get("use_ollama", True):
        try:
            chain = _build_chain(state["model"], state["base_url"], state["timeout"])
        except Exception as exc:  # bad import / config → offline fallback
            print(f"[freddi] LangChain/Ollama unavailable ({exc}); using offline "
                  "fallback for all rows.", file=sys.stderr)
            chain = None

    out_rows, ollama_used = [], False
    for i, row in enumerate(rows, start=1):
        explanation = ""
        if chain is not None:
            try:
                explanation = _clean(chain.invoke({"user_input": _user_message(row)}))
                ollama_used = True
            except Exception as exc:  # call failed → stop hammering, fall back
                print(f"[freddi] Ollama call failed ({exc}); switching to offline "
                      "fallback for remaining rows.", file=sys.stderr)
                chain = None
        if not explanation:
            explanation = fallback_explanation(row)

        out_row = {col: (row.get(col, "") or "") for col in PASSTHROUGH_COLUMNS}
        out_row["explanation"] = explanation
        out_row["manual_score"] = ""  # blank by contract; scored by hand later
        out_rows.append(out_row)
        print(f"[freddi] {i}/{len(rows)} {out_row['article_id']} -> done")
    return {"out_rows": out_rows, "ollama_used": ollama_used}


def write_output(state: ExplanationState) -> dict:
    """Write explanations.csv with exactly the contract columns (UTF-8, comma)."""
    path = state["output_path"]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    rows = state.get("out_rows", [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[freddi] wrote {len(rows)} rows to {path}")
    return {}


def build_graph(checkpointer):
    """Compile the agent's LangGraph: load_sample → explain → write_output."""
    from langgraph.graph import END, START, StateGraph

    b = StateGraph(ExplanationState)
    b.add_node("load_sample", load_sample)
    b.add_node("explain", explain)
    b.add_node("write_output", write_output)
    b.add_edge(START, "load_sample")
    b.add_edge("load_sample", "explain")
    b.add_edge("explain", "write_output")
    b.add_edge("write_output", END)
    return b.compile(checkpointer=checkpointer)


class ExplanationAgent(Agent):
    """Explanation agent (Freddi) behind the shared `.run()` interface. Construct
    once, then hand it the sample path:

        agent = ExplanationAgent()
        agent.run(sample_for_explanation="outputs/sample_for_explanation.csv")

    Set `use_ollama=False` to force the offline fallback (used by the tests).
    """

    def __init__(self, *, model=DEFAULT_MODEL, base_url=DEFAULT_OLLAMA_URL,
                 timeout=DEFAULT_TIMEOUT, use_ollama=True, output_path=DEFAULT_OUTPUT,
                 limit=None, checkpointer=None, thread_id="explanation"):
        self._defaults = {
            "model": model,
            "base_url": base_url,
            "timeout": timeout,
            "use_ollama": use_ollama,
            "output_path": output_path,
            "limit": limit,
        }
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, sample_for_explanation: str, output: str | None = None) -> dict:
        """`sample_for_explanation`: path to Jack's sample CSV. Writes
        explanations.csv and returns the final state dict."""
        state = {**self._defaults, "sample_path": sample_for_explanation}
        if output is not None:
            state["output_path"] = output
        return self._invoke(state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explanation Agent (Freddi) — Handoff 5")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="sample_for_explanation.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="explanations.csv output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds to wait per Ollama request (cold-load tolerance)")
    parser.add_argument("--no-ollama", action="store_true",
                        help="skip Ollama and always use the offline fallback")
    parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
    args = parser.parse_args(argv)

    agent = ExplanationAgent(
        model=args.model, base_url=args.ollama_url, timeout=args.timeout,
        use_ollama=not args.no_ollama, output_path=args.output, limit=args.limit,
    )
    state = agent.run(sample_for_explanation=args.input, output=args.output)
    if not state.get("ollama_used", False) and not args.no_ollama:
        print("[freddi] NOTE: all explanations came from the offline fallback "
              "(Ollama was not reachable).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
