"""Regression tests for the Explanation agent (Freddi).

Freezes the development checks as `assert` statements (team meeting 2026-06-28):
the agent reads Jack's sample, writes a contract-faithful `explanations.csv`,
passes the upstream columns through untouched, keeps `manual_score` blank, and —
crucially — STILL produces valid output when the LLM is unavailable, so the
pipeline ALWAYS hands a file back to Jack ("even if issue, proceed to Jack").

Everything runs OFFLINE against `mock_data/` with `use_ollama=False`, so no Ollama
server is needed. Tests that build the LangGraph need `langgraph` installed and are
skipped automatically where it is absent; the node-level tests run anywhere.
"""

import csv
import importlib.util

import pytest

import agents.freddi_explanation as fe

SAMPLE = "mock_data/sample_for_explanation.csv"

# The full-graph tests construct ExplanationAgent, which imports langgraph. Skip
# them cleanly where langgraph isn't installed (the node-level tests still run).
_HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
needs_langgraph = pytest.mark.skipif(not _HAS_LANGGRAPH, reason="langgraph not installed")


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_nodes(out_path, *, use_ollama=False, limit=None):
    """Drive the three nodes directly (no LangGraph) — lets the core behaviour be
    tested even where langgraph isn't installed."""
    state = {
        "sample_path": SAMPLE, "output_path": str(out_path), "use_ollama": use_ollama,
        "model": fe.DEFAULT_MODEL, "base_url": fe.DEFAULT_OLLAMA_URL,
        "timeout": 5, "limit": limit,
    }
    state.update(fe.load_sample(state))
    state.update(fe.explain(state))
    fe.write_output(state)
    return state


# --- "even if issue, proceed to Jack": offline still yields valid output ----

def test_offline_still_writes_contract_valid_output(tmp_path):
    out = tmp_path / "explanations.csv"
    state = _run_nodes(out, use_ollama=False)        # Ollama deliberately disabled

    assert out.exists(), "an explanations.csv must exist even with no LLM"
    rows = _read(out)
    assert len(rows) == len(state["rows"]) > 0
    assert [*rows[0].keys()] == fe.OUTPUT_COLUMNS     # exact contract columns + order
    assert all(r["explanation"].strip() for r in rows)   # every row explained
    assert all(r["manual_score"] == "" for r in rows)    # blank by contract
    assert {r["predicted_label"] for r in rows} <= {"up", "down", "neutral"}


def test_passthrough_columns_unmodified(tmp_path):
    out = tmp_path / "explanations.csv"
    _run_nodes(out, use_ollama=False)
    src = {r["article_id"]: r for r in _read(SAMPLE)}
    for r in _read(out):
        original = src[r["article_id"]]
        for col in fe.PASSTHROUGH_COLUMNS:
            assert r[col] == original[col], f"{col} must pass through byte-for-byte"


# --- Option A: the model never sees the actual next-day outcome --------------

def test_prompt_never_includes_actual_outcome():
    # A sentinel actual_label that would never otherwise appear in the prompt.
    row = {
        "article_title": "Acme delays product launch", "predicted_label": "neutral",
        "actual_label": "SENTINEL_OUTCOME", "confidence": "0.5",
        "prob_up": "0.2", "prob_down": "0.5", "prob_neutral": "0.3",
    }
    msg = fe._user_message(row)
    assert "SENTINEL_OUTCOME" not in msg          # actual outcome never reaches the model
    assert "neutral" in msg.lower()               # the prediction IS shown


def test_fallback_is_single_clean_option_a_sentence():
    # assert-style "known input -> known shape" check (the brief's calculator example):
    row = {"article_title": "Acme beats earnings", "predicted_label": "up",
           "actual_label": "SENTINEL_OUTCOME"}
    text = fe.fallback_explanation(row)
    assert text and "\n" not in text              # exactly one CSV-safe line
    assert "up" in text.lower()                   # reflects the prediction
    assert "SENTINEL_OUTCOME" not in text         # never references the outcome


# --- full LangGraph path through the public ExplanationAgent.run() API -------

@needs_langgraph
def test_agent_run_offline_lifecycle(tmp_path):
    """The public .run() wrapper (Jack's trigger point) drives the same graph and
    writes a contract-valid file, offline."""
    out = tmp_path / "explanations.csv"
    agent = fe.ExplanationAgent(use_ollama=False, output_path=str(out))
    state = agent.run(sample_for_explanation=SAMPLE)

    assert out.exists()
    rows = _read(out)
    assert [*rows[0].keys()] == fe.OUTPUT_COLUMNS
    assert len(rows) == len(state["rows"]) > 0
    assert all(r["explanation"].strip() for r in rows)
