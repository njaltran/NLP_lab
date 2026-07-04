"""Classifier Agent (Nadi) — generates and runs classifier.py.

Reads processed_data.csv and optionally retune_request.json, stamps the
tunable hyperparameters into classifier_template.py, writes the result to
classifier.py, and runs it to produce predictions_test.csv.

On retune cycles where Sabina's evaluation_report.json has non-empty code_notes,
the agent asks a local Ollama LLM to rewrite the classifier from her feedback
instead of only swapping the four constants — the "agentic" requirement.

See docs/data_contracts.md (Handoff 2).
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from agents.base import Agent
    from agents.state import PipelineState
except ModuleNotFoundError:
    from base import Agent
    from state import PipelineState

OUTPUT_DIR = "outputs"

# Path to the readable template file that classifier.py is generated from.
_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_template.py")

# Default values — must match the constants in classifier_template.py.
_DEFAULTS = {"threshold": 0.5, "max_length": 128, "focus_labels": [], "boost_factor": 1.25}

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 180.0

_CODE_GEN_SYSTEM_PROMPT = """\
You are an expert Python developer improving a FinBERT stock-move classifier.

### HARD CONSTRAINTS — preserve these exactly ###
- Function signatures: classify(title: str) -> dict  and  main(src: str, dst: str) -> None
- classify() return keys: predicted_label, confidence, prob_up, prob_down, prob_neutral
- main() filters rows where split == "test" (falls back to all rows when absent) and
  carries the split column to the last output column
- FINETUNED_DIR = "outputs/finbert_finetuned" must be checked; prefer it over "ProsusAI/finbert"
- Required imports: csv, os, sys, torch, AutoTokenizer, AutoModelForSequenceClassification

### OUTPUT ###
Output ONLY the complete Python file — no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def _build_code(threshold, max_length, focus_labels, boost_factor) -> str:
    """Read classifier_template.py and substitute the four tunable constants."""
    with open(_TEMPLATE_PATH, encoding="utf-8") as f:
        code = f.read()
    d = _DEFAULTS
    return (
        code
        .replace(f"MAX_LENGTH = {d['max_length']}", f"MAX_LENGTH = {max_length}")
        .replace(f"THRESHOLD = {d['threshold']}", f"THRESHOLD = {threshold}")
        .replace(f"FOCUS_LABELS = {repr(d['focus_labels'])}", f"FOCUS_LABELS = {repr(focus_labels)}")
        .replace(f"BOOST_FACTOR = {d['boost_factor']}", f"BOOST_FACTOR = {boost_factor}")
    )


def _read_code_notes() -> str:
    """Return Sabina's code_notes from evaluation_report.json, or '' if absent."""
    path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("proposal", {}).get("code_notes", "")
    except Exception:
        return ""


def _rewrite_with_ollama(base_code, code_notes, retune_params) -> "str | None":
    """Ask a local Ollama LLM to rewrite classifier.py from Sabina's feedback.

    Returns the new code as a string, or None when Ollama is unavailable.
    Uses the same langchain_ollama stack as Freddi's explanation agent.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
    except ImportError:
        return None

    suggested = retune_params.get("suggested_params", {})
    user_msg = (
        f"Current classifier.py:\n\n{base_code}\n\n"
        f"Evaluator feedback (code_notes): {code_notes}\n\n"
        f"Active params — threshold: {suggested.get('threshold', _DEFAULTS['threshold'])}, "
        f"focus_labels: {retune_params.get('focus_labels', [])}, "
        f"boost_factor: {suggested.get('boost_factor', _DEFAULTS['boost_factor'])}\n\n"
        "Rewrite classifier.py to address the feedback. Output only the Python code."
    )

    chain = (
        ChatPromptTemplate.from_messages([("system", _CODE_GEN_SYSTEM_PROMPT), ("user", "{msg}")])
        | ChatOllama(model=DEFAULT_OLLAMA_MODEL, base_url=DEFAULT_OLLAMA_URL,
                     temperature=0.2, client_kwargs={"timeout": DEFAULT_OLLAMA_TIMEOUT})
        | StrOutputParser()
    )
    result = chain.invoke({"msg": user_msg}).strip()

    # Strip markdown fences if the model wrapped its output.
    if result.startswith("```python"):
        result = result[len("```python"):]
    elif result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    return result.strip()


def _is_usable(code: str) -> bool:
    """True when code compiles and preserves the required interface."""
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError:
        return False
    return "def classify(" in code and "def main(" in code


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def generate_code(state: PipelineState) -> dict:
    """Generate (or LLM-rewrite) classifier.py from retune params + Sabina's feedback."""
    threshold = _DEFAULTS["threshold"]
    max_length = _DEFAULTS["max_length"]
    focus_labels = _DEFAULTS["focus_labels"]
    boost_factor = _DEFAULTS["boost_factor"]

    retune_req = state.get("retune_request")
    if retune_req and isinstance(retune_req, dict):
        suggested = retune_req.get("suggested_params", {})
        threshold = suggested.get("threshold", threshold)
        max_length = suggested.get("max_length", max_length)
        boost_factor = suggested.get("boost_factor", boost_factor)
        focus_labels = retune_req.get("focus_labels", focus_labels)

    code_path = state.get("classifier_code_path") or os.path.join(OUTPUT_DIR, "classifier.py")
    os.makedirs(os.path.dirname(code_path) or ".", exist_ok=True)

    # Template substitution is always the safe baseline.
    code = _build_code(threshold, max_length, focus_labels, boost_factor)

    # Agentic rewrite: when Sabina left observations and we're on a retune cycle,
    # ask Ollama to reason from her feedback and write genuinely new classifier logic.
    code_notes = _read_code_notes()
    if code_notes and retune_req and isinstance(retune_req, dict):
        print("[nadi] code_notes found — attempting LLM rewrite via Ollama")
        try:
            llm_code = _rewrite_with_ollama(code, code_notes, retune_req)
            if llm_code and _is_usable(llm_code):
                code = llm_code
                print("[nadi] LLM-rewritten classifier accepted")
            else:
                print("[nadi] LLM output rejected; using template fallback", file=sys.stderr)
        except Exception as e:
            print(f"[nadi] Ollama rewrite failed ({e}); using template fallback", file=sys.stderr)

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[nadi] Generated classifier code at: {code_path}")

    # Archive per-iteration so past retune attempts survive the next overwrite.
    iteration = retune_req.get("iteration", 0) if isinstance(retune_req, dict) else 0
    history_dir = os.path.join(os.path.dirname(code_path) or ".", "classifier_history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"classifier_iter{iteration}.py")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(code)

    return {
        "classifier_code_path": code_path,
        "classifier_history_path": history_path,
        "classifier_metadata": {
            "model_name": "ProsusAI/finbert",
            "fine_tuning_params": {
                "threshold": threshold,
                "max_length": max_length,
                "focus_labels": focus_labels,
                "boost_factor": boost_factor,
            },
        },
    }


def run_classifier(state: PipelineState) -> dict:
    """Run the generated classifier.py as a subprocess on processed_data.csv."""
    code_path = state["classifier_code_path"]
    data_path = state.get("processed_data_path") or "mock_data/processed_data.csv"
    pred_path = state.get("predictions_path") or os.path.join(OUTPUT_DIR, "predictions_test.csv")

    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    print(f"[nadi] Running: {code_path} {data_path} → {pred_path}")
    subprocess.run([sys.executable, code_path, data_path, pred_path], check=True)
    print(f"[nadi] Predictions saved to: {pred_path}")
    return {"predictions_path": pred_path}


# ---------------------------------------------------------------------------
# Graph + agent class
# ---------------------------------------------------------------------------

def build_graph(checkpointer):
    from langgraph.graph import END, START, StateGraph

    b = StateGraph(PipelineState)
    b.add_node("generate_code", generate_code)
    b.add_node("run_classifier", run_classifier)
    b.add_edge(START, "generate_code")
    b.add_edge("generate_code", "run_classifier")
    b.add_edge("run_classifier", END)
    return b.compile(checkpointer=checkpointer)


class ClassifierAgent(Agent):
    """Classifier Agent (Nadi) behind the shared `.run()` interface."""

    def __init__(self, *, checkpointer=None, thread_id="classifier"):
        super().__init__(checkpointer=checkpointer, thread_id=thread_id)

    def build_graph(self, checkpointer):
        return build_graph(checkpointer)

    def run(self, processed_data, classifier_code, predictions, retune_request=None) -> dict:
        state = {
            "processed_data_path": os.path.abspath(processed_data),
            "classifier_code_path": os.path.abspath(classifier_code),
            "predictions_path": os.path.abspath(predictions),
        }
        if retune_request is not None:
            if not os.path.exists(retune_request):
                print(f"[nadi] WARNING: retune_request not found: {retune_request}", file=sys.stderr)
                state["retune_request"] = "no retune applied"
            else:
                try:
                    with open(retune_request, encoding="utf-8") as f:
                        state["retune_request"] = json.load(f)
                except Exception as e:
                    print(f"[nadi] ERROR parsing retune_request: {e}", file=sys.stderr)
                    raise
        return self._invoke(state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run the Classifier Agent (Nadi).")
    p.add_argument("--processed-data", default="mock_data/processed_data.csv")
    p.add_argument("--classifier-code", default="outputs/classifier.py")
    p.add_argument("--predictions", default="outputs/predictions_test.csv")
    p.add_argument("--retune-request", default=None)
    args = p.parse_args()

    agent = ClassifierAgent()
    res = agent.run(
        processed_data=args.processed_data,
        classifier_code=args.classifier_code,
        predictions=args.predictions,
        retune_request=args.retune_request,
    )
    print("\nClassifier completed. Predictions at:", res["predictions_path"])
