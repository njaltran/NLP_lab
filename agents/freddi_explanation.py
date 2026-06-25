"""Explanation Agent (Freddi) — Handoff 5 of the stock-move prediction loop.

Reads `sample_for_explanation.csv` (from Jack, the Manager) and, for each row,
generates a plain-text justification of the model's prediction using Ollama.
Writes `explanations.csv` back to Jack.

Contract: docs/data_contracts.md, Handoffs 4 (input) and 5 (output).

Input columns  : article_id, article_title, predicted_label, actual_label,
                 confidence, prob_up, prob_down, prob_neutral
Output columns : article_id, article_title, predicted_label, actual_label,
                 confidence, explanation, manual_score

Rules honoured here:
  - First five columns are passed through byte-for-byte (do not modify).
  - `prob_*` are inputs to the reasoning only; they are NOT written to output.
  - `manual_score` is left blank (filled by hand later for 30-50 rows).
  - CSV is UTF-8, comma-separated; blanks written as empty string.

Ollama is used when its local server is reachable. If it is not (e.g. not yet
installed), a deterministic fallback explanation is produced instead, so the
agent runs end-to-end against mock_data before the real model is available.
Run `python agents/freddi_explanation.py --help` for options.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request

# Columns we read in (from Handoff 4) and the exact columns we must write out
# (Handoff 5). Order of OUTPUT_COLUMNS is the column order of explanations.csv.
INPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "prob_up", "prob_down", "prob_neutral",
]
OUTPUT_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "explanation", "manual_score",
]
# Columns passed through from the input untouched (Golden rule 1).
PASSTHROUGH_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label", "confidence",
]

DEFAULT_INPUT = "mock_data/sample_for_explanation.csv"
DEFAULT_OUTPUT = "explanations.csv"
DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def build_prompt(row: dict) -> str:
    """Build the Ollama prompt for one row — Option A, "explain the prediction only".

    The model is given the headline and the model's predicted next-day move, and
    explains why that headline could justify that move. It is deliberately NOT given
    the actual next-day outcome: that mirrors real prediction time (tomorrow's price
    is unknown) and keeps this to the assigned "explain your reasoning" task. The
    actual_label is still written to explanations.csv for the human graders — it just
    never reaches the model, so the explanation can't defend a known-wrong answer.
    """
    move_phrase = {
        "up": "an upward next-day move",
        "down": "a downward next-day move",
        "neutral": "little or no next-day move",
    }.get((row.get("predicted_label") or "").strip(), "the predicted next-day move")
    return (
        "You explain a stock-move model's prediction in plain language.\n"
        "Given a news headline and the model's predicted next-day move, write ONE "
        "clear sentence (max ~25 words) explaining why the headline could justify "
        "that move. Use only what the headline states — do not invent facts, numbers, "
        "or events. Do not add a preamble or restate the task.\n\n"
        f"Headline: {row.get('article_title', '')}\n"
        f"Predicted move: {move_phrase}\n"
        f"Model confidence: {row.get('confidence', '')}\n"
        f"Class probabilities -> up: {row.get('prob_up', '')}, "
        f"down: {row.get('prob_down', '')}, neutral: {row.get('prob_neutral', '')}\n\n"
        "Explanation:"
    )


def generate_with_ollama(prompt: str, model: str, base_url: str, timeout: float = 60.0) -> str:
    """Call the local Ollama HTTP API. Raises urllib.error.URLError if unreachable."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _clean(data.get("response", ""))


def fallback_explanation(row: dict) -> str:
    """Deterministic placeholder used when Ollama is unavailable.

    Produces a plausible, contract-valid sentence so the pipeline runs offline.
    Replaced automatically by real Ollama output once the server is reachable.
    """
    title = (row.get("article_title") or "the headline").strip()
    pred = (row.get("predicted_label") or "").strip()
    actual = (row.get("actual_label") or "").strip()
    matched = pred == actual

    direction = {
        "up": "an upward next-day move",
        "down": "a downward next-day move",
        "neutral": "little next-day movement",
    }.get(pred, "the predicted move")

    if matched:
        return _clean(
            f'The headline "{title}" reads as a {pred or "neutral"} signal, '
            f"which supports {direction}."
        )
    return _clean(
        f'The model predicted {pred or "neutral"} from "{title}", but the actual '
        f"move was {actual or 'different'}, so this call was missed."
    )


def _clean(text: str) -> str:
    """Collapse whitespace/newlines so each explanation is a single CSV-safe line."""
    return " ".join(text.split()).strip()


def explain_rows(rows: list[dict], model: str, base_url: str, use_ollama: bool) -> list[dict]:
    """Generate an explanation per row, returning rows shaped for OUTPUT_COLUMNS."""
    ollama_live = use_ollama  # may flip to False on first failure
    out = []
    for i, row in enumerate(rows, start=1):
        explanation = ""
        if ollama_live:
            try:
                explanation = generate_with_ollama(build_prompt(row), model, base_url)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                # Ollama not reachable / errored: warn once, switch to fallback.
                print(
                    f"[freddi] Ollama unavailable ({exc}); using offline fallback "
                    "for all rows.",
                    file=sys.stderr,
                )
                ollama_live = False
        if not explanation:
            explanation = fallback_explanation(row)

        out_row = {col: (row.get(col, "") or "") for col in PASSTHROUGH_COLUMNS}
        out_row["explanation"] = explanation
        out_row["manual_score"] = ""  # left blank by contract; scored by hand later
        out.append(out_row)
        print(f"[freddi] {i}/{len(rows)} {out_row['article_id']} -> done")
    return out


def read_input(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(
                f"[freddi] WARNING: input {path} is missing expected columns: "
                f"{missing}",
                file=sys.stderr,
            )
        return list(reader)


def write_output(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explanation Agent (Freddi) — Handoff 5")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="sample_for_explanation.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="explanations.csv output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument(
        "--no-ollama", action="store_true",
        help="skip Ollama and always use the offline fallback (handy for testing)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
    args = parser.parse_args(argv)

    rows = read_input(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[freddi] read {len(rows)} rows from {args.input}")

    out_rows = explain_rows(
        rows, model=args.model, base_url=args.ollama_url, use_ollama=not args.no_ollama
    )
    write_output(args.output, out_rows)
    print(f"[freddi] wrote {len(out_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
