# AGENTS.md

Team working agreement for building the stock-move prediction system. Read this before writing any agent code. Applies to every AI coding tool (Claude Code, Codex, Cursor, Copilot, …) and every human on the team.

## Mission

Predict next-day stock move (**up / down / neutral**) from financial-news headlines with FinBERT, then explain each prediction in plain text. Five agents pass files to each other in a loop. Full design: [`docs/architecture.md`](./docs/architecture.md). Exact file formats: [`docs/data_contracts.md`](./docs/data_contracts.md). How we work async as a team: [`docs/collaborating.md`](./docs/collaborating.md).

## Sources of truth — do not duplicate, link to them

| Topic | File |
|---|---|
| Flow, agent roles, eval axes | `docs/architecture.md` |
| Every handoff file's columns/types | `docs/data_contracts.md` |
| Valid sample inputs/outputs | `mock_data/` (one file per handoff) |
| How the team works async (non-technical) | `docs/collaborating.md` |
| Repo facts, commands, dlt pipeline | `CLAUDE.md` |

If code and these docs disagree, the docs win — fix the code or update the doc in the same change, never silently.

## Who owns what

| Agent | Owner | Builds | Reads | Writes |
|---|---|---|---|---|
| Manager | Jack | orchestration loop + threshold gate + final output | `evaluation_report.json`, `predictions_test.csv`, `explanations.csv` | `decision.json`, `retune_request.json`, `sample_for_explanation.csv`, `final_results.csv`, `final_report.json` |
| Processing | Aurora | join FNSPID headlines ↔ yfinance prices, label move | FNSPID, yfinance | `processed_data.csv` |
| Classifier | Nadi | FinBERT → up/down/neutral | `processed_data.csv`, `retune_request.json` | `predictions_test.csv` |
| Evaluator | Sabina | accuracy + per-class metrics | `predictions_test.csv` | `evaluation_report.json` |
| Explanation | Freddi | Ollama justification per row | `sample_for_explanation.csv` | `explanations.csv` |

You own one agent. You may change **only the files in your "Writes" column.** Touching another agent's output format requires their sign-off (see Golden rules).

## Golden rules

1. **The contract is law.** Column names, types, filenames, and label values (`up`/`down`/`neutral`, lowercase) are fixed by `docs/data_contracts.md`. Pass-through columns marked "do not modify" must arrive at the next agent byte-for-byte.
2. **Never rename a column or file alone.** Propose the change, update `docs/data_contracts.md` + `mock_data/`, and tell every downstream owner before merging. A rename that skips this breaks someone else's agent silently.
3. **Build against `mock_data/` first.** Each agent must read the mock input and produce output matching the mock output *before* the upstream agent exists. That is your integration test.
4. **Your output is someone's input.** Validate it against the contract before you call your agent done. Bad data flowing downstream is the failure mode this whole system is designed to prevent.
5. **Don't touch the dlt pipeline** (`yahoo_finance_pipeline.py`) for agent work — it's a separate ingestion layer. See `CLAUDE.md`.
6. **Stay in your lane.** No "while I'm here" edits to another agent's module. Surface problems; don't fix them across the boundary.

## Repo layout

```
docs/architecture.md       # design (read-only reference)
docs/data_contracts.md     # handoff formats (read-only reference)
docs/collaborating.md      # how the team works async (non-technical)
mock_data/              # valid sample for every handoff
yahoo_finance_pipeline.py  # dlt ingestion — NOT part of agent work
agents/                 # <- put your agent code here (create as needed)
```

Put each agent in its own module under `agents/` (e.g. `agents/aurora_processing.py`). Don't scatter agent code across the repo root.

## Setup & run

Python 3.13, uv-managed. From repo root:

```bash
uv sync                 # deps
uv run agents/<your_agent>.py
```

Add a new dependency to **both** `requirements.txt` and `pyproject.toml`.

## Definition of done (per agent)

- [ ] Reads its contract input from `mock_data/` and runs end to end.
- [ ] Output columns/types/filename match `docs/data_contracts.md` exactly.
- [ ] Labels lowercase; dates `YYYY-MM-DD`; nulls written as empty string; UTF-8, comma-separated.
- [ ] Pass-through columns from upstream are unmodified.
- [ ] Output diffs cleanly against the matching `mock_data/` file (same shape, plausible values).

## Working with AI tools

When you ask Claude Code / Codex / etc. to build your agent: point it at this file, `docs/architecture.md`, `docs/data_contracts.md`, and the relevant `mock_data/` files. Tell it which agent you own and that it must not change other agents' output formats. Have it test against mock data before claiming done.

## Conventions

- Branch per agent/feature; don't commit straight to `main`.
- Keep changes scoped to your agent (Golden rule 6).
- Commit messages: state what changed and why. (This repo omits AI co-author trailers.)
