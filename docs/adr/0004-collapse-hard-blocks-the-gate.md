# Collapse hard-blocks the gate

Previously, a collapsed class only lowered `report_score` — it could still reach `final_action = "proceed"` via `cap_hit` or `converged`, with `select_best` publishing the least-bad (still collapsed) iteration on disk. This branch makes collapse an unconditional block: the gate cannot return `proceed` while any class is collapsed, full stop.

This means the iteration cap can now be exhausted with nothing safe to publish. In that case the pipeline fails loudly — no `final_results.csv` / `final_report.json` is written — rather than shipping a model with near-zero recall on a class.

**Rejected alternatives:** (a) let the cap keep overriding collapse as before, silently publishing the best-of-a-bad-batch — rejected because a degenerate model with a plausible-looking `final_report.json` is worse than an obvious failure; (b) remove the iteration cap for the collapsed case and retrain indefinitely until it resolves — rejected because "keep training forever" isn't a real stopping condition and risks unbounded GPU spend chasing a class that may never un-collapse (e.g. a genuine data problem).
