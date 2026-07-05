"""Pipeline dashboard (Jack) — Marimo.

Reactive view of the stock-move pipeline's outputs plus a live trace of the
Manager's LangGraph. Reads the contract files the agents wrote (prefers
`outputs/`, falls back to `mock_data/` so it renders before a real run), and
re-streams the Manager graph against the current evaluation_report to show each
node's state update as it fires.

Read-only: it imports the Manager agent and reads contract files; it writes
nothing and changes no agent's output format. Run with:

    uv run marimo edit dashboard.py      # interactive
    uv run marimo run dashboard.py       # app view (read-only)
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, json, mo, os, pd


@app.cell
def _(json, os, pd):
    # Prefer real run outputs; fall back to mock_data so the dashboard renders
    # before the pipeline has ever run.
    DATA_DIR = "outputs" if os.path.exists("outputs/predictions_test.csv") else "mock_data"

    def _load_json(name):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(name):
        path = os.path.join(DATA_DIR, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    # final_report.json is deliberately NOT read: it can lag the latest run. Every
    # headline figure is computed from predictions_test.csv (ground truth) or read
    # from the per-split evaluator reports the loop rewrites each pass.
    preds = _load_csv("predictions_test.csv")
    explanations = _load_csv("explanations.csv")
    decision = _load_json("decision.json")

    # Sabina writes one report per split (validation_report / test_report);
    # evaluation_report.json mirrors whichever split the loop last gated on. Key
    # them by eval_split so we can show the metrics that match the chosen split.
    reports = {}
    for _f in ("validation_report.json", "test_report.json", "evaluation_report.json"):
        _r = _load_json(_f)
        if _r:
            reports[_r.get("eval_split", "test")] = _r

    # Shared correctness flag over the full frame; split filtering happens below.
    if not preds.empty:
        preds["correct"] = preds["label"] == preds["predicted_label"]

    LABELS = ["up", "down", "neutral"]  # canonical order, reused by chart + dropdown
    return DATA_DIR, LABELS, decision, explanations, preds, reports


@app.cell
def _(DATA_DIR, mo):
    mo.md(f"""
    # \U0001F4C9 Can FinBERT call tomorrow\'s move?

    Next-day **up / down / neutral** from a single news headline. The loop retuned
    FinBERT five times chasing a **51.6% accuracy target** (the all-neutral
    baseline) \u2014 this is where it landed, and why it stalled.
    Source: **`{DATA_DIR}/`**
    """)
    return


@app.cell
def beat_verdict(mo):
    mo.md("""
    ---
    ### ① The verdict &nbsp; · &nbsp; _where the loop landed vs the 51.6% gate_

    Pick a split — **test** is the number that counts; **validation** is what the
    loop gated on each pass.
    """)
    return


@app.cell
def split_ctl(mo, preds):
    # Everything below is scoped to ONE eval split. Validation (Sabina gate input)
    # and test are separate populations; mixing them is why earlier headline numbers
    # drifted from the reports. Pick the split once and every chart, KPI and table
    # downstream reacts to it.
    _counts = preds["split"].value_counts().to_dict() if (not preds.empty and "split" in preds) else {}
    _opts = {}
    for _s in ("test", "val"):
        if _s in _counts:
            _label = "Test" if _s == "test" else "Validation"
            _opts[f"{_label} \u00b7 {_counts[_s]:,} rows"] = _s
    if len(_counts) > 1:
        _opts["Both \u00b7 combined"] = "all"
    split_sel = mo.ui.radio(
        options=_opts or {"Test": "test"},
        value=next(iter(_opts)) if _opts else "Test",
        inline=True,
    )
    mo.hstack([mo.md("### Eval split"), split_sel], justify="start", align="center", gap=1)
    return (split_sel,)


@app.cell
def split_view(preds, reports, split_sel):
    # Derived view + the evaluator report matching the chosen split. Defined once
    # here so accuracy, confusion, per-class and the table can never disagree.
    split = split_sel.value
    if preds.empty:
        pv = preds
    elif split == "all":
        pv = preds
    else:
        pv = preds[preds["split"] == split]
    split_report = reports.get("test" if split == "all" else split, {})
    return pv, split, split_report


@app.cell
def _(LABELS, decision, mo, pv, split, split_report):
    # --- KPI row --- scoped to the selected split. Accuracy is recomputed from the
    # predictions so it can never drift from the matrix/table below, and it matches
    # the evaluator report for the same split.
    accuracy = pv["correct"].mean() if not pv.empty else 0.0
    iterations = decision.get("iteration", "\u2014")  # Manager rewrites this each pass
    below = split_report.get("below_threshold")
    gate = "\u2705 cleared" if below is False else ("\u26a0\ufe0f below target" if below else "\u2014")

    kpis = mo.hstack(
        [
            mo.stat(f"{accuracy:.0%}", label=f"Accuracy ({split})", bordered=True),
            mo.stat(iterations, label="Loop iterations", bordered=True),
            mo.stat(gate, label="Threshold gate", bordered=True),
            mo.stat(f"{len(pv):,}", label=f"{split.capitalize()} rows", bordered=True),
        ],
        gap=1,
        widths="equal",
    )

    # Per-class accuracy from the same view -- the down class collapsing is the whole
    # story of this run, so surface it instead of burying it in the matrix.
    _cls = (
        {l: pv.loc[pv["label"] == l, "correct"].mean() for l in LABELS if (pv["label"] == l).any()}
        if not pv.empty
        else {}
    )
    per_class = mo.hstack(
        [mo.stat(f"{_cls[l]:.0%}" if l in _cls else "\u2014", label=f"{l} accuracy", bordered=True) for l in LABELS],
        gap=1,
        widths="equal",
    ) if _cls else mo.md("")

    mo.vstack([kpis, per_class])
    return


@app.cell
def beat_stalled(mo):
    mo.md("""
    ---
    ### ② Why it stalled &nbsp; · &nbsp; _five retune passes, no lift_

    Accuracy drifted sideways and never cleared the gate. More tuning wasn't going
    to fix a data-shaped problem.
    """)
    return


@app.cell
def loop_history(alt, decision, mo, pd):
    # --- Loop convergence: accuracy per iteration, from the Manager history ---
    _hist = decision.get("accuracy_history", [])
    if len(_hist) > 1:
        _hdf = pd.DataFrame({"iteration": range(1, len(_hist) + 1), "accuracy": _hist})
        history_chart = mo.ui.altair_chart(
            alt.Chart(_hdf)
            .mark_line(point=True)
            .encode(
                x=alt.X("iteration:O", title="Loop iteration"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
                tooltip=["iteration", alt.Tooltip("accuracy:Q", format=".0%")],
            )
            .properties(width=420, height=180, title="Accuracy across the retune loop")
        )
    else:
        history_chart = mo.md("")
    history_chart
    return


@app.cell
def beat_culprit(mo):
    mo.md("""
    ---
    ### ③ The culprit &nbsp; · &nbsp; _the model won't say “down”_

    Look at the **down** column: FinBERT almost never predicts it, so it hedges to
    **neutral**. Confidence sits low across the board — the model is guessing.
    """)
    return


@app.cell
def _(LABELS, alt, mo, pv):
    # --- Confusion matrix: actual (label) vs predicted_label, scoped to split ---
    cm_df = pv.groupby(["label", "predicted_label"]).size().reset_index(name="count")
    confusion = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", sort=LABELS, title="Predicted"),
            y=alt.Y("label:N", sort=LABELS, title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["label", "predicted_label", "count"],
        )
        .properties(width=240, height=240, title="Confusion matrix")
    )
    cm_text = confusion.mark_text(baseline="middle").encode(
        text="count:Q",
        color=alt.condition(
            alt.datum.count > cm_df["count"].max() / 2,
            alt.value("white"),
            alt.value("black"),
        ),
    )
    confusion_view = mo.ui.altair_chart(confusion + cm_text) if not pv.empty else mo.md("_no predictions_")
    return (confusion_view,)


@app.cell
def _(alt, mo, pv):
    # --- Confidence histogram, split by correct/incorrect (scoped to split) ---
    if pv.empty:
        conf_hist = mo.md("_no predictions_")
    else:
        conf_hist = mo.ui.altair_chart(
            alt.Chart(pv)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence"),
                y=alt.Y("count():Q", title="Rows"),
                color=alt.Color(
                    "correct:N",
                    scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#d62728"]),
                    title="Correct",
                ),
                tooltip=["count()"],
            )
            .properties(width=280, height=240, title="Prediction confidence")
        )
    return (conf_hist,)


@app.cell
def culprit_charts(conf_hist, confusion_view, mo):
    # Confusion + confidence side by side: the down column is nearly empty, and the
    # confidence spread shows how little conviction is behind any call.
    mo.hstack([confusion_view, conf_hist], gap=2, justify="start", align="start")
    return


@app.cell
def beat_inspect(mo):
    mo.md("""
    ---
    ### ④ Inspect the calls &nbsp; · &nbsp; _drill into the rows_

    Filter the predictions and read the model's own plain-text rationale per row.
    """)
    return


@app.cell
def _(LABELS, mo, pv):
    # --- Widgets driving the predictions table (scoped to split) ---
    tickers = sorted(pv["ticker"].unique().tolist()) if not pv.empty else []
    ticker_filter = mo.ui.multiselect(options=tickers, label="Tickers (empty = all)")
    label_filter = mo.ui.dropdown(options=["all", *LABELS], value="all", label="Predicted label")
    only_wrong = mo.ui.switch(value=False, label="Only misclassified")

    controls = mo.vstack(
        [
            mo.md("## Predictions"),
            mo.hstack([ticker_filter, label_filter, only_wrong], gap=2, justify="start"),
        ]
    )
    controls
    return label_filter, only_wrong, ticker_filter


@app.cell
def _(label_filter, mo, only_wrong, pv, ticker_filter):
    # Reactive filter -- table redraws when any widget changes, no callbacks.
    if pv.empty:
        preds_table = mo.md("_no predictions_")
    else:
        view = pv
        if ticker_filter.value:
            view = view[view["ticker"].isin(ticker_filter.value)]
        if label_filter.value != "all":
            view = view[view["predicted_label"] == label_filter.value]
        if only_wrong.value:
            view = view[~view["correct"]]
        preds_table = mo.vstack(
            [
                mo.md(f"_{len(view)} of {len(pv)} rows_"),
                mo.ui.table(view, selection=None, page_size=15, label="predictions_test.csv"),
            ]
        )
    preds_table
    return


@app.cell
def _(explanations, mo):
    # --- Explanations: headline → call → rationale ---
    if explanations.empty:
        expl_view = mo.md("_no explanations yet (run Freddi)_")
    else:
        expl_view = mo.vstack(
            [
                mo.md("## Explanations"),
                mo.ui.table(explanations, selection=None, page_size=10, label="explanations.csv"),
            ]
        )
    expl_view
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ### ⑤ How the Manager decided &nbsp; · &nbsp; _the gate logic, re-run_

    Re-streams the Manager LangGraph against the current report — one row per
    node firing. Sandboxed: this trace writes **nothing** to `outputs/`.
    """)
    return


@app.cell
def _(DATA_DIR, json, mo, os, pd):
    # Re-stream the Manager graph against the current report to show each node fire
    # and the partial state it returned. The Manager write nodes persist to its
    # OUTPUT_DIR, so point that at a throwaway temp dir for this trace -- the
    # dashboard must NEVER overwrite the real run\'s decision.json / retune_request.
    report_path = os.path.join(DATA_DIR, "evaluation_report.json")

    if not os.path.exists(report_path):
        graph_trace = pd.DataFrame()
        final_state = {}
    else:
        import tempfile
        from agents import jack_manager
        from agents.jack_manager import ManagerAgent

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        _saved_out = jack_manager.OUTPUT_DIR
        jack_manager.OUTPUT_DIR = tempfile.mkdtemp(prefix="dashboard_trace_")
        try:
            mgr = ManagerAgent(thread_id="dashboard")
            init_state = {**mgr._defaults, "evaluation_report": report}
            trace_rows = [
                {
                    "node": node,
                    "keys_updated": ", ".join(update.keys()),
                    "final_action": update.get("final_action", ""),
                    "decision": update.get("decision", ""),
                    "iteration": update.get("iteration", ""),
                    "notes": update.get("notes", ""),
                }
                for event in mgr._graph.stream(init_state, mgr._config)
                for node, update in event.items()
            ]
            graph_trace = pd.DataFrame(trace_rows)
            final_state = mgr._graph.get_state(mgr._config).values
        finally:
            jack_manager.OUTPUT_DIR = _saved_out

    mo.ui.table(graph_trace, selection=None, label="node-by-node trace") if not graph_trace.empty else mo.md("_no evaluation_report to stream_")
    return (final_state,)


@app.cell
def _(final_state, mo):
    # --- Decision log + final merged state ---
    if not final_state:
        log_view = mo.md("")
    else:
        log = final_state.get("decision_log", [])
        log_view = mo.vstack(
            [
                mo.md("### Decision log"),
                mo.md("\n".join(f"- {line}" for line in log) or "_empty_"),
                mo.md("### Final Manager state"),
                mo.json(
                    {k: v for k, v in final_state.items() if k != "evaluation_report"}
                ),
            ]
        )
    log_view
    return


if __name__ == "__main__":
    app.run()
