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

    preds = _load_csv("predictions_test.csv")
    explanations = _load_csv("explanations.csv")
    evaluation = _load_json("evaluation_report.json")
    final_report = _load_json("final_report.json")
    decision = _load_json("decision.json")
    return DATA_DIR, evaluation, explanations, final_report, preds


@app.cell
def _(DATA_DIR, mo):
    mo.md(f"""
    # 📈 Stock-Move Pipeline Dashboard

    Predict next-day move (up / down / neutral) from news headlines, then
    explain each call. Source: **`{DATA_DIR}/`**
    """)
    return


@app.cell
def _(evaluation, final_report, mo, preds):
    # --- KPI row ---
    accuracy = final_report.get("final_accuracy", evaluation.get("accuracy", 0.0))
    iterations = final_report.get("loop_iterations", "—")
    below = evaluation.get("below_threshold")
    gate = "✅ cleared" if below is False else ("⚠️ below target" if below else "—")

    kpis = mo.hstack(
        [
            mo.stat(f"{accuracy:.0%}", label="Accuracy", bordered=True),
            mo.stat(iterations, label="Loop iterations", bordered=True),
            mo.stat(gate, label="Threshold gate", bordered=True),
            mo.stat(len(preds), label="Test rows", bordered=True),
        ],
        gap=1,
        widths="equal",
    )
    kpis
    return (accuracy,)


@app.cell
def _(alt, mo, preds):
    # --- Confusion matrix: actual (label) vs predicted_label ---
    labels = ["up", "down", "neutral"]
    cm = (
        preds.groupby(["label", "predicted_label"]).size().reset_index(name="count")
    )
    confusion = (
        alt.Chart(cm)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", sort=labels, title="Predicted"),
            y=alt.Y("label:N", sort=labels, title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["label", "predicted_label", "count"],
        )
        .properties(width=260, height=260, title="Confusion matrix")
    )
    text = confusion.mark_text(baseline="middle").encode(
        text="count:Q",
        color=alt.condition(
            alt.datum.count > cm["count"].max() / 2,
            alt.value("white"),
            alt.value("black"),
        ),
    )
    mo.ui.altair_chart(confusion + text) if not preds.empty else mo.md("_no predictions_")
    return


@app.cell
def _(alt, mo, preds):
    # --- Confidence histogram, split by correct/incorrect ---
    if preds.empty:
        conf_hist = mo.md("_no predictions_")
    else:
        d = preds.assign(correct=preds["label"] == preds["predicted_label"])
        conf_hist = mo.ui.altair_chart(
            alt.Chart(d)
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
            .properties(width=280, height=260, title="Prediction confidence")
        )
    conf_hist
    return


@app.cell
def _(alt, mo, preds):
    # --- Per-ticker accuracy ---
    if preds.empty:
        ticker_chart = mo.md("_no predictions_")
    else:
        per = (
            preds.assign(correct=(preds["label"] == preds["predicted_label"]).astype(int))
            .groupby("ticker")["correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        ticker_chart = mo.ui.altair_chart(
            alt.Chart(per)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", sort="-y", title="Ticker"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
                color=alt.Color("accuracy:Q", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["ticker", alt.Tooltip("accuracy:Q", format=".0%")],
            )
            .properties(width=400, height=260, title="Accuracy by ticker")
        )
    ticker_chart
    return


@app.cell
def _(mo):
    # --- Reactive control: confidence threshold ---
    conf_threshold = mo.ui.slider(
        0.0, 1.0, value=0.0, step=0.05, label="Min confidence", show_value=True
    )
    conf_threshold
    return (conf_threshold,)


@app.cell
def _(accuracy, conf_threshold, mo, preds):
    # Accuracy among only the rows the model was confident about — moves live
    # with the slider above (Marimo reactivity, no callback).
    if preds.empty:
        thresh_view = mo.md("_no predictions_")
    else:
        kept = preds[preds["confidence"] >= conf_threshold.value]
        if len(kept):
            acc = (kept["label"] == kept["predicted_label"]).mean()
            coverage = len(kept) / len(preds)
            thresh_view = mo.hstack(
                [
                    mo.stat(f"{acc:.0%}", label=f"Accuracy ≥ {conf_threshold.value:.2f}", bordered=True),
                    mo.stat(f"{coverage:.0%}", label="Coverage (rows kept)", bordered=True),
                    mo.stat(f"{acc - accuracy:+.0%}", label="vs. all rows", bordered=True),
                ],
                widths="equal",
            )
        else:
            thresh_view = mo.md(f"_no rows with confidence ≥ {conf_threshold.value:.2f}_")
    thresh_view
    return


@app.cell
def _(mo, preds):
    # --- Widgets driving the predictions table ---
    if preds.empty:
        ticker_filter = mo.ui.multiselect(options=[], label="Tickers")
        label_filter = mo.ui.dropdown(options=["all"], value="all", label="Predicted label")
        only_wrong = mo.ui.switch(value=False, label="Only misclassified")
    else:
        ticker_filter = mo.ui.multiselect(
            options=sorted(preds["ticker"].unique().tolist()),
            label="Tickers (empty = all)",
        )
        label_filter = mo.ui.dropdown(
            options=["all", "up", "down", "neutral"], value="all", label="Predicted label"
        )
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
def _(label_filter, mo, only_wrong, preds, ticker_filter):
    # Reactive filter — table redraws when any widget changes, no callbacks.
    if preds.empty:
        preds_table = mo.md("_no predictions_")
    else:
        view = preds
        if ticker_filter.value:
            view = view[view["ticker"].isin(ticker_filter.value)]
        if label_filter.value != "all":
            view = view[view["predicted_label"] == label_filter.value]
        if only_wrong.value:
            view = view[view["label"] != view["predicted_label"]]
        preds_table = mo.vstack(
            [
                mo.md(f"_{len(view)} of {len(preds)} rows_"),
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
    ## 🔀 Manager LangGraph — live state trace

    Re-streams the Manager graph against the current `evaluation_report.json`.
    Each row is one node firing and the partial state update it returned.
    """)
    return


@app.cell
def _(DATA_DIR, mo, os, pd):
    # Stream the Manager graph and capture each node's state update. Read-only
    # re-run against the current report — writes go to outputs/ exactly as a
    # normal run would, so guard on the report existing.
    report_path = os.path.join(DATA_DIR, "evaluation_report.json")

    if not os.path.exists(report_path):
        graph_trace = pd.DataFrame()
        final_state = {}
    else:
        import json as _json

        from agents.jack_manager import ManagerAgent

        with open(report_path, encoding="utf-8") as f:
            _report = _json.load(f)

        mgr = ManagerAgent(thread_id="dashboard")
        init = {**mgr._defaults, "evaluation_report": _report}

        rows = []
        for event in mgr._graph.stream(init, mgr._config):
            for node, update in event.items():
                rows.append(
                    {
                        "node": node,
                        "keys_updated": ", ".join(update.keys()),
                        "final_action": update.get("final_action", ""),
                        "decision": update.get("decision", ""),
                        "iteration": update.get("iteration", ""),
                        "notes": update.get("notes", ""),
                    }
                )
        graph_trace = pd.DataFrame(rows)
        final_state = mgr._graph.get_state(mgr._config).values

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
