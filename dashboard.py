"""Pipeline dashboard (Jack) — Marimo. Presentation edition.

Findings-first view of the stock-move pipeline, built to be projected to a
class: big numbers, one finding per chart, almost no prose. Reads only the
real contract files the agents wrote, fetched straight from the GitHub repo
(github.com/njaltran/NLP_lab/tree/main/outputs) instead of a local outputs/
run (never `mock_data/` — missing/unreachable files render as placeholders,
not fake numbers) plus the Manager's own `decision.json` history. Read-only:
writes nothing, changes no agent's output format. Run with:

    uv run marimo run dashboard.py       # app view (present this)
    uv run marimo edit dashboard.py      # interactive
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import ssl
    import urllib.request

    import altair as alt
    import certifi
    import marimo as mo
    import pandas as pd

    return alt, certifi, io, json, mo, pd, ssl, urllib


@app.cell
def _():
    # Data source: the GitHub repo's outputs/, not a local pipeline run —
    # https://github.com/njaltran/NLP_lab/tree/main/outputs
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/njaltran/NLP_lab/main/outputs"
    return (GITHUB_RAW_BASE,)


@app.cell
def _(alt):
    # --- Chart theme: one palette, applied to every chart. -------------------
    # Hues are categorical slots from a CVD-validated palette; ink/grid colors
    # are mid-tones that stay readable on both light and dark app themes.
    C = {
        "blue": "#2a78d6",      # primary series
        "blue_dark": "#1c5cab",
        "aqua": "#1baf7a",      # second series / "correct"
        "red": "#e34948",       # "wrong"
        "muted": "#898781",     # axis ink, works in light + dark
        "grid": "rgba(137,135,129,0.22)",
        "blues": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#2a78d6", "#256abf", "#184f95", "#0d366b"],
    }

    def themed(chart):
        """Apply the shared look: transparent surface, recessive axes."""
        return (
            chart.configure(background="transparent", font="system-ui, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(
                labelColor=C["muted"], titleColor=C["muted"],
                gridColor=C["grid"], domainColor=C["grid"], tickColor=C["grid"],
                labelFontSize=13, titleFontSize=13, titlePadding=10,
            )
            .configure_legend(labelColor=C["muted"], titleColor=C["muted"], labelFontSize=13)
            .configure_title(color=C["muted"], fontSize=14, fontWeight="normal", anchor="start")
        )

    PCT = alt.Axis(format=".0%")
    return C, PCT, themed


@app.cell
def _(GITHUB_RAW_BASE, certifi, io, json, pd, ssl, urllib):
    # Real run outputs ONLY — never mock_data. Mock numbers on a projector are
    # worse than an empty chart, so missing/unreachable files render as
    # placeholders.
    DATA_DIR = GITHUB_RAW_BASE
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _fetch(name):
        try:
            with urllib.request.urlopen(f"{DATA_DIR}/{name}", timeout=10, context=_ssl_ctx) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    def _load_json(name):
        raw = _fetch(name)
        return json.loads(raw) if raw else {}

    def _load_csv(name):
        raw = _fetch(name)
        return pd.read_csv(io.StringIO(raw)) if raw else pd.DataFrame()

    preds = _load_csv("predictions_test.csv")
    explanations = _load_csv("explanations.csv")
    evaluation = _load_json("evaluation_report.json")
    decision = _load_json("decision.json")
    finetunes = {
        "lr 1e-5 · 6 epochs": _load_json("finetune_report.json"),
        "lr 2e-5 · 10 epochs": _load_json("finetune_report_lr2e5_e10.json"),
    }
    finetunes = {k: v for k, v in finetunes.items() if v}

    # One shared correctness flag; headline accuracy is test-split only (the
    # loop tunes on val, so mixing splits would flatter nothing and confuse).
    if not preds.empty:
        preds["correct"] = preds["label"] == preds["predicted_label"]
    test_preds = preds[preds["split"] == "test"] if "split" in preds.columns else preds

    LABELS = ["up", "down", "neutral"]
    TARGET = 0.6  # target accuracy the gate uses (see retune_request.json)
    CHANCE = 1 / 3

    # Manager loop history: one validation accuracy per run, plus the params
    # each retune actually tried (run 1 ran the classifier defaults).
    hist = decision.get("accuracy_history", [])
    tried = [{}] + decision.get("tried_params", [])
    runs = pd.DataFrame(
        {
            "run": range(1, len(hist) + 1),
            "val_accuracy": hist,
            "params": [
                ", ".join(f"{k}={v}" for k, v in tried[i].items()) if i < len(tried) and tried[i] else "defaults"
                for i in range(len(hist))
            ],
        }
    )
    return (
        CHANCE,
        DATA_DIR,
        LABELS,
        TARGET,
        decision,
        evaluation,
        explanations,
        finetunes,
        preds,
        runs,
        test_preds,
    )


@app.cell
def _(CHANCE, decision, evaluation, mo, runs, test_preds):
    # --- Hero: the findings, as numbers. --------------------------------------
    acc = test_preds["correct"].mean() if not test_preds.empty else 0.0
    class_acc = evaluation.get("class_accuracy", {})
    down_acc = class_acc.get("down", 0.0)
    n_runs = decision.get("iteration", len(runs)) or len(runs)
    best_val = runs["val_accuracy"].max() if not runs.empty else 0.0
    first_val = runs["val_accuracy"].iloc[0] if not runs.empty else 0.0

    def tile(value, label, caption):
        return f"""
        <div style="flex:1;min-width:150px;padding:20px 22px;border-radius:14px;
                    border:1px solid rgba(137,135,129,.28);
                    background:rgba(137,135,129,.06);">
          <div style="font-size:46px;font-weight:750;line-height:1.05;
                      letter-spacing:-0.02em;">{value}</div>
          <div style="font-size:12px;text-transform:uppercase;letter-spacing:.09em;
                      color:#898781;margin-top:8px;">{label}</div>
          <div style="font-size:13px;color:#898781;margin-top:3px;">{caption}</div>
        </div>"""

    if not test_preds.empty:
        _y0, _y1 = test_preds["date"].min()[:4], test_preds["date"].max()[:4]
        years = _y0 if _y0 == _y1 else f"{_y0}–{_y1}"

    if test_preds.empty:
        tiles = (
            '<div style="font-size:15px;color:#898781;padding:14px 0;">'
            "waiting for a pipeline run — <code>outputs/</code> has no predictions yet</div>"
        )
    else:
        tiles = f"""
      <div style="display:flex;gap:14px;flex-wrap:wrap;">
        {tile(f"{acc:.0%}", "final test accuracy", f"chance = {CHANCE:.0%} · target 60%")}
        {tile(f"{best_val:.0%}", "best validation run", f"started at {first_val:.0%} · {n_runs} run{'s' if n_runs != 1 else ''} · kept")}
        {tile(f"{down_acc:.0%}", "accuracy on “down”", "the model can't see bad news")}
        {tile(f"{len(test_preds):,}", "test headlines", f"{years} · {test_preds['ticker'].nunique()} tickers")}
      </div>"""

    hero = mo.Html(f"""
    <div style="font-family:system-ui,sans-serif;padding:6px 0 2px;">
      <div style="font-size:13px;text-transform:uppercase;letter-spacing:.14em;
                  color:#2a78d6;font-weight:650;">Findings</div>
      <div style="font-size:34px;font-weight:760;letter-spacing:-0.02em;
                  line-height:1.15;margin:8px 0 4px;">
        Headlines beat the coin flip — not the market.
      </div>
      <div style="font-size:16px;color:#898781;margin-bottom:22px;">
        FinBERT on financial news · next-day move, up / down / neutral · 5 agents in a loop
      </div>
      {tiles}
    </div>""")
    hero
    return


@app.cell
def _(C, PCT, TARGET, alt, mo, runs, themed):
    # --- Finding 1: the Manager loop — tuning helped, over-tuning hurt. -------
    if runs.empty:
        runs_view = mo.md("_no loop history yet (run the Manager)_")
    else:
        best_run = int(runs.loc[runs["val_accuracy"].idxmax(), "run"])
        pts = runs.assign(
            role=["best" if r == best_run else "other" for r in runs["run"]]
        )
        base = alt.Chart(pts).encode(
            x=alt.X("run:O", title="run", axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y(
                "val_accuracy:Q",
                title="validation accuracy",
                scale=alt.Scale(domain=[0, 0.7]),
                axis=PCT,
            ),
            tooltip=[
                alt.Tooltip("run:O"),
                alt.Tooltip("val_accuracy:Q", format=".0%", title="accuracy"),
                alt.Tooltip("params:N", title="params tried"),
            ],
        )
        line = base.mark_line(color=C["blue"], strokeWidth=2.5)
        dots = base.mark_point(filled=True, size=110, color=C["blue"])
        ring = (
            base.transform_filter(alt.datum.role == "best")
            .mark_point(size=420, filled=False, stroke=C["blue_dark"], strokeWidth=2.5)
        )
        # Selective direct labels: start, the kept best, and the last run.
        labeled = pts[(pts["run"] == 1) | (pts["role"] == "best") | (pts["run"] == pts["run"].max())].copy()
        labeled["text"] = [
            f"{v:.0%}" + ("  ← kept" if role == "best" else "")
            for v, role in zip(labeled["val_accuracy"], labeled["role"])
        ]
        labels = (
            alt.Chart(labeled)
            .mark_text(dy=-16, fontSize=14, fontWeight=600, color=C["blue_dark"], align="left", dx=-8)
            .encode(x="run:O", y="val_accuracy:Q", text="text:N")
        )
        target_df = alt.Chart(alt.Data(values=[{"y": TARGET, "t": "target 60%"}]))
        target = target_df.mark_rule(strokeDash=[6, 5], color=C["muted"], strokeWidth=1.5).encode(y="y:Q")
        target_txt = target_df.mark_text(
            align="right", dy=-9, x="width", fontSize=12, color=C["muted"]
        ).encode(y="y:Q", text="t:N")

        best_acc = runs["val_accuracy"].max()
        last_acc = runs["val_accuracy"].iloc[-1]
        story = (
            f"run {best_run} peaked at {best_acc:.0%}, then over-tuning collapsed it — the Manager kept run {best_run}"
            if last_acc < best_acc
            else "each retune tries new hyperparameters from the Evaluator's schedule"
        )
        # NB: charts are displayed raw, not via mo.ui.altair_chart — marimo's
        # selection wrapper silently drops bar marks from layered charts.
        runs_view = themed(
            (line + dots + ring + labels + target + target_txt).properties(
                width=680,
                height=320,
                title=f"One point per Manager loop run — {story}",
            )
        )
    runs_view
    return


@app.cell
def _(C, LABELS, PCT, alt, evaluation, mo, pd, test_preds, themed):
    # --- Finding 2: where the 50% comes from — and where it doesn't. ----------
    per_class_acc = evaluation.get("class_accuracy", {})
    if not per_class_acc or test_preds.empty:
        failure_view = mo.md("_no evaluation yet_")
    else:
        ca = pd.DataFrame([{"label": k, "accuracy": v} for k, v in per_class_acc.items()])
        bars = (
            alt.Chart(ca)
            .mark_bar(cornerRadiusEnd=4, height=34, color=C["blue"])
            .encode(
                y=alt.Y("label:N", sort=LABELS, title=None,
                        axis=alt.Axis(labelFontSize=15, labelFontWeight=600, grid=False)),
                x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=PCT),
                tooltip=[alt.Tooltip("label:N"), alt.Tooltip("accuracy:Q", format=".0%")],
            )
        )
        bar_labels = bars.mark_text(align="left", dx=6, fontSize=15, fontWeight=600, color=C["blue_dark"]).encode(
            text=alt.Text("accuracy:Q", format=".0%")
        )
        per_class = themed(
            (bars + bar_labels).properties(
                width=300, height=210, title="Accuracy by true class — “down” has collapsed"
            )
        )

        cm = (
            test_preds.groupby(["label", "predicted_label"]).size().reset_index(name="count")
        )
        heat = (
            alt.Chart(cm)
            .mark_rect(cornerRadius=3)
            .encode(
                x=alt.X("predicted_label:N", sort=LABELS, title="predicted",
                        axis=alt.Axis(labelAngle=0, labelFontSize=14)),
                y=alt.Y("label:N", sort=LABELS, title="actual", axis=alt.Axis(labelFontSize=14)),
                color=alt.Color("count:Q", scale=alt.Scale(range=C["blues"]), legend=None),
                tooltip=["label", "predicted_label", "count"],
            )
        )
        heat_txt = heat.mark_text(fontSize=15, fontWeight=600).encode(
            text="count:Q",
            color=alt.condition(
                alt.datum.count > cm["count"].max() / 2, alt.value("white"), alt.value("#0b0b0b")
            ),
        )
        confusion = themed(
            (heat + heat_txt).properties(
                width=270, height=210, title="Nearly everything gets called “neutral”"
            )
        )
        failure_view = mo.hstack(
            [per_class, confusion],
            gap=2, widths="equal",
        )
    failure_view
    return


@app.cell
def _(C, CHANCE, PCT, alt, finetunes, mo, pd, themed):
    # --- Finding 3: fine-tuning — one epoch is enough; COVID breaks it. -------
    if not finetunes:
        ft_view = mo.md("_no fine-tune reports_")
    else:
        series_color = alt.Color(
            "series:N", title=None,
            scale=alt.Scale(domain=list(finetunes), range=[C["blue"], C["aqua"]]),
            legend=alt.Legend(orient="top-left"),
        )
        ep = pd.DataFrame(
            [
                {"series": name, "epoch": e["epoch"], "val_accuracy": e["val_accuracy"]}
                for name, rep in finetunes.items()
                for e in rep.get("epochs", [])
            ]
        )
        ep_base = alt.Chart(ep).encode(
            x=alt.X("epoch:O", title="epoch", axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y("val_accuracy:Q", title="validation accuracy",
                    scale=alt.Scale(domain=[0.25, 0.5]), axis=PCT),
            color=series_color,
            tooltip=["series", "epoch", alt.Tooltip("val_accuracy:Q", format=".1%")],
        )
        peak = ep.loc[ep.groupby("series")["val_accuracy"].idxmax()]
        peak_ring = (
            alt.Chart(peak)
            .mark_point(size=300, filled=False, strokeWidth=2.5)
            .encode(x="epoch:O", y="val_accuracy:Q", color=series_color)
        )
        curves = themed(
            (ep_base.mark_line(strokeWidth=2.5) + ep_base.mark_point(filled=True, size=70) + peak_ring)
            .properties(width=360, height=250,
                        title="Both runs peak at epoch 1 — more training just memorizes 2012–19")
        )

        te = pd.DataFrame(
            [{"series": n, "test_accuracy": r.get("test_accuracy")} for n, r in finetunes.items()]
        )
        te_bars = (
            alt.Chart(te)
            .mark_bar(cornerRadiusEnd=4, width=54)
            .encode(
                x=alt.X("series:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                y=alt.Y("test_accuracy:Q", scale=alt.Scale(domain=[0, 0.5]),
                        title="test accuracy", axis=PCT),
                color=series_color,
                tooltip=["series", alt.Tooltip("test_accuracy:Q", format=".1%")],
            )
        )
        te_labels = te_bars.mark_text(dy=-10, fontSize=14, fontWeight=600).encode(
            text=alt.Text("test_accuracy:Q", format=".0%"), color=series_color
        )
        chance_df = alt.Chart(alt.Data(values=[{"y": CHANCE, "t": "chance 33%"}]))
        chance = chance_df.mark_rule(strokeDash=[6, 5], color=C["muted"], strokeWidth=1.5).encode(y="y:Q")
        chance_txt = chance_df.mark_text(align="right", dy=-9, x="width", fontSize=12, color=C["muted"]).encode(
            y="y:Q", text="t:N"
        )
        covid = themed(
            (te_bars + te_labels + chance + chance_txt).properties(
                width=280, height=250,
                title="Tested on Feb–Jun 2020: COVID moves — below chance",
            )
        )
        ft_view = mo.hstack(
            [curves, covid], gap=2, widths="equal"
        )
    ft_view
    return


@app.cell
def _(mo):
    conf_threshold = mo.ui.slider(
        0.0, 1.0, value=0.0, step=0.05, label="**Only keep predictions with confidence ≥**", show_value=True
    )
    return (conf_threshold,)


@app.cell
def _(C, PCT, alt, conf_threshold, mo, pd, test_preds, themed):
    # --- Finding 4: confidence is honest — trade coverage for accuracy, live. -
    if test_preds.empty:
        conf_view = mo.md("_no predictions_")
    else:
        # Precomputed tradeoff curve; the slider drops a live marker on it.
        curve = pd.DataFrame(
            [
                {
                    "threshold": t / 20,
                    "accuracy": kept["correct"].mean() if len(kept) else None,
                    "coverage": len(kept) / len(test_preds),
                }
                for t in range(21)
                for kept in [test_preds[test_preds["confidence"] >= t / 20]]
            ]
        ).dropna()
        melted = curve.melt("threshold", var_name="measure", value_name="value")
        tradeoff = (
            alt.Chart(melted)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("threshold:Q", title="confidence cutoff"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=PCT),
                color=alt.Color(
                    "measure:N", title=None,
                    scale=alt.Scale(domain=["accuracy", "coverage"], range=[C["blue"], C["aqua"]]),
                    legend=alt.Legend(orient="top-right"),
                ),
                tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "measure",
                         alt.Tooltip("value:Q", format=".0%")],
            )
        )
        marker = (
            alt.Chart(alt.Data(values=[{"x": conf_threshold.value}]))
            .mark_rule(color=C["muted"], strokeWidth=1.5, strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        kept = test_preds[test_preds["confidence"] >= conf_threshold.value]
        acc_all = test_preds["correct"].mean()
        if len(kept):
            stats = mo.hstack(
                [
                    mo.stat(f"{kept['correct'].mean():.0%}", label="accuracy", bordered=True),
                    mo.stat(f"{len(kept) / len(test_preds):.0%}", label="coverage", bordered=True),
                    mo.stat(f"{kept['correct'].mean() - acc_all:+.0%}", label="vs. all rows", bordered=True),
                ],
                widths="equal",
            )
        else:
            stats = mo.md(f"_no rows with confidence ≥ {conf_threshold.value:.2f}_")
        conf_view = mo.vstack(
            [
                themed((tradeoff + marker).properties(
                    width=680, height=260,
                    title="Ask the model to be sure and it gets more accurate — on fewer headlines",
                )),
                conf_threshold,
                stats,
            ]
        )
    conf_view
    return


@app.cell
def _(explanations, mo):
    # --- The pipeline talks: one explanation card per prediction. -------------
    if explanations.empty:
        expl_view = mo.md("_no explanations yet (run Freddi)_")
    else:
        LABEL_TINT = {"up": "#1baf7a", "down": "#e34948", "neutral": "#898781"}

        def chip(text, color):
            return (
                f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
                f'font-size:12px;font-weight:650;color:{color};'
                f'border:1.5px solid {color};">{text}</span>'
            )

        def card(row):
            placeholder = "[PLACEHOLDER" in str(row["explanation"])
            text = str(row["explanation"]).split("]", 1)[-1].strip() if placeholder else row["explanation"]
            badge = (
                '<span style="font-size:11px;color:#898781;border:1px solid rgba(137,135,129,.4);'
                'border-radius:4px;padding:1px 6px;margin-left:8px;">placeholder — Ollama offline</span>'
                if placeholder else ""
            )
            ok = row["predicted_label"] == row["actual_label"]
            verdict = chip("correct", "#1baf7a") if ok else chip("was " + str(row["actual_label"]), "#e34948")
            return f"""
            <div style="flex:1;min-width:220px;padding:16px 18px;border-radius:12px;
                        border:1px solid rgba(137,135,129,.28);background:rgba(137,135,129,.06);">
              <div style="font-size:15px;font-weight:650;line-height:1.35;margin-bottom:10px;">
                “{row["article_title"]}”</div>
              <div style="margin-bottom:10px;">
                {chip("→ " + str(row["predicted_label"]), LABEL_TINT.get(row["predicted_label"], "#898781"))}
                &nbsp;{verdict}
              </div>
              <div style="font-size:13px;color:#898781;line-height:1.5;">{text}{badge}</div>
            </div>"""

        # One correct and one wrong call, so the class sees both faces.
        ok_rows = explanations[explanations["predicted_label"] == explanations["actual_label"]]
        bad_rows = explanations[explanations["predicted_label"] != explanations["actual_label"]]
        sample = [r for df in (ok_rows, bad_rows) for _, r in df.head(1).iterrows()]
        expl_view = mo.Html(
            '<div style="font-family:system-ui,sans-serif;">'
            '<div style="font-size:14px;color:#898781;margin-bottom:10px;">'
            f"Every prediction gets a plain-text justification — {len(explanations)} generated</div>"
            '<div style="display:flex;gap:14px;flex-wrap:wrap;">'
            + "".join(card(r) for r in sample)
            + "</div></div>"
        )
    expl_view
    return


@app.cell
def _(mo):
    # --- How it's built: five agents, one loop. --------------------------------
    mo.mermaid("""
    flowchart LR
        A["Processing<br/><small>headlines + prices</small>"] --> B["Classifier<br/><small>FinBERT</small>"]
        B --> C["Evaluator<br/><small>metrics + retune schedule</small>"]
        C --> D{"Manager<br/><small>threshold gate</small>"}
        D -- "retune (new params)" --> B
        D -- "proceed" --> E["Explanation<br/><small>Ollama</small>"]
        style D stroke:#2a78d6,stroke-width:2.5px
        style B stroke:#1baf7a,stroke-width:2px
    """)
    return


@app.cell
def _(DATA_DIR, decision, evaluation, explanations, mo, preds):
    # --- Appendix: raw data, out of the way. -----------------------------------
    mo.accordion(
        {
            f"Appendix — raw data (source: `{DATA_DIR}/`)": mo.vstack(
                [
                    mo.md("**predictions_test.csv** — search/sort built in"),
                    mo.ui.table(preds, selection=None, page_size=12),
                    mo.md("**explanations.csv**"),
                    mo.ui.table(explanations, selection=None, page_size=8),
                    mo.md("**decision.json / evaluation_report.json**"),
                    mo.hstack(
                        [
                            mo.json({k: v for k, v in decision.items() if k != "misclassified_ids"}),
                            mo.json({k: v for k, v in evaluation.items() if k != "misclassified_ids"}),
                        ],
                        widths="equal",
                    ),
                ]
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
