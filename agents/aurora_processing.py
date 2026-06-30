"""
Processing Agent
Owner: Aurora Ruci

Reads data/fnspid_raw.csv, fetches closing prices via yfinance, assigns
price-movement labels, and writes data/processed_data.csv.

Exports
-------
ProcessingAgent   Agent subclass — callers do ProcessingAgent().run()
processing_node   raw LangGraph node used inside ProcessingAgent.build_graph

Usage (standalone test):
    python agents/aurora_processing.py --threshold 0.01
"""

import os
import pickle
import sys
import pandas as pd
import yfinance as yf
from langgraph.graph import StateGraph, START, END

# Allow running as `python agents/aurora_processing.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.state import PipelineState
from agents.base import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_price_cache(tickers, date_min, date_max):
    cache, failed = {}, []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=date_min, end=date_max)
            if hist.empty:
                print(f"  EMPTY  {ticker}")
                failed.append(ticker)
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None).normalize()
            else:
                hist.index = hist.index.normalize()
            cache[ticker] = hist["Close"]
        except Exception as e:
            print(f"  FAILED {ticker}: {e}")
            failed.append(ticker)
    print(f"\nCached {len(cache)} / {len(tickers)} tickers. Failed: {failed or 'none'}")
    return cache


def _get_t_and_t1(ticker, date_str, cache):
    prices = cache.get(ticker)
    if prices is None or prices.empty:
        return None, None
    date = pd.Timestamp(date_str)
    future = prices.index[prices.index >= date]
    if len(future) < 2:
        return None, None
    return float(prices[future[0]]), float(prices[future[1]])


def _assign_label(pct_change, threshold):
    if pct_change > threshold * 100:
        return "up"
    elif pct_change < -threshold * 100:
        return "down"
    return "neutral"


# ---------------------------------------------------------------------------
# Processing node
# ---------------------------------------------------------------------------

def processing_node(state: PipelineState) -> dict:
    """LangGraph node — runs the full processing pipeline and returns the
    output file path as `processed_data_path` in the state."""

    threshold = state.get("threshold", 0.01)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = state.get("data_dir") or os.path.join(repo_root, "data")

    raw_path   = os.path.join(data_dir, "fnspid_raw.csv")
    cache_path = os.path.join(data_dir, "price_cache.pkl")
    out_path   = os.path.join(data_dir, "processed_data.csv")

    # 1. Load raw news data
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"fnspid_raw.csv not found in {data_dir}. "
            "Make sure data/fnspid_raw.csv exists in the repo."
        )
    print("[processing] Loading raw data ...")
    df = pd.read_csv(raw_path)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")

    # 2. Fetch or load price cache
    tickers = sorted(df["ticker"].unique().tolist())
    if os.path.exists(cache_path):
        print("[processing] Loading price cache ...")
        with open(cache_path, "rb") as f:
            price_cache = pickle.load(f)
        print(f"  {len(price_cache)} tickers in cache")
    else:
        date_min = pd.to_datetime(df["date"].min()) - pd.Timedelta(days=5)
        date_max = pd.to_datetime(df["date"].max()) + pd.Timedelta(days=10)
        print(f"[processing] Fetching prices for {len(tickers)} tickers ...")
        price_cache = _fetch_price_cache(tickers, date_min, date_max)
        with open(cache_path, "wb") as f:
            pickle.dump(price_cache, f)
        print(f"  Price cache saved to {cache_path}")

    # 3. Assign prices and labels
    print("[processing] Assigning prices and labels ...")
    results = [_get_t_and_t1(row["ticker"], row["date"], price_cache)
               for _, row in df.iterrows()]

    df["price_t"]    = [round(r[0], 2) if r[0] is not None else None for r in results]
    df["price_t1"]   = [round(r[1], 2) if r[1] is not None else None for r in results]
    df["pct_change"] = [
        round((r[1] - r[0]) / r[0] * 100, 4) if r[0] and r[1] else None
        for r in results
    ]
    df["label"] = [
        _assign_label(p, threshold) if pd.notna(p) else None
        for p in df["pct_change"]
    ]

    # 4. Drop rows with no price data
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} rows with no price data. Final: {len(df):,} rows")

    # 5. Drop outliers (1st–99th percentile).
    # IQR 1.5× was cutting ~10% of data because stock returns are fat-tailed;
    # percentile fences keep exactly 98% of rows regardless of distribution shape.
    lo, hi = df["pct_change"].quantile(0.01), df["pct_change"].quantile(0.99)
    before_outliers = len(df)
    df = df[
        (df["pct_change"] >= lo) &
        (df["pct_change"] <= hi)
    ].reset_index(drop=True)
    print(f"  Outliers dropped: {before_outliers - len(df):,}. Final: {len(df):,} rows")

    # 6. Export (Handoff 1 schema from data_contracts.md)
    output_cols = [
        "article_id", "date", "ticker", "article_title",
        "price_t", "price_t1", "pct_change", "label",
    ]
    df[output_cols].to_csv(out_path, index=False)
    print(f"[processing] Written to {out_path}")

    label_counts = df["label"].value_counts().to_dict()
    print(f"  Labels: {label_counts}")

    return {"processed_data_path": out_path}


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ProcessingAgent(Agent):
    def build_graph(self, checkpointer):
        builder = StateGraph(PipelineState)
        builder.add_node("processing", processing_node)
        builder.add_edge(START, "processing")
        builder.add_edge("processing", END)
        return builder.compile(checkpointer=checkpointer)

    def run(self, **inputs) -> dict:
        return self._invoke(inputs)


# ---------------------------------------------------------------------------
# CLI — builds agent only when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Processing Agent as a LangGraph node."
    )
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Price-change threshold in decimal form (default: 0.01 = ±1%%)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory containing fnspid_raw.csv (default: data/)")
    args = parser.parse_args()

    agent = ProcessingAgent()
    final_state = agent.run(
        threshold=args.threshold,
        data_dir=args.data_dir,
    )
    print("\nOutput file:", final_state["processed_data_path"])
