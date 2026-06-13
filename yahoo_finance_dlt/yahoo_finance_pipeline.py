"""Load Yahoo Finance OHLCV history into DuckDB with dlt."""

from typing import Iterator, Tuple

import dlt
import yfinance as yf

TICKERS = ("AAPL", "MSFT", "GOOG")


@dlt.resource(
    name="prices",
    primary_key=("ticker", "date"),
    write_disposition="merge",
)
def prices(
    tickers: Tuple[str, ...] = TICKERS,
    period: str = "1y",
    interval: str = "1d",
    cursor=dlt.sources.incremental("date"),
) -> Iterator[dict]:
    """Yield one row per ticker per bar (OHLCV)."""
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty:
            continue
        hist = hist.reset_index()  # move Date/Datetime index to a column
        date_col = "Datetime" if "Datetime" in hist.columns else "Date"
        for row in hist.to_dict("records"):
            yield {
                "ticker": ticker,
                "date": row[date_col].to_pydatetime(),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"],
                "dividends": row.get("Dividends"),
                "stock_splits": row.get("Stock Splits"),
            }


@dlt.resource(
    name="news",
    primary_key=("ticker", "id"),
    write_disposition="merge",
)
def news(
    tickers: Tuple[str, ...] = TICKERS,
    cursor=dlt.sources.incremental("pub_date"),
) -> Iterator[dict]:
    """Yield one row per ticker per news article."""
    for ticker in tickers:
        for item in yf.Ticker(ticker).news:
            c = item.get("content") or {}
            yield {
                "ticker": ticker,
                "id": item.get("id"),
                "title": c.get("title"),
                "summary": c.get("summary"),
                "content_type": c.get("contentType"),
                "pub_date": c.get("pubDate"),
                "url": (c.get("canonicalUrl") or {}).get("url"),
                "publisher": (c.get("provider") or {}).get("displayName"),
            }


@dlt.source(name="yahoo_finance")
def yahoo_finance(tickers: Tuple[str, ...] = TICKERS):
    yield prices(tickers=tickers)
    yield news(tickers=tickers)


if __name__ == "__main__":
    pipeline = dlt.pipeline(
        pipeline_name="yahoo_finance",
        destination="duckdb",
        dataset_name="market_data",
    )
    load_info = pipeline.run(yahoo_finance())
    print(load_info)
