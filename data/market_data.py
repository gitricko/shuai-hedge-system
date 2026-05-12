"""Daily OHLCV market data via yfinance.

3yr lookback, incremental updates: only fetch bars after the latest stored date
per ticker. Writes to `daily_prices` table.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from cache.db import conn_ctx, init_schema, upsert_many
from data.universe import get_benchmarks, get_universe

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 3 * 365
BATCH_SIZE = 50


def _last_stored_date(conn, ticker: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(date) AS d FROM daily_prices WHERE ticker = ?", (ticker,)
    ).fetchone()
    if not row or not row["d"]:
        return None
    return datetime.strptime(row["d"], "%Y-%m-%d").date()


def _fetch_batch(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Pull a multi-ticker frame from yfinance. Returns long-format DataFrame."""
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    if isinstance(df.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr not in df.columns.get_level_values(0):
                continue
            sub = df[tkr].dropna(how="all").reset_index()
            for r in sub.itertuples(index=False):
                rows.append(_row_dict(tkr, r))
    else:
        # Single ticker — flat columns
        sub = df.dropna(how="all").reset_index()
        tkr = tickers[0]
        for r in sub.itertuples(index=False):
            rows.append(_row_dict(tkr, r))
    return pd.DataFrame(rows)


def _row_dict(tkr: str, r) -> dict:
    return {
        "ticker": tkr,
        "date": pd.Timestamp(r.Date).strftime("%Y-%m-%d"),
        "open": float(r.Open) if pd.notna(r.Open) else None,
        "high": float(r.High) if pd.notna(r.High) else None,
        "low": float(r.Low) if pd.notna(r.Low) else None,
        "close": float(r.Close) if pd.notna(r.Close) else None,
        "adj_close": float(getattr(r, "Adj_Close", r.Close)) if pd.notna(r.Close) else None,
        "volume": float(r.Volume) if pd.notna(r.Volume) else None,
    }


def refresh(tickers: Iterable[str] | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> int:
    """Incrementally refresh daily prices.

    For each ticker: start from (last_stored + 1d) or (today - lookback_days)
    if absent. Returns the number of new bars inserted.
    """
    init_schema()
    if tickers is None:
        tickers = get_universe(include_benchmarks=False) + get_benchmarks()
    tickers = sorted(set(tickers))
    today = date.today()

    # Determine per-ticker start dates so we batch only those needing data.
    starts: dict[str, date] = {}
    with conn_ctx() as conn:
        for t in tickers:
            last = _last_stored_date(conn, t)
            start = last + timedelta(days=1) if last else today - timedelta(days=lookback_days)
            if start <= today:
                starts[t] = start

    if not starts:
        log.info("daily_prices already up-to-date for all tickers")
        return 0

    # Group by start-date so each batch shares a window.
    by_start: dict[date, list[str]] = {}
    for t, s in starts.items():
        by_start.setdefault(s, []).append(t)

    total = 0
    for start, tkrs in by_start.items():
        log.info("Fetching %d tickers from %s onward", len(tkrs), start)
        for i in tqdm(range(0, len(tkrs), BATCH_SIZE), desc=f"prices@{start}"):
            batch = tkrs[i : i + BATCH_SIZE]
            df = _fetch_batch(batch, start, today)
            if df.empty:
                continue
            rows = [
                (
                    row.ticker, row.date, row.open, row.high, row.low,
                    row.close, row.adj_close, row.volume,
                )
                for row in df.itertuples(index=False)
            ]
            with conn_ctx() as conn:
                upsert_many(
                    conn,
                    "daily_prices",
                    ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"],
                    rows,
                )
            total += len(rows)

    log.info("Inserted/updated %d daily price rows", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = refresh()
    print(f"Refreshed {n} bars.")
