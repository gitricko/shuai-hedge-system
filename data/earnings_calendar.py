"""Upcoming earnings dates for the next 30 days.

yfinance exposes per-ticker calendar info via Ticker.calendar (DataFrame in newer
versions, dict in older). We normalize both shapes.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from cache.db import conn_ctx, init_schema, upsert_many
from data._yf_utils import call_or_raise, throttled_iter, with_retry
from data.universe import get_universe

log = logging.getLogger(__name__)


def _extract_earnings_dates(cal) -> list[str]:
    """Return ISO date strings from whatever shape yfinance hands back."""
    if cal is None:
        return []
    if isinstance(cal, pd.DataFrame) and not cal.empty:
        if "Earnings Date" in cal.index:
            row = cal.loc["Earnings Date"]
            return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in row.dropna().tolist() if pd.notna(v)]
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date") or cal.get("earningsDate")
        if ed is None:
            return []
        if not isinstance(ed, (list, tuple)):
            ed = [ed]
        return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in ed]
    return []


@with_retry
def _fetch_calendar(ticker: str):
    return call_or_raise(lambda: yf.Ticker(ticker).calendar)


def refresh(tickers: Iterable[str] | None = None, lookahead_days: int = 30, target_rps: float = 6.0) -> int:
    init_schema()
    if tickers is None:
        tickers = get_universe(include_benchmarks=False)
    tickers = list(tickers)

    today = date.today()
    cutoff = today + timedelta(days=lookahead_days)
    today_iso = today.isoformat()

    rows: list[tuple] = []
    for ticker in tqdm(throttled_iter(tickers, target_rps), total=len(tickers), desc="earnings_cal"):
        try:
            cal = _fetch_calendar(ticker)
        except Exception as exc:
            log.warning("calendar fetch failed %s: %s", ticker, exc)
            continue
        for d_iso in _extract_earnings_dates(cal):
            try:
                d_obj = pd.Timestamp(d_iso).date()
            except Exception:
                continue
            if today <= d_obj <= cutoff:
                rows.append((ticker, d_iso, None, None, None, today_iso))

    if rows:
        with conn_ctx() as conn:
            upsert_many(
                conn,
                "earnings_calendar",
                ["ticker", "earnings_date", "eps_estimate", "revenue_estimate", "time_of_day", "snapshot_date"],
                rows,
            )
    log.info("earnings_calendar rows: %d", len(rows))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh()
