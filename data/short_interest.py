"""Daily short-interest snapshots from yfinance .info.

Stores one row per ticker per snapshot_date in the `short_interest` table.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import yfinance as yf
from tqdm import tqdm

from cache.db import conn_ctx, init_schema, upsert_many
from data._yf_utils import call_or_raise, throttled_iter, with_retry
from data.universe import get_universe

log = logging.getLogger(__name__)


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


@with_retry
def _fetch_info(ticker: str) -> dict:
    return call_or_raise(lambda: yf.Ticker(ticker).info or {})


def refresh(tickers: Iterable[str] | None = None, target_rps: float = 6.0) -> int:
    init_schema()
    if tickers is None:
        tickers = get_universe(include_benchmarks=False)
    tickers = list(tickers)
    today = date.today().isoformat()

    rows: list[tuple] = []
    for ticker in tqdm(throttled_iter(tickers, target_rps), total=len(tickers), desc="short_interest"):
        try:
            info = _fetch_info(ticker)
        except Exception as exc:
            log.warning("yf.info failed %s: %s", ticker, exc)
            continue
        rows.append((
            ticker,
            today,
            _safe_float(info.get("sharesShort")),
            _safe_float(info.get("shortRatio")),
            _safe_float(info.get("shortPercentOfFloat")),
            _safe_float(info.get("floatShares")),
        ))

    if rows:
        with conn_ctx() as conn:
            upsert_many(
                conn,
                "short_interest",
                ["ticker", "snapshot_date", "shares_short", "short_ratio", "short_pct_float", "float_shares"],
                rows,
            )
    log.info("short_interest snapshots: %d", len(rows))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh()
