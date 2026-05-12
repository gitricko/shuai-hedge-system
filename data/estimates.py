"""Analyst estimate snapshots from yfinance.

The estimate-revisions factor (Layer 2) needs >=30 days of daily snapshots to
compute 30/60/90-day deltas; this module appends one row per ticker per day.
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


def _f(x):
    try:
        return float(x) if x is not None else None
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
    for ticker in tqdm(throttled_iter(tickers, target_rps), total=len(tickers), desc="estimates"):
        try:
            info = _fetch_info(ticker)
        except Exception as exc:
            log.warning("info fetch failed %s: %s", ticker, exc)
            continue
        rows.append((
            ticker,
            today,
            _f(info.get("forwardEps")),
            _f(info.get("forwardPE")),
            _f(info.get("targetMeanPrice")),
            _f(info.get("targetHighPrice")),
            _f(info.get("targetLowPrice")),
            _f(info.get("recommendationMean")),
            int(info.get("numberOfAnalystOpinions") or 0) if info.get("numberOfAnalystOpinions") else None,
        ))

    if rows:
        with conn_ctx() as conn:
            upsert_many(
                conn,
                "analyst_estimates",
                ["ticker", "snapshot_date", "forward_eps", "forward_pe",
                 "target_mean", "target_high", "target_low",
                 "recommendation_mean", "num_analyst_opinions"],
                rows,
            )
    log.info("analyst_estimates snapshots: %d", len(rows))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh()
