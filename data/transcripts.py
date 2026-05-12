"""Earnings call transcripts via Financial Modeling Prep (optional).

If FMP_API_KEY is set in .env, fetches the latest transcript for each
candidate ticker. Without a key, gracefully no-ops with an INFO log.

Layer 3's earnings analyzer reads from the `earnings_transcripts` table.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import requests
from tqdm import tqdm

from cache.db import conn_ctx, init_schema, upsert_many

log = logging.getLogger(__name__)

FMP_TRANSCRIPT_URL = "https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"


def fmp_available() -> bool:
    return bool(os.environ.get("FMP_API_KEY"))


def _fetch_latest(ticker: str, api_key: str) -> dict | None:
    url = FMP_TRANSCRIPT_URL.format(ticker=ticker)
    resp = requests.get(url, params={"apikey": api_key, "limit": 1}, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


def refresh(tickers: Iterable[str]) -> int:
    """Fetch transcripts for the given (typically candidate) tickers only.

    Returns rows written. Skips silently if no FMP key is configured.
    """
    init_schema()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        log.info("FMP_API_KEY not set — skipping transcript pull")
        return 0

    tickers = list(tickers)
    rows: list[tuple] = []
    for ticker in tqdm(tickers, desc="transcripts"):
        try:
            entry = _fetch_latest(ticker, api_key)
        except Exception as exc:
            log.warning("FMP fetch failed %s: %s", ticker, exc)
            continue
        if not entry:
            continue
        rows.append((
            ticker,
            int(entry.get("year") or 0),
            int(entry.get("quarter") or 0),
            entry.get("date"),
            entry.get("content") or "",
            "FMP",
        ))

    if rows:
        with conn_ctx() as conn:
            upsert_many(
                conn,
                "earnings_transcripts",
                ["ticker", "fiscal_year", "fiscal_quarter", "call_date", "transcript", "source"],
                rows,
            )
    log.info("transcripts fetched: %d", len(rows))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh(["AAPL", "MSFT"])
