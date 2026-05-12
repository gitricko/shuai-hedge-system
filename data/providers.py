"""Provider abstraction layer.

Routes price/transcript/macro requests to the best configured backend:

  * Polygon  - daily prices (licensed exchange data) if POLYGON_API_KEY set
  * FMP      - transcripts + structured financials   if FMP_API_KEY set
  * FRED     - yield curve, credit spread, fed funds if FRED_API_KEY set
  * Fallback - yfinance for prices/fundamentals, SEC EDGAR for filings

This module is consulted by the other Layer 1 loaders to pick the active
provider for each data type. It does not itself orchestrate refreshes.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import requests

log = logging.getLogger(__name__)

POLYGON_AGG = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_}/{to}"
FMP_INCOME = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"


def active_price_provider() -> str:
    if os.environ.get("POLYGON_API_KEY"):
        log.info("Using Polygon for prices")
        return "polygon"
    log.info("Falling back to yfinance for prices")
    return "yfinance"


def active_transcript_provider() -> str:
    if os.environ.get("FMP_API_KEY"):
        return "fmp"
    return "none"


def active_macro_provider() -> str:
    if os.environ.get("FRED_API_KEY"):
        return "fred"
    return "none"


def polygon_daily_bars(ticker: str, start: date, end: date) -> list[dict]:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        return []
    url = POLYGON_AGG.format(ticker=ticker, from_=start.isoformat(), to=end.isoformat())
    resp = requests.get(url, params={"apiKey": key, "adjusted": "true", "limit": 50000}, timeout=30)
    if resp.status_code != 200:
        log.warning("polygon %s: %s", ticker, resp.status_code)
        return []
    out: list[dict] = []
    for bar in resp.json().get("results", []) or []:
        out.append({
            "ticker": ticker,
            "date": date.fromtimestamp(bar["t"] / 1000).isoformat(),
            "open": bar.get("o"), "high": bar.get("h"), "low": bar.get("l"),
            "close": bar.get("c"), "adj_close": bar.get("c"), "volume": bar.get("v"),
        })
    return out


def fmp_income_statement(ticker: str, period: str = "quarter", limit: int = 12) -> list[dict]:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        return []
    url = FMP_INCOME.format(ticker=ticker)
    resp = requests.get(url, params={"apikey": key, "period": period, "limit": limit}, timeout=30)
    if resp.status_code != 200:
        return []
    return resp.json() or []


def fred_series(series_id: str, start: Optional[date] = None) -> list[dict]:
    """e.g. BAMLH0A0HYM2 = high-yield credit spread, DGS10 = 10y treasury."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return []
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    if start:
        params["observation_start"] = start.isoformat()
    resp = requests.get(FRED_OBS, params=params, timeout=30)
    if resp.status_code != 200:
        return []
    return resp.json().get("observations", []) or []


def log_active_providers() -> None:
    log.info(
        "providers: prices=%s, transcripts=%s, macro=%s",
        active_price_provider(), active_transcript_provider(), active_macro_provider(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_active_providers()
