"""Insider transactions interpreter.

Reads the last 90 days of Form 4 transactions from `insider_transactions`
and asks Claude to distinguish routine 10b5-1 selling from meaningful
open-market buying.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from cache.db import conn_ctx
from analysis import cache as result_cache
from analysis.api_client import AnalysisClient

log = logging.getLogger(__name__)

ANALYZER = "insider"
WINDOW_DAYS = 90

SYSTEM_PROMPT = """You are the head of insider-activity research at
Meridian Capital Partners. You distinguish routine 10b5-1 plan sales,
option exercises, and tax withholdings from meaningful open-market signal
trades — especially CEO/CFO open-market purchases. Net dollar flow alone
is not the answer; pattern and seniority matter.

Return ONLY valid JSON with this exact structure:
{
  "signal_strength": "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL",
  "confidence": int (0-100),
  "key_transactions": [
    {"insider": "name", "title": "...", "code": "P|S|...", "shares": int, "dollar_value": float, "date": "YYYY-MM-DD"}
  ],
  "reasoning": "2-3 sentences explaining the call",
  "one_line_summary": "single sentence assessment"
}"""


def _load_window(ticker: str) -> tuple[pd.DataFrame, str] | None:
    cutoff = (date.today() - timedelta(days=WINDOW_DAYS)).isoformat()
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT insider_name, insider_title, transaction_code, transaction_date, "
            "shares, price, is_officer, is_director "
            "FROM insider_transactions WHERE ticker = ? AND transaction_date >= ? "
            "ORDER BY transaction_date DESC",
            conn, params=[ticker, cutoff],
        )
    if df.empty:
        return None
    artifact = f"{cutoff}_to_{date.today().isoformat()}"
    return df, artifact


def analyze(ticker: str, client: Optional[AnalysisClient] = None) -> dict | None:
    payload = _load_window(ticker)
    if payload is None:
        log.info("[insider] no transactions for %s in last %dd", ticker, WINDOW_DAYS)
        return None
    df, artifact = payload

    cached = result_cache.get(ANALYZER, ticker, artifact)
    if cached is not None:
        log.info("[insider] cache hit %s/%s", ticker, artifact)
        return cached

    df["dollar_value"] = (df["shares"].fillna(0) * df["price"].fillna(0)).round(0)
    table = df.to_csv(index=False)

    client = client or AnalysisClient()
    user = (
        f"Ticker: {ticker}\nWindow: {artifact}\n\n--- FORM 4 TRANSACTIONS ---\n{table}"
    )
    result = client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer=ANALYZER, ticker=ticker,
    )
    if result is None:
        return None

    result_cache.put(ANALYZER, ticker, artifact, result, model=client.model)
    return result
