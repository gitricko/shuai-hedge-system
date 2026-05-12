"""Forensic accounting analyzer.

Reads 8 quarters of fundamental metrics and asks Claude to look for earnings-
quality red flags: CFO/NI divergence, AR growing faster than revenue, balance
sheet deterioration, accruals manipulation.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from cache.db import conn_ctx
from analysis import cache as result_cache
from analysis.api_client import AnalysisClient

log = logging.getLogger(__name__)

ANALYZER = "filing"
N_QUARTERS = 8

SYSTEM_PROMPT = """You are a forensic accountant on the Meridian Capital
Partners short-research desk. You inspect 8 quarters of fundamental metrics
to detect earnings quality issues, balance-sheet deterioration, and accrual
manipulation. Concrete numbers and dated quarters beat generic commentary.

Score on 0-100 (higher is better quality):
  - earnings_quality_score: CFO vs NI alignment, recurring vs one-time items
  - balance_sheet_score: leverage, liquidity, working capital trends

Return ONLY valid JSON with this exact structure:
{
  "earnings_quality_score": int,
  "balance_sheet_score": int,
  "red_flags": ["specific red flag with quarter reference"],
  "green_flags": ["specific positive signal with quarter reference"],
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "one_line_summary": "single sentence assessment"
}"""


def _load_recent_quarters(ticker: str, n: int = N_QUARTERS) -> tuple[pd.DataFrame, str] | None:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT period_end, line_item, value FROM fundamentals "
            "WHERE ticker = ? AND period_type = 'Q' "
            "ORDER BY period_end DESC", conn, params=[ticker],
        )
    if df.empty:
        return None
    periods = sorted(df["period_end"].unique(), reverse=True)[:n]
    if len(periods) < 4:
        return None
    df = df[df["period_end"].isin(periods)]
    pivot = df.pivot_table(
        index="line_item", columns="period_end", values="value", aggfunc="last",
    )
    artifact = f"{periods[-1]}_to_{periods[0]}"
    return pivot, artifact


def analyze(ticker: str, client: Optional[AnalysisClient] = None) -> dict | None:
    payload = _load_recent_quarters(ticker)
    if payload is None:
        log.info("[filing] insufficient quarters for %s", ticker)
        return None
    pivot, artifact = payload

    cached = result_cache.get(ANALYZER, ticker, artifact)
    if cached is not None:
        log.info("[filing] cache hit %s/%s", ticker, artifact)
        return cached

    table = pivot.to_csv()  # compact, machine-readable
    client = client or AnalysisClient()
    user = (
        f"Ticker: {ticker}\nPeriods covered: {artifact}\n\n"
        f"--- 8Q FUNDAMENTALS (rows = line items, columns = period end) ---\n{table}"
    )
    result = client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer=ANALYZER, ticker=ticker,
    )
    if result is None:
        return None

    result_cache.put(ANALYZER, ticker, artifact, result, model=client.model)
    return result
