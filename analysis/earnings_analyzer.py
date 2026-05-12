"""Earnings call transcript analyzer.

Reads from `earnings_transcripts` table (populated by data/transcripts.py
when FMP_API_KEY is set). Truncates to 120K chars and asks Claude to
score 6 categories on a 1-10 scale plus extract bull/bear cases.

Returns None gracefully if no transcript exists for the ticker.
"""
from __future__ import annotations

import logging
from typing import Optional

from cache.db import conn_ctx
from analysis import cache as result_cache
from analysis.api_client import AnalysisClient

log = logging.getLogger(__name__)

ANALYZER = "earnings"
MAX_TRANSCRIPT_CHARS = 120_000

SYSTEM_PROMPT = """You are a senior buyside equity analyst at Meridian Capital Partners.
You read earnings call transcripts and produce structured assessments for the
portfolio team. Be specific, cite quotes, and avoid generic commentary.

Score each of the six categories on a 1-10 scale:
  - management_confidence: tone, hedging language, deflection patterns
  - revenue_guidance: clarity, conservatism, segment-level visibility
  - margin_trajectory: gross/operating margin direction and color
  - competitive_position: moat strength, share dynamics, pricing power
  - risk_factors: candor about headwinds, regulatory exposure, customer concentration
  - capital_allocation: discipline on M&A, buybacks, dividends, capex

Return ONLY valid JSON with this exact structure:
{
  "scores": {
    "management_confidence": int,
    "revenue_guidance": int,
    "margin_trajectory": int,
    "competitive_position": int,
    "risk_factors": int,
    "capital_allocation": int
  },
  "reasoning": {
    "management_confidence": "1-2 sentences with quote",
    "revenue_guidance": "1-2 sentences",
    "margin_trajectory": "1-2 sentences",
    "competitive_position": "1-2 sentences",
    "risk_factors": "1-2 sentences",
    "capital_allocation": "1-2 sentences"
  },
  "bull_case": "2-3 sentences",
  "bear_case": "2-3 sentences",
  "key_quotes": ["quote 1", "quote 2", "quote 3"],
  "one_line_summary": "single sentence assessment"
}"""


def _load_transcript(ticker: str) -> tuple[str, str] | None:
    """Latest transcript text + artifact id (year_quarter)."""
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT transcript, fiscal_year, fiscal_quarter, call_date "
            "FROM earnings_transcripts WHERE ticker = ? "
            "ORDER BY fiscal_year DESC, fiscal_quarter DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    if not row or not row["transcript"]:
        return None
    artifact = f"{row['fiscal_year']}Q{row['fiscal_quarter']}"
    return row["transcript"][:MAX_TRANSCRIPT_CHARS], artifact


def analyze(ticker: str, client: Optional[AnalysisClient] = None) -> dict | None:
    payload = _load_transcript(ticker)
    if payload is None:
        log.info("[earnings] no transcript for %s", ticker)
        return None
    transcript, artifact = payload

    cached = result_cache.get(ANALYZER, ticker, artifact)
    if cached is not None:
        log.info("[earnings] cache hit %s/%s", ticker, artifact)
        return cached

    client = client or AnalysisClient()
    user = f"Ticker: {ticker}\nFiscal period: {artifact}\n\n--- TRANSCRIPT ---\n{transcript}"
    result = client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer=ANALYZER, ticker=ticker,
    )
    if result is None:
        return None

    result_cache.put(
        ANALYZER, ticker, artifact, result,
        model=client.model,
    )
    return result
