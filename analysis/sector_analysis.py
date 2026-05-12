"""Per-sector aggregation across all four analyzers.

For a given GICS sector: gather all cached Claude analyses, ask Claude to
rank them by fundamental quality + positioning, surface a top long idea
and top short idea, and write a sector outlook.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from cache.db import conn_ctx
from analysis.api_client import AnalysisClient

log = logging.getLogger(__name__)

ANALYZER = "sector"

SYSTEM_PROMPT = """You are the sector lead at Meridian Capital Partners.
You synthesize per-ticker Claude analyses (earnings, filing, risk, insider)
into a sector view: who is best positioned, who is most exposed, and what
the sector-level macro tilt is for the next 90 days.

Return ONLY valid JSON with this exact structure:
{
  "rankings": [
    {"ticker": "...", "rank": int, "rationale": "1-2 sentences"}
  ],
  "top_long_idea": {"ticker": "...", "rationale": "2-3 sentences"},
  "top_short_idea": {"ticker": "...", "rationale": "2-3 sentences"},
  "sector_outlook": "3-4 sentences"
}"""


def _gather_for_sector(sector: str) -> dict[str, dict[str, dict]]:
    """Return {ticker: {analyzer: result_dict}} for all tickers in `sector`
    that have at least one cached analysis."""
    with conn_ctx() as conn:
        rows = conn.execute(
            """
            SELECT a.ticker, a.analyzer, a.result_json
            FROM analysis_results a
            JOIN universe u ON u.ticker = a.ticker
            WHERE u.gics_sector = ?
            """,
            (sector,),
        ).fetchall()
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        try:
            out.setdefault(r["ticker"], {})[r["analyzer"]] = json.loads(r["result_json"])
        except json.JSONDecodeError:
            continue
    return out


def analyze(sector: str, client: Optional[AnalysisClient] = None) -> dict | None:
    by_ticker = _gather_for_sector(sector)
    if not by_ticker:
        log.info("[sector] no cached analyses for %s", sector)
        return None

    client = client or AnalysisClient()
    payload = json.dumps(by_ticker, indent=2)[:80_000]
    user = f"Sector: {sector}\n\n--- PER-TICKER CLAUDE ANALYSES ---\n{payload}"
    return client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer=ANALYZER, ticker=sector,
    )
