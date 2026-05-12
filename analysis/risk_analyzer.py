"""10-K Risk Factors analyzer.

Reads the latest cached 10-K from `sec_filings` (downloaded by Layer 1 to
cache/edgar/...), extracts the Risk Factors section, and asks Claude to
separate material risks from boilerplate, flag new risks vs the prior 10-K,
and rate severity.
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from cache.db import conn_ctx
from analysis import cache as result_cache
from analysis.api_client import AnalysisClient

log = logging.getLogger(__name__)

ANALYZER = "risk"
MAX_RISK_CHARS = 80_000
RISK_HEADER_RE = re.compile(r"item\s*1a[\.\s\-]*risk\s*factors", re.IGNORECASE)
NEXT_ITEM_RE = re.compile(r"item\s*1b[\.\s]", re.IGNORECASE)


class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


def _strip_html(raw: bytes) -> str:
    try:
        decoded = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    s = _Stripper()
    s.feed(decoded)
    text = s.text()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_risk_section(text: str) -> str:
    m = RISK_HEADER_RE.search(text)
    if not m:
        return text[:MAX_RISK_CHARS]
    start = m.end()
    end_match = NEXT_ITEM_RE.search(text, pos=start + 200)  # guard against "Item 1B" right next to header
    end = end_match.start() if end_match else len(text)
    return text[start:end][:MAX_RISK_CHARS]


SYSTEM_PROMPT = """You are an SEC-disclosure analyst at Meridian Capital
Partners. You read 10-K Risk Factors sections and separate material,
company-specific risks from generic boilerplate. Be specific.

Return ONLY valid JSON with this exact structure:
{
  "new_risks": ["risk newly added vs prior filing (if known)"],
  "material_risks": ["concrete, company-specific risk"],
  "boilerplate_percentage": int,
  "risk_severity": "LOW" | "MEDIUM" | "HIGH",
  "one_line_summary": "single sentence assessment"
}"""


def _load_latest_10k(ticker: str) -> tuple[str, str] | None:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT accession, local_path FROM sec_filings "
            "WHERE ticker = ? AND form_type = '10-K' AND local_path IS NOT NULL "
            "ORDER BY filing_date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    if not row:
        return None
    path = Path(row["local_path"])
    if not path.exists():
        return None
    raw = path.read_bytes()
    text = _strip_html(raw)
    risk = _extract_risk_section(text)
    if not risk.strip():
        return None
    return risk, row["accession"]


def analyze(ticker: str, client: Optional[AnalysisClient] = None) -> dict | None:
    payload = _load_latest_10k(ticker)
    if payload is None:
        log.info("[risk] no 10-K cached for %s", ticker)
        return None
    risk_text, artifact = payload

    cached = result_cache.get(ANALYZER, ticker, artifact)
    if cached is not None:
        log.info("[risk] cache hit %s/%s", ticker, artifact)
        return cached

    client = client or AnalysisClient()
    user = f"Ticker: {ticker}\nFiling accession: {artifact}\n\n--- RISK FACTORS ---\n{risk_text}"
    result = client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer=ANALYZER, ticker=ticker,
    )
    if result is None:
        return None

    result_cache.put(ANALYZER, ticker, artifact, result, model=client.model)
    return result
