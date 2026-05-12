"""Markdown reports per LONG/SHORT candidate.

Saves to output/reports_{YYYYMMDD_HHMMSS}/{TICKER}.md with:
  * All Layer 2 sub-factor scores
  * All Claude summaries (one_line + bull/bear where applicable)
  * Upcoming earnings catalyst (from earnings_calendar)
  * Risk flags (from filing/risk analyzers)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from cache.db import REPO_ROOT, conn_ctx

log = logging.getLogger(__name__)


def _claude_summaries(ticker: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT analyzer, result_json FROM analysis_results WHERE ticker = ?",
            (ticker,),
        ).fetchall()
    for r in rows:
        try:
            out[r["analyzer"]] = json.loads(r["result_json"])
        except json.JSONDecodeError:
            continue
    return out


def _upcoming_earnings(ticker: str) -> str | None:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT earnings_date FROM earnings_calendar WHERE ticker = ? "
            "ORDER BY earnings_date LIMIT 1",
            (ticker,),
        ).fetchone()
    return row["earnings_date"] if row else None


def _scored_row(ticker: str) -> dict | None:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT * FROM scored_universe WHERE ticker = ? "
            "ORDER BY score_date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    return dict(row) if row else None


def render(ticker: str, side: str) -> str:
    s = _scored_row(ticker) or {}
    summaries = _claude_summaries(ticker)
    earnings = _upcoming_earnings(ticker)

    lines: list[str] = []
    lines.append(f"# {ticker} — {side} candidate")
    lines.append("")
    lines.append(f"**Sector:** {s.get('sector') or '—'}  ")
    lines.append(f"**Composite:** {s.get('composite', float('nan')):.1f}  ")
    if earnings:
        lines.append(f"**Upcoming earnings:** {earnings}  ")
    lines.append("")
    lines.append("## Layer 2 — Quantitative factor scores")
    lines.append("")
    lines.append("| Factor | Score |")
    lines.append("|---|---:|")
    for f in ("momentum", "quality", "value", "revisions", "insider", "growth", "short_int", "institutional"):
        v = s.get(f)
        if v is not None:
            lines.append(f"| {f} | {v:.1f} |")
    lines.append("")

    if summaries:
        lines.append("## Layer 3 — Claude AI analysis")
        lines.append("")
        for analyzer in ("earnings", "filing", "risk", "insider"):
            data = summaries.get(analyzer)
            if not data:
                continue
            lines.append(f"### {analyzer.capitalize()}")
            lines.append("")
            one_line = data.get("one_line_summary")
            if one_line:
                lines.append(f"> {one_line}")
                lines.append("")
            if "bull_case" in data:
                lines.append(f"**Bull case.** {data['bull_case']}")
                lines.append("")
            if "bear_case" in data:
                lines.append(f"**Bear case.** {data['bear_case']}")
                lines.append("")
            if "red_flags" in data:
                rf = data.get("red_flags") or []
                if rf:
                    lines.append("**Red flags:**")
                    for f in rf:
                        lines.append(f"- {f}")
                    lines.append("")
            if "material_risks" in data:
                mr = data.get("material_risks") or []
                if mr:
                    lines.append("**Material risks:**")
                    for f in mr[:5]:
                        lines.append(f"- {f}")
                    lines.append("")
            if "signal_strength" in data:
                lines.append(f"**Insider signal:** {data['signal_strength']} "
                             f"(confidence {data.get('confidence', '?')})")
                lines.append("")
    else:
        lines.append("## Layer 3 — Claude AI analysis")
        lines.append("")
        lines.append("_No Claude analyses available for this ticker._")
        lines.append("")

    return "\n".join(lines)


def write_reports(candidates: pd.DataFrame, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "output" / f"reports_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker, row in candidates.iterrows():
        side = row.get("side")
        if not side:
            continue
        md = render(str(ticker), str(side))
        (out_dir / f"{ticker}.md").write_text(md)

    log.info("Wrote %d reports to %s", len(candidates), out_dir)
    return out_dir
