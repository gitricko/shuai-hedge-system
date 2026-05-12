"""Layer 2 quant + Layer 3 Claude blend.

Final blend per spec:
  60% quantitative composite (Layer 2)
  40% Claude fundamental (mean of available analyzer scores, normalized 0-100)

If a ticker has zero Claude analyses available, its combined score is just
the quant composite — no penalty for missing AI input. After blending, we
re-rank within sector.
"""
from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from cache.db import REPO_ROOT, conn_ctx
from factors._utils import re_rank_within_sector, universe_with_sectors

log = logging.getLogger(__name__)

QUANT_WEIGHT = 0.60
CLAUDE_WEIGHT = 0.40
ANALYZERS = ("earnings", "filing", "risk", "insider")
OUTPUT_CSV = REPO_ROOT / "output" / "combined_score_latest.csv"


def _claude_score_per_ticker() -> pd.Series:
    """Mean Claude score (0-100) per ticker across available analyzers."""
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT ticker, analyzer, result_json FROM analysis_results "
            "WHERE analyzer IN (?, ?, ?, ?)",
            ANALYZERS,
        ).fetchall()

    scores: dict[str, list[float]] = {}
    for r in rows:
        try:
            data = json.loads(r["result_json"])
        except json.JSONDecodeError:
            continue
        s = _normalize_score(r["analyzer"], data)
        if s is not None:
            scores.setdefault(r["ticker"], []).append(s)

    return pd.Series({t: float(np.mean(v)) for t, v in scores.items()})


def _normalize_score(analyzer: str, data: dict) -> float | None:
    """Map each analyzer's output to a single 0-100 fundamental score."""
    if analyzer == "earnings":
        scores = data.get("scores", {})
        if not scores:
            return None
        # Six categories on 1-10; mean * 10 -> 0-100
        return float(np.mean(list(scores.values()))) * 10.0
    if analyzer == "filing":
        eq = data.get("earnings_quality_score")
        bs = data.get("balance_sheet_score")
        vals = [v for v in (eq, bs) if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else None
    if analyzer == "risk":
        sev = (data.get("risk_severity") or "").upper()
        # Lower severity = higher score for the long book
        return {"LOW": 80.0, "MEDIUM": 50.0, "HIGH": 20.0}.get(sev)
    if analyzer == "insider":
        sig = (data.get("signal_strength") or "").upper()
        return {
            "STRONG_BUY": 90.0, "BUY": 70.0, "NEUTRAL": 50.0,
            "SELL": 30.0, "STRONG_SELL": 10.0,
        }.get(sig)
    return None


def compute(scored: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build combined scores. `scored` is the Layer 2 output; if None, read
    the most recent row per ticker from `scored_universe`."""
    sectors = universe_with_sectors()
    if scored is None:
        with conn_ctx() as conn:
            scored = pd.read_sql_query(
                "SELECT * FROM scored_universe su WHERE score_date = "
                "(SELECT MAX(score_date) FROM scored_universe WHERE ticker = su.ticker)",
                conn,
            ).set_index("ticker")
    if scored is None or scored.empty:
        log.error("No Layer 2 scored_universe data; run scoring first.")
        return pd.DataFrame()

    quant = scored["composite"].astype(float).reindex(sectors.index)
    claude = _claude_score_per_ticker().reindex(sectors.index)

    use_blend = claude.notna()
    blended = pd.Series(index=sectors.index, dtype=float)
    blended[use_blend] = QUANT_WEIGHT * quant[use_blend] + CLAUDE_WEIGHT * claude[use_blend]
    blended[~use_blend] = quant[~use_blend]   # 100% quant when no Claude input

    final = re_rank_within_sector(blended, sectors).fillna(50.0)

    out = pd.DataFrame({
        "sector": sectors,
        "quant_composite": quant,
        "claude_score": claude,
        "blended_raw": blended,
        "final_score": final,
    }).sort_values("final_score", ascending=False)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index_label="ticker")
    log.info("Wrote %d rows to %s", len(out), OUTPUT_CSV)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
