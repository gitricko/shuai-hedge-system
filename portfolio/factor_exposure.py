"""Factor exposure & spread monitor.

For a given target portfolio (signed weights), compute the weighted-average
factor score across the long book and short book. Flag any factor whose
long-short spread exceeds 1 std dev from its historical baseline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cache.db import conn_ctx

log = logging.getLogger(__name__)

FACTORS = ("momentum", "quality", "value", "revisions",
           "insider", "growth", "short_int", "institutional")


def _latest_scores() -> pd.DataFrame:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su WHERE score_date = "
            "(SELECT MAX(score_date) FROM scored_universe)",
            conn,
        )
    return df.set_index("ticker") if not df.empty else df


def _historical_spreads(window_days: int = 60) -> pd.Series:
    """Approximate historical std-dev of long-short spreads per factor.
    Uses cross-sectional std-dev across all tickers as a proxy when no
    history is yet available."""
    df = _latest_scores()
    if df.empty:
        return pd.Series({f: 20.0 for f in FACTORS})
    return df[list(FACTORS)].std()


def compute(weights: pd.Series) -> pd.DataFrame:
    """Returns DataFrame indexed by factor with long/short exposures, spread,
    historical std, and z-score."""
    scores = _latest_scores()
    if scores.empty:
        log.warning("No scored_universe — factor_exposure skipped")
        return pd.DataFrame()

    aligned = scores.reindex(weights.index)
    long_w = weights[weights > 0]
    short_w = weights[weights < 0].abs()

    rows = []
    historical_std = _historical_spreads()
    for f in FACTORS:
        if f not in aligned.columns:
            continue
        s = aligned[f]
        long_avg = (long_w * s.reindex(long_w.index)).sum() / max(long_w.sum(), 1e-9)
        short_avg = (short_w * s.reindex(short_w.index)).sum() / max(short_w.sum(), 1e-9)
        spread = long_avg - short_avg
        std = float(historical_std.get(f, 20.0)) or 20.0
        rows.append({
            "factor": f,
            "long_book_avg": float(long_avg),
            "short_book_avg": float(short_avg),
            "spread": float(spread),
            "historical_std": std,
            "z_score": float(spread / std) if std else 0.0,
            "alert": abs(spread) > std,
        })
    return pd.DataFrame(rows).set_index("factor")
