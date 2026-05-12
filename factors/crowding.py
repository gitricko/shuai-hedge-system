"""Crowding detection — pairwise correlations between factor return spreads.

Logic per factor:
  1. Each day, sort all stocks by that factor's score (using today's snapshot
     as a proxy — a true historical factor return needs back-scoring).
  2. Top quintile minus bottom quintile = synthetic factor return for that day.
  3. Roll across 60 days to get a return series.
  4. Pairwise correlation matrix across factors.

Compare to academic baselines:
    momentum/value   ≈ -0.30
    momentum/quality ≈ +0.10
Flag pairs whose realized 60-day correlation deviates by > 0.40 — that signals
factor crowding (typical pre-quant-quake behavior).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from factors._utils import universe_with_sectors

log = logging.getLogger(__name__)

ACADEMIC_BASELINES: dict[tuple[str, str], float] = {
    ("momentum", "value"):   -0.30,
    ("momentum", "quality"): +0.10,
    ("value", "quality"):    +0.20,
}
DEVIATION_THRESHOLD = 0.40
WINDOW_DAYS = 60


def _load_close_panel(tickers: list[str], days: int = WINDOW_DAYS + 1) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    sql = (
        f"SELECT ticker, date, adj_close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) ORDER BY date"
    )
    with conn_ctx() as cn:
        df = pd.read_sql_query(sql, cn, params=tickers)
    if df.empty:
        return pd.DataFrame()
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return panel.tail(days)


def _factor_quintile_spread(panel: pd.DataFrame, scores: pd.Series) -> pd.Series:
    """Top-quintile - bottom-quintile equally-weighted daily return."""
    rets = panel.pct_change().dropna(how="all")
    top_cut = scores.quantile(0.80)
    bot_cut = scores.quantile(0.20)
    top = scores[scores >= top_cut].index
    bot = scores[scores <= bot_cut].index
    top_ret = rets[top.intersection(rets.columns)].mean(axis=1)
    bot_ret = rets[bot.intersection(rets.columns)].mean(axis=1)
    return (top_ret - bot_ret).rename("spread")


def detect(scored_universe: pd.DataFrame) -> dict:
    """Return crowding diagnostics: pairwise correlations + flagged pairs.

    `scored_universe` is the DataFrame from factors.composite.compute_all().
    """
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()
    panel = _load_close_panel(tickers)
    if panel.empty:
        return {"correlations": pd.DataFrame(), "flags": []}

    spreads = pd.DataFrame()
    for factor in ["momentum", "value", "quality", "growth", "revisions"]:
        if factor in scored_universe.columns:
            spreads[factor] = _factor_quintile_spread(panel, scored_universe[factor])

    if spreads.empty or spreads.shape[1] < 2:
        return {"correlations": pd.DataFrame(), "flags": []}

    corr = spreads.corr()
    flags: list[dict] = []
    for (a, b), baseline in ACADEMIC_BASELINES.items():
        if a in corr.index and b in corr.columns:
            actual = corr.loc[a, b]
            if abs(actual - baseline) > DEVIATION_THRESHOLD:
                flags.append({
                    "pair": f"{a}/{b}",
                    "actual": round(float(actual), 3),
                    "baseline": baseline,
                    "deviation": round(float(actual - baseline), 3),
                })

    return {"correlations": corr, "flags": flags}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from factors.composite import compute_all

    su = compute_all(save=False)
    diag = detect(su)
    print("Pairwise correlations:")
    print(diag["correlations"])
    print("\nCrowding flags:")
    for f in diag["flags"]:
        print(f"  {f['pair']:<20} actual={f['actual']:+.2f} baseline={f['baseline']:+.2f} dev={f['deviation']:+.2f}")
