"""Growth factor — 5 sub-factors.

  1. Revenue growth YoY
  2. Earnings growth YoY
  3. Revenue growth acceleration (latest YoY minus 4Q-ago YoY)
  4. R&D intensity (R&D / revenue) — high R&D in tech/health tends to outperform
  5. Free cash flow growth YoY — harder to manipulate than earnings

Layer 1's `fundamental_ratios` table already pre-computes revenue/earnings YoY
and rd_intensity, so we mostly read from there.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from factors._utils import (
    NEUTRAL,
    equal_weight_subfactors,
    fill_neutral,
    sector_percentile_rank,
    universe_with_sectors,
)

log = logging.getLogger(__name__)


def _latest_ratio(ratio_name: str, period_type: str = "Q") -> pd.Series:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            """
            SELECT ticker, value FROM fundamental_ratios fr
            WHERE ratio = ? AND period_type = ?
              AND period_end = (
                  SELECT MAX(period_end) FROM fundamental_ratios
                  WHERE ticker = fr.ticker AND ratio = fr.ratio AND period_type = fr.period_type
              )
            """,
            cn, params=[ratio_name, period_type],
        )
    return df.set_index("ticker")["value"] if not df.empty else pd.Series(dtype=float)


def _ratio_n_periods_ago(ratio_name: str, n: int, period_type: str = "Q") -> pd.Series:
    """Value of ratio from n quarters ago, per ticker."""
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, period_end, value FROM fundamental_ratios "
            "WHERE ratio = ? AND period_type = ? ORDER BY ticker, period_end DESC",
            cn, params=[ratio_name, period_type],
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["rk"] = df.groupby("ticker").cumcount()
    return df[df["rk"] == n].set_index("ticker")["value"]


def _fcf_growth_yoy() -> pd.Series:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, period_end, value FROM fundamentals "
            "WHERE line_item = 'Free Cash Flow' AND period_type = 'Q' "
            "ORDER BY ticker, period_end DESC",
            cn,
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["rk"] = df.groupby("ticker").cumcount()
    pivot = df[df["rk"].isin([0, 4])].pivot(index="ticker", columns="rk", values="value")
    pivot.columns = ["latest", "prior"]
    return ((pivot["latest"] - pivot["prior"]) / pivot["prior"].abs()).replace([np.inf, -np.inf], np.nan)


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    raw = pd.DataFrame(index=tickers)
    raw["rev_growth_yoy"] = _latest_ratio("revenue_growth_yoy").reindex(tickers)
    raw["earn_growth_yoy"] = _latest_ratio("earnings_growth_yoy").reindex(tickers)

    rev_now = _latest_ratio("revenue_growth_yoy").reindex(tickers)
    rev_4q = _ratio_n_periods_ago("revenue_growth_yoy", n=4).reindex(tickers)
    raw["rev_acceleration"] = rev_now - rev_4q

    raw["rd_intensity"] = _latest_ratio("rd_intensity").reindex(tickers)
    raw["fcf_growth_yoy"] = _fcf_growth_yoy().reindex(tickers)

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    scored["growth"] = equal_weight_subfactors(scored, raw.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
