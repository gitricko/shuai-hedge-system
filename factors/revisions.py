"""Estimate Revisions factor — 3 sub-factors.

  1. 30-day change in consensus next-quarter EPS
  2. 60-day change
  3. 90-day change

Degenerate (all scores = 50) until ~30 days of snapshots accumulate. Equal-
weights whichever deltas are computable today.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

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


def _snapshot_panel() -> pd.DataFrame:
    """Return wide panel: ticker × snapshot_date of forward_eps."""
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, snapshot_date, forward_eps FROM analyst_estimates",
            cn,
        )
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index="ticker", columns="snapshot_date", values="forward_eps", aggfunc="last"
    ).sort_index(axis=1)


def _delta(panel: pd.DataFrame, days: int) -> pd.Series:
    """Latest value minus value `days` ago, divided by abs(prior)."""
    if panel.empty:
        return pd.Series(dtype=float)
    cols = pd.to_datetime(panel.columns)
    latest = panel.iloc[:, -1]
    cutoff = cols[-1] - pd.Timedelta(days=days)
    eligible = panel.iloc[:, cols <= cutoff]
    if eligible.shape[1] == 0:
        return pd.Series(np.nan, index=panel.index)
    prior = eligible.iloc[:, -1]
    return ((latest - prior) / prior.abs()).replace([np.inf, -np.inf], np.nan)


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    panel = _snapshot_panel().reindex(tickers)
    raw = pd.DataFrame(index=tickers)
    available_windows: list[str] = []
    for col, days in [("rev_30d", 30), ("rev_60d", 60), ("rev_90d", 90)]:
        d = _delta(panel, days)
        if d.notna().sum() > 0:
            raw[col] = d
            available_windows.append(col)
        else:
            log.info("Revisions: %d-day window has no eligible snapshots — degenerate", days)

    if not available_windows:
        log.warning("Revisions factor degenerate (no historical snapshots yet)")
        return pd.DataFrame({"revisions": NEUTRAL}, index=tickers)

    scored = pd.DataFrame(index=tickers)
    for col in available_windows:
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    scored["revisions"] = equal_weight_subfactors(scored, available_windows)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
