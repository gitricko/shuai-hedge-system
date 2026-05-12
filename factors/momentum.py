"""Momentum factor — 6 sub-factors.

All inputs come from the `daily_prices` table populated by Layer 1.

  1. 12-1 month return (skip recent month to avoid 1-month reversal)
  2. 6-month return
  3. 3-month return
  4. Acceleration: recent 3m return minus older 3m return
  5. 52-week-high proximity (close / 252-day high) — George & Hwang 2004
  6. Relative strength vs sector ETF: stock 6m minus sector ETF 6m

Each sub-factor is sector-percentile-ranked 0-100 then averaged.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from data.universe import get_sector_map
from factors._utils import (
    NEUTRAL,
    SECTOR_TO_ETF,
    equal_weight_subfactors,
    fill_neutral,
    sector_percentile_rank,
    universe_with_sectors,
)

log = logging.getLogger(__name__)

# Trading-day approximations
M1, M3, M6, M12 = 21, 63, 126, 252


def _load_close_panel(tickers: list[str]) -> pd.DataFrame:
    """Wide ticker × date adj_close panel (most recent ~400 trading days)."""
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
    return panel


def _return_n_days(panel: pd.DataFrame, n: int) -> pd.Series:
    if len(panel) < n + 1:
        return pd.Series(np.nan, index=panel.columns)
    last = panel.iloc[-1]
    prior = panel.iloc[-(n + 1)]
    return (last / prior) - 1


def _return_skip(panel: pd.DataFrame, skip: int, total: int) -> pd.Series:
    """Return from t-total to t-skip."""
    if len(panel) < total + 1:
        return pd.Series(np.nan, index=panel.columns)
    end = panel.iloc[-(skip + 1)]
    start = panel.iloc[-(total + 1)]
    return (end / start) - 1


def _proximity_to_52w_high(panel: pd.DataFrame) -> pd.Series:
    if len(panel) < M12:
        return pd.Series(np.nan, index=panel.columns)
    window = panel.iloc[-M12:]
    last = panel.iloc[-1]
    high = window.max()
    return last / high


def compute(score_date: str | None = None) -> pd.DataFrame:
    """Return DataFrame indexed by ticker with 6 sub-factor scores + composite.

    All scores are 0-100 sector-percentile ranks.
    """
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()
    etfs = sorted(set(SECTOR_TO_ETF.values()))

    panel = _load_close_panel(tickers + etfs)
    if panel.empty:
        log.warning("No price data — momentum scores will be neutral")
        empty = pd.DataFrame({c: NEUTRAL for c in [
            "ret_12_1m", "ret_6m", "ret_3m", "acceleration",
            "prox_52w_high", "rel_strength", "momentum",
        ]}, index=tickers)
        return empty

    stock_panel = panel.reindex(columns=tickers)

    raw = pd.DataFrame(index=tickers)
    raw["ret_12_1m"] = _return_skip(stock_panel, skip=M1, total=M12)
    raw["ret_6m"] = _return_n_days(stock_panel, M6)
    raw["ret_3m"] = _return_n_days(stock_panel, M3)
    raw["acceleration"] = (
        _return_n_days(stock_panel, M3) - _return_skip(stock_panel, skip=M3, total=2 * M3)
    )
    raw["prox_52w_high"] = _proximity_to_52w_high(stock_panel)

    # Relative strength: stock 6m return minus its sector ETF 6m return.
    etf_6m = _return_n_days(panel.reindex(columns=etfs), M6)
    rel = pd.Series(np.nan, index=tickers)
    for ticker, sector in sectors.items():
        etf = SECTOR_TO_ETF.get(sector)
        if etf and etf in etf_6m.index:
            rel[ticker] = raw["ret_6m"].get(ticker, np.nan) - etf_6m[etf]
    raw["rel_strength"] = rel

    # Sector-relative percentile rank for each sub-factor.
    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    scored["momentum"] = equal_weight_subfactors(scored, raw.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
    print(f"\n{len(out)} tickers scored")
