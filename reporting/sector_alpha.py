"""Sector-relative performance — per-sector 90d alpha.

For each sector:
    alpha = avg_return_on_picks - sector_etf_return

Sum across sectors = total stock-selection alpha. Tracks how many
sectors are net winners vs losers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from factors._utils import SECTOR_TO_ETF

log = logging.getLogger(__name__)

WINDOW_DAYS = 90


def _period_return(panel: pd.DataFrame, days: int) -> pd.Series:
    if panel.empty or len(panel) < days + 1:
        return pd.Series(dtype=float)
    start = panel.iloc[-(days + 1)]
    end = panel.iloc[-1]
    return (end / start) - 1


def compute(window: int = WINDOW_DAYS) -> pd.DataFrame:
    """Returns DataFrame indexed by sector with avg_pick_return, etf_return, alpha."""
    with conn_ctx() as conn:
        positions = pd.read_sql_query("SELECT * FROM portfolio_positions", conn)
        prices = pd.read_sql_query(
            "SELECT ticker, date, adj_close FROM daily_prices ORDER BY date", conn,
        )
    if positions.empty or prices.empty:
        return pd.DataFrame()

    panel = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index().tail(window + 1)
    returns = _period_return(panel, window)

    out_rows = []
    for sector, group in positions.groupby("sector"):
        tickers = group["ticker"].tolist()
        side_signs = group.set_index("ticker")["side"].map({"LONG": 1, "SHORT": -1})
        valid = [t for t in tickers if t in returns.index]
        if not valid:
            continue
        signed = returns.reindex(valid) * side_signs.reindex(valid)
        avg_pick_return = float(signed.mean())
        etf = SECTOR_TO_ETF.get(sector)
        etf_ret = float(returns.get(etf, np.nan)) if etf else np.nan
        # For shorts, the ETF reference flips too
        net_signed_etf = (side_signs.reindex(valid).mean()) * etf_ret if etf_ret == etf_ret else np.nan
        alpha = avg_pick_return - net_signed_etf if net_signed_etf == net_signed_etf else avg_pick_return
        out_rows.append({
            "sector": sector,
            "n_picks": int(len(valid)),
            "avg_pick_return": avg_pick_return,
            "sector_etf_return": etf_ret,
            "alpha": alpha,
        })
    df = pd.DataFrame(out_rows).set_index("sector").sort_values("alpha", ascending=False)
    return df


def summary() -> dict:
    df = compute()
    if df.empty:
        return {"total_alpha": 0.0, "winners": 0, "losers": 0, "by_sector": []}
    winners = int((df["alpha"] > 0).sum())
    losers = int((df["alpha"] < 0).sum())
    return {
        "total_alpha": float(df["alpha"].sum()),
        "winners": winners, "losers": losers,
        "by_sector": df.reset_index().to_dict("records"),
    }
