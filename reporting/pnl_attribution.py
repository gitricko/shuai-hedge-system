"""Daily P&L attribution — four-bucket decomposition.

    daily_return = beta + sector + factor + alpha_residual

  * beta    — net_portfolio_beta × SPY_return
  * sector  — Brinson-style: per-sector portfolio weight × sector ETF return,
              summed across sectors
  * factor  — projection onto Layer 2 factor return spreads
              (top-quintile minus bottom-quintile per factor, daily)
  * alpha   — what's left after the above. Pure stock selection.

Output: persisted to output/daily_attribution.csv (append on each run).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cache.db import REPO_ROOT, conn_ctx
from factors._utils import SECTOR_TO_ETF
from portfolio import state as port_state
from portfolio.beta import compute_betas

log = logging.getLogger(__name__)

OUTPUT_PATH = REPO_ROOT / "output" / "daily_attribution.csv"


def _latest_returns(tickers: list[str], n_days: int = 5) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, adj_close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return df
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index().tail(n_days + 1)
    return panel.pct_change().dropna(how="all")


def _factor_returns(window: int = 60) -> pd.DataFrame:
    """Synthesize daily factor returns from quintile spreads of latest scores
    applied to that-day stock returns. Cheap proxy until Layer 5's factor
    risk model output is wired in here."""
    with conn_ctx() as conn:
        scored = pd.read_sql_query(
            "SELECT * FROM scored_universe su "
            "WHERE score_date = (SELECT MAX(score_date) FROM scored_universe)",
            conn,
        ).set_index("ticker")
    if scored.empty:
        return pd.DataFrame()
    rets = _latest_returns(scored.index.tolist(), n_days=window)
    if rets.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=rets.index)
    for f in ("momentum", "quality", "value", "growth", "short_int"):
        if f not in scored.columns:
            continue
        top = scored[scored[f] >= scored[f].quantile(0.80)].index
        bot = scored[scored[f] <= scored[f].quantile(0.20)].index
        out[f] = (rets[top.intersection(rets.columns)].mean(axis=1)
                  - rets[bot.intersection(rets.columns)].mean(axis=1))
    return out


def attribute_today(*, nav: float = 100_000_000.0) -> dict:
    """Run attribution for the most recent trading day."""
    positions = port_state.get_positions()
    if positions.empty:
        return {"date": None, "total": 0.0, "beta": 0.0, "sector": 0.0,
                "factor": 0.0, "alpha": 0.0}

    signs = positions["side"].map({"LONG": 1, "SHORT": -1})
    weights = pd.Series((positions["weight"] * signs).values, index=positions["ticker"])

    # Get returns for portfolio + benchmarks
    universe = list(weights.index)
    benchmarks = [SECTOR_TO_ETF[s] for s in positions["sector"].dropna().unique()
                  if s in SECTOR_TO_ETF]
    benchmarks = list(set(benchmarks + ["SPY"]))
    rets = _latest_returns(universe + benchmarks, n_days=2)
    if rets.empty:
        return {"date": None, "total": 0.0, "beta": 0.0, "sector": 0.0,
                "factor": 0.0, "alpha": 0.0}
    today = rets.index[-1]
    r_today = rets.loc[today]

    # Total portfolio return = sum(w_i * r_i)
    portfolio_ret = float((weights * r_today.reindex(weights.index).fillna(0.0)).sum())

    # Beta component: net_beta × SPY
    betas = compute_betas(universe)
    net_beta = float((weights * betas.reindex(weights.index).fillna(1.0)).sum())
    spy_ret = float(r_today.get("SPY", 0.0))
    beta_contrib = net_beta * spy_ret

    # Sector (Brinson-style): sum over sectors of (sector_weight × sector_etf_return)
    sector_map = positions.set_index("ticker")["sector"]
    sector_contrib = 0.0
    for sector in sector_map.unique():
        if sector not in SECTOR_TO_ETF:
            continue
        etf = SECTOR_TO_ETF[sector]
        if etf not in r_today.index:
            continue
        in_sector = sector_map[sector_map == sector].index
        s_weight = float(weights.reindex(in_sector).sum())
        sector_contrib += s_weight * float(r_today[etf])
    # Subtract the already-counted beta so sector is incremental
    sector_contrib -= beta_contrib

    # Factor: project onto factor return spreads
    factor_panel = _factor_returns()
    factor_contrib = 0.0
    if not factor_panel.empty and today in factor_panel.index:
        with conn_ctx() as conn:
            scored = pd.read_sql_query(
                "SELECT * FROM scored_universe su WHERE score_date = "
                "(SELECT MAX(score_date) FROM scored_universe)", conn,
            ).set_index("ticker")
        for f, daily_spread in factor_panel.loc[today].items():
            if f in scored.columns:
                exposure = float((weights * (scored[f].reindex(weights.index) - 50) / 50).sum())
                factor_contrib += exposure * float(daily_spread)

    alpha = portfolio_ret - beta_contrib - sector_contrib - factor_contrib

    out = {
        "date": str(today.date()) if hasattr(today, "date") else str(today),
        "total": portfolio_ret,
        "beta": beta_contrib,
        "sector": sector_contrib,
        "factor": factor_contrib,
        "alpha": alpha,
    }
    _persist(out)
    return out


def _persist(row: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        existing = existing[existing["date"] != row["date"]]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(OUTPUT_PATH, index=False)


def history() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
