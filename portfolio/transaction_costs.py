"""Transaction cost model — three components per ticker, expressed in bps.

  1. Commission       : flat $/share (default $0 on Alpaca)
  2. Spread cost      : 5% of avg daily (high - low) range as fraction of price
  3. Market impact    : coef * sqrt(trade_size / ADV) * daily_vol_bps,  coef=0.10

The estimate is plugged into the MVO objective so the optimizer sees
expected returns net-of-cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 20


def _load_cfg() -> dict:
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml"))
    return cfg.get("portfolio", {}).get("transaction_costs", {})


def _recent_window(tickers: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, high, low, close, volume FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("ticker").tail(days).reset_index(drop=True)


@dataclass
class CostEstimate:
    spread_bps: float
    impact_bps: float
    commission_bps: float
    total_bps: float


def estimate(
    tickers: list[str],
    trade_dollars: dict[str, float],
    *,
    portfolio_nav: float | None = None,
) -> pd.DataFrame:
    """Return DataFrame indexed by ticker with bps cost components.

    `trade_dollars[ticker]` is the absolute dollar size of the proposed trade.
    """
    cfg = _load_cfg()
    spread_pct = float(cfg.get("spread_pct_of_hl", 0.05))
    impact_coef = float(cfg.get("market_impact_coef", 0.10))
    comm_per_share = float(cfg.get("commission_per_share", 0.0))

    win = _recent_window(tickers)
    if win.empty:
        return pd.DataFrame(index=tickers)

    grp = win.groupby("ticker")
    avg_hl_pct = (grp["high"].mean() - grp["low"].mean()) / grp["close"].mean()
    avg_volume = grp["volume"].mean()
    avg_close = grp["close"].mean()
    daily_ret_std = grp.apply(lambda d: float(np.std(d["close"].pct_change().dropna(), ddof=1) or 0.0))
    daily_vol_bps = (daily_ret_std * 10_000).fillna(0.0)
    adv_dollars = (avg_volume * avg_close).fillna(0.0)

    out = pd.DataFrame(index=tickers)
    out["spread_bps"] = (avg_hl_pct.reindex(tickers).fillna(0.005) * spread_pct * 10_000)
    sizes = pd.Series({t: float(trade_dollars.get(t, 0.0)) for t in tickers})
    pa = (sizes / adv_dollars.reindex(tickers).replace({0: np.nan})).fillna(0.0).clip(lower=0)
    out["impact_bps"] = (impact_coef * np.sqrt(pa) * daily_vol_bps.reindex(tickers).fillna(50)).fillna(0.0)
    out["commission_bps"] = (
        comm_per_share / avg_close.reindex(tickers).replace({0: np.nan}) * 10_000
    ).fillna(0.0)
    out["total_bps"] = out["spread_bps"] + out["impact_bps"] + out["commission_bps"]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = estimate(["AAPL", "MSFT", "TSLA"], {"AAPL": 1_000_000, "MSFT": 500_000, "TSLA": 5_000_000})
    print(out)
