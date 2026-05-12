"""Insider Activity factor — 3 sub-factors.

  1. Net dollar flow over the last 90 days (purchases − sales)
  2. CEO/CFO weighted 3× vs other insiders
  3. Cluster-buy bonus (3+ insiders open-market buying within 30 days)

Only `transaction_code IN ('P','S')` counts (open-market purchases / sales).
A/M/F (grants, option exercises, tax withholdings) are ignored.

Tickers with no insider data get sector median (50).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from data.sec_data import cluster_buy_tickers
from factors._utils import (
    NEUTRAL,
    equal_weight_subfactors,
    fill_neutral,
    sector_percentile_rank,
    universe_with_sectors,
)

log = logging.getLogger(__name__)


def _is_top_officer(title: str | None) -> bool:
    if not title:
        return False
    t = title.upper()
    return any(k in t for k in ("CEO", "CHIEF EXECUTIVE", "CFO", "CHIEF FINANCIAL"))


def compute(window_days: int = 90) -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            """
            SELECT ticker, insider_title, transaction_code, shares, price, transaction_date
            FROM insider_transactions
            WHERE transaction_date >= ?
              AND transaction_code IN ('P', 'S')
            """,
            cn, params=[cutoff],
        )

    raw = pd.DataFrame(index=tickers)
    if df.empty:
        log.info("No qualifying Form 4 transactions in last %dd — neutral scores", window_days)
        scored = pd.DataFrame({"insider": NEUTRAL}, index=tickers)
        return scored

    df["dollar"] = df["shares"] * df["price"]
    df["sign"] = np.where(df["transaction_code"] == "P", 1, -1)
    df["weighted_dollar"] = df["dollar"] * df["sign"] * df["insider_title"].apply(
        lambda t: 3.0 if _is_top_officer(t) else 1.0
    )

    net_flow = df.groupby("ticker")["dollar"].apply(
        lambda s: (s * df.loc[s.index, "sign"]).sum()
    )
    weighted_flow = df.groupby("ticker")["weighted_dollar"].sum()

    raw["net_flow_90d"] = net_flow.reindex(tickers)
    raw["weighted_flow_90d"] = weighted_flow.reindex(tickers)

    cluster = set(cluster_buy_tickers(window_days=30, threshold=3))
    raw["cluster_buy"] = pd.Series(
        {t: 1.0 if t in cluster else 0.0 for t in tickers}
    )

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    scored["insider"] = equal_weight_subfactors(scored, raw.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
