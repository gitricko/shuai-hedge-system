"""Institutional Flow factor — 3 sub-factors.

  1. Number of tracked funds holding the ticker
  2. Net change in aggregate holdings vs prior quarter
  3. Multi-fund opening flag (3+ funds opening new positions same ticker)

NOTE: Layer 1's institutional_holdings table stores CUSIPs in the `ticker`
column (13-F doesn't carry tickers — only CUSIPs). Until a CUSIP→ticker
mapping is wired in, this factor will return mostly neutral scores. The
plumbing is in place — once data/cusip_map.py is added, scoring activates.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from data.institutional import multi_fund_openings
from factors._utils import (
    NEUTRAL,
    equal_weight_subfactors,
    fill_neutral,
    sector_percentile_rank,
    universe_with_sectors,
)

log = logging.getLogger(__name__)


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT fund_cik, ticker, report_date, shares FROM institutional_holdings",
            cn,
        )

    if df.empty:
        log.info("No institutional holdings yet — neutral scores")
        return pd.DataFrame({"institutional": NEUTRAL}, index=tickers)

    df = df.sort_values("report_date")
    latest_period = df["report_date"].max()
    prior_period = df.loc[df["report_date"] < latest_period, "report_date"].max()

    latest = df[df["report_date"] == latest_period]
    prior = df[df["report_date"] == prior_period] if prior_period else pd.DataFrame()

    n_funds = latest.groupby("ticker")["fund_cik"].nunique()
    latest_total = latest.groupby("ticker")["shares"].sum()
    prior_total = prior.groupby("ticker")["shares"].sum() if not prior.empty else pd.Series(dtype=float)
    net_change = (latest_total - prior_total).reindex(latest_total.index)

    multi_open = set(multi_fund_openings(threshold=3))

    raw = pd.DataFrame(index=tickers)
    raw["n_funds"] = n_funds.reindex(tickers)
    raw["net_change"] = net_change.reindex(tickers)
    raw["multi_open"] = pd.Series(
        {t: 1.0 if t in multi_open else 0.0 for t in tickers}
    )

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        if raw[col].notna().sum() == 0:
            continue
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    if scored.empty:
        return pd.DataFrame({"institutional": NEUTRAL}, index=tickers)

    scored["institutional"] = equal_weight_subfactors(scored, scored.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
