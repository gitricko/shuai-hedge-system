"""Short Interest factor — 3 sub-factors.

  1. Short percent of float
  2. Days-to-cover (short ratio)
  3. Change in short interest vs prior period

For LONG candidates, *declining* short interest scores higher.
For SHORT candidates, *rising* short interest scores higher.
The composite layer applies the directional flip — this module just emits the
raw and ranked sub-factors with neutral (long) orientation.
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


def _snapshot_panel() -> pd.DataFrame:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, snapshot_date, shares_short, short_ratio, short_pct_float "
            "FROM short_interest",
            cn,
        )
    return df


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    df = _snapshot_panel()
    if df.empty:
        log.warning("No short interest data — neutral scores")
        return pd.DataFrame({"short_int": NEUTRAL}, index=tickers)

    df = df.sort_values("snapshot_date")
    latest = df.drop_duplicates("ticker", keep="last").set_index("ticker")
    prior = (
        df.groupby("ticker").nth(-2)
        if df.groupby("ticker").size().max() > 1
        else pd.DataFrame()
    )

    raw = pd.DataFrame(index=tickers)
    raw["short_pct_float"] = latest.reindex(tickers)["short_pct_float"]
    raw["days_to_cover"] = latest.reindex(tickers)["short_ratio"]
    if not prior.empty and "shares_short" in prior.columns:
        raw["si_change"] = (
            latest.reindex(tickers)["shares_short"]
            - prior.reindex(tickers)["shares_short"]
        )
    else:
        raw["si_change"] = np.nan

    # All three sub-factors invert for the long book (less short pressure = better).
    inverted = {"short_pct_float", "days_to_cover", "si_change"}

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        if raw[col].notna().sum() == 0:
            continue
        ranked = sector_percentile_rank(raw[col], sectors, invert=(col in inverted))
        scored[col] = fill_neutral(ranked)

    if scored.empty:
        return pd.DataFrame({"short_int": NEUTRAL}, index=tickers)

    scored["short_int"] = equal_weight_subfactors(scored, scored.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
