"""Shared scoring utilities.

Every factor module returns a DataFrame indexed by ticker with columns =
sub-factor names + a final composite. Sub-factor scores are 0-100 percentile
ranks WITHIN each GICS sector (so a financial stock is ranked vs other
financials, not vs tech). Tickers with missing data get the sector median (50).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from data.universe import get_sector_map

# GICS sector -> sector ETF used for relative-strength comparisons.
SECTOR_TO_ETF: dict[str, str] = {
    "Information Technology":     "XLK",
    "Financials":                 "XLF",
    "Health Care":                "XLV",
    "Energy":                     "XLE",
    "Industrials":                "XLI",
    "Communication Services":     "XLC",
    "Consumer Discretionary":     "XLY",
    "Consumer Staples":           "XLP",
    "Materials":                  "XLB",
    "Real Estate":                "XLRE",
    "Utilities":                  "XLU",
}

NEUTRAL = 50.0  # sector median fallback when data is missing


def sector_percentile_rank(values: pd.Series, sectors: pd.Series, invert: bool = False) -> pd.Series:
    """Map raw factor values to 0-100 percentile rank within each sector.

    NaN values stay NaN (callers should fill with NEUTRAL where appropriate).
    Higher value -> higher rank by default; pass invert=True to flip.
    """
    df = pd.DataFrame({"v": values, "s": sectors})
    if invert:
        df["v"] = -df["v"]
    df["rank"] = df.groupby("s")["v"].transform(
        lambda x: x.rank(pct=True, method="average") * 100
    )
    return df["rank"]


def fill_neutral(series: pd.Series) -> pd.Series:
    return series.fillna(NEUTRAL)


def equal_weight_subfactors(scores_df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Mean across sub-factor columns, ignoring NaN. Returns 0-100 score."""
    cols = [c for c in columns if c in scores_df.columns]
    if not cols:
        return pd.Series(NEUTRAL, index=scores_df.index)
    return scores_df[cols].mean(axis=1, skipna=True).fillna(NEUTRAL)


def universe_with_sectors() -> pd.Series:
    """Return Series mapping ticker -> GICS sector for the S&P 500 universe."""
    sector_map = get_sector_map()
    return pd.Series(sector_map, name="sector")


def re_rank_within_sector(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    """After blending, percentile-rerank within sector for the final 0-100 score."""
    return sector_percentile_rank(scores, sectors)
