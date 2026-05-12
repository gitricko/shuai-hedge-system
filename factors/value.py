"""Value factor — 6 sub-factors.

  1. Forward earnings yield (1 / forward P/E)
  2. Book-to-price (book equity / market cap)
  3. FCF yield (TTM FCF / market cap)
  4. EV/EBITDA (inverted — lower multiple is more attractive)
  5. Shareholder yield (TTM buybacks + dividends / market cap)
  6. Sales-to-EV (revenue / EV) — works where P/E breaks on negative earnings

Market cap = latest close × shares outstanding (from fundamentals).
EV         = market cap + total debt − cash equivalents.
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


def _latest_close() -> pd.Series:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            """
            SELECT ticker, adj_close FROM daily_prices dp
            WHERE date = (SELECT MAX(date) FROM daily_prices WHERE ticker = dp.ticker)
            """,
            cn,
        )
    return df.set_index("ticker")["adj_close"] if not df.empty else pd.Series(dtype=float)


def _latest_fundamental(line_items: list[str], period_type: str = "Q") -> pd.DataFrame:
    """Return wide DataFrame: ticker × line_item with most-recent value."""
    placeholders = ",".join(["?"] * len(line_items))
    sql = (
        f"SELECT ticker, line_item, value, period_end FROM fundamentals "
        f"WHERE line_item IN ({placeholders}) AND period_type = ?"
    )
    with conn_ctx() as cn:
        df = pd.read_sql_query(sql, cn, params=[*line_items, period_type])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("period_end").drop_duplicates(["ticker", "line_item"], keep="last")
    return df.pivot(index="ticker", columns="line_item", values="value")


def _ttm_sum(line_item: str) -> pd.Series:
    """Trailing-twelve-month sum across the 4 most recent quarterly rows."""
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, period_end, value FROM fundamentals "
            "WHERE line_item = ? AND period_type = 'Q' ORDER BY ticker, period_end DESC",
            cn, params=[line_item],
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["rk"] = df.groupby("ticker").cumcount()
    return df[df["rk"] < 4].groupby("ticker")["value"].sum()


def _latest_estimate() -> pd.DataFrame:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            """
            SELECT ticker, forward_eps, forward_pe FROM analyst_estimates ae
            WHERE snapshot_date = (
                SELECT MAX(snapshot_date) FROM analyst_estimates WHERE ticker = ae.ticker
            )
            """,
            cn,
        )
    return df.set_index("ticker") if not df.empty else pd.DataFrame()


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    close = _latest_close().reindex(tickers)
    funds = _latest_fundamental([
        "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
        "Diluted Average Shares", "EBIT", "Total Revenue",
    ])
    funds = funds.reindex(tickers)

    ttm_ebitda = (_ttm_sum("EBIT") + _ttm_sum("Reconciled Depreciation")).reindex(tickers)
    ttm_fcf = _ttm_sum("Free Cash Flow").reindex(tickers)
    ttm_buybacks = _ttm_sum("Repurchase Of Capital Stock").reindex(tickers).abs()
    ttm_divs = _ttm_sum("Cash Dividends Paid").reindex(tickers).abs()
    ttm_revenue = _ttm_sum("Total Revenue").reindex(tickers)

    estimates = _latest_estimate().reindex(tickers)

    shares = funds.get("Diluted Average Shares", pd.Series(np.nan, index=tickers))
    market_cap = close * shares
    debt = funds.get("Total Debt", pd.Series(np.nan, index=tickers))
    cash = funds.get("Cash And Cash Equivalents", pd.Series(np.nan, index=tickers))
    enterprise_value = market_cap + debt.fillna(0) - cash.fillna(0)
    book_equity = funds.get("Stockholders Equity", pd.Series(np.nan, index=tickers))

    raw = pd.DataFrame(index=tickers)
    raw["fwd_earnings_yield"] = 1.0 / estimates["forward_pe"].replace({0: np.nan}) \
        if "forward_pe" in estimates.columns else np.nan
    raw["book_to_price"] = book_equity / market_cap
    raw["fcf_yield"] = ttm_fcf / market_cap
    # EV/EBITDA — invert here so higher value = more attractive
    raw["ev_ebitda_inv"] = ttm_ebitda / enterprise_value.replace({0: np.nan})
    raw["shareholder_yield"] = (ttm_buybacks.fillna(0) + ttm_divs.fillna(0)) / market_cap
    raw["sales_to_ev"] = ttm_revenue / enterprise_value.replace({0: np.nan})

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        ranked = sector_percentile_rank(raw[col], sectors)
        scored[col] = fill_neutral(ranked)

    scored["value"] = equal_weight_subfactors(scored, raw.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
