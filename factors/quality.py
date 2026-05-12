"""Quality factor — 8 sub-factors.

  1. ROE stability      — std-dev of trailing 12-quarter ROE (inverted, lower=better)
  2. Gross margin level — most recent quarterly gross margin
  3. Gross margin trend — latest minus 4 quarters ago
  4. Debt-to-equity     — inverted (lower leverage scores higher)
  5. CFO / NI           — higher = real cash earnings
  6. Accruals ratio     — (NI − CFO) / TA, inverted (high accruals predict underperformance)
  7. Piotroski F-Score  — 9-point binary check (1 each)
  8. Altman Z-Score     — distress indicator: 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MV/TL + 1.0*Sales/TA
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


def _quarterly_history(line_item: str, n: int = 12) -> pd.DataFrame:
    """Wide DataFrame: ticker × period_end (most-recent first), value of line_item."""
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, period_end, value FROM fundamentals "
            "WHERE line_item = ? AND period_type = 'Q' ORDER BY ticker, period_end DESC",
            cn, params=[line_item],
        )
    if df.empty:
        return pd.DataFrame()
    df["rk"] = df.groupby("ticker").cumcount()
    return df[df["rk"] < n].pivot(index="ticker", columns="rk", values="value")


def _latest_close() -> pd.Series:
    with conn_ctx() as cn:
        df = pd.read_sql_query(
            "SELECT ticker, adj_close FROM daily_prices dp "
            "WHERE date = (SELECT MAX(date) FROM daily_prices WHERE ticker = dp.ticker)",
            cn,
        )
    return df.set_index("ticker")["adj_close"] if not df.empty else pd.Series(dtype=float)


def _piotroski(ni, cfo, roa_now, roa_prev, de_now, de_prev, cr_now, cr_prev,
               shares_now, shares_prev, gm_now, gm_prev, at_now, at_prev) -> pd.Series:
    """Score 0-9 across the 9 Piotroski signals."""
    score = pd.Series(0, index=ni.index, dtype=float)
    score += (ni > 0).astype(int)
    score += (cfo > 0).astype(int)
    score += (roa_now > roa_prev).astype(int)
    score += (cfo > ni).astype(int)
    score += (de_now < de_prev).astype(int)
    score += (cr_now > cr_prev).astype(int)
    score += (shares_now <= shares_prev).astype(int)
    score += (gm_now > gm_prev).astype(int)
    score += (at_now > at_prev).astype(int)
    return score


def compute() -> pd.DataFrame:
    sectors = universe_with_sectors()
    tickers = sectors.index.tolist()

    # Pull 12 quarters of NI, CFO, equity for ROE stability + ratios.
    ni_q = _quarterly_history("Net Income", 12).reindex(tickers)
    eq_q = _quarterly_history("Stockholders Equity", 12).reindex(tickers)
    cfo_q = _quarterly_history("Operating Cash Flow", 12).reindex(tickers)
    ta_q = _quarterly_history("Total Assets", 12).reindex(tickers)
    rev_q = _quarterly_history("Total Revenue", 12).reindex(tickers)
    gp_q = _quarterly_history("Gross Profit", 12).reindex(tickers)
    debt_q = _quarterly_history("Total Debt", 12).reindex(tickers)
    cur_a_q = _quarterly_history("Current Assets", 12).reindex(tickers)
    cur_l_q = _quarterly_history("Current Liabilities", 12).reindex(tickers)
    shares_q = _quarterly_history("Diluted Average Shares", 12).reindex(tickers)
    wc_q = _quarterly_history("Working Capital", 12).reindex(tickers)
    re_q = _quarterly_history("Retained Earnings", 12).reindex(tickers)
    ebit_q = _quarterly_history("EBIT", 12).reindex(tickers)
    liab_q = _quarterly_history("Total Liabilities Net Minority Interest", 12).reindex(tickers)

    roe = (ni_q / eq_q).replace([np.inf, -np.inf], np.nan)
    roa = (ni_q / ta_q).replace([np.inf, -np.inf], np.nan)
    gm = (gp_q / rev_q).replace([np.inf, -np.inf], np.nan)
    de = (debt_q / eq_q).replace([np.inf, -np.inf], np.nan)
    cr = (cur_a_q / cur_l_q).replace([np.inf, -np.inf], np.nan)
    asset_t = (rev_q / ta_q).replace([np.inf, -np.inf], np.nan)

    raw = pd.DataFrame(index=tickers)
    # 1. ROE stability — std of last 12 quarters (lower is better, so we invert at rank time)
    raw["roe_stability"] = roe.std(axis=1, skipna=True)
    raw["gm_level"] = gm.iloc[:, 0] if not gm.empty else np.nan
    raw["gm_trend"] = gm.iloc[:, 0] - gm.iloc[:, 4] if gm.shape[1] >= 5 else np.nan
    raw["debt_to_equity"] = de.iloc[:, 0] if not de.empty else np.nan
    raw["cfo_to_ni"] = (cfo_q.iloc[:, 0] / ni_q.iloc[:, 0]) if not ni_q.empty else np.nan
    raw["accruals"] = ((ni_q.iloc[:, 0] - cfo_q.iloc[:, 0]) / ta_q.iloc[:, 0]) \
        if not ta_q.empty else np.nan

    # 7. Piotroski F-Score
    if all(d.shape[1] >= 5 for d in (ni_q, cfo_q, roa, de, cr, shares_q, gm, asset_t)):
        raw["piotroski"] = _piotroski(
            ni_q.iloc[:, 0], cfo_q.iloc[:, 0],
            roa.iloc[:, 0], roa.iloc[:, 4],
            de.iloc[:, 0], de.iloc[:, 4],
            cr.iloc[:, 0], cr.iloc[:, 4],
            shares_q.iloc[:, 0], shares_q.iloc[:, 4],
            gm.iloc[:, 0], gm.iloc[:, 4],
            asset_t.iloc[:, 0], asset_t.iloc[:, 4],
        )
    else:
        raw["piotroski"] = np.nan

    # 8. Altman Z-Score
    close = _latest_close().reindex(tickers)
    market_cap = close * shares_q.iloc[:, 0]
    z = (
        1.2 * (wc_q.iloc[:, 0] / ta_q.iloc[:, 0])
        + 1.4 * (re_q.iloc[:, 0] / ta_q.iloc[:, 0])
        + 3.3 * (ebit_q.iloc[:, 0] / ta_q.iloc[:, 0])
        + 0.6 * (market_cap / liab_q.iloc[:, 0])
        + 1.0 * (rev_q.iloc[:, 0] / ta_q.iloc[:, 0])
    )
    raw["altman_z"] = z

    # Sub-factors that need inversion (lower = better)
    inverted = {"roe_stability", "debt_to_equity", "accruals"}

    scored = pd.DataFrame(index=tickers)
    for col in raw.columns:
        ranked = sector_percentile_rank(raw[col], sectors, invert=(col in inverted))
        scored[col] = fill_neutral(ranked)

    scored["quality"] = equal_weight_subfactors(scored, raw.columns)
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = compute()
    print(out.head(10))
