"""Fundamental statements + 24 derived ratios.

Pulls quarterly + annual income statement, balance sheet, and cash flow via
yfinance, normalizes line items into the long-format `fundamentals` table, and
computes derived ratios (ROE, ROA, margins, growth, leverage, cash quality, etc.)
into `fundamental_ratios`.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from cache.db import conn_ctx, init_schema, upsert_many
from data.universe import get_universe

log = logging.getLogger(__name__)

INCOME_KEYS = [
    "Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income",
    "Net Income", "EBIT", "Research And Development", "Diluted Average Shares",
]
BALANCE_KEYS = [
    "Total Assets", "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest",
    "Stockholders Equity", "Current Assets", "Current Liabilities",
    "Cash And Cash Equivalents", "Accounts Receivable", "Inventory",
    "Total Debt", "Long Term Debt", "Working Capital", "Retained Earnings",
    "Common Stock Equity", "Net Tangible Assets",
]
CASHFLOW_KEYS = [
    "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
    "Free Cash Flow", "Capital Expenditure", "Repurchase Of Capital Stock",
    "Cash Dividends Paid", "Net Income From Continuing Operations",
]


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in df.columns]
    return df


def _statement_rows(ticker: str, statement: str, df: pd.DataFrame, period_type: str) -> list[tuple]:
    df = _normalize_index(df)
    rows: list[tuple] = []
    for line_item, series in df.iterrows():
        for period_end, value in series.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            rows.append((ticker, period_end, period_type, statement, str(line_item), float(value), None))
    return rows


def _fetch_one(ticker: str) -> dict[str, pd.DataFrame]:
    """Pull all four statements (quarterly + annual) for a ticker."""
    t = yf.Ticker(ticker)
    return {
        ("income", "Q"): t.quarterly_financials,
        ("income", "A"): t.financials,
        ("balance", "Q"): t.quarterly_balance_sheet,
        ("balance", "A"): t.balance_sheet,
        ("cashflow", "Q"): t.quarterly_cashflow,
        ("cashflow", "A"): t.cashflow,
    }


def _safe(x):
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return float(x)


def _div(a, b):
    a, b = _safe(a), _safe(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _compute_ratios(ticker: str) -> list[tuple]:
    """Compute the 24 derived ratios from the fundamentals table.

    Aligns income/balance/cashflow line items by (period_end, period_type) and
    emits one row per ratio per period.
    """
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT period_end, period_type, statement, line_item, value "
            "FROM fundamentals WHERE ticker = ?", (ticker,),
        ).fetchall()

    if not rows:
        return []

    df = pd.DataFrame([dict(r) for r in rows])
    pivot = df.pivot_table(
        index=["period_end", "period_type"],
        columns="line_item",
        values="value",
        aggfunc="last",
    )

    def col(name: str) -> pd.Series:
        return pivot[name] if name in pivot.columns else pd.Series(np.nan, index=pivot.index)

    rev = col("Total Revenue")
    cogs = col("Cost Of Revenue")
    gp = col("Gross Profit")
    op_inc = col("Operating Income")
    ni = col("Net Income")
    ebit = col("EBIT")
    rd = col("Research And Development")
    shares = col("Diluted Average Shares")

    assets = col("Total Assets")
    liab = col("Total Liabilities Net Minority Interest")
    equity = col("Stockholders Equity")
    cur_assets = col("Current Assets")
    cur_liab = col("Current Liabilities")
    ar = col("Accounts Receivable")
    debt = col("Total Debt")
    wc = col("Working Capital")
    re_earn = col("Retained Earnings")

    cfo = col("Operating Cash Flow")
    fcf = col("Free Cash Flow")
    capex = col("Capital Expenditure")
    buybacks = col("Repurchase Of Capital Stock")
    divs = col("Cash Dividends Paid")

    ratios = pd.DataFrame(index=pivot.index)
    ratios["roe"] = ni / equity
    ratios["roa"] = ni / assets
    ratios["gross_margin"] = gp / rev
    ratios["operating_margin"] = op_inc / rev
    ratios["net_margin"] = ni / rev
    ratios["debt_to_equity"] = debt / equity
    ratios["current_ratio"] = cur_assets / cur_liab
    ratios["ar_to_revenue"] = ar / rev
    ratios["cfo_to_ni"] = cfo / ni
    # Accruals = (NI - CFO) / Total Assets
    ratios["accruals_ratio"] = (ni - cfo) / assets
    ratios["fcf_yield_proxy"] = fcf / assets   # proper yield needs market cap; placeholder
    ratios["asset_turnover"] = rev / assets
    ratios["rd_intensity"] = rd / rev
    ratios["retained_earnings"] = re_earn
    ratios["working_capital"] = wc
    ratios["total_liabilities"] = liab
    ratios["ebit"] = ebit
    ratios["rd_expense"] = rd
    ratios["shares_outstanding"] = shares
    ratios["dividends_paid"] = divs.abs()
    ratios["buybacks"] = buybacks.abs()

    # Growth: compare to value 4 quarters / 1 year ago within same period_type.
    def yoy(series: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=series.index)
        for (period_end, period_type), v in series.items():
            shift = 4 if period_type == "Q" else 1
            sub = series.xs(period_type, level="period_type")
            sub_sorted = sub.sort_index()
            try:
                idx = sub_sorted.index.get_loc(period_end)
            except KeyError:
                continue
            if idx - shift >= 0:
                prior = sub_sorted.iloc[idx - shift]
                if pd.notna(v) and pd.notna(prior) and prior != 0:
                    out.loc[(period_end, period_type)] = (v - prior) / abs(prior)
        return out

    def qoq(series: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=series.index)
        sub_q = series.xs("Q", level="period_type", drop_level=False).sort_index()
        for i in range(1, len(sub_q)):
            v, prior = sub_q.iloc[i], sub_q.iloc[i - 1]
            if pd.notna(v) and pd.notna(prior) and prior != 0:
                out.loc[sub_q.index[i]] = (v - prior) / abs(prior)
        return out

    ratios["revenue_growth_yoy"] = yoy(rev)
    ratios["revenue_growth_qoq"] = qoq(rev)
    ratios["earnings_growth_yoy"] = yoy(ni)
    ratios["earnings_growth_qoq"] = qoq(ni)

    out_rows: list[tuple] = []
    for (period_end, period_type), row in ratios.iterrows():
        for ratio_name, value in row.items():
            v = _safe(value)
            if v is None:
                continue
            out_rows.append((ticker, period_end, period_type, ratio_name, v))
    return out_rows


def refresh(tickers: Iterable[str] | None = None) -> int:
    init_schema()
    if tickers is None:
        tickers = get_universe(include_benchmarks=False)
    tickers = list(tickers)

    fund_rows: list[tuple] = []
    ratio_rows: list[tuple] = []
    failures = 0

    for ticker in tqdm(tickers, desc="fundamentals"):
        try:
            statements = _fetch_one(ticker)
        except Exception as exc:
            log.warning("fundamentals fetch failed for %s: %s", ticker, exc)
            failures += 1
            continue

        for (statement, period_type), df in statements.items():
            try:
                fund_rows.extend(_statement_rows(ticker, statement, df, period_type))
            except Exception as exc:
                log.warning("parse failed %s/%s/%s: %s", ticker, statement, period_type, exc)

        # Flush per ticker to keep memory bounded
        if fund_rows:
            with conn_ctx() as conn:
                upsert_many(
                    conn,
                    "fundamentals",
                    ["ticker", "period_end", "period_type", "statement", "line_item", "value", "currency"],
                    fund_rows,
                )
            fund_rows = []

        try:
            ratio_rows.extend(_compute_ratios(ticker))
        except Exception as exc:
            log.warning("ratio compute failed for %s: %s", ticker, exc)

        if ratio_rows:
            with conn_ctx() as conn:
                upsert_many(
                    conn,
                    "fundamental_ratios",
                    ["ticker", "period_end", "period_type", "ratio", "value"],
                    ratio_rows,
                )
            ratio_rows = []

    with conn_ctx() as conn:
        n_fund = conn.execute("SELECT COUNT(*) AS c FROM fundamentals").fetchone()["c"]
        n_ratio = conn.execute("SELECT COUNT(*) AS c FROM fundamental_ratios").fetchone()["c"]
    log.info("fundamentals=%d, ratios=%d, failures=%d", n_fund, n_ratio, failures)
    return n_fund + n_ratio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh()
