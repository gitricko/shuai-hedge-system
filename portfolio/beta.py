"""Rolling 60-day beta calculator.

Per-stock beta vs SPY using ordinary least squares on daily log returns.
Book-level aggregates: long-book beta, short-book beta, net portfolio beta.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from cache.db import conn_ctx

log = logging.getLogger(__name__)

WINDOW = 60
BENCHMARK = "SPY"


def _load_prices(tickers: list[str]) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    sql = (
        f"SELECT ticker, date, adj_close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) ORDER BY date"
    )
    with conn_ctx() as conn:
        df = pd.read_sql_query(sql, conn, params=tickers)
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def compute_betas(tickers: Iterable[str], window: int = WINDOW) -> pd.Series:
    """Latest 60-day OLS beta per ticker vs SPY. Tickers without enough
    history get NaN."""
    tickers = list(set(tickers) | {BENCHMARK})
    panel = _load_prices(tickers)
    if panel.empty or BENCHMARK not in panel.columns:
        return pd.Series(dtype=float)

    rets = np.log(panel / panel.shift(1)).dropna(how="all").iloc[-window:]
    spy = rets[BENCHMARK]
    spy_var = float(np.var(spy.dropna(), ddof=1))
    if spy_var == 0:
        return pd.Series(dtype=float)

    out: dict[str, float] = {}
    for tkr in tickers:
        if tkr == BENCHMARK:
            continue
        s = rets[tkr].dropna() if tkr in rets.columns else pd.Series(dtype=float)
        if len(s) < max(20, window // 3):
            out[tkr] = float("nan")
            continue
        common = s.index.intersection(spy.index)
        x, y = spy.loc[common], s.loc[common]
        cov = float(np.cov(x, y, ddof=1)[0, 1])
        out[tkr] = cov / spy_var
    return pd.Series(out, name="beta")


def book_beta(weights: pd.Series, betas: pd.Series) -> dict:
    """Aggregate to long-book / short-book / net portfolio beta.

    `weights` must use signed convention: +ve for longs, -ve for shorts.
    """
    aligned = pd.concat([weights.rename("w"), betas.rename("b")], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"long": 0.0, "short": 0.0, "net": 0.0}
    long_mask = aligned["w"] > 0
    short_mask = aligned["w"] < 0
    long_beta = (aligned.loc[long_mask, "w"] * aligned.loc[long_mask, "b"]).sum()
    short_beta = (aligned.loc[short_mask, "w"] * aligned.loc[short_mask, "b"]).sum()
    return {
        "long": float(long_beta),
        "short": float(short_beta),
        "net": float(long_beta + short_beta),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    betas = compute_betas(["AAPL", "MSFT", "JPM", "XOM"])
    print(betas)
