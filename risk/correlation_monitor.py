"""Pairwise correlation monitor + effective number of bets.

  * 60-day rolling pairwise correlation within each book (long, short).
  * Alert when within-book average |corr| > 0.60.
  * Effective number of bets = exp(entropy(normalized_eigenvalues)).
    Eigen-decompose the correlation matrix; the entropy of its
    eigenvalue distribution measures how many "independent directions"
    the book actually has. A perfectly-diversified 20-name book has
    effective_bets ≈ 20; a single-factor cluster has effective_bets ≈ 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx

log = logging.getLogger(__name__)


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {})


@dataclass
class CorrelationReport:
    long_avg_corr: float
    short_avg_corr: float
    long_effective_bets: float
    short_effective_bets: float
    alert: bool
    reasons: list[str]


def _load_returns(tickers: list[str], window: int) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, adj_close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return df
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index().tail(window + 1)
    return panel.pct_change().dropna(how="all")


def _avg_off_diagonal_abs(corr: pd.DataFrame) -> float:
    """Mean |corr| over upper triangle."""
    if corr.shape[0] < 2:
        return 0.0
    m = corr.values
    iu = np.triu_indices_from(m, k=1)
    vals = np.abs(m[iu])
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def _effective_bets(corr: pd.DataFrame) -> float:
    """exp(entropy(eigenvalues/sum)).

    Higher value = more independent risk sources. Bounded above by N.
    """
    if corr.shape[0] < 2:
        return float(corr.shape[0])
    eigvals = np.linalg.eigvalsh(corr.fillna(0).values)
    eigvals = np.clip(eigvals, 1e-9, None)
    p = eigvals / eigvals.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def evaluate(weights: pd.Series) -> CorrelationReport:
    cfg = _cfg()
    window = int(cfg.get("correlation_window", 60))
    threshold = float(cfg.get("correlation_monitor_avg_alert", 0.60))

    longs = [t for t in weights.index if weights[t] > 0]
    shorts = [t for t in weights.index if weights[t] < 0]

    reasons: list[str] = []
    long_corr_avg = short_corr_avg = 0.0
    long_eff = short_eff = 0.0

    if len(longs) >= 2:
        rets = _load_returns(longs, window)
        if not rets.empty and rets.shape[1] >= 2:
            corr = rets.corr()
            long_corr_avg = _avg_off_diagonal_abs(corr)
            long_eff = _effective_bets(corr)

    if len(shorts) >= 2:
        rets = _load_returns(shorts, window)
        if not rets.empty and rets.shape[1] >= 2:
            corr = rets.corr()
            short_corr_avg = _avg_off_diagonal_abs(corr)
            short_eff = _effective_bets(corr)

    alert = False
    if long_corr_avg > threshold:
        alert = True
        reasons.append(f"long book avg |corr| {long_corr_avg:.2f} > {threshold:.2f}")
    if short_corr_avg > threshold:
        alert = True
        reasons.append(f"short book avg |corr| {short_corr_avg:.2f} > {threshold:.2f}")

    return CorrelationReport(
        long_avg_corr=long_corr_avg, short_avg_corr=short_corr_avg,
        long_effective_bets=long_eff, short_effective_bets=short_eff,
        alert=alert, reasons=reasons,
    )
