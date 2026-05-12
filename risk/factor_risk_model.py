"""Barra-style cross-sectional factor risk model.

For each trading day t in a 120-day lookback we run:

    r_{i,t} = alpha_t + sum_k beta_{k,t} * F_{k,i} + epsilon_{i,t}

where F_{k,i} is stock i's z-scored exposure to factor k (the same 8
factor scores produced by Layer 2, standardized). The cross-section is
solved by OLS, giving us a daily factor-return series for each k.

From that we derive:
  * factor returns         — daily series per factor
  * factor covariance      — annualized cov of factor returns
  * specific variance      — per-stock residual variance, annualized
  * portfolio total var    = w' X F X' w   +   sum(w_i^2 * spec_var_i)
  * MCTR_i (marginal contribution to total risk)

The predicted covariance matrix Sigma = X*F*X' + diag(spec) can be
fed back to Layer 4's MVO optimizer for forward-looking optimization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from cache.db import conn_ctx

log = logging.getLogger(__name__)

FACTORS = ("momentum", "quality", "value", "revisions",
           "insider", "growth", "short_int", "institutional")
TRADING_DAYS_PER_YEAR = 252


@dataclass
class RiskModelResult:
    factor_returns: pd.DataFrame      # date × factor
    factor_cov: pd.DataFrame          # factor × factor, annualized
    specific_var: pd.Series           # ticker, annualized
    predicted_cov: pd.DataFrame       # ticker × ticker
    exposures: pd.DataFrame           # ticker × factor (z-scored)


def _load_score_panel() -> pd.DataFrame:
    """Cross-sectional factor exposures (z-scored)."""
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su "
            "WHERE score_date = (SELECT MAX(score_date) FROM scored_universe)",
            conn,
        )
    if df.empty:
        return df
    df = df.set_index("ticker")
    return df[list(FACTORS)]


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each factor cross-sectionally to mean 0 / std 1."""
    out = df.copy()
    for c in df.columns:
        s = df[c]
        mu, sd = s.mean(), s.std(ddof=1)
        out[c] = (s - mu) / sd if sd else 0.0
    return out


def _load_returns(tickers: list[str], lookback: int) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, adj_close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return df
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = np.log(panel / panel.shift(1)).dropna(how="all").iloc[-lookback:]
    return rets


def fit(lookback: int = 120) -> RiskModelResult:
    """Estimate the cross-sectional factor model over `lookback` trading days."""
    exposures_raw = _load_score_panel()
    if exposures_raw.empty:
        raise RuntimeError("scored_universe is empty; run Layer 2 first")
    X = _zscore_columns(exposures_raw).fillna(0.0)
    tickers = X.index.tolist()

    rets = _load_returns(tickers, lookback)
    if rets.empty:
        raise RuntimeError("No daily_prices coverage for scored tickers")

    common = [t for t in tickers if t in rets.columns]
    X = X.loc[common]
    rets = rets[common]
    n_factors = X.shape[1]

    # Per-day cross-sectional OLS: r_i = alpha + X_i @ f
    # Add intercept column
    X_aug = np.column_stack([np.ones(len(X)), X.values])  # n_stocks × (1 + n_factors)
    factor_ret_rows: list[np.ndarray] = []
    residuals_rows: list[np.ndarray] = []

    for d in rets.index:
        y = rets.loc[d].values
        mask = ~np.isnan(y)
        if mask.sum() < n_factors + 2:
            continue
        beta, *_ = np.linalg.lstsq(X_aug[mask], y[mask], rcond=None)
        # beta[0] = alpha (market), beta[1:] = factor returns
        factor_ret_rows.append(np.concatenate([[d], beta[1:]]))
        full = np.full_like(y, np.nan)
        full[mask] = y[mask] - X_aug[mask] @ beta
        residuals_rows.append(full)

    if not factor_ret_rows:
        raise RuntimeError("Insufficient cross-sectional coverage for any day")

    factor_returns = pd.DataFrame(
        [r[1:] for r in factor_ret_rows],
        index=pd.to_datetime([r[0] for r in factor_ret_rows]),
        columns=list(X.columns),
    ).astype(float)
    residuals = pd.DataFrame(residuals_rows, index=factor_returns.index, columns=common)

    factor_cov = factor_returns.cov() * TRADING_DAYS_PER_YEAR
    specific_var = residuals.var(ddof=1) * TRADING_DAYS_PER_YEAR

    # Predicted covariance: X @ F @ X.T + diag(specific)
    X_mat = X.values
    F = factor_cov.values
    sysmat = X_mat @ F @ X_mat.T
    predicted_cov = pd.DataFrame(sysmat, index=common, columns=common)
    for t in common:
        predicted_cov.at[t, t] += float(specific_var.get(t, 0.0) or 0.0)

    return RiskModelResult(
        factor_returns=factor_returns,
        factor_cov=factor_cov,
        specific_var=specific_var.fillna(0.0),
        predicted_cov=predicted_cov,
        exposures=X,
    )


def portfolio_risk(weights: pd.Series, model: RiskModelResult) -> dict:
    """Return total / factor / specific variance + MCTR per holding."""
    common = weights.index.intersection(model.predicted_cov.index)
    w = weights.loc[common].values
    Sigma = model.predicted_cov.loc[common, common].values

    total_var = float(w @ Sigma @ w)
    if total_var <= 0:
        return {"total_var": 0.0, "factor_var": 0.0, "specific_var": 0.0,
                "vol_annualized": 0.0, "mctr": pd.Series(dtype=float)}

    X = model.exposures.loc[common].values
    F = model.factor_cov.values
    factor_var = float(w @ X @ F @ X.T @ w)
    specific_var = float(np.sum((w ** 2) * model.specific_var.loc[common].values))

    sigma_p = float(np.sqrt(total_var))
    # MCTR_i = w_i * (Sigma @ w)_i / sigma_p
    mctr = pd.Series((w * (Sigma @ w)) / sigma_p, index=common, name="mctr").sort_values(ascending=False)

    return {
        "total_var": total_var,
        "factor_var": factor_var,
        "specific_var": specific_var,
        "vol_annualized": sigma_p,
        "mctr": mctr,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = fit(120)
    print("Factor cov (annualized):")
    print(model.factor_cov.round(4))
    print(f"\nMedian specific var: {model.specific_var.median():.4f}")
