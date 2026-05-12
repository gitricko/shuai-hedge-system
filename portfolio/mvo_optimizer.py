"""Markowitz mean-variance optimizer via SLSQP.

Universe: top N longs + bottom N shorts from latest scored_universe.
Decision variable: signed weights (positive = long, negative = short).

Inputs
  mu     : expected return per ticker (composite mapped linearly:
           score 100 -> +15%/yr, score 0 -> -15%/yr) NET of transaction costs.
  Sigma  : 120-day historical covariance matrix (daily-return scaled).
  lambda : risk aversion (default 1.0).

Objective (maximize): mu*w - lambda * w' Sigma w
  Re-expressed for scipy.minimize as -(mu*w) + lambda * w'Sigma w.

Constraints
  long_gross  = sum(w[long_idx])   == target_long
  short_gross = sum(-w[short_idx]) == target_short
  per-position bounds [min_pct, max_pct] on |w|
  |sum(w * beta)| <= max_beta
  |sum(w_in_sector)| <= max_sector_net  (per sector)
  sum(|w_in_sector_per_side|) <= max_sector_single_side  (per sector × side)

On non-convergence the conviction-tilt optimizer is used as a fallback.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from cache.db import REPO_ROOT, conn_ctx
from portfolio.beta import compute_betas
from portfolio import optimizer as conviction
from portfolio import transaction_costs as tc

log = logging.getLogger(__name__)


def _load_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("portfolio", {})


def _latest_scored() -> pd.DataFrame:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su WHERE score_date = "
            "(SELECT MAX(score_date) FROM scored_universe)",
            conn,
        )
    return df.set_index("ticker") if not df.empty else df


def _expected_returns(scores: pd.Series, top: float, bot: float) -> pd.Series:
    """Linear map composite[0,100] -> expected_return[bot, top]."""
    return bot + (scores / 100.0) * (top - bot)


def _covariance(tickers: list[str], window: int) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, adj_close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return pd.DataFrame()
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = np.log(panel / panel.shift(1)).dropna(how="all").iloc[-window:]
    rets = rets.dropna(axis=1, thresh=window // 2)
    cov = rets.cov() * 252  # annualized
    return cov.loc[cov.index.intersection(tickers), cov.columns.intersection(tickers)]


def optimize() -> pd.Series:
    cfg = _load_cfg()
    n_long = int(cfg.get("num_longs", 20))
    n_short = int(cfg.get("num_shorts", 20))
    gross = float(cfg.get("gross_target", 1.50))
    net = float(cfg.get("net_target", 0.05))
    target_long = (gross + net) / 2.0
    target_short = (gross - net) / 2.0
    max_pos = float(cfg.get("max_position", 0.05))
    min_pos = float(cfg.get("min_position", 0.005))
    max_beta = float(cfg.get("max_beta", 0.15))
    max_sec_net = float(cfg.get("max_sector_net", 0.05))
    max_sec_side = float(cfg.get("max_sector_single_side", 0.25))
    lambda_ra = float(cfg.get("mvo_risk_aversion", 1.0))
    cov_window = int(cfg.get("cov_lookback_days", 120))

    scored = _latest_scored()
    if scored.empty:
        log.error("No scored_universe; cannot run MVO")
        return pd.Series(dtype=float)

    longs = scored[scored["side"] == "LONG"].sort_values("composite", ascending=False).head(n_long)
    shorts = scored[scored["side"] == "SHORT"].sort_values("composite").head(n_short)
    if longs.empty or shorts.empty:
        log.error("Insufficient candidates for MVO; falling back to conviction-tilt")
        return conviction.optimize()

    universe = list(longs.index) + list(shorts.index)
    sectors = pd.concat([longs["sector"], shorts["sector"]])
    side_map = {t: "LONG" for t in longs.index}
    side_map.update({t: "SHORT" for t in shorts.index})

    expected_top = float(cfg.get("expected_return_at_top", 0.15))
    expected_bot = float(cfg.get("expected_return_at_bot", -0.15))
    mu = _expected_returns(
        pd.concat([longs["composite"], shorts["composite"]]),
        expected_top, expected_bot,
    )

    cov = _covariance(universe, cov_window)
    common = [t for t in universe if t in cov.index]
    if len(common) < len(universe) * 0.6:
        log.warning("Covariance coverage poor (%d/%d) — falling back to conviction",
                    len(common), len(universe))
        return conviction.optimize()
    universe = common
    sectors = sectors.reindex(universe)
    mu = mu.reindex(universe).fillna(0.0).values
    Sigma = cov.reindex(universe, axis=0).reindex(universe, axis=1).fillna(0.0).values
    Sigma = (Sigma + Sigma.T) / 2.0  # enforce symmetry

    # Net of transaction costs (assume each name traded at target_long/short / N notional)
    nav = 1.0  # weights are fractions of NAV
    trade_dollars = {t: max_pos * nav for t in universe}  # conservative upper bound
    cost_df = tc.estimate(universe, trade_dollars, portfolio_nav=nav)
    cost_pct = (cost_df.reindex(universe)["total_bps"].fillna(20) / 10_000).values
    mu = mu - cost_pct  # subtract cost from expected return

    betas = compute_betas(universe).reindex(universe).fillna(1.0).values

    long_idx = np.array([i for i, t in enumerate(universe) if side_map[t] == "LONG"])
    short_idx = np.array([i for i, t in enumerate(universe) if side_map[t] == "SHORT"])

    # Decision variable: |w| for each name; sign is set by side. So x >= 0.
    n = len(universe)
    x0 = np.zeros(n)
    x0[long_idx] = target_long / max(len(long_idx), 1)
    x0[short_idx] = target_short / max(len(short_idx), 1)

    sign = np.zeros(n)
    sign[long_idx] = +1
    sign[short_idx] = -1

    def to_weights(x: np.ndarray) -> np.ndarray:
        return x * sign

    def neg_obj(x: np.ndarray) -> float:
        w = to_weights(x)
        return -float(mu @ w) + lambda_ra * float(w @ Sigma @ w)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda x: x[long_idx].sum() - target_long},
        {"type": "eq", "fun": lambda x: x[short_idx].sum() - target_short},
        # |sum(w * beta)| <= max_beta
        {"type": "ineq", "fun": lambda x: max_beta - abs((to_weights(x) * betas).sum())},
    ]

    # Per-sector net + single-side caps
    for sector_name in sorted(set(sectors.dropna())):
        idx = np.array([i for i, t in enumerate(universe) if sectors.loc[t] == sector_name])
        if len(idx) == 0:
            continue
        long_in = idx[np.isin(idx, long_idx)]
        short_in = idx[np.isin(idx, short_idx)]
        constraints.append({
            "type": "ineq",
            "fun": (lambda x, i=idx: max_sec_net - abs(to_weights(x)[i].sum())),
        })
        if len(long_in):
            constraints.append({
                "type": "ineq",
                "fun": (lambda x, i=long_in: max_sec_side - x[i].sum()),
            })
        if len(short_in):
            constraints.append({
                "type": "ineq",
                "fun": (lambda x, i=short_in: max_sec_side - x[i].sum()),
            })

    bounds = [(min_pos, max_pos) for _ in range(n)]

    result = minimize(
        neg_obj, x0, method="SLSQP",
        constraints=constraints, bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-7},
    )

    if not result.success:
        log.warning("MVO did not converge (%s); falling back to conviction", result.message)
        return conviction.optimize()

    weights = pd.Series(to_weights(result.x), index=universe, name="weight")
    weights.attrs["regime"] = "mvo"
    return weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    w = optimize()
    print(w.round(4).to_string())
    print(f"\nGross={w.abs().sum():.3f}  Net={w.sum():.3f}")
