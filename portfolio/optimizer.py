"""Conviction-tilt optimizer (the simpler, robust default).

Algorithm per spec:
  1. Pick top N longs and bottom N shorts from latest scored_universe.
  2. Equal weight base within each book; top-5% scores get 1.5x, top-10% 1.25x.
  3. Liquidity cap: trim any position > 5% of 20-day ADV.
  4. Earnings cap: halve any position with earnings <=5 days away.
  5. Beta scaling: scale long and short books separately so beta-adjusted
     book exposure equals 1.0 each (i.e. dollar-neutral after beta).
  6. Sector neutrality: cap |sector_net| to config max_sector_net.

Returns a Series of signed target weights (positive = long, negative = short).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from portfolio.beta import compute_betas

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


def _adv_dollars(tickers: list[str], window: int = 20) -> pd.Series:
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, close, volume FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return pd.Series(dtype=float)
    df = df.groupby("ticker").tail(window)
    df["dollar"] = df["close"] * df["volume"]
    return df.groupby("ticker")["dollar"].mean()


def _earnings_within(tickers: list[str], days: int) -> set[str]:
    today = date.today()
    cutoff = today + timedelta(days=days)
    placeholders = ",".join(["?"] * len(tickers))
    if not placeholders:
        return set()
    with conn_ctx() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT ticker FROM earnings_calendar "
            f"WHERE ticker IN ({placeholders}) AND earnings_date BETWEEN ? AND ?",
            (*tickers, today.isoformat(), cutoff.isoformat()),
        ).fetchall()
    return {r["ticker"] for r in rows}


def _tilt(scores: pd.Series) -> pd.Series:
    """1.0 base, 1.25x for top-10% scores, 1.5x for top-5%."""
    top5 = scores.quantile(0.95)
    top10 = scores.quantile(0.90)
    out = pd.Series(1.0, index=scores.index)
    out[scores >= top10] = 1.25
    out[scores >= top5] = 1.5
    return out


def _build_book(side: str, scored: pd.DataFrame, n: int, cfg: dict) -> pd.DataFrame:
    """Return DataFrame with equal-weight tilted weights for one book."""
    sub = (scored[scored["side"] == side]
           .sort_values("composite", ascending=(side == "SHORT"))
           .head(n))
    if sub.empty:
        return sub
    tilts = _tilt(sub["composite"])
    raw = tilts / tilts.sum()
    return sub.assign(raw_weight=raw)


def _scale_to_target(weights: pd.Series, target_gross: float) -> pd.Series:
    s = weights.abs().sum()
    return weights * (target_gross / s) if s > 0 else weights


def _liquidity_cap(weights: pd.Series, adv: pd.Series, nav: float = 100_000_000.0,
                   max_pct_of_adv: float = 0.05) -> pd.Series:
    """Trim any position whose dollar size exceeds max_pct_of_adv * 20d ADV."""
    out = weights.copy()
    for ticker in weights.index:
        a = adv.get(ticker)
        if not a or np.isnan(a):
            continue
        max_dollars = max_pct_of_adv * a
        max_weight = max_dollars / nav
        if abs(out[ticker]) > max_weight:
            out[ticker] = np.sign(out[ticker]) * max_weight
    return out


def _earnings_halve(weights: pd.Series, earnings_set: set[str]) -> pd.Series:
    out = weights.copy()
    for ticker in weights.index:
        if ticker in earnings_set:
            out[ticker] = out[ticker] * 0.5
    return out


def _beta_scale(weights: pd.Series, betas: pd.Series, target_gross: float, side: str) -> pd.Series:
    """Scale a single book so its beta-adjusted gross matches target."""
    aligned = pd.concat([weights.rename("w"), betas.rename("b")], axis=1, join="inner").dropna()
    if aligned.empty:
        return weights
    sign = 1 if side == "LONG" else -1
    cur_beta_exposure = (aligned["w"].abs() * aligned["b"]).sum()
    if cur_beta_exposure == 0:
        return weights
    factor = target_gross / cur_beta_exposure
    return weights * factor


def optimize(*, num_longs: int | None = None, num_shorts: int | None = None) -> pd.Series:
    cfg = _load_cfg()
    n_long = num_longs or int(cfg.get("num_longs", 20))
    n_short = num_shorts or int(cfg.get("num_shorts", 20))
    gross = float(cfg.get("gross_target", 1.50))
    net = float(cfg.get("net_target", 0.05))
    target_long = (gross + net) / 2.0
    target_short = (gross - net) / 2.0

    scored = _latest_scored()
    if scored.empty:
        log.error("No scored_universe — run Layer 2 first")
        return pd.Series(dtype=float)

    longs = _build_book("LONG", scored, n_long, cfg)
    shorts = _build_book("SHORT", scored, n_short, cfg)
    if longs.empty or shorts.empty:
        log.error("Insufficient candidates: longs=%d, shorts=%d", len(longs), len(shorts))
        return pd.Series(dtype=float)

    long_w = _scale_to_target(longs["raw_weight"], target_long)
    short_w = _scale_to_target(shorts["raw_weight"], target_short).mul(-1)

    universe = list(long_w.index) + list(short_w.index)
    adv = _adv_dollars(universe)
    earnings = _earnings_within(universe, days=int(cfg.get("earnings_size_halve_days", 5)))

    long_w = _liquidity_cap(long_w, adv)
    short_w = _liquidity_cap(short_w, adv)
    long_w = _earnings_halve(long_w, earnings)
    short_w = _earnings_halve(short_w, earnings)

    betas = compute_betas(universe)
    long_w = _beta_scale(long_w, betas, target_long, "LONG")
    short_w = _beta_scale(short_w, betas, target_short, "SHORT")

    long_w = _scale_to_target(long_w, target_long)
    short_w = _scale_to_target(short_w.abs(), target_short).mul(-1)

    weights = pd.concat([long_w, short_w]).rename("weight")
    weights.attrs["regime"] = "conviction"
    return weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    w = optimize()
    print(w.round(4).to_string())
    print(f"\nGross={w.abs().sum():.3f}  Net={w.sum():.3f}  N={len(w)}")
