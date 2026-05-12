"""Stress testing — 3 historical + 3 synthetic scenarios.

Historical (use actual ticker-level returns over the period, cached as parquet):
  1. 2008 Financial Crisis    (2008-09-01 → 2009-03-31)
  2. 2020 Covid Crash         (2020-02-20 → 2020-04-30)
  3. 2022 Rate Hikes          (2022-01-01 → 2022-10-31)

Synthetic (apply a shock vector directly):
  4. Sector Shock        — most-concentrated sector returns -30%
  5. Momentum Reversal   — top quintile -20%, bottom +20% (the quant quake)
  6. Short Squeeze       — every short up 30% simultaneously

Output: estimated portfolio P&L ($ and %), broken into long-book vs
short-book contributions.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from cache.db import REPO_ROOT, conn_ctx

log = logging.getLogger(__name__)

STRESS_CACHE_DIR = REPO_ROOT / "cache" / "stress"


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {})


@dataclass
class ScenarioResult:
    name: str
    pnl_pct: float        # as fraction of NAV
    pnl_dollars: float
    long_contribution: float
    short_contribution: float
    note: str = ""


def _load_or_fetch_period(name: str, start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    """Cumulative log-return per ticker over the period. Cached as pickle
    (stdlib — no pyarrow needed)."""
    STRESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.lower().replace(" ", "_")
    cache_path = STRESS_CACHE_DIR / f"{safe}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if set(tickers).issubset(cached.index):
                return cached.loc[tickers]
        except Exception as exc:
            log.warning("stress cache read failed %s: %s", cache_path, exc)

    log.info("Fetching %s prices for stress test (%s → %s)", len(tickers), start, end)
    try:
        df = yf.download(
            tickers=tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
    except Exception as exc:
        log.warning("yfinance fetch failed for stress period %s: %s", name, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    rows: dict[str, float] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr not in df.columns.get_level_values(0):
                continue
            sub = df[tkr]["Close"].dropna()
            if len(sub) < 2:
                continue
            rows[tkr] = float(np.log(sub.iloc[-1] / sub.iloc[0]))
    else:
        sub = df["Close"].dropna()
        if len(sub) >= 2:
            rows[tickers[0]] = float(np.log(sub.iloc[-1] / sub.iloc[0]))

    out = pd.DataFrame.from_dict(rows, orient="index", columns=["log_return"])
    if not out.empty:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(out, f)
        except Exception as exc:
            log.warning("stress cache write failed: %s", exc)
    return out


def _sector_of(ticker: str) -> str:
    with conn_ctx() as conn:
        row = conn.execute("SELECT gics_sector FROM universe WHERE ticker = ?", (ticker,)).fetchone()
    return row["gics_sector"] if row else ""


def _compute_scenario_pnl(weights: pd.Series, returns: pd.Series, nav: float) -> tuple[float, float, float]:
    """Returns (long_pnl_pct, short_pnl_pct, total_pnl_pct)."""
    aligned = pd.concat([weights.rename("w"), returns.rename("r")], axis=1, join="inner").dropna()
    if aligned.empty:
        return 0.0, 0.0, 0.0
    long_pnl = float((aligned.loc[aligned["w"] > 0, "w"] * aligned.loc[aligned["w"] > 0, "r"]).sum())
    short_pnl = float((aligned.loc[aligned["w"] < 0, "w"] * aligned.loc[aligned["w"] < 0, "r"]).sum())
    return long_pnl, short_pnl, long_pnl + short_pnl


def run(weights: pd.Series, *, nav: float = 100_000_000.0) -> list[ScenarioResult]:
    cfg = _cfg()
    results: list[ScenarioResult] = []
    if weights.empty:
        return results

    tickers = weights.index.tolist()

    # Historical scenarios
    for scenario in cfg.get("stress_scenarios", {}).get("historical", []):
        rets = _load_or_fetch_period(scenario["name"], scenario["start"], scenario["end"], tickers)
        if rets.empty:
            log.warning("Skipping %s — no data", scenario["name"])
            continue
        long_p, short_p, total = _compute_scenario_pnl(weights, rets["log_return"], nav)
        results.append(ScenarioResult(
            name=scenario["name"],
            pnl_pct=total, pnl_dollars=total * nav,
            long_contribution=long_p * nav, short_contribution=short_p * nav,
        ))

    # Synthetic scenarios
    syn = cfg.get("stress_scenarios", {}).get("synthetic", {})

    # 4. Sector shock — most concentrated sector by gross exposure
    sectors = {t: _sector_of(t) for t in tickers}
    sector_gross = {}
    for t, s in sectors.items():
        sector_gross[s] = sector_gross.get(s, 0.0) + abs(weights[t])
    if sector_gross:
        worst_sector = max(sector_gross, key=sector_gross.get)
        shock_pct = float(syn.get("sector_shock_pct", -0.30))
        rets = pd.Series({t: shock_pct if sectors.get(t) == worst_sector else 0.0 for t in tickers})
        long_p, short_p, total = _compute_scenario_pnl(weights, rets, nav)
        results.append(ScenarioResult(
            name=f"Sector Shock ({worst_sector})",
            pnl_pct=total, pnl_dollars=total * nav,
            long_contribution=long_p * nav, short_contribution=short_p * nav,
            note=f"sector_gross={sector_gross[worst_sector]:.0%}",
        ))

    # 5. Momentum reversal — needs composite scores
    with conn_ctx() as conn:
        scored = pd.read_sql_query(
            "SELECT ticker, composite FROM scored_universe su "
            "WHERE score_date = (SELECT MAX(score_date) FROM scored_universe)",
            conn,
        ).set_index("ticker")
    if not scored.empty:
        in_book = scored.reindex(tickers)["composite"].dropna()
        top_q = in_book.quantile(0.80)
        bot_q = in_book.quantile(0.20)
        top_shock = float(syn.get("momentum_top_shock", -0.20))
        bot_shock = float(syn.get("momentum_bot_shock", +0.20))
        rets = pd.Series({
            t: (top_shock if in_book.get(t, 50) >= top_q
                else bot_shock if in_book.get(t, 50) <= bot_q else 0.0)
            for t in tickers
        })
        long_p, short_p, total = _compute_scenario_pnl(weights, rets, nav)
        results.append(ScenarioResult(
            name="Momentum Reversal (quant quake)",
            pnl_pct=total, pnl_dollars=total * nav,
            long_contribution=long_p * nav, short_contribution=short_p * nav,
        ))

    # 6. Short squeeze — every short up 30%
    squeeze_pct = float(syn.get("short_squeeze_pct", +0.30))
    rets = pd.Series({t: squeeze_pct if weights[t] < 0 else 0.0 for t in tickers})
    long_p, short_p, total = _compute_scenario_pnl(weights, rets, nav)
    results.append(ScenarioResult(
        name="Short Squeeze",
        pnl_pct=total, pnl_dollars=total * nav,
        long_contribution=long_p * nav, short_contribution=short_p * nav,
    ))

    return results
