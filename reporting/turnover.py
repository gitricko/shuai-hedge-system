"""Turnover analytics + FIFO tax estimate.

  * Turnover = sum(|notional traded|) / avg_gross_exposure over window
  * Trailing 30d and 90d, annualized via (252 / window) scaling
  * Tax estimate: realized round-trip P&L bucketed into short-term
    (<= 365d, 37%) vs long-term (> 365d, 20%) gains
"""
from __future__ import annotations

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from reporting.position_attribution import round_trips


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("reporting", {})


def _history_window(days: int) -> pd.DataFrame:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, action_date, action, notional FROM portfolio_history "
            "WHERE action_date >= ?",
            conn, params=[cutoff.date().isoformat()],
        )
    return df


def turnover(window_days: int, avg_gross_exposure: float = 1.5e8) -> dict:
    df = _history_window(window_days)
    if df.empty:
        return {"traded_dollar": 0.0, "turnover_window": 0.0, "annualized": 0.0,
                "budget": _cfg().get("turnover_budget", 0.30)}
    traded = float(df["notional"].abs().sum())
    t = traded / max(avg_gross_exposure, 1.0)
    annualized = t * (252.0 / max(window_days, 1))
    return {
        "traded_dollar": traded,
        "turnover_window": t,
        "annualized": annualized,
    }


def tax_estimate() -> dict:
    """FIFO realized P&L split into short-term vs long-term tax buckets."""
    trips = round_trips()
    if trips.empty:
        return {"short_term_gains": 0.0, "long_term_gains": 0.0,
                "short_term_tax": 0.0, "long_term_tax": 0.0, "total_tax": 0.0}
    trips["holding_days"] = (pd.to_datetime(trips["close_date"]) - pd.to_datetime(trips["open_date"])).dt.days
    short = trips[trips["holding_days"] <= 365]["realized_pl"].sum()
    long_ = trips[trips["holding_days"] > 365]["realized_pl"].sum()
    rates = _cfg().get("tax_rates", {"short_term": 0.37, "long_term": 0.20})
    st_tax = max(float(short), 0.0) * float(rates.get("short_term", 0.37))
    lt_tax = max(float(long_), 0.0) * float(rates.get("long_term", 0.20))
    return {
        "short_term_gains": float(short),
        "long_term_gains": float(long_),
        "short_term_tax": st_tax,
        "long_term_tax": lt_tax,
        "total_tax": st_tax + lt_tax,
    }
