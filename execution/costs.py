"""Slippage tracker.

slippage_bps = (fill_price - signal_price) / signal_price * 10_000

For a BUY (positive qty) the convention is: positive slippage_bps =
paid more than signal (cost). For SELL/SHORT it flips. We normalize by
side so positive bps is always "worse than signal" / cost.

The Dashboard pulls 30-day rolling avg/median/p95 and the worst 5 fills.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx, init_schema

log = logging.getLogger(__name__)


def _window_days() -> int:
    return int(
        yaml.safe_load(open(REPO_ROOT / "config.yaml"))
            .get("execution", {})
            .get("slippage_window_days", 30)
    )


def compute_slippage_bps(*, fill_price: float, signal_price: float, side: str) -> float:
    """Returns cost-positive slippage in bps."""
    if signal_price <= 0:
        return 0.0
    raw = (fill_price - signal_price) / signal_price * 10_000.0
    return raw if side in ("BUY", "BUY_TO_COVER") else -raw


def record_fill(
    fill_id: str, order_id: str, ticker: str,
    fill_price: float, fill_qty: float,
    signal_price: float, side: str,
    fill_timestamp: Optional[str] = None,
) -> float:
    init_schema()
    bps = compute_slippage_bps(fill_price=fill_price, signal_price=signal_price, side=side)
    ts = fill_timestamp or datetime.now(timezone.utc).isoformat()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fills "
            "(fill_id, order_id, ticker, fill_price, fill_qty, slippage_bps, fill_timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fill_id, order_id, ticker, fill_price, fill_qty, bps, ts),
        )
    return bps


def rolling_stats() -> dict:
    """Latest 30-day rolling avg/median/p95/total $ slippage."""
    init_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_window_days())).isoformat()
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, fill_price, fill_qty, slippage_bps, fill_timestamp "
            "FROM fills WHERE fill_timestamp >= ? ORDER BY fill_timestamp DESC",
            conn, params=[cutoff],
        )
    if df.empty:
        return {"n_fills": 0, "avg_bps": 0.0, "median_bps": 0.0, "p95_bps": 0.0,
                "total_dollar": 0.0}
    df["dollar_cost"] = df["slippage_bps"].abs() / 10_000.0 * df["fill_price"].abs() * df["fill_qty"].abs()
    return {
        "n_fills": int(len(df)),
        "avg_bps": float(df["slippage_bps"].mean()),
        "median_bps": float(df["slippage_bps"].median()),
        "p95_bps": float(df["slippage_bps"].quantile(0.95)),
        "total_dollar": float(df["dollar_cost"].sum()),
    }


def worst_fills(n: int = 5) -> pd.DataFrame:
    init_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_window_days())).isoformat()
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, fill_price, fill_qty, slippage_bps, fill_timestamp "
            "FROM fills WHERE fill_timestamp >= ? "
            "ORDER BY slippage_bps DESC LIMIT ?",
            conn, params=[cutoff, n],
        )
    return df
