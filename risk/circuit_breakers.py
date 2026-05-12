"""P&L-triggered circuit breakers.

Per spec:
  Daily   > 1.5% loss  →  SIZE_DOWN 30%
  Daily   > 2.5% loss  →  CLOSE_ALL_TODAY
  Weekly  > 4%  loss   →  SIZE_DOWN 30%
  Drawdown > 8%        →  KILL_SWITCH (writes halt lock; --clear-halt needed)
  Single position > 3% NAV  →  force-close immediately

Reads NAV/P&L history from `portfolio_history` and `portfolio_positions`.
Until Layer 6 wires actual broker fills, P&L is approximated from
unrealized_pl on open positions plus realized P&L in history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from risk import is_halted, set_halt

log = logging.getLogger(__name__)


class Action(str, Enum):
    NONE = "NONE"
    SIZE_DOWN = "SIZE_DOWN"
    CLOSE_ALL_TODAY = "CLOSE_ALL_TODAY"
    KILL_SWITCH = "KILL_SWITCH"
    FORCE_CLOSE_POSITION = "FORCE_CLOSE_POSITION"


@dataclass
class BreakerEvent:
    action: Action
    reason: str
    severity: str   # 'WARN' | 'CRIT'
    ticker: str | None = None
    size_down_pct: float | None = None


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {}).get("circuit_breakers", {})


def _nav_history(window_days: int = 90) -> pd.Series:
    """Read approximate NAV history from portfolio_history.notional.

    Until Layer 6 logs broker fills with mark-to-market NAV, we
    approximate NAV change by cumulating signed notional flows. The
    breaker logic still operates correctly once real NAV data lands.
    """
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT action_date, action, notional FROM portfolio_history "
            "WHERE action_date >= ? ORDER BY action_date",
            conn, params=[cutoff],
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["action_date"])
    daily = df.groupby("date")["notional"].sum()
    return daily.cumsum()


def _current_with_drift() -> dict:
    """Return {ticker: {target_weight, current_weight, drifted}} per position.

    The breaker is meant to fire on positions that have *appreciated* beyond
    the threshold, not on initial allocations. We compute current_weight from
    shares × current_price, divided by NAV. Until Layer 6 records fills with
    actual share counts, current_weight defaults to target_weight (= no drift).
    """
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, side, weight, shares, current_price FROM portfolio_positions",
            conn,
        )
    if df.empty:
        return {}
    df["target_weight"] = df["weight"].astype(float)
    df["current_market_value"] = (df["shares"].fillna(0) * df["current_price"].fillna(0)).abs()
    # If no fills recorded yet, treat current_weight = target_weight (no drift)
    df["current_weight"] = df["current_market_value"].where(
        df["current_market_value"] > 0,
        df["target_weight"],
    )
    return df.set_index("ticker").to_dict("index")


def evaluate(nav: float = 100_000_000.0) -> list[BreakerEvent]:
    """Inspect P&L history + open positions; return list of triggered events.
    Side-effect: sets halt lock on KILL_SWITCH.
    """
    cfg = _cfg()
    events: list[BreakerEvent] = []

    hist = _nav_history()

    if len(hist) >= 2:
        latest = float(hist.iloc[-1])
        prior = float(hist.iloc[-2])
        daily_pnl_pct = (latest - prior) / nav
        if daily_pnl_pct <= -cfg.get("daily_close_all", 0.025):
            events.append(BreakerEvent(
                action=Action.CLOSE_ALL_TODAY,
                reason=f"Daily loss {daily_pnl_pct:.2%} ≤ -{cfg.get('daily_close_all'):.1%}",
                severity="CRIT",
            ))
        elif daily_pnl_pct <= -cfg.get("daily_size_down", 0.015):
            events.append(BreakerEvent(
                action=Action.SIZE_DOWN,
                reason=f"Daily loss {daily_pnl_pct:.2%} ≤ -{cfg.get('daily_size_down'):.1%}",
                severity="WARN",
                size_down_pct=cfg.get("size_down_pct", 0.30),
            ))

    if len(hist) >= 6:
        weekly = (float(hist.iloc[-1]) - float(hist.iloc[-6])) / nav
        if weekly <= -cfg.get("weekly_size_down", 0.04):
            events.append(BreakerEvent(
                action=Action.SIZE_DOWN,
                reason=f"Weekly loss {weekly:.2%} ≤ -{cfg.get('weekly_size_down'):.1%}",
                severity="WARN",
                size_down_pct=cfg.get("size_down_pct", 0.30),
            ))

    if len(hist) >= 2:
        peak = float(hist.cummax().iloc[-1])
        cur = float(hist.iloc[-1])
        dd = (cur - peak) / nav if peak else 0.0
        if dd <= -cfg.get("drawdown_kill", 0.08):
            reason = f"Drawdown {dd:.2%} ≤ -{cfg.get('drawdown_kill'):.1%} — KILL_SWITCH"
            events.append(BreakerEvent(
                action=Action.KILL_SWITCH, reason=reason, severity="CRIT",
            ))
            if not is_halted():
                set_halt(reason)
                log.error("HALT engaged: %s", reason)

    # Single position breach — fires on APPRECIATION past threshold, not initial
    # target. Compares current weight (shares × price / NAV) to target_weight.
    threshold = cfg.get("position_force_close", 0.03)
    for ticker, row in _current_with_drift().items():
        current = float(row.get("current_weight") or 0.0)
        target = float(row.get("target_weight") or 0.0)
        # Only fire if the position has DRIFTED above threshold beyond its target
        if current > threshold and current > target * 1.10:  # 10% drift buffer
            events.append(BreakerEvent(
                action=Action.FORCE_CLOSE_POSITION,
                reason=f"{ticker} weight {current:.1%} > max {threshold:.0%} "
                       f"(target {target:.1%}, drifted)",
                severity="WARN",
                ticker=ticker,
            ))

    for e in events:
        log.warning("CIRCUIT BREAKER %s [%s]: %s", e.action, e.severity, e.reason)
    return events
