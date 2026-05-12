"""Risk state I/O — cache/risk_state.json.

Holds the live risk snapshot read by run_risk_check.py and written by
each monitor: daily/weekly P&L, drawdown, breaker usage, factor
exposures, MCTR top contributors, alerts list. JSON (not SQLite) because
it's a single point-in-time blob the dashboard polls.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from cache.db import REPO_ROOT

log = logging.getLogger(__name__)

STATE_PATH = REPO_ROOT / "cache" / "risk_state.json"


def _empty() -> dict:
    return {
        "as_of": None,
        "pnl": {"daily": None, "weekly": None, "mtd": None, "ytd": None},
        "drawdown": {"current": None, "max_60d": None},
        "circuit_breakers": {"fired_today": [], "size_down_active": False},
        "exposures": {"gross": None, "net": None, "long_beta": None, "short_beta": None, "net_beta": None},
        "factor": {"spreads": {}, "z_scores": {}, "alerts": []},
        "correlation": {"long_avg": None, "short_avg": None, "effective_bets": None, "alert": False},
        "tail_risk": {"vix": None, "credit_spread_z": None, "directive": None},
        "mctr_top": [],
        "alerts": [],
    }


def load() -> dict:
    if not STATE_PATH.exists():
        return _empty()
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("risk_state.json unreadable (%s); returning empty", exc)
        return _empty()


def save(state: dict) -> None:
    state["as_of"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def update(**deltas: Any) -> dict:
    """Merge top-level keys (one level deep)."""
    state = load()
    for k, v in deltas.items():
        if isinstance(v, dict) and isinstance(state.get(k), dict):
            state[k].update(v)
        else:
            state[k] = v
    save(state)
    return state


def append_alert(message: str, severity: str = "WARN") -> None:
    state = load()
    state.setdefault("alerts", []).append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "message": message,
    })
    save(state)


if __name__ == "__main__":
    s = load()
    print(json.dumps(s, indent=2, default=str))
