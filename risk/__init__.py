"""Layer 5 — Risk Management.

Modules:
  * risk_state            — JSON I/O for cache/risk_state.json
  * factor_risk_model     — Barra cross-sectional regression
  * pre_trade             — 8 hard veto checks per trade
  * circuit_breakers      — P&L-triggered halts
  * factor_monitor        — factor spread z-score alerts
  * correlation_monitor   — 60d corr + effective number of bets
  * tail_risk             — VIX + credit-spread triggers
  * stress_test           — 3 historical + 3 synthetic scenarios

All hard limits live in config.yaml under `risk.*` so tightening is a
diff to config, not code. Halts are file-based (cache/.halt) so any
process — Layer 6 execution, Layer 4 rebalance — can check `is_halted()`
before acting.
"""
from __future__ import annotations

from pathlib import Path

from cache.db import REPO_ROOT

HALT_LOCK = REPO_ROOT / "cache" / ".halt"


def is_halted() -> bool:
    return HALT_LOCK.exists()


def reason_for_halt() -> str:
    if not HALT_LOCK.exists():
        return ""
    try:
        return HALT_LOCK.read_text().strip()
    except OSError:
        return "halt file exists (unreadable)"


def set_halt(reason: str) -> None:
    HALT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    HALT_LOCK.write_text(reason)


def clear_halt() -> bool:
    if HALT_LOCK.exists():
        HALT_LOCK.unlink()
        return True
    return False
