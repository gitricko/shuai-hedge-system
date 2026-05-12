"""VIX-conditional factor weights.

Three regimes by VIX level (latest close of ^VIX in daily_prices):

  Low Vol  (VIX < 15):  momentum 0.20→0.28, value  0.15→0.10
  Normal   (15-25):     default weights
  High Vol (VIX > 25):  quality  0.20→0.28, value  0.15→0.22, momentum 0.20→0.10

Total weight always sums to 1.0.
"""
from __future__ import annotations

import logging

from cache.db import conn_ctx

log = logging.getLogger(__name__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum": 0.20,
    "quality": 0.20,
    "value": 0.15,
    "revisions": 0.15,
    "insider": 0.10,
    "growth": 0.10,
    "short_int": 0.05,
    "institutional": 0.05,
}

LOW_VOL_WEIGHTS: dict[str, float] = {**DEFAULT_WEIGHTS, "momentum": 0.28, "value": 0.10}
HIGH_VOL_WEIGHTS: dict[str, float] = {
    **DEFAULT_WEIGHTS, "quality": 0.28, "value": 0.22, "momentum": 0.10,
}


def latest_vix() -> float | None:
    with conn_ctx() as cn:
        row = cn.execute(
            "SELECT adj_close FROM daily_prices WHERE ticker='^VIX' "
            "ORDER BY date DESC LIMIT 1"
        ).fetchone()
    return float(row["adj_close"]) if row and row["adj_close"] is not None else None


def select_weights(enable: bool = True) -> tuple[str, dict[str, float]]:
    """Return (regime_name, weight_dict). Pass enable=False for default weights."""
    if not enable:
        return "default", DEFAULT_WEIGHTS

    vix = latest_vix()
    if vix is None:
        log.info("VIX unavailable — using default weights")
        return "default", DEFAULT_WEIGHTS
    if vix < 15:
        return "low_vol", LOW_VOL_WEIGHTS
    if vix > 25:
        return "high_vol", HIGH_VOL_WEIGHTS
    return "normal", DEFAULT_WEIGHTS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    name, w = select_weights()
    print(f"Regime: {name}, VIX={latest_vix()}")
    for f, x in w.items():
        print(f"  {f:<14} {x:.2f}")
