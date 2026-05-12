"""Tail-risk monitor.

Reads VIX and (when FRED_API_KEY is available) the high-yield credit
spread series, then issues gross-reduction directives. No override.

  VIX >= 25  →  REDUCE_GROSS_20%
  VIX >= 35  →  REDUCE_GROSS_50%
  Credit-spread z >= 1σ widening   →  REDUCE_GROSS_20%

The most severe directive wins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from data.providers import fred_series

log = logging.getLogger(__name__)


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {}).get("tail_risk", {})


@dataclass
class TailRiskReport:
    vix: float | None
    credit_spread_z: float | None
    directive: str | None
    reasons: list[str]


def _latest_vix() -> float | None:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT adj_close FROM daily_prices WHERE ticker = '^VIX' "
            "ORDER BY date DESC LIMIT 1",
        ).fetchone()
    return float(row["adj_close"]) if row and row["adj_close"] is not None else None


def _credit_spread_z(series_id: str, lookback_days: int = 252) -> float | None:
    """Pull HY OAS from FRED, compute current z-score vs trailing year.
    Returns None if FRED key absent or no observations."""
    obs = fred_series(series_id)
    if not obs:
        return None
    df = pd.DataFrame(obs)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).tail(lookback_days)
    if len(df) < 30:
        return None
    cur = float(df["value"].iloc[-1])
    mu = float(df["value"].mean())
    sd = float(df["value"].std(ddof=1))
    return (cur - mu) / sd if sd else None


def evaluate() -> TailRiskReport:
    cfg = _cfg()
    vix = _latest_vix()
    spread_id = cfg.get("credit_spread_series", "BAMLH0A0HYM2")
    cs_z = _credit_spread_z(spread_id)

    reasons: list[str] = []
    directives: list[tuple[int, str]] = []  # (severity_rank, directive)

    vix_warn = float(cfg.get("vix_warn", 25))
    vix_alert = float(cfg.get("vix_alert", 35))
    cs_threshold = float(cfg.get("credit_spread_z_warn", 1.0))

    if vix is not None:
        if vix >= vix_alert:
            directives.append((2, "REDUCE_GROSS_50%"))
            reasons.append(f"VIX {vix:.1f} >= {vix_alert:.0f}")
        elif vix >= vix_warn:
            directives.append((1, "REDUCE_GROSS_20%"))
            reasons.append(f"VIX {vix:.1f} >= {vix_warn:.0f}")

    if cs_z is not None and cs_z >= cs_threshold:
        directives.append((1, "REDUCE_GROSS_20%"))
        reasons.append(f"credit-spread z {cs_z:+.2f} >= {cs_threshold:+.1f}σ")

    directive = max(directives, default=(0, None), key=lambda x: x[0])[1]
    return TailRiskReport(
        vix=vix, credit_spread_z=cs_z, directive=directive, reasons=reasons,
    )
