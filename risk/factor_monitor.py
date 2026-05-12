"""Factor spread monitor.

For each factor, computes the long-book minus short-book exposure on the
current portfolio and z-scores it against the universe's cross-sectional
std. Alerts when |z| > 1.5 sigma. When a flagged factor also has a
crowding warning from factors.crowding, the alert priority is HIGH.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from factors.crowding import detect as detect_crowding
from portfolio import factor_exposure

log = logging.getLogger(__name__)


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {})


def _latest_scored() -> pd.DataFrame:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su "
            "WHERE score_date = (SELECT MAX(score_date) FROM scored_universe)",
            conn,
        )
    return df.set_index("ticker") if not df.empty else df


def evaluate(weights: pd.Series) -> list[dict]:
    """Return list of factor-spread alerts (z > 1.5 by default)."""
    cfg = _cfg()
    z_threshold = float(cfg.get("factor_monitor_z_alert", 1.5))

    exposure_df = factor_exposure.compute(weights)
    if exposure_df.empty:
        return []

    scored = _latest_scored()
    crowded = set()
    if not scored.empty:
        try:
            diag = detect_crowding(scored)
            crowded = {f["pair"] for f in diag.get("flags", [])}
        except Exception as exc:
            log.warning("crowding detect failed: %s", exc)

    alerts: list[dict] = []
    for factor, row in exposure_df.iterrows():
        z = float(row["z_score"])
        if abs(z) > z_threshold:
            in_crowd = any(factor in pair for pair in crowded)
            alerts.append({
                "factor": factor,
                "spread": float(row["spread"]),
                "z_score": z,
                "priority": "HIGH" if in_crowd else "MEDIUM",
                "crowded": in_crowd,
            })
    return alerts
