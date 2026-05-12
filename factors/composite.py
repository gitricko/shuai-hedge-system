"""Composite factor — weighted blend + final sector re-rank.

Default weights (overridable by regime_weights.select_weights()):
    Momentum 0.20, Quality 0.20, Value 0.15, Revisions 0.15,
    Insider  0.10, Growth  0.10, Short_int 0.05, Institutional 0.05.

After blending, the composite is re-ranked within sector for the final 0-100
score. Top quintile = LONG, bottom quintile = SHORT, middle = neutral.

Output: scored_universe table + output/scored_universe_latest.csv.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from cache.db import REPO_ROOT, conn_ctx, init_schema, upsert_many
from factors._utils import (
    NEUTRAL,
    re_rank_within_sector,
    universe_with_sectors,
)
from factors.regime_weights import DEFAULT_WEIGHTS, select_weights

log = logging.getLogger(__name__)

OUTPUT_CSV = REPO_ROOT / "output" / "scored_universe_latest.csv"


FACTOR_MODULES = [
    "momentum", "quality", "value", "revisions",
    "insider", "growth", "short_int", "institutional",
]


def _import_factor(name: str):
    """Lazy-import each factor module to keep startup fast and isolate failures."""
    if name == "short_int":
        from factors import short_interest as mod
    else:
        mod = __import__(f"factors.{name}", fromlist=["compute"])
    return mod


def _compute_factor(name: str, sectors: pd.Series) -> pd.Series:
    try:
        mod = _import_factor(name)
        df = mod.compute()
    except Exception as exc:
        log.exception("Factor %s failed: %s", name, exc)
        return pd.Series(NEUTRAL, index=sectors.index)

    parent_col = "short_int" if name == "short_int" else name
    if parent_col not in df.columns:
        log.warning("Factor %s missing parent column — using neutral", name)
        return pd.Series(NEUTRAL, index=sectors.index)
    return df[parent_col].reindex(sectors.index).fillna(NEUTRAL)


def compute_all(use_regime: bool = False, save: bool = True) -> pd.DataFrame:
    init_schema()
    sectors = universe_with_sectors()

    regime, weights = select_weights(enable=use_regime)
    log.info("Composite weights regime=%s", regime)

    factor_scores = pd.DataFrame(index=sectors.index)
    for name in FACTOR_MODULES:
        log.info("Computing factor: %s", name)
        factor_scores[name] = _compute_factor(name, sectors)

    blended = sum(factor_scores[name] * weights[name] for name in FACTOR_MODULES)
    composite = re_rank_within_sector(blended, sectors).fillna(NEUTRAL)

    quintile_top = composite.quantile(0.80)
    quintile_bot = composite.quantile(0.20)
    side = pd.Series(index=composite.index, dtype="object")
    side[composite >= quintile_top] = "LONG"
    side[composite <= quintile_bot] = "SHORT"

    out = pd.DataFrame({
        "sector": sectors,
        "composite": composite,
        "side": side,
        **{name: factor_scores[name] for name in FACTOR_MODULES},
    }).sort_values("composite", ascending=False)

    if save:
        out.to_csv(OUTPUT_CSV, index_label="ticker")
        _persist_to_db(out)
        log.info("Wrote %d rows to %s", len(out), OUTPUT_CSV)

    return out


def _persist_to_db(out: pd.DataFrame) -> None:
    today = date.today().isoformat()
    rows = [
        (
            ticker, today, r["sector"], r["composite"], r["side"],
            r["momentum"], r["quality"], r["value"], r["revisions"],
            r["insider"], r["growth"], r["short_int"], r["institutional"],
        )
        for ticker, r in out.iterrows()
    ]
    cols = [
        "ticker", "score_date", "sector", "composite", "side",
        "momentum", "quality", "value", "revisions",
        "insider", "growth", "short_int", "institutional",
    ]
    with conn_ctx() as conn:
        upsert_many(conn, "scored_universe", cols, rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = compute_all()
    print("\nTop 5 LONGS:")
    print(out[out["side"] == "LONG"].head(5)[["sector", "composite"]])
    print("\nBottom 5 (SHORTS):")
    print(out[out["side"] == "SHORT"].tail(5)[["sector", "composite"]])
