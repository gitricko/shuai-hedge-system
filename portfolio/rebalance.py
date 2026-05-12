"""Rebalance generator.

Compares current portfolio_positions to target weights, generates a trade
list. Applies a 30% turnover budget — when proposed turnover exceeds the
budget, prioritize the trades whose underlying composite score moved most.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from portfolio import state, transaction_costs as tc

log = logging.getLogger(__name__)


def _load_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("portfolio", {})


@dataclass
class Trade:
    ticker: str
    side: str             # 'LONG' or 'SHORT' — direction we want to be in
    action: str           # 'OPEN', 'CLOSE', 'INCREASE', 'DECREASE'
    weight_delta: float   # signed delta in NAV fraction
    score_delta: float    # |composite_now - composite_at_entry| (priority key)
    est_cost_bps: float


def _latest_scores() -> pd.DataFrame:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su WHERE score_date = "
            "(SELECT MAX(score_date) FROM scored_universe)",
            conn,
        )
    return df.set_index("ticker") if not df.empty else df


def generate(target_weights: pd.Series, *, nav: float = 100_000_000.0) -> pd.DataFrame:
    cfg = _load_cfg()
    turnover_budget = float(cfg.get("turnover_budget", 0.30))

    current = state.get_positions().set_index("ticker") if not state.get_positions().empty else pd.DataFrame()
    scores = _latest_scores()

    cur_weights = (current["weight"] if not current.empty else pd.Series(dtype=float))
    cur_signs = (current["side"].map({"LONG": 1, "SHORT": -1}) if not current.empty else pd.Series(dtype=float))
    cur_signed = cur_weights * cur_signs if not current.empty else pd.Series(dtype=float)

    universe = sorted(set(target_weights.index) | set(cur_signed.index))
    target = target_weights.reindex(universe).fillna(0.0)
    current_w = cur_signed.reindex(universe).fillna(0.0)
    delta = target - current_w

    cost_df = tc.estimate(universe, {t: abs(delta.get(t, 0.0)) * nav for t in universe})

    trades: list[Trade] = []
    for ticker in universe:
        d = float(delta.get(ticker, 0.0))
        if abs(d) < 1e-6:
            continue
        target_v = float(target.get(ticker, 0.0))
        current_v = float(current_w.get(ticker, 0.0))
        side = "LONG" if (target_v + current_v) / 2 >= 0 else "SHORT"

        if abs(current_v) < 1e-6 and abs(target_v) > 1e-6:
            action = "OPEN"
        elif abs(current_v) > 1e-6 and abs(target_v) < 1e-6:
            action = "CLOSE"
        elif abs(target_v) > abs(current_v):
            action = "INCREASE"
        else:
            action = "DECREASE"

        cur_score = float(scores.loc[ticker, "composite"]) if ticker in scores.index else 50.0
        if not current.empty and ticker in current.index:
            try:
                import json
                fs = json.loads(current.loc[ticker, "factor_scores_at_entry"] or "{}")
                entry_score = float(fs.get("composite", 50.0))
            except Exception:
                entry_score = 50.0
        else:
            entry_score = 50.0
        score_delta = abs(cur_score - entry_score)

        trades.append(Trade(
            ticker=ticker, side=side, action=action, weight_delta=d,
            score_delta=score_delta,
            est_cost_bps=float(cost_df.loc[ticker, "total_bps"]) if ticker in cost_df.index else 20.0,
        ))

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["abs_delta"] = df["weight_delta"].abs()
    proposed_turnover = df["abs_delta"].sum() / 2.0  # one-way turnover

    if proposed_turnover > turnover_budget:
        log.warning("Proposed turnover %.1f%% > budget %.1f%% — pruning by score change",
                    proposed_turnover * 100, turnover_budget * 100)
        df = df.sort_values(["action", "score_delta"], ascending=[True, False])
        df["cum_turnover"] = df["abs_delta"].cumsum() / 2.0
        df = df[df["cum_turnover"] <= turnover_budget]
        df = df.drop(columns=["cum_turnover"])

    return df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
