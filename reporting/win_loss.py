"""Win/loss analysis sliced multiple ways.

For closed round-trips, computes:
  * Overall win rate, P/L ratio (avg_win / avg_loss)
  * Sliced by: side, holding period bucket, sector, VIX regime at entry,
    entry-time composite quintile
  * Winning + losing streak lengths
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cache.db import conn_ctx
from reporting.position_attribution import round_trips

HOLDING_BUCKETS = [
    ("1-5d", 0, 5),
    ("5-20d", 5, 20),
    ("20-60d", 20, 60),
    ("60d+", 60, 10_000),
]


def _holding_days(open_date: str, close_date: str) -> int:
    try:
        return (pd.Timestamp(close_date) - pd.Timestamp(open_date)).days
    except Exception:
        return 0


def _bucketize(d: int) -> str:
    for name, lo, hi in HOLDING_BUCKETS:
        if lo <= d < hi:
            return name
    return "60d+"


def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["holding_days"] = df.apply(lambda r: _holding_days(r["open_date"], r["close_date"]), axis=1)
    df["bucket"] = df["holding_days"].map(_bucketize)
    df["realized_return"] = df["realized_pl"] / (df["entry_price"] * df["shares"]).replace({0: np.nan})

    # Sector + VIX at entry
    with conn_ctx() as conn:
        sectors = pd.read_sql_query("SELECT ticker, gics_sector FROM universe", conn).set_index("ticker")
        vix_hist = pd.read_sql_query(
            "SELECT date, adj_close AS vix FROM daily_prices WHERE ticker='^VIX'", conn,
        )
    df["sector"] = df["ticker"].map(sectors["gics_sector"].to_dict()).fillna("Unknown")

    vix_map = {pd.Timestamp(r["date"]).date(): float(r["vix"]) for _, r in vix_hist.iterrows()}
    df["vix_at_entry"] = df["open_date"].apply(
        lambda d: vix_map.get(pd.Timestamp(d).date(), np.nan)
    )
    df["vix_regime"] = pd.cut(
        df["vix_at_entry"], bins=[-np.inf, 15, 25, np.inf],
        labels=["LowVol", "Normal", "HighVol"],
    )
    return df


def _summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "win_rate": None, "pl_ratio": None,
                "avg_win": None, "avg_loss": None, "total_pl": 0.0}
    wins = df[df["realized_pl"] > 0]["realized_pl"]
    losses = df[df["realized_pl"] < 0]["realized_pl"]
    return {
        "n": int(len(df)),
        "win_rate": float(len(wins) / len(df)) if len(df) else None,
        "pl_ratio": float(wins.mean() / abs(losses.mean())) if len(losses) and len(wins) else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "total_pl": float(df["realized_pl"].sum()),
    }


def _streaks(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"max_win_streak": 0, "max_loss_streak": 0}
    signs = np.sign(df.sort_values("close_date")["realized_pl"].values)
    cur_w = cur_l = max_w = max_l = 0
    for s in signs:
        if s > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif s < 0:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            cur_w = cur_l = 0
    return {"max_win_streak": int(max_w), "max_loss_streak": int(max_l)}


def analyze() -> dict:
    trips = _add_context(round_trips())
    if trips.empty:
        return {"overall": _summary(trips), "by_side": {}, "by_bucket": {},
                "by_sector": {}, "by_vix_regime": {}, "streaks": _streaks(trips)}

    return {
        "overall": _summary(trips),
        "by_side": {k: _summary(g) for k, g in trips.groupby("side")},
        "by_bucket": {k: _summary(g) for k, g in trips.groupby("bucket")},
        "by_sector": {k: _summary(g) for k, g in trips.groupby("sector")},
        "by_vix_regime": {
            str(k): _summary(g) for k, g in trips.groupby("vix_regime", observed=True)
        },
        "streaks": _streaks(trips),
    }
