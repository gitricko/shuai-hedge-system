"""Per-position P&L attribution + FIFO round-trips.

  * Mark-to-market every open position from latest close
  * FIFO matching of opens and closes from portfolio_history to compute
    realized P&L per round-trip
  * Best/worst per side
  * Spearman correlation between composite_at_entry and realized return
    (predictive-power metric)
"""
from __future__ import annotations

import json
import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from cache.db import conn_ctx

log = logging.getLogger(__name__)


def _latest_close() -> pd.Series:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, adj_close FROM daily_prices dp "
            "WHERE date = (SELECT MAX(date) FROM daily_prices WHERE ticker = dp.ticker)",
            conn,
        )
    return df.set_index("ticker")["adj_close"] if not df.empty else pd.Series(dtype=float)


def mark_to_market() -> pd.DataFrame:
    """Returns DataFrame indexed by ticker with current MV + unrealized P&L."""
    with conn_ctx() as conn:
        pos = pd.read_sql_query("SELECT * FROM portfolio_positions", conn)
    if pos.empty:
        return pos
    close = _latest_close().reindex(pos["ticker"]).fillna(0.0).values
    pos = pos.set_index("ticker")
    pos["current_price"] = close
    sign = pos["side"].map({"LONG": 1, "SHORT": -1})
    pos["unrealized_pl_per_share"] = (pos["current_price"] - pos["entry_price"]) * sign
    pos["unrealized_pl"] = pos["unrealized_pl_per_share"] * pos["shares"]
    pos["market_value"] = pos["current_price"] * pos["shares"] * sign
    return pos[["sector", "side", "weight", "shares", "entry_price",
                "current_price", "unrealized_pl", "market_value"]]


def round_trips() -> pd.DataFrame:
    """Match closes (or position-flip events) against prior opens FIFO per
    ticker. Returns one row per closed round-trip with realized P&L."""
    with conn_ctx() as conn:
        hist = pd.read_sql_query(
            "SELECT * FROM portfolio_history ORDER BY action_date, ticker", conn,
        )
    if hist.empty:
        return hist

    trips: list[dict] = []
    by_ticker: dict[str, deque] = {}
    for _, row in hist.iterrows():
        t = row["ticker"]
        side = row["side"] or ""
        shares = float(row["shares_delta"] or 0.0)
        price = float(row["price"] or 0.0)
        action = row["action"] or ""
        date_ = row["action_date"]

        opens = by_ticker.setdefault(t, deque())
        if action in ("OPEN", "INCREASE") or (action == "" and shares > 0):
            opens.append({"side": side, "shares": abs(shares), "price": price, "date": date_})
        elif action in ("CLOSE", "DECREASE") and opens:
            remaining = abs(shares)
            while remaining > 0 and opens:
                lot = opens[0]
                take = min(lot["shares"], remaining)
                # Sign: LONG round-trip realized = (exit - entry) * take
                sign = 1 if lot["side"] == "LONG" else -1
                pnl = sign * (price - lot["price"]) * take
                trips.append({
                    "ticker": t, "side": lot["side"],
                    "open_date": lot["date"], "close_date": date_,
                    "shares": take, "entry_price": lot["price"], "exit_price": price,
                    "realized_pl": pnl,
                })
                lot["shares"] -= take
                remaining -= take
                if lot["shares"] <= 0:
                    opens.popleft()
    return pd.DataFrame(trips)


def best_worst_per_side(n: int = 5) -> dict:
    df = mark_to_market()
    if df.empty:
        return {"long_best": pd.DataFrame(), "long_worst": pd.DataFrame(),
                "short_best": pd.DataFrame(), "short_worst": pd.DataFrame()}
    longs = df[df["side"] == "LONG"].sort_values("unrealized_pl", ascending=False)
    shorts = df[df["side"] == "SHORT"].sort_values("unrealized_pl", ascending=False)
    return {
        "long_best": longs.head(n),
        "long_worst": longs.tail(n).iloc[::-1],
        "short_best": shorts.head(n),
        "short_worst": shorts.tail(n).iloc[::-1],
    }


def predictive_power() -> dict:
    """Spearman correlation between entry-time composite score and realized
    return for closed trades."""
    trips = round_trips()
    if trips.empty:
        return {"n": 0, "spearman": None}
    trips["realized_return"] = trips["realized_pl"] / (trips["entry_price"] * trips["shares"])

    # Look up factor_scores_at_entry — current snapshot doesn't store per-trip
    # entry score, so we approximate via the position record's snapshot when
    # the ticker is still open, falling back to latest scored_universe.
    with conn_ctx() as conn:
        snaps = pd.read_sql_query(
            "SELECT ticker, factor_scores_at_entry FROM portfolio_positions", conn,
        )
    snap_map: dict[str, float] = {}
    for _, r in snaps.iterrows():
        try:
            snap_map[r["ticker"]] = float(json.loads(r["factor_scores_at_entry"] or "{}").get("composite", np.nan))
        except Exception:
            continue
    if not snap_map:
        return {"n": int(len(trips)), "spearman": None}

    trips["entry_composite"] = trips["ticker"].map(snap_map)
    valid = trips.dropna(subset=["entry_composite", "realized_return"])
    if len(valid) < 5:
        return {"n": int(len(valid)), "spearman": None}
    return {
        "n": int(len(valid)),
        "spearman": float(valid["entry_composite"].corr(valid["realized_return"], method="spearman")),
    }
