"""SQLite-backed portfolio state.

Tables (defined in cache/db.py):
  * portfolio_positions   — current positions, one row per ticker
  * portfolio_history     — append-only log of every position action
  * position_approvals    — risk-veto approvals (Layer 5 will write here)

Corporate-action handling: split adjustments and dividend cash entries are
recorded as separate `portfolio_history` rows so the entry price stays
historical while shares scale.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from cache.db import conn_ctx, init_schema

log = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    side: str                    # 'LONG' or 'SHORT'
    shares: float
    weight: float
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    sector: str = ""
    factor_scores_at_entry: dict = field(default_factory=dict)

    @property
    def unrealized_pl(self) -> float:
        if self.current_price == 0:
            return 0.0
        sign = 1 if self.side == "LONG" else -1
        return sign * (self.current_price - self.entry_price) * self.shares


def get_positions() -> pd.DataFrame:
    init_schema()
    with conn_ctx() as conn:
        return pd.read_sql_query("SELECT * FROM portfolio_positions ORDER BY ticker", conn)


def upsert_position(p: Position) -> None:
    init_schema()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_positions "
            "(ticker, side, shares, weight, entry_price, entry_date, "
            "current_price, unrealized_pl, sector, factor_scores_at_entry) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p.ticker, p.side, p.shares, p.weight, p.entry_price, p.entry_date,
                p.current_price, p.unrealized_pl, p.sector,
                json.dumps(p.factor_scores_at_entry),
            ),
        )


def close_position(ticker: str, exit_price: float, exit_date: Optional[str] = None) -> None:
    exit_date = exit_date or date.today().isoformat()
    init_schema()
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT side, shares FROM portfolio_positions WHERE ticker = ?", (ticker,),
        ).fetchone()
        if not row:
            return
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_history "
            "(ticker, action_date, action, side, shares_delta, price, notional, notes) "
            "VALUES (?, ?, 'CLOSE', ?, ?, ?, ?, '')",
            (ticker, exit_date, row["side"], -row["shares"], exit_price,
             row["shares"] * exit_price),
        )
        conn.execute("DELETE FROM portfolio_positions WHERE ticker = ?", (ticker,))


def record_action(ticker: str, action: str, side: str, shares_delta: float,
                  price: float, notes: str = "") -> None:
    init_schema()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_history "
            "(ticker, action_date, action, side, shares_delta, price, notional, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker, date.today().isoformat(), action, side,
                shares_delta, price, abs(shares_delta * price), notes,
            ),
        )


def mark_to_market(prices: dict[str, float]) -> None:
    """Update current_price on every open position from a {ticker: close} map."""
    init_schema()
    with conn_ctx() as conn:
        for ticker, price in prices.items():
            row = conn.execute(
                "SELECT side, shares, entry_price FROM portfolio_positions WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            if not row:
                continue
            sign = 1 if row["side"] == "LONG" else -1
            pl = sign * (price - row["entry_price"]) * row["shares"]
            conn.execute(
                "UPDATE portfolio_positions SET current_price = ?, unrealized_pl = ? "
                "WHERE ticker = ?",
                (price, pl, ticker),
            )


def apply_split(ticker: str, ratio: float) -> None:
    """Adjust shares/entry_price for a stock split (e.g. ratio=2.0 for 2-for-1)."""
    init_schema()
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT shares, entry_price FROM portfolio_positions WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        if not row:
            return
        new_shares = row["shares"] * ratio
        new_entry = row["entry_price"] / ratio
        conn.execute(
            "UPDATE portfolio_positions SET shares = ?, entry_price = ? WHERE ticker = ?",
            (new_shares, new_entry, ticker),
        )
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_history "
            "(ticker, action_date, action, side, shares_delta, price, notional, notes) "
            "VALUES (?, ?, 'SPLIT', '', ?, 0, 0, ?)",
            (ticker, date.today().isoformat(),
             new_shares - row["shares"], f"split ratio {ratio}"),
        )


def portfolio_nav(cash: float = 0.0) -> float:
    df = get_positions()
    if df.empty:
        return cash
    long_mv = (df.loc[df["side"] == "LONG", "shares"] * df.loc[df["side"] == "LONG", "current_price"]).sum()
    short_mv = (df.loc[df["side"] == "SHORT", "shares"] * df.loc[df["side"] == "SHORT", "current_price"]).sum()
    return cash + long_mv - short_mv
