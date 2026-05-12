"""Pre-trade veto — 8 hard checks. Any failure rejects the trade.

ABSOLUTE VETO — no override. Closing/covering trades are always approved
(they reduce risk by definition).

Checks per spec:
  1. Halt lock exists                              → REJECT
  2. Earnings blackout: <=2d veto, 5d size cut 50% → REJECT or DOWNSIZE
  3. Liquidity > 5% of 20-day ADV                  → REJECT
  4. Position > 5% AUM                             → REJECT
  5. Sector > 25%                                  → REJECT
  6. Gross > 165% or net outside [-10%, +15%]      → REJECT
  7. |net beta| > 0.20                             → REJECT
  8. Pairwise correlation > 0.80 with existing     → REJECT

Every rejection is logged with timestamp + reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from portfolio import state as port_state
from portfolio.beta import compute_betas
from risk import is_halted, reason_for_halt

log = logging.getLogger(__name__)


def _load_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("risk", {}).get("pre_trade", {})


@dataclass
class TradeRequest:
    ticker: str
    side: str          # 'LONG' or 'SHORT' (target side)
    weight_delta: float  # signed change in target weight as fraction of NAV
    is_closing: bool = False  # closing/covering — auto-approve


@dataclass
class VetoResult:
    approved: bool
    sized_down: bool = False
    new_weight_delta: float | None = None
    reasons: list[str] = field(default_factory=list)


def _earnings_within(ticker: str, days: int) -> bool:
    today = date.today()
    cutoff = today + timedelta(days=days)
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT 1 FROM earnings_calendar WHERE ticker = ? AND earnings_date BETWEEN ? AND ? LIMIT 1",
            (ticker, today.isoformat(), cutoff.isoformat()),
        ).fetchone()
    return row is not None


def _adv_dollars(ticker: str, window: int = 20) -> float:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT close, volume FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            conn, params=[ticker, window],
        )
    if df.empty:
        return 0.0
    return float((df["close"] * df["volume"]).mean())


def _current_weights() -> pd.Series:
    df = port_state.get_positions()
    if df.empty:
        return pd.Series(dtype=float)
    signs = df["side"].map({"LONG": 1, "SHORT": -1})
    return pd.Series((df["weight"] * signs).values, index=df["ticker"])


def _sector_of(ticker: str) -> str:
    with conn_ctx() as conn:
        row = conn.execute("SELECT gics_sector FROM universe WHERE ticker = ?", (ticker,)).fetchone()
    return row["gics_sector"] if row else ""


def _pairwise_max_correlation(ticker: str, existing: list[str], window: int = 60) -> float:
    tickers = [ticker] + existing
    placeholders = ",".join(["?"] * len(tickers))
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            f"SELECT ticker, date, adj_close FROM daily_prices WHERE ticker IN ({placeholders}) ORDER BY date",
            conn, params=tickers,
        )
    if df.empty:
        return 0.0
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index().tail(window)
    rets = panel.pct_change().dropna(how="all")
    if ticker not in rets.columns or rets.shape[1] < 2:
        return 0.0
    corrs = rets.corr()[ticker].drop(ticker).abs()
    return float(corrs.max()) if not corrs.empty else 0.0


def evaluate(
    trade: TradeRequest,
    *,
    nav: float = 100_000_000.0,
    target_after: Optional[pd.Series] = None,
) -> VetoResult:
    """Run all 8 checks. `target_after` is the proposed portfolio AFTER the
    trade is applied (signed weights). If not supplied, we synthesize it
    from current positions + the delta."""
    cfg = _load_cfg()
    result = VetoResult(approved=True, new_weight_delta=trade.weight_delta)

    # Closing trades skip checks 2-8 (always approved unless halt lock).
    if is_halted() and not trade.is_closing:
        result.approved = False
        result.reasons.append(f"halt_lock: {reason_for_halt()}")
        _log_rejection(trade, result.reasons)
        return result

    if trade.is_closing:
        return result  # auto-approve closing/covering trades

    # 2. Earnings blackout
    earnings_2d = _earnings_within(trade.ticker, 2)
    earnings_5d = _earnings_within(trade.ticker, cfg.get("earnings_blackout_days", 5))
    if earnings_2d:
        result.approved = False
        result.reasons.append("earnings within 2d — veto")
    elif earnings_5d:
        # Size cut, not full veto
        cut = float(cfg.get("earnings_size_cut", 0.5))
        result.sized_down = True
        result.new_weight_delta = trade.weight_delta * cut
        result.reasons.append(f"earnings within 5d — size cut to {cut:.0%}")

    # 3. Liquidity (this trade alone)
    adv = _adv_dollars(trade.ticker)
    if adv > 0:
        trade_dollars = abs(result.new_weight_delta or trade.weight_delta) * nav
        adv_pct = trade_dollars / adv
        if adv_pct > cfg.get("max_liquidity_pct_adv", 0.05):
            result.approved = False
            result.reasons.append(f"trade is {adv_pct:.1%} of 20d ADV (max {cfg.get('max_liquidity_pct_adv', 0.05):.0%})")

    # Build proposed weights for portfolio-level checks
    current = _current_weights()
    if target_after is None:
        target = current.copy()
        target.loc[trade.ticker] = target.get(trade.ticker, 0.0) + (result.new_weight_delta or trade.weight_delta)
    else:
        target = target_after.copy()

    # 4. Per-position
    max_pos = cfg.get("max_position_pct_nav", 0.05)
    new_weight = abs(target.get(trade.ticker, 0.0))
    if new_weight > max_pos:
        result.approved = False
        result.reasons.append(f"position {new_weight:.1%} > max {max_pos:.0%}")

    # 5. Sector
    sectors = {t: _sector_of(t) for t in target.index if abs(target[t]) > 1e-6}
    sector_gross: dict[str, float] = {}
    for t, s in sectors.items():
        sector_gross[s] = sector_gross.get(s, 0.0) + abs(target.get(t, 0.0))
    max_sec = cfg.get("max_sector_pct_nav", 0.25)
    breached_sectors = {s: v for s, v in sector_gross.items() if v > max_sec}
    if breached_sectors:
        result.approved = False
        result.reasons.append(f"sector caps breached: {breached_sectors}")

    # 6. Gross / net
    gross = float(target.abs().sum())
    net = float(target.sum())
    if gross > cfg.get("max_gross", 1.65):
        result.approved = False
        result.reasons.append(f"gross {gross:.0%} > max {cfg.get('max_gross', 1.65):.0%}")
    if not (cfg.get("min_net", -0.10) <= net <= cfg.get("max_net", 0.15)):
        result.approved = False
        result.reasons.append(f"net {net:+.0%} outside [{cfg.get('min_net'):+.0%}, {cfg.get('max_net'):+.0%}]")

    # 7. Net beta
    betas = compute_betas(target.index.tolist())
    aligned = pd.concat([target.rename("w"), betas.rename("b")], axis=1, join="inner").dropna()
    if not aligned.empty:
        net_beta = float((aligned["w"] * aligned["b"]).sum())
        if abs(net_beta) > cfg.get("max_net_beta", 0.20):
            result.approved = False
            result.reasons.append(f"|net beta| {abs(net_beta):.2f} > {cfg.get('max_net_beta', 0.20):.2f}")

    # 8. Pairwise correlation
    existing = [t for t in current.index if t != trade.ticker and abs(current[t]) > 1e-6]
    if existing:
        max_corr = _pairwise_max_correlation(trade.ticker, existing)
        max_allowed = cfg.get("max_pairwise_corr", 0.80)
        if max_corr > max_allowed:
            result.approved = False
            result.reasons.append(f"pairwise corr {max_corr:.2f} > {max_allowed:.2f}")

    if not result.approved:
        _log_rejection(trade, result.reasons)
    elif result.sized_down:
        log.info("Pre-trade size-down %s: %s", trade.ticker, result.reasons)

    return result


def _log_rejection(trade: TradeRequest, reasons: list[str]) -> None:
    log.warning("REJECTED %s %s Δw=%+.4f: %s",
                trade.side, trade.ticker, trade.weight_delta, " | ".join(reasons))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tr = TradeRequest(ticker="AAPL", side="LONG", weight_delta=0.04)
    r = evaluate(tr)
    print(r)
