"""Per-trade execution pipeline.

For each requested trade:
  1. Pre-trade veto (risk.pre_trade.evaluate) — REJECT or DOWNSIZE
  2. Short availability (skip non-shortable opens of new shorts)
  3. Limit price: close * (1 ± 0.001)
  4. Chunk into slices ≤ 2% of 20-day ADV
  5. Submit as limit DAY order; record signal_price for slippage
  6. Poll every 5s up to 120s
  7. On timeout: cancel + retry (max 3 attempts)
  8. On fill (full or partial): record fill, update slippage stats
  9. Log every transition: timestamp, ticker, side, qty, limit, fill, slippage
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from execution import costs, order_manager as om, short_check
from execution.broker import Broker
from risk.pre_trade import TradeRequest, VetoResult, evaluate as veto_evaluate

log = logging.getLogger(__name__)


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("execution", {})


def _signal_price(ticker: str) -> float | None:
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    return float(row["close"]) if row else None


def _adv_dollars(ticker: str, window: int = 20) -> float:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT close, volume FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            conn, params=[ticker, window],
        )
    if df.empty:
        return 0.0
    return float((df["close"] * df["volume"]).mean())


@dataclass
class TradeOrder:
    """Output of rebalance.generate(). One row per ticker to trade."""
    ticker: str
    side: str               # 'LONG' or 'SHORT' (target side)
    intent: str             # 'OPEN_LONG' | 'CLOSE_LONG' | 'OPEN_SHORT' | 'COVER_SHORT'
    target_dollars: float   # signed: +ve adds long exposure, -ve adds short
    is_closing: bool = False


def _broker_side(intent: str) -> str:
    return "BUY" if intent in ("OPEN_LONG", "COVER_SHORT") else "SELL"


def _split_into_chunks(qty: float, signal: float, adv: float, chunk_pct: float) -> list[float]:
    """Slice an order so no chunk exceeds chunk_pct * 20d ADV (dollar-based)."""
    if adv <= 0 or qty <= 0:
        return [qty]
    chunk_dollars_max = chunk_pct * adv
    notional = qty * signal
    if notional <= chunk_dollars_max:
        return [qty]
    n_chunks = int(notional / chunk_dollars_max) + 1
    return [qty / n_chunks] * n_chunks


def _submit_limit(broker: Broker, ticker: str, side: str, qty: float, limit: float) -> Optional[str]:
    """Submit one limit DAY order. Returns broker order id, or None on error."""
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    req = LimitOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit, 2),
    )
    try:
        resp = broker.client.submit_order(req)
        return str(resp.id)
    except Exception as exc:
        log.warning("submit_order failed for %s %s qty=%g: %s", side, ticker, qty, exc)
        return None


def _poll_until_done(
    broker: Broker, alpaca_order_id: str,
    *, poll_seconds: int, deadline_seconds: int,
) -> dict:
    """Poll the broker until filled / cancelled / deadline. Returns final status dict."""
    deadline = time.time() + deadline_seconds
    last_status = None
    while time.time() < deadline:
        if om.should_shutdown():
            return {"status": "SHUTDOWN", "filled_qty": 0.0, "filled_avg_price": 0.0}
        try:
            o = broker.client.get_order_by_id(alpaca_order_id)
        except Exception as exc:
            log.warning("get_order_by_id failed %s: %s", alpaca_order_id, exc)
            time.sleep(poll_seconds)
            continue
        last_status = str(o.status)
        if last_status in ("filled", "canceled", "expired", "rejected"):
            return {
                "status": last_status,
                "filled_qty": float(o.filled_qty or 0.0),
                "filled_avg_price": float(o.filled_avg_price or 0.0),
            }
        time.sleep(poll_seconds)
    return {"status": last_status or "TIMEOUT", "filled_qty": 0.0, "filled_avg_price": 0.0}


def execute_one(
    broker: Broker,
    trade: TradeOrder,
    *,
    dry_run: bool = False,
) -> dict:
    """Run the full pipeline for a single ticker. Returns a result dict."""
    cfg = _cfg()
    result = {"ticker": trade.ticker, "intent": trade.intent, "fills": [],
              "rejected": False, "reasons": []}

    # 1. Pre-trade veto
    nav = float(cfg.get("nav_assumption", 100_000_000.0))
    weight_delta = trade.target_dollars / nav
    veto_req = TradeRequest(
        ticker=trade.ticker, side=trade.side,
        weight_delta=weight_delta, is_closing=trade.is_closing,
    )
    veto: VetoResult = veto_evaluate(veto_req, nav=nav)
    if not veto.approved:
        result["rejected"] = True
        result["reasons"] = veto.reasons
        log.warning("[%s] VETO REJECT: %s", trade.ticker, veto.reasons)
        return result
    if veto.sized_down and veto.new_weight_delta is not None:
        log.info("[%s] sized down by veto: %s", trade.ticker, veto.reasons)
        trade.target_dollars = veto.new_weight_delta * nav

    # 2. Short availability — only on opening a new short
    if trade.intent == "OPEN_SHORT":
        if not short_check.is_shortable(broker, trade.ticker):
            result["rejected"] = True
            result["reasons"] = ["not shortable"]
            return result

    # 3. Limit price
    signal = _signal_price(trade.ticker)
    if signal is None:
        result["rejected"] = True
        result["reasons"] = ["no signal price available"]
        return result

    broker_side = _broker_side(trade.intent)
    offset = float(cfg.get("limit_offset", 0.001))
    # BUY: bid slightly above (paying-up); SELL: post slightly below (selling-down)
    limit = signal * (1 + offset) if broker_side == "BUY" else signal * (1 - offset)

    qty = abs(trade.target_dollars) / signal
    if qty < 1:
        result["rejected"] = True
        result["reasons"] = [f"qty {qty:.3f} < 1 share — skipping"]
        return result
    qty = round(qty, 0)

    # 4. Chunking
    adv = _adv_dollars(trade.ticker)
    chunks = _split_into_chunks(
        qty, signal, adv, chunk_pct=float(cfg.get("chunk_pct_adv", 0.02)),
    )
    if len(chunks) > 1:
        log.info("[%s] chunked into %d slices of ~%.0f shares", trade.ticker, len(chunks), chunks[0])

    # 5-8. Submit + poll + retry per chunk
    max_retries = int(cfg.get("max_retries", 3))
    tif = int(cfg.get("time_in_force_seconds", 120))
    poll = int(cfg.get("poll_interval_seconds", 5))

    om.install_signal_handler()

    for chunk_qty in chunks:
        if om.should_shutdown():
            log.warning("[%s] shutdown requested — stopping", trade.ticker)
            break

        for attempt in range(1, max_retries + 1):
            order_id = om.new_order_id()
            record = om.OrderRecord(
                order_id=order_id, ticker=trade.ticker, side=broker_side,
                intent=trade.intent, qty=chunk_qty, limit_price=limit,
                signal_price=signal, status=om.OrderStatus.PENDING,
            )

            if dry_run:
                log.info("[DRY_RUN %s] %s %s qty=%g limit=%.2f signal=%.2f adv=$%.0f",
                         trade.ticker, broker_side, trade.intent, chunk_qty, limit, signal, adv)
                record.notes = "dry_run"
                record.status = om.OrderStatus.FILLED
                om.persist(record)
                result["fills"].append({"qty": chunk_qty, "price": limit, "slippage_bps": 0.0})
                break

            alpaca_id = _submit_limit(broker, trade.ticker, broker_side, chunk_qty, limit)
            if not alpaca_id:
                record.status = om.OrderStatus.REJECTED
                record.notes = "submit failed"
                om.persist(record)
                continue
            record.alpaca_order_id = alpaca_id
            om.persist(record)
            log.info("[%s] submitted attempt=%d alpaca=%s qty=%g limit=%.2f",
                     trade.ticker, attempt, alpaca_id, chunk_qty, limit)

            final = _poll_until_done(broker, alpaca_id, poll_seconds=poll, deadline_seconds=tif)

            if final["status"] == "filled" and final["filled_qty"] > 0:
                fill_price = final["filled_avg_price"]
                bps = costs.record_fill(
                    fill_id=f"fil_{uuid.uuid4().hex[:16]}", order_id=order_id,
                    ticker=trade.ticker, fill_price=fill_price,
                    fill_qty=final["filled_qty"], signal_price=signal, side=broker_side,
                )
                om.update_status(order_id, om.OrderStatus.FILLED, notes=f"slippage {bps:.1f}bps")
                result["fills"].append({"qty": final["filled_qty"], "price": fill_price, "slippage_bps": bps})
                log.info("[%s] FILLED %.0f @ $%.2f slippage=%+.1f bps",
                         trade.ticker, final["filled_qty"], fill_price, bps)
                break
            else:
                # Cancel + retry
                try:
                    broker.client.cancel_order_by_id(alpaca_id)
                except Exception:
                    pass
                om.update_status(order_id, om.OrderStatus.CANCELLED,
                                 notes=f"timeout attempt {attempt}")
                log.info("[%s] timeout, attempt %d/%d — cancelling", trade.ticker, attempt, max_retries)
        else:
            log.warning("[%s] giving up after %d attempts", trade.ticker, max_retries)
            result["reasons"].append(f"unfilled after {max_retries} attempts")

    return result
