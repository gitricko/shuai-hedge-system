"""Order state machine + SIGINT-safe shutdown.

State transitions (all persisted to `orders` table):
    PENDING  → PARTIAL → FILLED
    PENDING  → CANCELLED   (timeout, SIGINT, or retry exhaust)
    PENDING  → REJECTED    (broker validation fails)

SIGINT handler:
    1. Stop accepting new trades.
    2. Cancel all pending orders at the broker.
    3. Leave any partial positions intact — let the user decide whether
       to flatten manually in the next run.
    4. Mark cancelled orders in `orders` with notes='SIGINT'.
"""
from __future__ import annotations

import logging
import signal
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Event
from typing import Optional

from cache.db import conn_ctx, init_schema

log = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderRecord:
    order_id: str
    ticker: str
    side: str            # 'BUY' or 'SELL'
    intent: str          # 'OPEN_LONG' | 'CLOSE_LONG' | 'OPEN_SHORT' | 'COVER_SHORT'
    qty: float
    limit_price: float
    signal_price: float
    alpaca_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    notes: str = ""


# Global shutdown flag — set by SIGINT, checked by the executor loop
_shutdown = Event()


def request_shutdown(*_args) -> None:
    log.warning("SIGINT received — requesting graceful shutdown")
    _shutdown.set()


def install_signal_handler() -> None:
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def should_shutdown() -> bool:
    return _shutdown.is_set()


def reset_shutdown() -> None:
    _shutdown.clear()


def new_order_id() -> str:
    return f"ord_{uuid.uuid4().hex[:16]}"


def persist(record: OrderRecord) -> None:
    init_schema()
    now = datetime.now(timezone.utc).isoformat()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO orders "
            "(order_id, alpaca_order_id, ticker, side, intent, qty, limit_price, "
            "signal_price, status, submitted_at, filled_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.order_id, record.alpaca_order_id, record.ticker, record.side,
                record.intent, record.qty, record.limit_price, record.signal_price,
                record.status.value,
                now if record.status == OrderStatus.PENDING else None,
                now if record.status == OrderStatus.FILLED else None,
                record.notes,
            ),
        )


def update_status(order_id: str, status: OrderStatus, *, notes: str | None = None) -> None:
    init_schema()
    now = datetime.now(timezone.utc).isoformat()
    fields = ["status = ?"]
    params: list = [status.value]
    if status == OrderStatus.FILLED:
        fields.append("filled_at = ?")
        params.append(now)
    if notes is not None:
        fields.append("notes = ?")
        params.append(notes)
    params.append(order_id)
    with conn_ctx() as conn:
        conn.execute(
            f"UPDATE orders SET {', '.join(fields)} WHERE order_id = ?", params,
        )


def pending_orders() -> list[str]:
    init_schema()
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT order_id FROM orders WHERE status IN ('PENDING', 'PARTIAL')"
        ).fetchall()
    return [r["order_id"] for r in rows]


def cancel_all_pending_local(reason: str = "SIGINT") -> int:
    init_schema()
    with conn_ctx() as conn:
        cur = conn.execute(
            "UPDATE orders SET status = 'CANCELLED', notes = ? "
            "WHERE status IN ('PENDING', 'PARTIAL')",
            (reason,),
        )
        return cur.rowcount
