"""Short availability cache.

Calls Alpaca's asset endpoint per ticker, caches `shortable` +
`easy_to_borrow` flags for 7 days in the `short_availability` table.
Skipping short orders for non-shortable tickers is logged so they
surface in the run summary.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import yaml

from cache.db import REPO_ROOT, conn_ctx, init_schema

log = logging.getLogger(__name__)


def _ttl_days() -> int:
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("execution", {})
    return int(cfg.get("short_check_ttl_days", 7))


def _cached(ticker: str) -> dict | None:
    init_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_ttl_days())).isoformat()
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT shortable, easy_to_borrow, refreshed_at FROM short_availability "
            "WHERE ticker = ?", (ticker,),
        ).fetchone()
    if not row:
        return None
    if row["refreshed_at"] < cutoff:
        return None
    return {
        "shortable": bool(row["shortable"]),
        "easy_to_borrow": bool(row["easy_to_borrow"]),
    }


def _store(ticker: str, shortable: bool, etb: bool) -> None:
    init_schema()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO short_availability "
            "(ticker, shortable, easy_to_borrow, refreshed_at) VALUES (?, ?, ?, ?)",
            (ticker, 1 if shortable else 0, 1 if etb else 0,
             datetime.now(timezone.utc).isoformat()),
        )


def check(broker, ticker: str) -> dict:
    """Returns {shortable, easy_to_borrow}. Uses cache when fresh."""
    cached = _cached(ticker)
    if cached is not None:
        return cached
    try:
        asset = broker.client.get_asset(ticker)
    except Exception as exc:
        log.warning("get_asset failed for %s: %s", ticker, exc)
        # On API failure, conservative: not shortable
        return {"shortable": False, "easy_to_borrow": False}
    result = {
        "shortable": bool(getattr(asset, "shortable", False)),
        "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
    }
    _store(ticker, result["shortable"], result["easy_to_borrow"])
    return result


def is_shortable(broker, ticker: str) -> bool:
    r = check(broker, ticker)
    if not r["shortable"]:
        log.info("[short_check] %s NOT shortable — skipping", ticker)
        return False
    if not r["easy_to_borrow"]:
        log.info("[short_check] %s shortable but hard-to-borrow", ticker)
    return True
