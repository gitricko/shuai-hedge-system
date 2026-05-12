"""Alpaca broker connection.

Safety rules:
  * PAPER trading is the default.
  * Live trading requires BOTH `execution.mode: "live"` in config.yaml AND
    a typed confirmation ("YES I UNDERSTAND THE RISKS") at runtime.
  * Without explicit live opt-in, every code path here points at
    paper-api.alpaca.markets.

On startup we sync against the broker's account/positions endpoints so
local state doesn't drift silently from reality.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import yaml

from cache.db import REPO_ROOT

log = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
LIVE_CONFIRMATION_PHRASE = "YES I UNDERSTAND THE RISKS"


def _cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("execution", {})


def _is_live_mode() -> bool:
    return _cfg().get("mode", "paper").lower() == "live"


def confirm_live_or_abort(typed_phrase: Optional[str] = None) -> bool:
    """Caller must type LIVE_CONFIRMATION_PHRASE to enable live trading.
    Returns True if confirmation matches; False otherwise (caller falls
    back to paper or aborts)."""
    if not _is_live_mode():
        return False
    expected = LIVE_CONFIRMATION_PHRASE
    if typed_phrase is None:
        try:
            typed_phrase = input(
                f"\n*** LIVE TRADING REQUESTED ***\n"
                f"Type the exact phrase to proceed:\n  '{expected}'\n> "
            )
        except EOFError:
            return False
    return typed_phrase.strip() == expected


@dataclass
class BrokerSnapshot:
    """Lightweight snapshot of broker state for startup sync."""
    paper: bool
    account_id: str
    cash: float
    equity: float
    long_market_value: float
    short_market_value: float
    open_positions: dict[str, dict]   # ticker -> {qty, side, avg_entry_price}


class Broker:
    """Thin wrapper around alpaca-py TradingClient with our safety rails."""

    def __init__(self, *, live_confirmation: Optional[str] = None):
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not (api_key and secret_key):
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        self.paper = True
        if _is_live_mode():
            if not confirm_live_or_abort(live_confirmation):
                log.warning("Live confirmation failed — falling back to PAPER")
            else:
                self.paper = False

        # Import here so module imports don't pull in alpaca-py for code that
        # only needs config — keeps `python run_data.py` startup fast.
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, secret_key, paper=self.paper)
        log.info("Alpaca broker initialized (paper=%s)", self.paper)

    @property
    def client(self):
        return self._client

    def sync(self) -> BrokerSnapshot:
        """Pull account + open positions from Alpaca. Source of truth."""
        acct = self._client.get_account()
        positions = self._client.get_all_positions()
        open_positions = {
            p.symbol: {
                "qty": float(p.qty),
                "side": "LONG" if float(p.qty) > 0 else "SHORT",
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        }
        snap = BrokerSnapshot(
            paper=self.paper,
            account_id=str(acct.id),
            cash=float(acct.cash),
            equity=float(acct.equity),
            long_market_value=float(acct.long_market_value),
            short_market_value=float(acct.short_market_value),
            open_positions=open_positions,
        )
        log.info("Broker sync: equity=$%.0f cash=$%.0f long=$%.0f short=$%.0f positions=%d",
                 snap.equity, snap.cash, snap.long_market_value, snap.short_market_value,
                 len(open_positions))
        return snap

    def cancel_all_pending(self) -> int:
        """Cancel every open order. Used by SIGINT handler + retry logic."""
        try:
            cancelled = self._client.cancel_orders()
            n = len(cancelled) if cancelled else 0
            log.info("Cancelled %d open orders", n)
            return n
        except Exception as exc:
            log.warning("cancel_all_pending failed: %s", exc)
            return 0
