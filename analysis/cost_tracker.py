"""Per-run cost tracking with hard ceiling.

Reads `usage` from every Anthropic response and accumulates token counts.
Uses the per-1M-token prices from config.yaml so costs surface in $USD.

Hard ceiling (`cost_ceiling_usd` in config) aborts the run via `CostExceeded`
to prevent runaway spend.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cache.db import REPO_ROOT, conn_ctx, init_schema

log = logging.getLogger(__name__)


class CostExceeded(RuntimeError):
    """Raised when cumulative cost exceeds the configured ceiling."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class CostTracker:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ceiling_usd: float = 25.0
    prices: dict[str, float] = field(default_factory=dict)
    totals: TokenUsage = field(default_factory=TokenUsage)
    cumulative_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_config(cls) -> "CostTracker":
        cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml"))
        a = cfg.get("analysis", {})
        return cls(
            ceiling_usd=float(a.get("cost_ceiling_usd", 25.0)),
            prices=dict(a.get("prices_per_million", {})),
        )

    def _cost_of(self, u: TokenUsage) -> float:
        p = self.prices
        return (
            u.input_tokens       * p.get("input", 3.0)        +
            u.output_tokens      * p.get("output", 15.0)      +
            u.cache_write_tokens * p.get("cache_write", 3.75) +
            u.cache_read_tokens  * p.get("cache_read", 0.30)
        ) / 1_000_000.0

    def record(self, response_usage, *, analyzer: str = "", ticker: str = "") -> float:
        """Record one API call. Returns the call's cost in USD.

        `response_usage` is `response.usage` from `messages.create()` — a
        Pydantic model with input_tokens, output_tokens, cache_*_input_tokens.
        """
        u = TokenUsage(
            input_tokens=getattr(response_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response_usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(response_usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response_usage, "cache_creation_input_tokens", 0) or 0,
        )
        cost = self._cost_of(u)

        with self._lock:
            self.totals.add(u)
            self.cumulative_usd += cost
            current_cum = self.cumulative_usd

        self._persist(analyzer, ticker, u, cost, current_cum)

        if current_cum > self.ceiling_usd:
            raise CostExceeded(
                f"Cost ceiling ${self.ceiling_usd:.2f} exceeded "
                f"(cumulative ${current_cum:.4f}). Aborting run."
            )
        return cost

    def estimate(self, prompt_tokens: int, output_tokens: int = 1000) -> float:
        """Cheap estimate before making a call (assumes no cache yet)."""
        u = TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens)
        return self._cost_of(u)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_tokens": self.totals.input_tokens,
            "output_tokens": self.totals.output_tokens,
            "cache_read_tokens": self.totals.cache_read_tokens,
            "cache_write_tokens": self.totals.cache_write_tokens,
            "cumulative_usd": round(self.cumulative_usd, 4),
            "ceiling_usd": self.ceiling_usd,
        }

    def _persist(self, analyzer: str, ticker: str, u: TokenUsage, cost: float, cumulative: float) -> None:
        init_schema()
        with conn_ctx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cost_log (run_id, timestamp, analyzer, ticker, "
                "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
                "cost_usd, cumulative_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_id,
                    datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    analyzer or "", ticker or "",
                    u.input_tokens, u.output_tokens,
                    u.cache_read_tokens, u.cache_write_tokens,
                    round(cost, 6), round(cumulative, 6),
                ),
            )
