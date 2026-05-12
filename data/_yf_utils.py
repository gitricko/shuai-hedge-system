"""Shared yfinance throttle + retry helpers.

yfinance scrapes Yahoo's frontend, which rate-limits per-IP at roughly 50-100
requests/sec across the whole client. After ~500 calls in quick succession the
endpoint returns 429s. We solve this with:

  * `throttled_iter()` — sleeps between calls to stay under target_rps.
  * `with_retry()`     — exponential backoff on 429 / transient errors.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def throttled_iter(items: Iterable[T], target_rps: float = 8.0) -> Iterable[T]:
    """Yield items at no more than target_rps per second."""
    interval = 1.0 / target_rps
    last = 0.0
    for item in items:
        elapsed = time.time() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        last = time.time()
        yield item


class RateLimitError(Exception):
    """Raised by callers when yfinance signals 429."""


def is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg


with_retry = retry(
    retry=retry_if_exception_type((RateLimitError, ConnectionError, TimeoutError)),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
    stop=stop_after_attempt(4),
    reraise=True,
)


def call_or_raise(fn: Callable[..., R], *args, **kwargs) -> R:
    """Run fn; convert any rate-limit message into RateLimitError so tenacity retries."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if is_rate_limit(exc):
            raise RateLimitError(str(exc)) from exc
        raise
