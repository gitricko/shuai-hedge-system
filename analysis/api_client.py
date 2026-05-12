"""Anthropic SDK wrapper for Layer 3 analyzers.

Responsibilities:
  * Single shared client (reuses HTTP connections)
  * Prompt caching: every call places `cache_control: ephemeral` on the
    SYSTEM prompt — analyzer instructions stay constant per analyzer, while
    the user message (ticker-specific data) varies. The first call per
    analyzer pays the 1.25x cache-write premium; every subsequent call
    reads at ~0.1x.
  * Retry: SDK auto-retries 429/5xx with exponential backoff (max_retries=4).
  * JSON extraction tolerant of fenced markdown and prose-wrapped JSON.
  * Cost tracker integration on every successful call.

Default model: claude-sonnet-4-6 (configurable via config.yaml).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import anthropic
import yaml

from cache.db import REPO_ROOT
from analysis.cost_tracker import CostTracker

log = logging.getLogger(__name__)


def _load_cfg() -> dict:
    return yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("analysis", {})


class AnalysisClient:
    """Thin wrapper around `anthropic.Anthropic` with our defaults baked in."""

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        cfg = _load_cfg()
        self.model: str = cfg.get("model", "claude-sonnet-4-6")
        self.max_tokens: int = int(cfg.get("max_tokens", 4096))
        self.effort: str = cfg.get("effort", "medium")
        self.cost_tracker: CostTracker = cost_tracker or CostTracker.from_config()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set; analyzers will fail at call time")
        # SDK retries 429 / connection / 5xx automatically (default max_retries=2).
        # We bump to 4 for resilience under sustained 429.
        self._client = anthropic.Anthropic(max_retries=4) if api_key else None

    # ------------------------------------------------------------------
    # Token estimation (no API call) — used by --estimate-cost
    # ------------------------------------------------------------------
    def estimate_tokens(self, system: str, user: str) -> int:
        if not self._client:
            return 0
        try:
            resp = self._client.messages.count_tokens(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.input_tokens
        except Exception as exc:
            log.warning("count_tokens failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Main entry — analyzers call this with stable system + variable user
    # ------------------------------------------------------------------
    def analyze_json(
        self,
        *,
        system: str,
        user: str,
        analyzer: str,
        ticker: str,
    ) -> dict | None:
        """Call Claude with `system` cached, `user` un-cached, expect JSON back.

        Returns the parsed JSON dict, or None on parse failure.
        """
        if not self._client:
            log.error("Anthropic client unavailable (no API key) for %s/%s", analyzer, ticker)
            return None

        # Effort param is supported on Sonnet 4.6; it sits inside output_config.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # System block carries cache_control so the analyzer prompt is
            # cached across all tickers in a single run.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            log.error("Claude API error %s for %s/%s: %s",
                      getattr(exc, "status_code", "?"), analyzer, ticker, exc)
            return None
        except Exception as exc:
            log.exception("Unexpected error calling Claude for %s/%s: %s", analyzer, ticker, exc)
            return None

        # Record cost (may raise CostExceeded — propagate up to abort the run)
        cost_for_call = self.cost_tracker.record(
            response.usage, analyzer=analyzer, ticker=ticker,
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        parsed = extract_json(text)
        if parsed is None:
            log.warning(
                "JSON parse failed for %s/%s (%d chars). Cost so far: $%.4f",
                analyzer, ticker, len(text), self.cost_tracker.cumulative_usd,
            )
        return parsed


# ----------------------------------------------------------------------
# Robust JSON extraction
# ----------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> dict | list | None:
    """Tolerant JSON extractor.

    Handles three shapes:
      1. Raw JSON object/array
      2. Fenced markdown (```json ... ``` or ``` ... ```)
      3. Prose-wrapped — find the first balanced {...} / [...] in the text
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None
