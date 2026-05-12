"""SQLite-backed analysis result cache.

Keyed by (analyzer, ticker, artifact_id). The artifact_id is a short hash of
the input data (transcript text, fundamentals window, risk-factor section,
insider window) — re-running on unchanged input is a free cache hit.

TTL eviction defaults to 30 days; older entries are deleted lazily on read.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from cache.db import REPO_ROOT, conn_ctx, init_schema

log = logging.getLogger(__name__)


def _ttl_days() -> int:
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml"))
    return int(cfg.get("analysis", {}).get("cache_ttl_days", 30))


def artifact_hash(*parts: str) -> str:
    """Stable short hash of the variable input portion."""
    h = hashlib.sha256()
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8") if isinstance(p, str) else str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def get(analyzer: str, ticker: str, artifact_id: str) -> dict | None:
    init_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_ttl_days())).isoformat()
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT result_json, created_at FROM analysis_results "
            "WHERE analyzer = ? AND ticker = ? AND artifact_id = ?",
            (analyzer, ticker, artifact_id),
        ).fetchone()
    if not row:
        return None
    if row["created_at"] < cutoff:
        log.debug("cache stale for %s/%s/%s", analyzer, ticker, artifact_id)
        return None
    try:
        return json.loads(row["result_json"])
    except json.JSONDecodeError:
        return None


def put(
    analyzer: str,
    ticker: str,
    artifact_id: str,
    result: dict,
    *,
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    init_schema()
    now = datetime.now(timezone.utc).isoformat()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analysis_results "
            "(analyzer, ticker, artifact_id, result_json, model, created_at, "
            "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analyzer, ticker, artifact_id, json.dumps(result),
                model, now,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd,
            ),
        )


def evict_stale() -> int:
    init_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_ttl_days())).isoformat()
    with conn_ctx() as conn:
        cur = conn.execute("DELETE FROM analysis_results WHERE created_at < ?", (cutoff,))
        return cur.rowcount


if __name__ == "__main__":
    n = evict_stale()
    print(f"Evicted {n} stale analysis cache entries.")
