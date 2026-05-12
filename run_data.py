"""Layer 1 entry point — orchestrates all data refreshes.

Usage:
    python run_data.py                   # full refresh
    python run_data.py --no-filings      # skip SEC pull (fast daily)
    python run_data.py --no-13f          # skip institutional 13-F pull
    python run_data.py --forms 10-K,4    # selective SEC forms

Logs to output/run.log. Prints a summary of all stage counts at the end.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from cache.db import REPO_ROOT, conn_ctx, init_schema
from data import (
    earnings_calendar,
    estimates,
    fundamentals,
    institutional,
    market_data,
    providers,
    sec_data,
    short_interest,
    transcripts,
    universe,
)

LOG_PATH = REPO_ROOT / "output" / "run.log"


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    handlers = [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _record_step(run_id: str, step: str, started: float, status: str, rows: int, msg: str = "") -> None:
    finished = time.time()
    with conn_ctx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO run_log (run_id, step, started_at, finished_at, status, rows_affected, message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, step,
                datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
                status, rows, msg,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 1 data refresh orchestrator")
    parser.add_argument("--no-filings", action="store_true", help="Skip SEC EDGAR pull")
    parser.add_argument("--no-13f", action="store_true", help="Skip 13-F institutional pull")
    parser.add_argument("--forms", default="", help="Comma-separated SEC form types to pull (e.g. 10-K,4)")
    parser.add_argument("--no-transcripts", action="store_true", help="Skip FMP transcripts")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()
    init_schema()

    log = logging.getLogger("run_data")
    run_id = uuid.uuid4().hex[:12]
    log.info("=== Meridian Layer 1 refresh start (run_id=%s) ===", run_id)
    providers.log_active_providers()

    summary: dict[str, int] = {}
    forms = [f.strip() for f in args.forms.split(",") if f.strip()] or None

    steps = [
        ("universe",          lambda: universe.refresh()),
        ("market_data",       lambda: market_data.refresh()),
        ("fundamentals",      lambda: fundamentals.refresh()),
        ("short_interest",    lambda: short_interest.refresh()),
        ("estimates",         lambda: estimates.refresh()),
        ("earnings_calendar", lambda: earnings_calendar.refresh()),
    ]

    if not args.no_transcripts:
        # Without candidate selection (Layer 2 not built yet) we no-op unless the
        # caller explicitly seeds tickers. Keeping the step here for orchestration.
        steps.append(("transcripts", lambda: transcripts.refresh([])))

    if not args.no_filings:
        steps.append(("sec_filings", lambda: sec_data.refresh_filings(forms=forms)[0]))

    if not args.no_13f:
        steps.append(("institutional_13f", lambda: institutional.refresh_13f()))

    for name, fn in steps:
        started = time.time()
        try:
            result = fn()
            rows = int(result) if isinstance(result, (int, float)) else 0
            summary[name] = rows
            _record_step(run_id, name, started, "ok", rows)
            log.info("[%s] ok rows=%d", name, rows)
        except Exception as exc:
            log.exception("[%s] FAILED: %s", name, exc)
            summary[name] = -1
            _record_step(run_id, name, started, "error", 0, str(exc))

    print("\n=== Refresh summary ===")
    for name, rows in summary.items():
        status = "FAIL" if rows < 0 else f"{rows:>10,d} rows"
        print(f"  {name:<22s} {status}")
    log.info("=== Meridian Layer 1 refresh complete (run_id=%s) ===", run_id)
    return 0 if all(v >= 0 for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
