"""Layer 2 entry point — scoring engine.

Usage:
    python run_scoring.py                  # full universe, default weights
    python run_scoring.py --regime         # VIX-conditional weights
    python run_scoring.py --ticker AAPL    # single-stock breakdown
    python run_scoring.py --sector "Information Technology"   # filter

Writes:
    output/scored_universe_latest.csv
    SQLite: scored_universe table

Prints summary: top 5 longs / shorts, crowding warnings, degenerate factors.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cache.db import REPO_ROOT
from factors.composite import compute_all
from factors.crowding import detect
from factors.regime_weights import latest_vix, select_weights


def _setup_logging() -> None:
    log_path = REPO_ROOT / "output" / "scoring.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _format_top(df: pd.DataFrame, side: str, n: int = 5) -> str:
    sub = df[df["side"] == side]
    if side == "SHORT":
        sub = sub.sort_values("composite")
    sub = sub.head(n)
    cols = ["sector", "composite", "momentum", "quality", "value"]
    return sub[cols].to_string(float_format=lambda x: f"{x:6.1f}")


def _print_ticker(df: pd.DataFrame, ticker: str) -> int:
    if ticker not in df.index:
        print(f"Ticker {ticker} not in scored universe.")
        return 1
    row = df.loc[ticker]
    print(f"\n=== {ticker} ({row['sector']}) ===")
    print(f"Composite: {row['composite']:.1f}    Side: {row['side'] or '—'}")
    print("\nFactor breakdown:")
    for f in ["momentum", "quality", "value", "revisions",
              "insider", "growth", "short_int", "institutional"]:
        print(f"  {f:<14} {row[f]:6.1f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 2 scoring engine")
    parser.add_argument("--regime", action="store_true",
                        help="Use VIX-conditional weights")
    parser.add_argument("--ticker", help="Show breakdown for a single ticker")
    parser.add_argument("--sector", help="Filter output to a single GICS sector")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write CSV/DB — for read-only inspection")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()
    log = logging.getLogger("run_scoring")

    regime_name, _ = select_weights(enable=args.regime)
    vix = latest_vix()
    log.info("=== Meridian Layer 2 scoring start (regime=%s, VIX=%s) ===",
             regime_name, vix)

    df = compute_all(use_regime=args.regime, save=not args.no_save)

    if args.ticker:
        return _print_ticker(df, args.ticker.upper())

    if args.sector:
        df = df[df["sector"] == args.sector]
        if df.empty:
            print(f"No tickers in sector '{args.sector}'.")
            return 1

    diag = detect(df)

    print("\n=== Top 5 LONG candidates ===")
    print(_format_top(df, "LONG"))
    print("\n=== Top 5 SHORT candidates ===")
    print(_format_top(df, "SHORT"))

    if diag["flags"]:
        print("\n⚠ Crowding warnings:")
        for f in diag["flags"]:
            print(f"  {f['pair']:<20} actual={f['actual']:+.2f} baseline={f['baseline']:+.2f}")
    else:
        print("\nNo crowding warnings.")

    long_n = (df["side"] == "LONG").sum()
    short_n = (df["side"] == "SHORT").sum()
    print(f"\nUniverse: {len(df)} | LONGS: {long_n} | SHORTS: {short_n}")
    print(f"Output: {REPO_ROOT / 'output' / 'scored_universe_latest.csv'}")

    log.info("=== Scoring complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
