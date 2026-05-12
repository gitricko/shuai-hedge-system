"""Layer 3 entry point — Claude AI analysis.

Usage:
    python run_analysis.py --estimate-cost     # dry-run, count tokens only
    python run_analysis.py --ticker AAPL       # analyze single ticker
    python run_analysis.py --sector "Information Technology"
    python run_analysis.py                     # full run on top/bottom quintile

Reads candidates from `scored_universe` (Layer 2 output). For each candidate
runs the four analyzers (earnings/filing/risk/insider), then computes the
combined 60/40 quant + Claude score and writes per-ticker markdown reports.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from cache.db import REPO_ROOT, conn_ctx
from analysis import (
    combined_score,
    earnings_analyzer,
    filing_analyzer,
    insider_analyzer,
    report_generator,
    risk_analyzer,
    sector_analysis,
)
from analysis.api_client import AnalysisClient
from analysis.cost_tracker import CostExceeded, CostTracker

log = logging.getLogger("run_analysis")


def _setup_logging() -> None:
    log_path = REPO_ROOT / "output" / "analysis.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _candidates(num: int = 20) -> pd.DataFrame:
    """Top `num` LONG and bottom `num` SHORT from the latest Layer 2 run."""
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scored_universe su WHERE score_date = "
            "(SELECT MAX(score_date) FROM scored_universe)",
            conn,
        ).set_index("ticker")
    if df.empty:
        return df
    longs = df[df["side"] == "LONG"].sort_values("composite", ascending=False).head(num)
    shorts = df[df["side"] == "SHORT"].sort_values("composite").head(num)
    return pd.concat([longs, shorts])


def _run_per_ticker(ticker: str, client: AnalysisClient) -> dict:
    """Run all four analyzers for one ticker. Returns analyzer->result map."""
    out: dict = {}
    for name, fn in [
        ("earnings", earnings_analyzer.analyze),
        ("filing",   filing_analyzer.analyze),
        ("risk",     risk_analyzer.analyze),
        ("insider",  insider_analyzer.analyze),
    ]:
        try:
            result = fn(ticker, client=client)
            if result is not None:
                out[name] = result
        except CostExceeded:
            raise
        except Exception as exc:
            log.exception("analyzer %s failed for %s: %s", name, ticker, exc)
    return out


def _estimate_cost(candidates: pd.DataFrame, client: AnalysisClient) -> float:
    """Cheap estimate via count_tokens() on a representative sample."""
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("analysis", {})
    prices = cfg.get("prices_per_million", {})
    sample = candidates.head(1)
    if sample.empty:
        return 0.0

    ticker = sample.index[0]
    # Stub: assume average input ~30K, output ~1.5K per analyzer call
    per_call_input = 30_000
    per_call_output = 1_500
    n_analyzers = 4
    n_tickers = len(candidates)

    input_cost = per_call_input * n_analyzers * n_tickers * prices.get("input", 3.0) / 1e6
    cache_savings = per_call_input * n_analyzers * (n_tickers - 1) * (prices.get("input", 3.0) - prices.get("cache_read", 0.30)) / 1e6
    output_cost = per_call_output * n_analyzers * n_tickers * prices.get("output", 15.0) / 1e6

    estimated = input_cost - cache_savings + output_cost
    print(f"\n=== Cost estimate ===")
    print(f"  Candidates:           {n_tickers}")
    print(f"  Analyzers per ticker: {n_analyzers}")
    print(f"  Avg input tokens:     {per_call_input:,d} (cached after first call)")
    print(f"  Avg output tokens:    {per_call_output:,d}")
    print(f"  Estimated cost:       ${estimated:.2f}  (ceiling ${cfg.get('cost_ceiling_usd', 25):.2f})")
    print()
    return estimated


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 3 Claude analysis")
    parser.add_argument("--estimate-cost", action="store_true", help="Print cost estimate, don't call API")
    parser.add_argument("--ticker", help="Single ticker mode")
    parser.add_argument("--sector", help="Sector aggregation mode")
    parser.add_argument("--num", type=int, default=20, help="Candidates per side (default 20)")
    parser.add_argument("--no-reports", action="store_true", help="Skip markdown report generation")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()

    tracker = CostTracker.from_config()
    client = AnalysisClient(cost_tracker=tracker)
    log.info("=== Meridian Layer 3 analysis start (run_id=%s, model=%s) ===",
             tracker.run_id, client.model)

    if args.ticker:
        candidates = pd.DataFrame({"side": ["LONG"]}, index=[args.ticker.upper()])
    elif args.sector:
        with conn_ctx() as conn:
            tickers = [r["ticker"] for r in conn.execute(
                "SELECT ticker FROM universe WHERE gics_sector = ? AND is_benchmark = 0",
                (args.sector,),
            ).fetchall()]
        candidates = pd.DataFrame({"side": ["LONG"] * len(tickers)}, index=tickers)
    else:
        candidates = _candidates(args.num)

    if candidates.empty:
        print("No candidates found. Run `python run_scoring.py` first.")
        return 1

    if args.estimate_cost:
        _estimate_cost(candidates, client)
        return 0

    print(f"Analyzing {len(candidates)} tickers with {client.model}...")
    n_done = 0
    try:
        for ticker in candidates.index:
            results = _run_per_ticker(str(ticker), client)
            n_done += 1
            n_results = len(results)
            log.info("[%d/%d] %s: %d analyzers complete (cum cost $%.4f)",
                     n_done, len(candidates), ticker, n_results,
                     tracker.cumulative_usd)
    except CostExceeded as exc:
        log.error("ABORTED: %s", exc)

    print("\n=== Cost summary ===")
    for k, v in tracker.summary().items():
        print(f"  {k:<20} {v}")

    # Sector aggregation when requested
    if args.sector:
        agg = sector_analysis.analyze(args.sector, client=client)
        if agg:
            out_path = REPO_ROOT / "output" / f"sector_{args.sector.replace(' ', '_')}.json"
            out_path.write_text(json.dumps(agg, indent=2))
            print(f"\nSector summary -> {out_path}")

    # Combined 60/40 score + reports
    if not args.no_reports and not args.ticker:
        log.info("Computing combined quant + Claude scores")
        combined = combined_score.compute()
        out_dir = report_generator.write_reports(candidates)
        print(f"\nReports written to {out_dir}")

    log.info("=== Layer 3 complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
