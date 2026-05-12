"""Layer 4 entry point — portfolio construction.

Usage:
    python run_portfolio.py --whatif                   # show proposed trades, don't commit
    python run_portfolio.py --rebalance                # commit target weights to portfolio_positions
    python run_portfolio.py --current                  # show current positions
    python run_portfolio.py --optimize-method mvo      # use Markowitz (default: conviction)

Reads scored_universe (Layer 2) for candidates and target factor scores.
Writes portfolio_positions on --rebalance. The --whatif mode is the default
safety net — always preview before committing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from cache.db import REPO_ROOT
from portfolio import (
    factor_exposure,
    optimizer as conviction_optimizer,
    rebalance,
    rebalance_schedule,
    state,
)
from portfolio import mvo_optimizer
from portfolio.beta import book_beta, compute_betas


def _setup_logging() -> None:
    log_path = REPO_ROOT / "output" / "portfolio.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _print_current() -> int:
    df = state.get_positions()
    if df.empty:
        print("No open positions.")
        return 0
    print(df.to_string(index=False))
    return 0


def _commit(target_weights: pd.Series) -> None:
    """Write target weights into portfolio_positions (paper accounting only —
    actual share counts come from Layer 6 when broker fills land)."""
    today = date.today().isoformat()
    from cache.db import conn_ctx
    with conn_ctx() as conn:
        scored = pd.read_sql_query(
            "SELECT ticker, sector, composite, momentum, quality, value FROM scored_universe su "
            "WHERE score_date = (SELECT MAX(score_date) FROM scored_universe)",
            conn,
        ).set_index("ticker")
        # Wipe stale targets, then insert
        conn.execute("DELETE FROM portfolio_positions")
    for ticker, w in target_weights.items():
        if abs(w) < 1e-6:
            continue
        row = scored.loc[ticker] if ticker in scored.index else None
        sector = row["sector"] if row is not None else ""
        snapshot = {
            "composite": float(row["composite"]) if row is not None else 50.0,
            "momentum": float(row["momentum"]) if row is not None else 50.0,
            "quality": float(row["quality"]) if row is not None else 50.0,
            "value": float(row["value"]) if row is not None else 50.0,
        }
        state.upsert_position(state.Position(
            ticker=str(ticker),
            side="LONG" if w > 0 else "SHORT",
            shares=0.0,                      # filled when broker executes
            weight=abs(float(w)),
            entry_price=0.0,
            entry_date=today,
            sector=sector,
            factor_scores_at_entry=snapshot,
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 4 portfolio construction")
    parser.add_argument("--whatif", action="store_true", help="Show trades, don't commit (default)")
    parser.add_argument("--rebalance", action="store_true", help="Commit target weights to DB")
    parser.add_argument("--current", action="store_true", help="Print current positions")
    parser.add_argument("--optimize-method", choices=("conviction", "mvo"), default=None)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()
    log = logging.getLogger("run_portfolio")

    if args.current:
        return _print_current()

    method = args.optimize_method
    if method is None:
        import yaml
        method = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("portfolio", {}).get("optimize_method", "conviction")

    log.info("=== Layer 4 portfolio construction (method=%s) ===", method)

    if method == "mvo":
        target = mvo_optimizer.optimize()
    else:
        target = conviction_optimizer.optimize()

    if target.empty:
        log.error("Optimizer returned no weights")
        return 1

    print(f"\n=== Target weights ({method}) ===")
    print(f"  Gross: {target.abs().sum() * 100:.1f}%   "
          f"Net: {target.sum() * 100:+.1f}%   N: {len(target)}")

    longs = target[target > 0].sort_values(ascending=False)
    shorts = target[target < 0].sort_values()
    print(f"\nTop 5 LONG by weight:")
    print(longs.head(5).round(4).to_string())
    print(f"\nTop 5 SHORT by weight:")
    print(shorts.head(5).round(4).to_string())

    # Beta + factor exposure diagnostics
    betas = compute_betas(target.index.tolist())
    bb = book_beta(target, betas)
    print(f"\nBeta — long: {bb['long']:+.3f}  short: {bb['short']:+.3f}  net: {bb['net']:+.3f}")

    fe = factor_exposure.compute(target)
    if not fe.empty:
        alerts = fe[fe["alert"]]
        print(f"\nFactor spreads (alerts: {len(alerts)} factor(s) > 1 std):")
        print(fe[["long_book_avg", "short_book_avg", "spread", "z_score"]].round(2).to_string())

    # Calendar advisory
    adv = rebalance_schedule.advisory(target.index.tolist())
    if adv["earnings"] or adv["fomc"] or adv["opex"]:
        print(f"\nCalendar advisories:")
        if adv["earnings"]:
            print(f"  Earnings within 2d: {adv['earnings']}")
        if adv["fomc"]:
            print(f"  FOMC within 5d: {adv['fomc']}")
        if adv["opex"]:
            print(f"  Monthly opex within 3d: {adv['opex']}")

    # Trade list
    trades = rebalance.generate(target)
    if not trades.empty:
        print(f"\n=== Proposed trades ({len(trades)}) ===")
        print(trades[["ticker", "side", "action", "weight_delta", "est_cost_bps"]].head(20).to_string(index=False))

    if args.rebalance:
        _commit(target)
        log.info("Committed %d target weights to portfolio_positions", len(target))
        print(f"\n✓ Committed {len(target)} target weights.")
    else:
        print(f"\n(--whatif mode — nothing committed. Use --rebalance to write to DB.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
