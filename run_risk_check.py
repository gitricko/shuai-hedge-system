"""Layer 5 entry point — full risk dashboard.

Usage:
    python run_risk_check.py              # full risk snapshot
    python run_risk_check.py --stress     # include stress scenarios
    python run_risk_check.py --tail-only  # just VIX + credit spread
    python run_risk_check.py --clear-halt # clear the halt lock

Reads target weights from portfolio_positions (Layer 4 output). Writes
cache/risk_state.json with the latest snapshot.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from cache.db import REPO_ROOT, conn_ctx
from portfolio import state as port_state
from portfolio.beta import book_beta, compute_betas
from risk import (
    clear_halt,
    is_halted,
    reason_for_halt,
    circuit_breakers,
    correlation_monitor,
    factor_monitor,
    factor_risk_model,
    risk_state,
    stress_test,
    tail_risk,
)


def _setup_logging() -> None:
    log_path = REPO_ROOT / "output" / "risk.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _current_weights() -> pd.Series:
    df = port_state.get_positions()
    if df.empty:
        return pd.Series(dtype=float)
    signs = df["side"].map({"LONG": 1, "SHORT": -1})
    return pd.Series((df["weight"] * signs).values, index=df["ticker"], name="weight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 5 risk check")
    parser.add_argument("--stress", action="store_true", help="Run stress scenarios")
    parser.add_argument("--tail-only", action="store_true", help="Only tail-risk monitor")
    parser.add_argument("--clear-halt", action="store_true", help="Clear halt lock (override KILL_SWITCH)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()
    log = logging.getLogger("run_risk_check")

    if args.clear_halt:
        cleared = clear_halt()
        print("✓ Halt cleared" if cleared else "(no halt was set)")
        return 0

    if is_halted():
        print(f"⚠ HALT ACTIVE: {reason_for_halt()}")
        print("Use --clear-halt to override after investigating.")
        # continue anyway so we report current state

    print("\n=== Tail risk ===")
    tail = tail_risk.evaluate()
    print(f"  VIX:               {tail.vix}")
    print(f"  Credit-spread z:   {tail.credit_spread_z}")
    print(f"  Directive:         {tail.directive or 'NONE'}")
    if tail.reasons:
        for r in tail.reasons:
            print(f"    - {r}")

    if args.tail_only:
        risk_state.update(tail_risk={
            "vix": tail.vix,
            "credit_spread_z": tail.credit_spread_z,
            "directive": tail.directive,
        })
        return 0

    weights = _current_weights()
    if weights.empty:
        print("\n(No positions in portfolio_positions — run `python run_portfolio.py --rebalance` first.)")
        return 0

    print(f"\n=== Portfolio risk on {len(weights)} positions ===")
    print(f"  Gross: {weights.abs().sum() * 100:.1f}%   Net: {weights.sum() * 100:+.1f}%")

    # Beta
    betas = compute_betas(weights.index.tolist())
    bb = book_beta(weights, betas)
    print(f"  Beta — long {bb['long']:+.3f}  short {bb['short']:+.3f}  net {bb['net']:+.3f}")

    # Factor risk model
    print("\n=== Barra factor risk model ===")
    try:
        model = factor_risk_model.fit(lookback=120)
        pr = factor_risk_model.portfolio_risk(weights, model)
        print(f"  Total var (ann):     {pr['total_var']:.4f}")
        print(f"  Vol (ann):           {pr['vol_annualized']:.4f}")
        if pr["total_var"]:
            print(f"  Factor share:        {pr['factor_var'] / pr['total_var'] * 100:.1f}%")
            print(f"  Specific share:      {pr['specific_var'] / pr['total_var'] * 100:.1f}%")
        # Top 5 MCTR
        top_mctr = pr["mctr"].head(5)
        print("  Top 5 MCTR contributors:")
        for t, v in top_mctr.items():
            print(f"    {t:<6}  MCTR={v:+.4f}  weight={weights[t]:+.4f}")
    except Exception as exc:
        log.exception("factor risk model failed: %s", exc)

    # Factor + correlation monitors
    print("\n=== Factor spread alerts ===")
    f_alerts = factor_monitor.evaluate(weights)
    if f_alerts:
        for a in f_alerts:
            print(f"  [{a['priority']}] {a['factor']:<14} spread={a['spread']:+.1f}  z={a['z_score']:+.2f}  crowded={a['crowded']}")
    else:
        print("  (no factor-spread breaches)")

    print("\n=== Correlation monitor ===")
    corr_report = correlation_monitor.evaluate(weights)
    print(f"  Long  avg |corr| {corr_report.long_avg_corr:.2f}   effective bets {corr_report.long_effective_bets:.1f}")
    print(f"  Short avg |corr| {corr_report.short_avg_corr:.2f}   effective bets {corr_report.short_effective_bets:.1f}")
    if corr_report.alert:
        for r in corr_report.reasons:
            print(f"  ⚠ {r}")

    # Circuit breakers
    print("\n=== Circuit breakers ===")
    events = circuit_breakers.evaluate()
    if events:
        for e in events:
            print(f"  [{e.severity}] {e.action.value}  {e.reason}")
    else:
        print("  (no breakers triggered)")

    if args.stress:
        print("\n=== Stress tests ===")
        scenarios = stress_test.run(weights)
        for s in scenarios:
            print(f"  {s.name:<35}  P&L {s.pnl_pct * 100:+.2f}%  "
                  f"(long {s.long_contribution / 1e6:+.2f}M / short {s.short_contribution / 1e6:+.2f}M)")

    # Persist snapshot
    risk_state.update(
        as_of=date.today().isoformat(),
        exposures={
            "gross": float(weights.abs().sum()),
            "net": float(weights.sum()),
            "long_beta": bb["long"],
            "short_beta": bb["short"],
            "net_beta": bb["net"],
        },
        tail_risk={
            "vix": tail.vix,
            "credit_spread_z": tail.credit_spread_z,
            "directive": tail.directive,
        },
        factor={
            "alerts": f_alerts,
        },
        correlation={
            "long_avg": corr_report.long_avg_corr,
            "short_avg": corr_report.short_avg_corr,
            "effective_bets": corr_report.long_effective_bets,
            "alert": corr_report.alert,
        },
    )
    print(f"\nRisk snapshot written to {risk_state.STATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
