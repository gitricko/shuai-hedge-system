"""Layer 6 entry point — Alpaca paper-trading execution.

Usage:
    python run_execution.py --dry-run     # log what would happen, no orders
    python run_execution.py --execute     # actually place orders (paper default)

Reads target weights from portfolio_positions (Layer 4 output), generates
a trade list (delta vs current Alpaca positions on --execute, or vs empty
on first run), runs each trade through the executor pipeline:
  pre-trade veto -> short check -> limit -> chunk -> submit -> poll -> retry.

Final report:
  - Fills (count, total slippage bps, dollar cost)
  - Rejections and reasons
  - Open orders left in flight (should be 0 unless SIGINT)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd
import yaml
from dotenv import load_dotenv

from cache.db import REPO_ROOT, conn_ctx
from execution import costs, order_manager as om
from execution.broker import Broker
from execution.executor import TradeOrder, execute_one
from portfolio import state as port_state


def _setup_logging() -> None:
    log_path = REPO_ROOT / "output" / "execution.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _build_trade_list(broker: Broker, nav: float) -> list[TradeOrder]:
    """Diff target portfolio (portfolio_positions) vs broker positions."""
    targets = port_state.get_positions()
    snap = broker.sync() if broker else None
    broker_pos = snap.open_positions if snap else {}

    # Signed dollar exposure for each target ticker
    target_dollars: dict[str, float] = {}
    for _, row in targets.iterrows():
        sign = 1 if row["side"] == "LONG" else -1
        target_dollars[row["ticker"]] = sign * float(row["weight"]) * nav

    current_dollars: dict[str, float] = {}
    for ticker, info in broker_pos.items():
        current_dollars[ticker] = float(info["market_value"])

    universe = sorted(set(target_dollars.keys()) | set(current_dollars.keys()))
    trades: list[TradeOrder] = []
    for ticker in universe:
        target = target_dollars.get(ticker, 0.0)
        current = current_dollars.get(ticker, 0.0)
        delta = target - current
        if abs(delta) < 100:  # ignore micro-deltas
            continue
        target_side = "LONG" if target >= 0 else "SHORT"
        is_closing = (abs(target) < 1.0)
        if is_closing:
            intent = "CLOSE_LONG" if current > 0 else "COVER_SHORT"
        elif current == 0:
            intent = "OPEN_LONG" if target > 0 else "OPEN_SHORT"
        elif (target > 0) == (current > 0):
            intent = "OPEN_LONG" if target > 0 else "OPEN_SHORT"  # increase
        else:
            # Side flip — close existing, then open the new side. Simplified
            # as a single intent here; the broker handles direction.
            intent = "OPEN_LONG" if target > 0 else "OPEN_SHORT"
        trades.append(TradeOrder(
            ticker=ticker, side=target_side, intent=intent,
            target_dollars=delta, is_closing=is_closing,
        ))
    return trades


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 6 execution (Alpaca paper trading)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Log what would happen, no orders")
    g.add_argument("--execute", action="store_true", help="Place orders (paper by default)")
    parser.add_argument("--max-trades", type=int, default=None,
                        help="Cap number of trades (useful for incremental rollout)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    _setup_logging()
    log = logging.getLogger("run_execution")

    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("execution", {})
    nav = float(cfg.get("nav_assumption", 100_000_000.0))

    try:
        broker = Broker()
    except RuntimeError as exc:
        log.error(str(exc))
        return 1
    except Exception as exc:
        log.exception("Broker init failed: %s", exc)
        return 1

    log.info("=== Layer 6 execution start (paper=%s, dry_run=%s) ===",
             broker.paper, args.dry_run)

    snap = broker.sync()
    log.info("Broker NAV=$%.0f cash=$%.0f positions=%d",
             snap.equity, snap.cash, len(snap.open_positions))

    trades = _build_trade_list(broker, nav=nav)
    if args.max_trades:
        trades = trades[: args.max_trades]

    if not trades:
        print("\nNo trades to execute — portfolio aligned with target.")
        return 0

    print(f"\n=== {'DRY RUN: ' if args.dry_run else ''}{len(trades)} trades queued ===")
    for t in trades[:30]:
        print(f"  {t.intent:<13} {t.ticker:<6}  ${t.target_dollars:+,.0f}")

    fills = 0
    rejects = 0
    total_slippage_dollars = 0.0
    rejection_reasons: list[tuple[str, list[str]]] = []

    for trade in trades:
        if om.should_shutdown():
            log.warning("Shutdown flag set — stopping after %d trades", fills + rejects)
            break
        res = execute_one(broker, trade, dry_run=args.dry_run)
        if res["rejected"]:
            rejects += 1
            rejection_reasons.append((res["ticker"], res["reasons"]))
        else:
            fills += len(res["fills"])
            for f in res["fills"]:
                total_slippage_dollars += abs(f["slippage_bps"]) / 10_000 * f["price"] * f["qty"]

    print(f"\n=== Execution summary ===")
    print(f"  Mode:             {'PAPER' if broker.paper else 'LIVE'}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"  Fills:            {fills}")
    print(f"  Rejections:       {rejects}")
    print(f"  Total slippage:   ${total_slippage_dollars:,.2f}")

    if rejection_reasons:
        print(f"\n  Rejection details (first 10):")
        for tkr, reasons in rejection_reasons[:10]:
            print(f"    {tkr}: {' | '.join(reasons)}")

    stats = costs.rolling_stats()
    if stats["n_fills"]:
        print(f"\n=== 30-day slippage rolling ===")
        print(f"  Fills:        {stats['n_fills']}")
        print(f"  Avg bps:      {stats['avg_bps']:+.2f}")
        print(f"  Median bps:   {stats['median_bps']:+.2f}")
        print(f"  p95 bps:      {stats['p95_bps']:+.2f}")
        print(f"  Total $:      ${stats['total_dollar']:,.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
