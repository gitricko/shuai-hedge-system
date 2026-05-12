"""Institutional-format tear sheet (markdown).

Pulls together the daily P&L attribution history, win/loss summary,
factor + sector exposures, turnover, and core metrics vs SPY. Writes
to output/tear_sheet.md.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cache.db import REPO_ROOT, conn_ctx
from portfolio import factor_exposure, state as port_state
from portfolio.beta import book_beta, compute_betas
from reporting import pnl_attribution, sector_alpha, turnover, win_loss

log = logging.getLogger(__name__)

OUTPUT_PATH = REPO_ROOT / "output" / "tear_sheet.md"


def _portfolio_returns() -> pd.Series:
    hist = pnl_attribution.history()
    if hist.empty:
        return pd.Series(dtype=float)
    hist = hist.set_index("date").sort_index()
    return hist["total"]


def _spy_returns() -> pd.Series:
    with conn_ctx() as conn:
        df = pd.read_sql_query(
            "SELECT date, adj_close FROM daily_prices WHERE ticker='SPY' ORDER BY date", conn,
        )
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index(pd.to_datetime(df["date"]))["adj_close"]
    return s.pct_change().dropna()


def _metrics(rets: pd.Series, freq: int = 252) -> dict:
    if rets.empty:
        return {"ann_return": None, "ann_vol": None, "sharpe": None, "max_dd": None}
    mu = float(rets.mean()) * freq
    sigma = float(rets.std(ddof=1)) * np.sqrt(freq) if rets.std() else 0.0
    sharpe = mu / sigma if sigma else None
    curve = (1 + rets).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    return {"ann_return": mu, "ann_vol": sigma, "sharpe": sharpe, "max_dd": dd}


def _monthly_grid(rets: pd.Series) -> pd.DataFrame:
    if rets.empty:
        return pd.DataFrame()
    monthly = (1 + rets).resample("MS").prod() - 1
    grid = monthly.to_frame("ret")
    grid["year"] = grid.index.year
    grid["month"] = grid.index.month
    return grid.pivot_table(index="year", columns="month", values="ret")


def render() -> str:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rets = _portfolio_returns()
    spy = _spy_returns()
    fund_metrics = _metrics(rets)
    spy_metrics = _metrics(spy.reindex(rets.index).dropna() if not rets.empty else spy)

    pos = port_state.get_positions()
    weights = pd.Series(dtype=float)
    if not pos.empty:
        signs = pos["side"].map({"LONG": 1, "SHORT": -1})
        weights = pd.Series((pos["weight"] * signs).values, index=pos["ticker"])

    factor_df = factor_exposure.compute(weights) if not weights.empty else pd.DataFrame()
    sector_df = sector_alpha.compute()
    wl = win_loss.analyze()
    to_30 = turnover.turnover(30)
    to_90 = turnover.turnover(90)
    tax = turnover.tax_estimate()

    lines: list[str] = []
    lines.append("# Meridian Capital Partners — Tear Sheet")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")

    lines.append("## Performance vs SPY")
    lines.append("| Metric | Fund | SPY |")
    lines.append("|---|---:|---:|")
    for k, label in [("ann_return", "Annualized return"), ("ann_vol", "Annualized vol"),
                     ("sharpe", "Sharpe"), ("max_dd", "Max drawdown")]:
        f_v = fund_metrics.get(k)
        s_v = spy_metrics.get(k)
        fmt = (lambda v: f"{v:+.2%}" if v is not None and k != "sharpe" else (f"{v:.2f}" if v is not None else "—"))
        lines.append(f"| {label} | {fmt(f_v)} | {fmt(s_v)} |")
    lines.append("")

    if not rets.empty:
        grid = _monthly_grid(rets)
        if not grid.empty:
            lines.append("## Monthly returns")
            lines.append(grid.fillna(0).map(lambda v: f"{v*100:+.1f}%").to_markdown())
            lines.append("")

    if not weights.empty:
        betas = compute_betas(weights.index.tolist())
        bb = book_beta(weights, betas)
        lines.append("## Book exposures")
        lines.append(f"- Gross: **{weights.abs().sum()*100:.1f}%**  Net: **{weights.sum()*100:+.1f}%**")
        lines.append(f"- Beta — long {bb['long']:+.2f}  short {bb['short']:+.2f}  net **{bb['net']:+.2f}**")
        lines.append("")

    if not factor_df.empty:
        lines.append("## Factor exposures")
        keep = factor_df[["long_book_avg", "short_book_avg", "spread", "z_score"]].round(2)
        lines.append(keep.to_markdown())
        lines.append("")

    if not sector_df.empty:
        lines.append("## Sector alpha (90d)")
        s = sector_df[["n_picks", "avg_pick_return", "sector_etf_return", "alpha"]].round(4)
        lines.append(s.to_markdown())
        lines.append(f"\n**Total alpha**: {sector_df['alpha'].sum():+.2%}")
        lines.append("")

    overall = wl["overall"]
    if overall.get("n"):
        lines.append("## Win/Loss")
        wr = overall.get("win_rate")
        pl = overall.get("pl_ratio")
        lines.append(f"- N trades: {overall['n']}")
        lines.append(f"- Win rate: {wr * 100:.1f}%" if wr is not None else "- Win rate: —")
        lines.append(f"- P/L ratio: {pl:.2f}" if pl is not None else "- P/L ratio: —")
        lines.append(f"- Streaks — wins: {wl['streaks']['max_win_streak']}  losses: {wl['streaks']['max_loss_streak']}")
        lines.append("")

    lines.append("## Turnover")
    lines.append(f"- 30d turnover: {to_30['turnover_window']*100:.1f}%  (annualized {to_30['annualized']*100:.0f}%)")
    lines.append(f"- 90d turnover: {to_90['turnover_window']*100:.1f}%  (annualized {to_90['annualized']*100:.0f}%)")
    lines.append(f"- Tax estimate (realized): ${tax['total_tax']:,.0f}")
    lines.append("")

    text = "\n".join(lines)
    OUTPUT_PATH.write_text(text)
    return text


if __name__ == "__main__":
    print(render())
