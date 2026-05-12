"""Streamlit dashboard — 6 pages.

Run with:
    streamlit run dashboard/app.py

Pages:
    I.  Overview         — KPIs, NAV, gross/net/beta gauges
    II. Positions        — current book table
    III. Factor & Risk   — Layer 5 outputs (factor model, MCTR, correlation)
    IV. Performance      — equity curve vs SPY, monthly grid, attribution
    V.  Execution        — open orders, slippage, recent trades
    VI. LP Letter        — formal daily letter; regenerate button
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root on path so we can import sibling packages
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.db import conn_ctx                              # noqa: E402
from execution import costs as exec_costs                  # noqa: E402
from portfolio import factor_exposure, state as port_state  # noqa: E402
from portfolio.beta import book_beta, compute_betas         # noqa: E402
from reporting import (                                     # noqa: E402
    lp_letter,
    pnl_attribution,
    position_attribution,
    sector_alpha,
    turnover as turnover_mod,
    win_loss,
)
from risk import is_halted, reason_for_halt, risk_state     # noqa: E402

st.set_page_config(page_title="Meridian Capital Partners", layout="wide")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _signed_weights() -> pd.Series:
    df = port_state.get_positions()
    if df.empty:
        return pd.Series(dtype=float)
    signs = df["side"].map({"LONG": 1, "SHORT": -1})
    return pd.Series((df["weight"] * signs).values, index=df["ticker"])


def _during_market_hours() -> bool:
    now = datetime.now().time()
    return dtime(9, 30) <= now <= dtime(16, 0)


# Auto-refresh during market hours
if _during_market_hours():
    st.markdown(
        "<script>setTimeout(() => window.location.reload(), 300000);</script>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Sidebar — navigation + halt banner
# ----------------------------------------------------------------------
PAGES = ["0. Start Here", "I. Overview", "II. Positions", "III. Factor & Risk",
         "IV. Performance", "V. Execution", "VI. LP Letter"]
page = st.sidebar.radio("Pages", PAGES)

st.sidebar.markdown("---")
if is_halted():
    st.sidebar.error(f"⚠ HALT ACTIVE\n\n{reason_for_halt()}")
else:
    st.sidebar.success("✓ No halt")

st.sidebar.markdown(f"_Last refresh: {datetime.now().strftime('%H:%M:%S')}_")
if _during_market_hours():
    st.sidebar.caption("Auto-refresh every 5 min (market hours)")


# ----------------------------------------------------------------------
# PAGE 0 — Start Here (plain-English intro for non-finance users)
# ----------------------------------------------------------------------
def render_intro():
    st.title("👋 Start Here")
    st.markdown(
        "Welcome to **Meridian Capital Partners** — a simulated hedge fund built "
        "from the ground up. This page explains what this tool does and how to "
        "use it, assuming you've never invested before."
    )

    st.markdown("## What is this tool?")
    st.markdown(
        "Every day, this system looks at the 500 largest U.S. companies "
        "(the **S&P 500**), scores each one from 0 to 100, and tells you which "
        "ones look strong (to **buy**) and which look weak (to **bet against**)."
        "\n\n"
        "It then builds a portfolio, checks the risk, and — if you connect a "
        "broker — can place the trades automatically (with fake money in "
        "*paper-trading* mode by default)."
    )

    st.markdown("## The big idea in 30 seconds")
    st.info(
        "**Long** = buying a stock because you think it will go UP.  \n"
        "**Short** = borrowing a stock, selling it now, buying it back later "
        "for cheaper — profitable when the stock goes DOWN.  \n"
        "**Hedge fund** = a fund that does BOTH at the same time. If the "
        "market crashes, the shorts make money, cushioning the longs."
    )

    st.markdown("### A simple example")
    st.markdown(
        "Imagine the system ranks all 500 stocks today:"
        "\n\n"
        "- It picks the **top 20** (e.g., Apple, Costco, Google) — these are **longs**\n"
        "- It picks the **bottom 20** (e.g., Nike, some struggling retailer) — these are **shorts**\n"
        "\n"
        "If the market drops 5% next week:\n"
        "- Your longs might drop 4% (better than market) → you lose 4%\n"
        "- Your shorts might drop 6% (worse than market) → you *make* 6%\n"
        "- **Net result:** roughly flat instead of -5%. That's the hedge."
    )

    st.markdown("## What gets scored?")
    st.markdown(
        "Each stock is graded on **8 different angles**, in plain English:"
    )
    st.markdown(
        "| Factor | What it asks |\n"
        "|---|---|\n"
        "| **Momentum** | Has the price been going up lately? |\n"
        "| **Quality** | Is this a healthy, profitable business? |\n"
        "| **Value** | Is it cheap relative to its earnings/assets? |\n"
        "| **Growth** | Is the business getting bigger? |\n"
        "| **Estimate Revisions** | Are Wall Street analysts getting more optimistic? |\n"
        "| **Insider Activity** | Are the CEO/CFO buying their own stock? (bullish signal) |\n"
        "| **Short Interest** | Are other people betting against this stock? |\n"
        "| **Institutional Flow** | Are big hedge funds piling in or bailing out? |\n"
    )
    st.markdown(
        "Each factor gives a 0–100 score. The system combines them with weights "
        "(Momentum and Quality each get 20%, Value 15%, etc.) into one **composite** "
        "score per stock. Top 20% of scores = LONG candidates. Bottom 20% = SHORT."
    )

    st.markdown("## What each page shows you")
    st.markdown(
        "| Page | What it answers |\n"
        "|---|---|\n"
        "| **I. Overview** | What's the portfolio doing right now? Are exposures balanced? |\n"
        "| **II. Positions** | Which 40 stocks am I holding? Which are winning/losing? |\n"
        "| **III. Factor & Risk** | How risky is this? What could go wrong? |\n"
        "| **IV. Performance** | How am I doing over time? Win rate? Sharpe ratio? |\n"
        "| **V. Execution** | What trades happened? How much was I overcharged on price? |\n"
        "| **VI. LP Letter** | A formal daily letter you could send to investors |\n"
    )

    st.markdown("## How to actually use it")
    st.markdown(
        "Open a terminal in the project folder. The daily workflow is four commands:"
    )
    st.code(
        "# 1. Download fresh prices (~10 min)\n"
        ".venv/bin/python run_data.py --no-filings --no-13f\n\n"
        "# 2. Re-rank all 503 stocks (~2 seconds)\n"
        ".venv/bin/python run_scoring.py\n\n"
        "# 3. Build the portfolio — pick which 40 to hold (~10 sec)\n"
        ".venv/bin/python run_portfolio.py --rebalance\n\n"
        "# 4. Check the risk before trading anything (~30 sec)\n"
        ".venv/bin/python run_risk_check.py",
        language="bash",
    )
    st.markdown(
        "After that, refresh this dashboard. The numbers on every page update."
    )

    st.markdown("### One-off questions")
    st.code(
        "# What's the system's view on a specific stock?\n"
        ".venv/bin/python run_scoring.py --ticker AAPL\n\n"
        "# Preview proposed trades WITHOUT committing them\n"
        ".venv/bin/python run_portfolio.py --whatif\n\n"
        "# Test the portfolio against historical disasters (2008, COVID, etc.)\n"
        ".venv/bin/python run_risk_check.py --stress",
        language="bash",
    )

    st.markdown("## A quick glossary")
    with st.expander("Click to expand — terms you'll see across the pages"):
        st.markdown(
            "- **Gross exposure**: total money at risk. 150% gross on a $100 fund "
            "means $150 of stocks are in play ($75 long + $75 short, roughly).\n"
            "- **Net exposure**: longs minus shorts. +5% net means you're slightly "
            "betting the market goes up; 0% is fully market-neutral.\n"
            "- **Beta**: how much a stock moves when the market moves. Beta of 1.0 "
            "matches the market; 2.0 moves twice as much; 0.0 is uncorrelated.\n"
            "- **Sharpe ratio**: return per unit of risk. Above 1.0 is good, above "
            "2.0 is excellent.\n"
            "- **Drawdown**: peak-to-trough loss. \"8% drawdown\" means the fund "
            "is down 8% from its high-water mark.\n"
            "- **NAV**: Net Asset Value — the total dollar value of the fund.\n"
            "- **Slippage**: when you wanted to buy at $100 but actually paid "
            "$100.05 — that 5¢ extra is slippage cost.\n"
            "- **MCTR**: Marginal Contribution to Risk — which single stock is "
            "adding the most risk to your portfolio.\n"
            "- **VIX**: the market's \"fear gauge.\" Above 25 = nervous; "
            "above 35 = panic; below 15 = complacent.\n"
            "- **Sector**: industry category. The system tracks 11 sectors "
            "(Tech, Health Care, Financials, Energy, etc.).\n"
            "- **Quintile**: a 20% slice. \"Top quintile\" = best 20% of scores.\n"
            "- **Form 4**: a legal filing CEOs/CFOs submit when they buy or sell "
            "shares in their own company. Big insider buys are a bullish signal.\n"
            "- **10-K / 10-Q**: annual / quarterly reports companies file with the SEC.\n"
            "- **Long/Short equity**: the strategy this system implements — buy "
            "good stocks, short bad ones."
        )

    st.markdown("## Important things to know")
    st.warning(
        "**This is a research tool, not investment advice.** The system can be "
        "wrong. It's been running for days, not years. Before risking real "
        "money: paper-trade for at least 3 months, study the results, understand "
        "every page of this dashboard."
    )
    st.markdown(
        "- **Paper trading is the default.** If you connect to Alpaca, it uses "
        "fake money unless you explicitly switch to live mode AND type "
        "`YES I UNDERSTAND THE RISKS` at the prompt.\n"
        "- **You don't need any API keys to use the core system.** Scoring, "
        "portfolio construction, and risk all work on local data. API keys are "
        "only needed for (a) AI-written letters and analysis (Anthropic), or "
        "(b) actually placing trades (Alpaca).\n"
        "- **Halt button:** If anything looks broken, Layer 5's kill-switch can "
        "halt all trading. Look for the red banner in the sidebar."
    )

    st.markdown("## Want to dig deeper?")
    st.markdown(
        "- Open `README_CLAUDE_PRO.md` in the repo for the full original spec.\n"
        "- Open `PROGRESS.md` to see what's built and what's pending.\n"
        "- All code is on GitHub: https://github.com/gnuef-bot/Hedge-fund-system"
    )

    st.success("Ready? Click **I. Overview** in the sidebar to see today's portfolio.")


# ----------------------------------------------------------------------
# PAGE I — Overview
# ----------------------------------------------------------------------
def render_overview():
    st.title("Meridian Capital Partners")
    st.caption("Multi-agent LLM-powered long/short equity hedge fund")

    weights = _signed_weights()
    if weights.empty:
        st.warning("No positions yet. Run `python run_portfolio.py --rebalance`.")
        return

    betas = compute_betas(weights.index.tolist())
    bb = book_beta(weights, betas)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gross", f"{weights.abs().sum() * 100:.1f}%")
    c2.metric("Net", f"{weights.sum() * 100:+.1f}%")
    c3.metric("Net beta", f"{bb['net']:+.2f}")
    c4.metric("Positions", len(weights))

    risk = risk_state.load()
    tail = risk.get("tail_risk", {})
    c5.metric("VIX", f"{tail.get('vix', '—')}",
              delta=tail.get("directive") or "no directive",
              delta_color="inverse" if tail.get("directive") else "off")

    st.markdown("### Factor exposures")
    fe = factor_exposure.compute(weights)
    if not fe.empty:
        st.dataframe(
            fe[["long_book_avg", "short_book_avg", "spread", "z_score", "alert"]]
                .style.format("{:.2f}", subset=["long_book_avg", "short_book_avg", "spread", "z_score"]),
            use_container_width=True,
        )

    s = sector_alpha.summary()
    if s.get("by_sector"):
        st.markdown(f"### Sector alpha (90d)  · total **{s['total_alpha']*100:+.2f}%**")
        st.dataframe(pd.DataFrame(s["by_sector"]).set_index("sector"), use_container_width=True)


# ----------------------------------------------------------------------
# PAGE II — Positions
# ----------------------------------------------------------------------
def render_positions():
    st.title("Positions")
    df = position_attribution.mark_to_market()
    if df.empty:
        st.info("No open positions.")
        return
    st.dataframe(df.style.format({
        "weight": "{:.2%}", "entry_price": "${:,.2f}", "current_price": "${:,.2f}",
        "unrealized_pl": "${:,.0f}", "market_value": "${:,.0f}",
    }), use_container_width=True)

    bw = position_attribution.best_worst_per_side(5)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Best 5 longs**")
        st.dataframe(bw["long_best"][["unrealized_pl"]])
        st.markdown("**Worst 5 longs**")
        st.dataframe(bw["long_worst"][["unrealized_pl"]])
    with cols[1]:
        st.markdown("**Best 5 shorts**")
        st.dataframe(bw["short_best"][["unrealized_pl"]])
        st.markdown("**Worst 5 shorts**")
        st.dataframe(bw["short_worst"][["unrealized_pl"]])


# ----------------------------------------------------------------------
# PAGE III — Factor & Risk
# ----------------------------------------------------------------------
def render_factor_risk():
    st.title("Factor & Risk")
    risk = risk_state.load()
    exp = risk.get("exposures") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Gross", f"{(exp.get('gross') or 0) * 100:.1f}%")
    c2.metric("Net", f"{(exp.get('net') or 0) * 100:+.1f}%")
    c3.metric("Net beta", f"{exp.get('net_beta') or 0:+.2f}")

    st.markdown("### Factor alerts")
    alerts = (risk.get("factor") or {}).get("alerts") or []
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.success("No factor-spread breaches.")

    st.markdown("### Correlation")
    corr = risk.get("correlation") or {}
    cc1, cc2 = st.columns(2)
    cc1.metric("Long avg |corr|", f"{corr.get('long_avg') or 0:.2f}")
    cc2.metric("Short avg |corr|", f"{corr.get('short_avg') or 0:.2f}")
    st.metric("Effective bets (long)", f"{corr.get('effective_bets') or 0:.1f}")

    st.markdown("### Tail risk")
    tail = risk.get("tail_risk") or {}
    st.json(tail)


# ----------------------------------------------------------------------
# PAGE IV — Performance
# ----------------------------------------------------------------------
def render_performance():
    st.title("Performance")
    hist = pnl_attribution.history()
    if hist.empty:
        st.info("No attribution history yet. Run `python run_scoring.py` then "
                "`python -m reporting.pnl_attribution`.")
    else:
        hist = hist.sort_values("date")
        hist["equity"] = (1 + hist["total"]).cumprod()
        st.line_chart(hist.set_index("date")[["equity"]], use_container_width=True)

        st.markdown("### Attribution decomposition")
        st.bar_chart(hist.set_index("date")[["beta", "sector", "factor", "alpha"]],
                     use_container_width=True)

    wl = win_loss.analyze()
    overall = wl.get("overall", {})
    if overall.get("n"):
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Trades", overall.get("n", 0))
        cc2.metric("Win rate", f"{(overall.get('win_rate') or 0) * 100:.1f}%")
        pl = overall.get("pl_ratio")
        cc3.metric("P/L ratio", f"{pl:.2f}" if pl else "—")
        cc4.metric("Total realized P&L", f"${overall.get('total_pl') or 0:,.0f}")

        st.markdown("### By holding period")
        st.dataframe(pd.DataFrame(wl.get("by_bucket", {})).T, use_container_width=True)

    st.markdown("### Turnover")
    t30 = turnover_mod.turnover(30)
    t90 = turnover_mod.turnover(90)
    cc1, cc2 = st.columns(2)
    cc1.metric("30d turnover", f"{t30['turnover_window']*100:.1f}%",
               delta=f"annualized {t30['annualized']*100:.0f}%")
    cc2.metric("90d turnover", f"{t90['turnover_window']*100:.1f}%",
               delta=f"annualized {t90['annualized']*100:.0f}%")


# ----------------------------------------------------------------------
# PAGE V — Execution
# ----------------------------------------------------------------------
def render_execution():
    st.title("Execution")
    stats = exec_costs.rolling_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fills (30d)", stats["n_fills"])
    c2.metric("Avg slippage", f"{stats['avg_bps']:+.1f} bps")
    c3.metric("p95 slippage", f"{stats['p95_bps']:+.1f} bps")
    c4.metric("Total $ cost", f"${stats['total_dollar']:,.0f}")

    st.markdown("### Worst 5 fills (30d)")
    st.dataframe(exec_costs.worst_fills(5), use_container_width=True)

    st.markdown("### Recent orders (200)")
    with conn_ctx() as conn:
        orders = pd.read_sql_query(
            "SELECT ticker, side, intent, qty, limit_price, signal_price, "
            "status, submitted_at, filled_at, notes "
            "FROM orders ORDER BY submitted_at DESC LIMIT 200", conn,
        )
    if orders.empty:
        st.info("No orders recorded yet. Run `python run_execution.py --execute`.")
    else:
        st.dataframe(orders, use_container_width=True)


# ----------------------------------------------------------------------
# PAGE VI — LP Letter
# ----------------------------------------------------------------------
def render_letter():
    st.title("Daily LP Letter")
    today = datetime.now().date()
    letter_path = lp_letter.LETTERS_DIR / f"{today.isoformat()}.md"

    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("Regenerate"):
            with st.spinner("Calling Claude..."):
                lp_letter.generate(force=True)
            st.rerun()
    with cols[1]:
        st.caption(f"Cached at {letter_path}")

    if not letter_path.exists():
        st.info("No letter yet for today — click Regenerate to author one.")
        return
    st.markdown(letter_path.read_text())


# ----------------------------------------------------------------------
# Page router
# ----------------------------------------------------------------------
PAGE_MAP = {
    "0. Start Here": render_intro,
    "I. Overview": render_overview,
    "II. Positions": render_positions,
    "III. Factor & Risk": render_factor_risk,
    "IV. Performance": render_performance,
    "V. Execution": render_execution,
    "VI. LP Letter": render_letter,
}
PAGE_MAP[page]()
