# Meridian Capital Partners — Hedge Fund System

> **Type:** Multi-Agent LLM-Powered Long/Short Equity Hedge Fund  
> **Stack:** Anthropic Sonnet 4-5 + Python + SQLite + Streamlit + Alpaca  
> **Source:** OCR from 15-image slide deck · May 2026  
> **Scale:** 5 sources · 390K price bars · 50K insider txns · 20K holdings

> source: https://www.youtube.com/watch?v=ANUXcTgrpg0&t=11s
> source: https://photos.app.goo.gl/9KFyapvKaDfcGTnM8
---

## 📐 Architecture — 7 Layers

```
┌─────────────────────────────────────────────────────────────┐
│               LAYER 7 — REPORTING & DASHBOARD               │
│        P&L · Tear Sheet · LP Letter · JARVIS Commentary    │
├─────────────────────────────────────────────────────────────┤
│                  LAYER 6 — EXECUTION                       │
│            Alpaca · Limit Orders · Slippage Tracking        │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 5 — RISK MANAGEMENT                   │
│       Barra Factor Risk · Pre-Trade Veto · Circuit Breakers│
├─────────────────────────────────────────────────────────────┤
│              LAYER 4 — PORTFOLIO CONSTRUCTION               │
│          MVO Optimizer · Conviction-Tilt · Rebalance        │
├─────────────────────────────────────────────────────────────┤
│              LAYER 3 — AI QUALITATIVE ANALYSIS             │
│      Claude Earnings · Filing · Insider · Risk Analyzers    │
├─────────────────────────────────────────────────────────────┤
│                  LAYER 2 — SCORING ENGINE                  │
│         8 Factors · 27 Sub-Factors · Sector-Relative        │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 1 — DATA INFRASTRUCTURE              │
│            5 Sources · 390K bars · 50K insider txns          │
└─────────────────────────────────────────────────────────────┘
```

---

## LAYER 1 — DATA INFRASTRUCTURE

*Build Layer 1 of a long/short equity hedge fund system called "Meridian Capital Partners."
Project folder: "ls_equity_fund." This layer handles ALL data ingestion — no scoring, no analysis — just pulling data from 5 sources into a local SQLite database.*

### Project Structure

```
Project Structure/
├── data/            # Layer 1 (data ingestion) - this layer
├── factors/         # Layer 2 (scoring engine)
├── analysis/        # Layer 3 (Claude AI analysis)
├── portfolio/       # Layer 4 (portfolio construction)
├── risk/            # Layer 5 (risk management)
├── execution/       # Layer 6 (Alpaca execution)
├── reporting/       # Layer 7 (reports)
├── dashboard/       # Layer 7 (Streamlit dashboard)
├── cache/           # SQLite + cached files
├── output/          # CSVs, logs, reports
├── config.yaml      # All parameters
└── .env             # API keys (gitignored)
```

---

### 5 Data Sources

#### 1: **Universe** (data/universe.py):
Scrape current S&P 500 list from Wikipedia. Store ticker, company name, GICS sector, sub-industry. Cache locally, refresh weekly. Also maintain benchmark tickers: SPY, QQQ, IWM, DIA, sector ETFs (XLK, XLF, XLV, XLE, XLI, XLC, XLY, XLP, XLB, XLRE, XLU), ^VIX, TLT, HYG.

#### 2: **Market Data + Fundamentals** (data/market_data.py + data/fundamentals.py):
Daily OHLCV via yfinance for all universe + benchmarks. 3yr lookback. Incremental updates — only fetch new data since last stored date. SQLite table "daily_prices". Fundamentals: quarterly + annual income stmt, balance sheet, cash flow via yfinance. Calculate 24 derived ratios: ROE, ROA, gross/operating/net margin, revenue growth YoY/QoQ, earnings growth YoY/QoQ, debt/equity, FCF yield, current ratio, AR/revenue, CFO/NI, accruals ratio, retained earnings, working capital, total liabilities, EBIT, R&D expense, shares outstanding, dividends paid, buybacks, asset turnover.

#### 3: **SEC Filings + Insider** (data/sec_data.py):
Connect to SEC EDGAR EFTS API. Headers: User-Agent with email, 8 req/sec rate limit. For each ticker fetch: latest 10-K (full doc for Risk Factors), latest 10-Q (MD&A), recent 8-K filings, Form 4 insider transactions (last 180 days). Parse Form 4 XML into "insider_transactions" table: ticker, insider name, insider title, transaction_type, transaction_code, shares, price, date, ownership type.Distinguish open-market purchases (code P) from grants/exercises (A, M, F). Flag CEO/CFO purchases. Flag cluster buying (3+ insiders within 30 days same ticker). Add --no-filings flag to skip SEC for fast daily runs. Add --forms flag for selective pulls.

#### 4: **Institutional Holdings** (data/institutional.py):
Fetch 13-F filings from SEC EDGAR for 9 hedge funds: Citadel, Point72, Bridgewater, Tiger Global, Third Point, Berkshire Hathaway, Appaloosa, Baupost, Pershing Square. Parse: fund name, ticker, shares held, market value, report_date. Calculate per ticker: number of tracked funds holding, net change from prior quarter. Flag tickers with 3+ funds opening new positions simultaneously. Add --no-13f flag for fast daily runs.

#### 5: **Short Interest + Estimates** (data/short_interest.py + data/estimates.py):
Fetch from yfinance .info: shares_short, short ratio, short percent of float. Daily snapshots in SQLite table "short_interest". Refresh daily.

#### 6: **Earnings Transcripts** (data/transcripts.py):
Fetch forward EPS estimate, price target consensus via yfinance. Store as daily snapshots in "analyst_estimates" table. Revisions factor needs 30+ days of snapshots to compute 30/60/90-day deltas. Refresh daily.

#### 7: **Earnings Calendar** (data/earnings_calendar.py): 
Fetch upcoming earnings dates for next 30 days across the universe. Refresh daily.

#### 8: **Earnings Transcripts** (data/transcript.py):
If FMP_API_KEY in .env, fetch latest transcript from Financial Modeling Prep API. Store in "earnings_transcripts" table. Only fetch for long/short candidates, not entire universe. If no FMP key, skip gracefully and log. 

#### 9: **Provider Abstraction** (data/providers.py):
Create provider layer that routes to best available data source: 
- If POLYGON_API_KEY in .env: use Polygon for daily prices (licensed exchange data)
- If FMP_API_KEY in .env: use FMP for transcripts + structured financials
- If FRED_API_KEY in .env: use FRED for yield curve, credit spread, fed funds rate.
- Default fallback: yfinance for prices/fundamentals, SEC EDGAR for filings. 
Log which provider is active: "Using Polygon for prices" or "Falling back to yfinance"

=== ENTRY POINT ===
run_data.py:
- Arguments: --no-filings, --no-13f
- Run all data refreshes in order: universe -> prices -> fundamentals -> short interest -> estimates -> earnings calendar -> transcripts -> SEC filings (unless --no-filings) -> 13-F (unless --no-13f). Log everything to output/run.log.

Print summary: tickers updated, price bars added, filings cached, insider txns parsed.

---

## LAYER 2 — SCORING ENGINE

Build Layer 2 of the Meridian Capital Partners hedge fund. Layer 1 (data) is built.
Build the scoring engine: 8 factors with 27 sub-factors. All scores are 0-100 percentile rank WITHIN each GICS sector.

### Composite Weights

| Factor | Weight | Sub-Factors |
|--------|--------|-------------|
| **Momentum** | 20% | 6 |
| **Quality** | 20% | 8 |
| **Value** | 15% | 6 |
| **Estimate Revisions** | 15% | 3 |
| **Insider Activity** | 10% | 3 |
| **Growth** | 10% | 5 |
| **Short Interest** | 5% | 3 |
| **Institutional Flow** | 5% | 3 |

### Factor Details

#### 1. ﻿﻿﻿Momentum (factors/momentum.py) - 6 sub-factors:
12-1 month return (skip recent 1mo to avoid reversal), 6-month return, 3-month return, acceleration (recent 3m minus older 3m), 52-week-high proximity (price / 52w high - George & Hwang 2004), relative strength vs sector ETF (6m stock return minus sector ETF return - isolates stock-specific momentum from sector beta).

#### 2. ﻿﻿﻿Value (factors/value.py) - 6 sub-factors:
Forward earnings yield (1/forward P/E), book-to-price, FCF yield, EV/EBITDA (invert), shareholder yield (TTM buybacks + dividends / mkt cap), sales-to-EV (revenue / EV - works where P/E breaks on negative or volatile earnings).

#### 3. ﻿﻿﻿Quality (factors/quality.py) - 8 sub-factors:
ROE stability (std dev of 12Q ROEs, invert), gross margin level, gross margin trend(latest minus 4Q ago), debt/equity (invert), CFO/NI (higher = real cash earnings),accruals ratio ((NI-CFO)/TA, invert - high accruals predict underperformance), Piotroski F-Score (1-9): 9 binary signals - positive ROA, positive CFO, rising ROA, CFO > NI, falling D/E, rising current ratio, no dilution, rising gross margin, rising asset turnover. Color code: green >=7, amber <=3.Altman Z-Score: 1.2*(WC/TA)+1.4*(RE/TA) +3.3*(EBIT/TA) +0. 6* (MktCap/TL)+1.0* (Sales/TA) .
>2.99 = "safe" (green), 1.81-2.99 = "grey zone", <1.81 = "distress" (amber).

#### 4. Growth (factors/growth.py) - 5 sub-factors:
Revenue growth YoY, earnings growth YoY,
revenue growth acceleration (latest YoY minus
4Q-ago YoY), R&D intensity (R&D expense / revenue - high R&D in tech/healthcare tends to outperform long-term), free cash flow growth YoY (harder to manipulate than earnings).

#### 5. Estimate Revisions (factors/revisions.py) - 3 sub-factors:
30-day change in consensus next-Q EPS, 60-day change,
90-day change. Degenerate (all
scores = 50) until ~30 days of snapshots accumulate. Equal-weight available deltas.

#### 6. Short Interest (factors/short_interest.py) - 3 sub-factors:
Short percent of float, days to cover, change in short interest vs prior period.
For LONGS: declining short interest scores higher. For SHORTS: increasing scores higher.

#### 7. ﻿﻿﻿Insider Activity (factors/insider-py) - 3 sub-factors:
Net dollar flow over 90 days from Form 4 data. CEO/CFO open-market purchases weighted3x vs other insiders. Cluster-buy flag (3+ insiders within 30 days) = bonus. Only counttransaction code P (purchase) and S (sale), ignore A/M/F. No data = sector median (50).

#### 8. ﻿﻿﻿Institutional Flow (factors/institutional.py) - 3 sub-factors:
Number of tracked funds holding, net change in aggregate holdings vs prior quarter, multi-fund simultaneous opening flag (3+ funds opening new positions same ticker).

All factors: equal-weight sub-factors within each parent, then sector percentile rank 0-100.

=== COMPOSITE + EXTRAS ===

#### 9. ﻿﻿﻿Composite Score (factors/composite.py):
Weighted blend: Momentum 0.20, Quality 0.20, Value 0.15, Estimate Revisions 0.15, Insider Activity 0.10, Growth 0.10, Short Interest 0.05, Institutional Flow 0.05.After blending, re-rank within sector for final 0-100 composite.Top quintile = LONG candidates. Bottom quintile = SHORT candidates.Output: scored_universe_latest.csv with ALL sub-factor scores, composite, LONG/SHORT flag.

#### 10. ﻿﻿﻿﻿Regime-Conditional Weights (factors/regime_weights.py) :
Low Vol (VIX < 15): boost momentum 0.20->0.28, cut value 0.15->0.10.Normal (15-25): default weights. High Vol (VIX > 25): boost quality 0.20->0.28 and value 0.15->0.22, cut momentum 0.20->0.10.
Config flag: regime_conditional_weights.

#### 11. Crowding Detection (factors/crowding-py) :
Synthesize daily factor returns: top-quintile minus bottom-quintile per factor, 60-day rolling. Pairwise correlations between all factor return series.
Compare to academic
baselines (momentum/value ~-0.3, momentum/quality ~+0.1). Flag when deviation > 0.4.

Entry point: run_scoring•py -ticker AAPL for single stock mode.
Print summary: top 5 longs, top 5 shorts, crowding warnings, degenerate factor warnings.

---

## LAYER 3 — Claude AI ANALYSIS

Build Layer 3 of the Meridian Capital Partners hedge fund. Layers 1-2 are built.
Build the Claude API qualitative analysis layer - the AI analyst that reads filings, financials, and insider data. Use Anthropic SDK. Key: ANTHROPIC_API_KEY in •env.
Default model: claude-sonnet-4-5 (configurable). Use prompt caching on system prompts.

#### 1. ﻿﻿﻿API Client (analysis/api_client.py) :
Anthropic SDK wrapper. Prompt caching (cache_control: ephemeral) on every system prompt.Retry on 429/5xx with exponential backoff. JSON extraction: handle raw JSON, fences, and prose-wrapped JSON. Token count estimator for cost prediction.

#### 2. ﻿﻿﻿Cost Tracker (analysis/cost_tracker.py) :
Read response. usage after every call. Track: input/output/cache-write/cache-read tokens.Hard cost ceiling from config (default: $25/run) - abort if exceeded.

#### 3. ﻿﻿﻿Analysis Cache (analysis/cache.py) :
SQLite table "analysis results" keyed by (analyzer,
ticker, artifact_id). TTL-based
eviction (default 30 days). Re-running same artifact = free cache hit.

#### 4. Earnings Call Analyzer (analysis/earnings_analyzer.py) :
Input: transcript from data/transcripts (requires FMP key). Truncate to 120K chars.
Score 1-10: Management Confidence, Revenue Guidance, Margin Trajectory,
Competitive
Position, Risk Factors, Capital Allocation. Output JSON with per-category reasoning, bull_case, bear_case, key_quotes, one_line_summary. Return None if no transcript.

#### 5. Filing Analyzer (analysis/filing_analyzer-py) :
Input: 8 quarters of fundamental metrics. Forensic accounting review. Assess: earnings quality (CF0 vs NI), revenue quality (AR vs revenue), balance sheet health, accruals.
Output JSON: earnings_quality_score, balance_sheet_score, red/green flags, risk_level.

#### 6. Risk Analyzer (analysis/risk_analyzer.py) :
Input: 10-K Risk Factors section (strip HTML, cap 80k chars). Seperate material risks from boilerplate.
Flag new risks vs prior filing. Output JSON: new risks, material_risks, boilerplate_percentage, risk_severity, one_line_summary. Return None if no 10-K cached.

#### 7. Insider Analyzer (analysis/insider_analyzer.py) :
Input: Form 4 data (last 90 days). Interpret: routine selling vs meaningful buying.
Output JSON: signal_strength (STRONG_BUY to STRONG_SELL), confidence, key_transactions, reasoning, one_line_summary. Return None if no insider data.

#### 8. Sector Analysis (analysis/sector_analysis.py):
Per sector: gather all Claude results, rank by fundamental quality and positioning.
Output: rankings with reasoning, top_long_idea, top_short_idea, sector_outlook.

#### 9. Combined Score (analysis/combined_score.py) :
60% quantitative composite (Layer 2) + 40% Claude fundamental (avg of available analyzers). If no Claude analysis available, use 100% quantitative - no penalty. Re-rank within sector.

#### 10. Report Generator (analysis/report_generator-py) :
Per LONG/SHORT candidate: markdown report with all scores, Claude summaries, upcoming catalysts, risk flags. Save to output/reports_{timestamp}/{TICKER}.md.

Entry: run_analysis.py --estimate-cost | -ticker AAPL | --sector Technology | full run
Estimated cost for full run (20 long + 20 short candidates): $2-5 using Sonnet.

---

## LAYER 4 — PORTFOLIO CONSTRUCTION

Build Layer 4 of the Meridian Capital Partners hedge fund. Layers 1-3 are built.
Build portfolio construction with TWO optimization methods: MVO and conviction-tilt.

#### 1. MVO Optimizer (portfolio/mvo_optimizer.py) :
Markowitz via scipy.optimize.minimize (SLSQP). Inputs: expected returns (composite score
mapped linearly: score 100 = +15%/yr, score 0 = -15%/yr), covariance matrix (120-day
historical - replaced by factor-cov from Layer 5 later), risk aversion lambda (default
1.0), transaction costs per ticker subtracted from gross expected return.
Objective: maximize mu*w - lambda*w*Sigma*w
Constraints: long weights sum to target_long_gross, short weights to target_short_gross, per-position [min_pct, max_pct], |w*beta| <= 0.15, |sector_net| <= 5%, single-side sector <= 25%. On non-convergence: log warning, use conviction-tilt fallback.
CLI flag: --optimize-method mvo or -- optimize-method conviction

#### 2. Conviction-Tilt Optimizer (portfolio/optimizer-py):
Equal weight base within each book. Top 5% scores get 1.5x,
top 10% get 1.25x.
Liquidity: no position > 5% of 20-day ADV. Earnings: halve size if earnings in 5 days.
Beta adjustment: scale so beta-adjusted exposure matches beta 1.0. Sector neutral.

#### 3. Transaction Cost Model (portfolio/transaction_costs.py):
Three components per ticker in bps: commission ($0 Alpaca), spread cost (5% ot avg daily
H-L range), market impact (coef * sqrt(trade_size/ADV) * daily_vol_bps, coef=0.10).
Fed into MVO objective so optimizer sees net-of-cost expected returns.

#### 4. Rebalance Schedule (portfolio/rebalance_schedule.py) :
Check events: positions with earnings in 2 days, FOMC meeting within 5 days (hardcode 2026 dates), monthly options expiration within 3 days (third Friday). Return advisory warnings - does not block trading.

#### 5. Portfolio State (portfolio/state.py) :
SQLite tables: "portfolio_positions",
"portfolio_history", "position_approvals".
Track: ticker, shares, entry_price, entry_date, current_price, unrealized_pl, sector, factor_scores_at_entry. Handle corporate actions.

#### 6. Beta Calculator (portfolio/beta-py):
Rolling 60-day beta per stock vs SPY. Portfolio-level: long book beta,
short book beta,
net portfolio beta.

#### 7. Factor Exposure Calculator (portfolio/factor_exposure.py) :
Weighted average of each factor score across long and short book. Flag if any spread
exceeds 1 std dev from historical.

#### 8. Rebalance Generator (portfolio/rebalance.py) :
Compare current to target, generate trade list. Apply turnover budget (max 30%).
Prioritize largest score changes. Estimate transaction costs per trade.
Include --whatif mode (show proposed changes without committing).

Entry: run_portfolio.py --rebalance | --whatif | --current | --optimize-method mvo
Config: num_longs=20, num_shorts=20, max_position=5%, max_sector=25%, gross=150%, net=[0%, +10%], max_beta=0.15, turnover_budget=30%, mvo_risk_aversion=1.0

---

## LAYER 5 — RISK MANAGEMENT

Build Layer 5 of the Meridian Capital Partners hedge fund. Layers 1-4 are built.
Build risk management with ABSOLUTE VETO POWER plus Barra-style factor risk model.

### == FACTOR RISK MODEL ===

#### 1. Factor Risk Model (risk/factor_risk_model.py) :
Barra-style cross-sectional regression. For each day t in 120-day lookback:
r_i,t = alpha_t + sum_k beta_k,t * F_k,i + epsilon_i,t
F_k,i = stock i standardized factor exposure (z-scored from 0-100 sector ranks).
Produces: factor returns (daily series), factor covariance matrix (annualized),
specific variance per stock (annualized). Portfolio: factor_var = exp*F*exp,
specific_var = sum(w_i^2 * spec_var_i), total_var = factor_var + specific_var.
MCTR_1 = w_İ * Cov(r_i, r_P) / sigma_p. Flag where MCTR% > 1.5x weight%.
Feed predicted cov matrix (X*F*X + diag(specific)) to Layer 4 MVO optimizer.

### == RISK CHECKS (ABSOLUTE VETO - NO OVERRIDE) ==
#### 2. Pre-Trade Veto (risk/pre_trade.py) - 8 checks, ANY failure = REJECT:
1. Halt lock exists? 2. Earnings blackout (5d = 50% size cut)
3. Liquidity <= 5% ADV
4. Position <= 5% AUM
5. Sector <= 25% 6. Gross <= 165%,
net [-10%, +15%]
7. Inet beta | <= 0.20
8. Pairwise correlation <= 0.80 with existing positions.
Closing/covering trades always approved. Log every rejection with timestamp and reason.

#### 3. Circuit Breakers (risk/circuit_breakers.py) - fire on actual dollar losses:
Daily > 1.5% -> SIZE_DOWN 30%.
Daily > 2.5% -> CLOSE_ALL_TODAY.
Weekly > 4% -> SIZE_DOWN 30%.
Drawdown > 8% -> KILL_SWITCH (lock file, --clear-halt).
Single position > 3% NAV -> force-close immediately.

#### 4. Factor Monitor (risk/factor_monitor.py) :
Z-score each factor spread (long minus short) vs universe cross-sectional std.Alert when |z| > 1.5 sigma. Cross-check vs crowding warnings = HIGH priority alert.

#### 5. Correlation Monitor (risk/correlation_monitor.py):
60-day rolling pairwise correlations within each book. Alert if avg within-book > 0.60.
Effective number of bets: exp (entropy (eigenvalue_distribution)).

#### 6. Tail Risk Monitor (risk/tail_risk.py):
VIX >= 25 -> REDUCE_GROSS_20%.
VIX >= 35 -> REDUCE_GROSS_50%.
Credit spread z-score ›= 1 sigma widening
- REDUCE_GROSS_20%.
No override possible.
If FRED_API_KEY available, pull BAMLHOAOHYM2 for actual high-yield credit spread.

#### 7. Stress Testing (risk/stress_test.py) - 6 scenarios:
Historical: 2008 Financial Crisis (Sep 08 - Mar 09), 2020 Covid Crash (Feb-Apr 20),
2022 Rate Hikes (Jan-Oct 22). Use actual stock-level returns from yfinance, cache parquet.
Synthetic: Sector Shock (-30% most concentrated sector), Momentum Reversal (top quintile
-20%, bottom +20% - the quant quake), Short Squeeze (all shorts +30% simultaneously) .
Report estimated P&L ($, %) broken into long book and short book contributions.

#### 8. Risk State (risk/risk_state.py):
Maintain cache/risk_state. json with: daily/weekly P&L, drawdown,
circuit breaker usage, factor exposures, risk decomposition, per-factor contributions, top MCTR positions, alerts.

Entry: run_risk_check.py --stress | --tail-only | --clear-halt

---

## LAYER 6 — EXECUTION

Build Layer 6 of the Meridian Capital Partners hedge fund. Layers 1-5 are built.
Build the Alpaca paper trading execution layer.

#### 1. Broker Connection (execution/broker-py) :
Alpaca API using keys from . env (ALPACA_API_KEY, ALPACA_SECRET_KEY). DEFAULT TO PAPER TRADING - hardcode paper base URL. Live requires: mode: live in config AND typing "YES I UNDERSTAND THE RISKS". Sync portfolio state with Alpaca on startup. Exponential backoff.

#### 2. Order Executor (execution/executor•py) :
Per trade: a) pre-trade veto check b) short availability check
c) limit price: close * ( 1 +/- 0.001) d) chunk orders > 2% ADV e) 120s time-in-force
f) poll every 5s g) cancel + retry on timeout (3x max) h) record
signal_price for slippage calculation.
Log every order: timestamp, ticker, side, shares, limit, fill, slippage_bps, status.

#### 3. Slippage Tracker (execution/costs.py) :
slippage = (fill - signal) / signal * 10,000 bps. 30-day rolling: avg, median, p95,
total dollar cost. Surface worst 5 fills for dashboard.

#### 4. Short Availability (execution/short_check.py):
Check Alpaca "shortable" + "easy_to_borrow" flags. Cache 7 days. Log and skip if not.

#### 5. Order Manager (execution/order_manager.py) :
Track pending/partial/filled/cancelled. SIGINT -> cancel pending, keep positions, log.

Entry: run_execution.py --dry-run (log what would happen) | --execute (place orders)


---

## LAYER 7 — REPORTING & DASHBOARD

Build Layer 7 of the Meridian Capital Partners hedge fund. Layers 1-6 are built.
Build BOTH the reporting engine AND the Streamlit dashboard with JARVIS persona.

### === REPORTING ENGINE ===

#### 1. Daily P&L Attribution (reporting/pnl_attribution.py):
Decompose: daily_return = beta + sector + factor + alpha_residual. Beta: net_beta *
SPY_return. Sector: Brinson-style. Factor: regression on factor return spreads. Alpha:
residual after subtracting all three. Persist to output/daily_attribution.csv.

#### 2. Position Attribution: mark-to-market, FIFO round-trips, best/worst per side.
Predictive power: Spearman correlation between entry-time score and realized return.

#### 3. Win/Loss Analysis: win rate, P/L ratio. 
Sliced by: side, holding period (1-5d/5-20d/20-60d/60d+), sector, VIX regime at entry, factor quintile at entry. Streaks.

#### 4. Sector-Relative Performance: per sector 90d, your picks alpha. 
Sum across sectors = total alpha. Track winner/loser sector counts.sector ETF = stock-selection

### 5. Turnover Analytics: trailing 30/90d turnover, annualized, vs budget from config.
Tax estimate via FIFO: short-term gains @ 37%, long-term @ 20%.

#### 6. Tear Sheet: markdown institutional format
metrics vs SPY, monthly returns grid, equity curve, drawdown, rolling 12mo Sharpe, factor + sector exposures, turnover.

#### 7. Claude Weekly Commentary: JARVIS-authored, fires on configurable weekday (default Fri).

#### 8. Daily LP Letter: 3-4 paragraphs, letterhead, signature block, compliance footer

#### PAGE IV - PERFORMANCE:
Equity curve vs SPY (rebased to 100), monthly returns grid (green/red heatmap), drawdown chart, P&L attribution bars (Beta/Sector/Factor/Alpha), rolling 12mo Sharpe, sector relative alpha chart with total alpha KPI + winner/loser counts, turnover panel (30d/ annualized/budget/tax), transaction cost panel (estimated vs actual vs model error), best/worst 5 contributors, win/loss panel, Claude weekly commentary card.

#### PAGE V - EXECUTION:
KPI row (filled orders 30d, avg slippage bps, total slippage $, open orders count), open orders table (polling Alpaca), recent trades log (last 200 orders), worst 5 fills, short availability panel per current short, daily notional turnover table.

#### PAGE VI - LETTER:
Formal daily LP letter. Letterhead: fund name, domicile (Delaware), inception, AUM,
doc ID (MCP-IM-{YYYY}- {MMDD}), date. "CONFIDENTIAL • LIMITED PARTNERS ONLY" stamp.
"Dear Limited Partners," + 3-4 paragraph body from Claude in JARVIS voice. Signature block + compliance footer. "Regenerate letter" button. Cache by date.

AUTO-REFRESH: Every 5 minutes during market hours (9:30am - 4:00pm ET) •

##### DAILY AUTOMATION:
macoS launchd plist at ~/Library/LaunchAgents/com.user.hedgefund.daily.plist
Weekdays at 17:15 local. Runs: run_scoring.py --no-filings --no-13f
Refreshes prices, short interest, estimates, calendar, rescores all factors. ~10min