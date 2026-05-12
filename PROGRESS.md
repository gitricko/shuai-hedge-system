# Meridian Capital Partners — Build Progress

> Multi-layer LLM-powered long/short equity hedge fund per `README_CLAUDE_PRO.md`.
> 7 layers total. Update this file as you complete work.

---

## Layer status

| # | Layer                       | Status          | Entry point          |
|---|-----------------------------|-----------------|----------------------|
| 1 | Data Infrastructure         | ✅ Complete      | `run_data.py`        |
| 2 | Scoring Engine              | ✅ Complete      | `run_scoring.py`     |
| 3 | Claude AI Analysis          | ✅ Complete      | `run_analysis.py`    |
| 4 | Portfolio Construction      | ✅ Complete      | `run_portfolio.py`   |
| 5 | Risk Management             | ✅ Complete      | `run_risk_check.py`  |
| 6 | Execution (Alpaca)          | ✅ Complete (needs keys) | `run_execution.py` |
| 7 | Reporting & Dashboard       | ✅ Complete      | `streamlit run dashboard/app.py` |

---

## Where I left off

**Current position:** ALL 7 LAYERS COMPLETE. Streamlit dashboard live at
http://localhost:8502 when started. Code complete; runtime usage needs
API keys to unlock the optional Layer 3/6 paths.

**Most recent action:** Layer 7 committed — reporting modules
(P&L attribution, position attribution, win/loss, sector alpha, turnover,
tear sheet, LP letter, weekly commentary), 6-page Streamlit dashboard,
macOS launchd daily-automation plist.

**Next steps:**
- Add `ANTHROPIC_API_KEY` to `.env` to enable Claude analyzers + JARVIS letters
- Add `ALPACA_API_KEY/SECRET` to `.env` to enable Layer 6 execution
- Install launchd job: `bash scripts/install_launchd.sh`
- Run dashboard: `.venv/bin/streamlit run dashboard/app.py`

---

## Setup recap (skip if your `.venv` works)

```bash
# venv lives at .venv (Python 3.14)
source .venv/bin/activate          # or use .venv/bin/python directly
pip install -r requirements.txt    # only if dependencies missing
cp .env.example .env               # then edit SEC_USER_AGENT_EMAIL etc.
```

---

## Database state (as of last check)

| Table                      | Rows         |
|----------------------------|--------------|
| universe                   | 521          |
| daily_prices               | 388,943      |
| fundamentals               | 738,082      |
| fundamental_ratios         | 93,357       |
| short_interest             | 503          |
| analyst_estimates          | 503          |
| earnings_calendar          | 56           |
| sec_filings                | 3,764+       |
| insider_transactions       | (retry running) |
| institutional_holdings     | (retry running) |
| portfolio_positions        | 0 (whatif only) |
| analysis_results           | 0            |

---

## How to resume work

1. **Where am I?** — read this file (top to bottom)
2. **What's the data state?** — `.venv/bin/python -c "from cache.db import conn_ctx; ..."` or just re-run `run_data.py --no-filings --no-13f` for a fast incremental refresh
3. **What's the latest scoring?** — `cat output/scored_universe_latest.csv | head -20`
4. **What's committed?** — `git log --oneline`
5. **What's in flight?** — check Claude conversation, or `ls -lt output/*.log | head`

---

## Recent commits

- (next) — Layer 7: Reporting + dashboard + launchd automation
- `1208951` — Layer 6: Execution (Alpaca paper trading)
- `d176fe2` — Layer 5: Risk Management
- `daf29dc` — Layer 4: Portfolio Construction
- `cd1d01a` — Fix Form 4 parser (was downloading XSL display version)
- `6871ae4` — PROGRESS.md
- `21457b7` — Layer 3: Claude AI qualitative analysis layer
- `92e2e40` — Initial commit: Layer 1 + Layer 2

---

## Outstanding decisions / TODOs

- [ ] Add `ANTHROPIC_API_KEY` to `.env` to actually run Layer 3 analyzers (~$5 per 40-candidate run)
- [ ] Optional: `FMP_API_KEY` for earnings call transcripts (paid)
- [ ] Optional: `POLYGON_API_KEY` for licensed exchange data instead of yfinance
- [ ] Optional: `FRED_API_KEY` for macro series (yield curve, credit spread)
- [ ] CUSIP→ticker mapping for institutional_holdings table (Layer 2 institutional factor partially blocked on this)
- [ ] Get Alpaca paper-trading API keys from https://alpaca.markets and add to `.env` to actually run Layer 6
- [ ] Run `bash scripts/install_launchd.sh` to enable nightly auto-refresh
- [ ] (Optional polish) Fix sector_alpha aggregation — currently sums sector alphas without portfolio weighting
