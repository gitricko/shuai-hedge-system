"""SQLite database helpers for Meridian Capital Partners.

Provides a single source of truth for connections and schema initialization.
All Layer 1 modules import `get_conn()` and `init_schema()` from here.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_db_path() -> Path:
    cfg = _load_config()
    db_rel = cfg.get("database", {}).get("path", "cache/meridian.db")
    return REPO_ROOT / db_rel


def get_conn() -> sqlite3.Connection:
    """Open a connection with sane pragmas. Caller closes."""
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def conn_ctx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS universe (
        ticker TEXT PRIMARY KEY,
        company_name TEXT,
        gics_sector TEXT,
        gics_sub_industry TEXT,
        is_benchmark INTEGER DEFAULT 0,
        last_refreshed TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume REAL,
        PRIMARY KEY (ticker, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date)",
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        ticker TEXT NOT NULL,
        period_end TEXT NOT NULL,
        period_type TEXT NOT NULL,           -- 'Q' or 'A'
        statement TEXT NOT NULL,             -- 'income', 'balance', 'cashflow'
        line_item TEXT NOT NULL,
        value REAL,
        currency TEXT,
        PRIMARY KEY (ticker, period_end, period_type, statement, line_item)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamental_ratios (
        ticker TEXT NOT NULL,
        period_end TEXT NOT NULL,
        period_type TEXT NOT NULL,
        ratio TEXT NOT NULL,
        value REAL,
        PRIMARY KEY (ticker, period_end, period_type, ratio)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sec_filings (
        accession TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        cik TEXT,
        form_type TEXT NOT NULL,
        filing_date TEXT,
        period_of_report TEXT,
        primary_doc TEXT,
        local_path TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_filings_ticker_form ON sec_filings(ticker, form_type)",
    """
    CREATE TABLE IF NOT EXISTS insider_transactions (
        accession TEXT NOT NULL,
        ticker TEXT NOT NULL,
        insider_name TEXT,
        insider_title TEXT,
        is_officer INTEGER DEFAULT 0,
        is_director INTEGER DEFAULT 0,
        is_ten_pct_owner INTEGER DEFAULT 0,
        transaction_date TEXT,
        transaction_type TEXT,        -- 'A' = acquired, 'D' = disposed
        transaction_code TEXT,        -- 'P' purchase, 'S' sale, 'A' grant, 'M' option exercise, 'F' tax, etc.
        ownership_type TEXT,          -- 'D' direct, 'I' indirect
        shares REAL,
        price REAL,
        PRIMARY KEY (accession, insider_name, transaction_date, transaction_code, shares, price)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_insider_ticker_date ON insider_transactions(ticker, transaction_date)",
    """
    CREATE TABLE IF NOT EXISTS institutional_holdings (
        fund_name TEXT NOT NULL,
        fund_cik TEXT,
        ticker TEXT NOT NULL,
        report_date TEXT NOT NULL,
        shares REAL,
        market_value REAL,
        PRIMARY KEY (fund_cik, ticker, report_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inst_ticker_date ON institutional_holdings(ticker, report_date)",
    """
    CREATE TABLE IF NOT EXISTS short_interest (
        ticker TEXT NOT NULL,
        snapshot_date TEXT NOT NULL,
        shares_short REAL,
        short_ratio REAL,
        short_pct_float REAL,
        float_shares REAL,
        PRIMARY KEY (ticker, snapshot_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analyst_estimates (
        ticker TEXT NOT NULL,
        snapshot_date TEXT NOT NULL,
        forward_eps REAL,
        forward_pe REAL,
        target_mean REAL,
        target_high REAL,
        target_low REAL,
        recommendation_mean REAL,
        num_analyst_opinions INTEGER,
        PRIMARY KEY (ticker, snapshot_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS earnings_calendar (
        ticker TEXT NOT NULL,
        earnings_date TEXT NOT NULL,
        eps_estimate REAL,
        revenue_estimate REAL,
        time_of_day TEXT,
        snapshot_date TEXT,
        PRIMARY KEY (ticker, earnings_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS earnings_transcripts (
        ticker TEXT NOT NULL,
        fiscal_year INTEGER,
        fiscal_quarter INTEGER,
        call_date TEXT,
        transcript TEXT,
        source TEXT,
        PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_log (
        run_id TEXT NOT NULL,
        step TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        status TEXT,
        rows_affected INTEGER,
        message TEXT,
        PRIMARY KEY (run_id, step)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_scores (
        ticker TEXT NOT NULL,
        score_date TEXT NOT NULL,
        factor TEXT NOT NULL,        -- 'momentum', 'value', ..., 'composite'
        sub_factor TEXT,             -- '' for parent factor, '12_1m_return' for sub
        score REAL,                  -- 0-100 percentile within sector
        raw_value REAL,              -- pre-rank input, for debugging
        PRIMARY KEY (ticker, score_date, factor, sub_factor)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scores_date ON factor_scores(score_date, factor)",
    """
    CREATE TABLE IF NOT EXISTS scored_universe (
        ticker TEXT NOT NULL,
        score_date TEXT NOT NULL,
        sector TEXT,
        composite REAL,
        side TEXT,                   -- 'LONG', 'SHORT', or NULL
        momentum REAL,
        quality REAL,
        value REAL,
        revisions REAL,
        insider REAL,
        growth REAL,
        short_int REAL,
        institutional REAL,
        PRIMARY KEY (ticker, score_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scored_date_comp ON scored_universe(score_date, composite DESC)",
    """
    CREATE TABLE IF NOT EXISTS analysis_results (
        analyzer TEXT NOT NULL,         -- 'earnings', 'filing', 'risk', 'insider'
        ticker TEXT NOT NULL,
        artifact_id TEXT NOT NULL,      -- transcript/filing/period hash
        result_json TEXT NOT NULL,
        model TEXT,
        created_at TEXT NOT NULL,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cache_read_tokens INTEGER,
        cache_write_tokens INTEGER,
        cost_usd REAL,
        PRIMARY KEY (analyzer, ticker, artifact_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_analysis_created ON analysis_results(created_at)",
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,           -- our internal id
        alpaca_order_id TEXT,                -- broker's id (nullable until submitted)
        ticker TEXT NOT NULL,
        side TEXT NOT NULL,                  -- 'BUY' or 'SELL'
        intent TEXT NOT NULL,                -- 'OPEN_LONG','CLOSE_LONG','OPEN_SHORT','COVER_SHORT'
        qty REAL NOT NULL,
        limit_price REAL,
        signal_price REAL,                   -- close at time of decision; for slippage
        status TEXT NOT NULL,                -- 'PENDING','PARTIAL','FILLED','CANCELLED','REJECTED'
        submitted_at TEXT,
        filled_at TEXT,
        notes TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_orders_ticker_status ON orders(ticker, status)",
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        ticker TEXT NOT NULL,
        fill_price REAL,
        fill_qty REAL,
        slippage_bps REAL,
        fill_timestamp TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS short_availability (
        ticker TEXT PRIMARY KEY,
        shortable INTEGER,
        easy_to_borrow INTEGER,
        refreshed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_positions (
        ticker TEXT PRIMARY KEY,
        side TEXT NOT NULL,            -- 'LONG' or 'SHORT'
        shares REAL NOT NULL,
        weight REAL,                   -- target weight as fraction of NAV
        entry_price REAL,
        entry_date TEXT,
        current_price REAL,
        unrealized_pl REAL,
        sector TEXT,
        factor_scores_at_entry TEXT    -- JSON snapshot of Layer 2 scores
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_history (
        ticker TEXT NOT NULL,
        action_date TEXT NOT NULL,
        action TEXT NOT NULL,          -- 'OPEN','CLOSE','RESIZE','DIV','SPLIT'
        side TEXT,
        shares_delta REAL,
        price REAL,
        notional REAL,
        notes TEXT,
        PRIMARY KEY (ticker, action_date, action, shares_delta)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS position_approvals (
        ticker TEXT NOT NULL,
        approval_date TEXT NOT NULL,
        approved INTEGER NOT NULL,     -- 1 or 0
        approver TEXT,
        reason TEXT,
        PRIMARY KEY (ticker, approval_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cost_log (
        run_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        analyzer TEXT,
        ticker TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cache_read_tokens INTEGER,
        cache_write_tokens INTEGER,
        cost_usd REAL,
        cumulative_usd REAL,
        PRIMARY KEY (run_id, timestamp, analyzer, ticker)
    )
    """,
]


def init_schema() -> None:
    """Idempotent schema setup."""
    with conn_ctx() as conn:
        for stmt in SCHEMA_STATEMENTS:
            conn.execute(stmt)


def upsert_many(conn: sqlite3.Connection, table: str, columns: list[str], rows: list[tuple]) -> int:
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(columns))
    col_list = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    cur = conn.executemany(sql, rows)
    return cur.rowcount


if __name__ == "__main__":
    init_schema()
    print(f"Schema initialized at {get_db_path()}")
