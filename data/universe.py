"""S&P 500 universe + benchmark tickers.

Scrapes the canonical Wikipedia S&P 500 list and stores ticker, company name,
GICS sector, sub-industry. Benchmarks (SPY, QQQ, sector ETFs, ^VIX, TLT, HYG)
are loaded from config.yaml and tagged with is_benchmark = 1.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yaml

from cache.db import REPO_ROOT, conn_ctx, init_schema, upsert_many

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (Meridian Capital research crawler)"


def _load_cfg() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _benchmark_tickers(cfg: dict) -> list[str]:
    u = cfg.get("universe", {})
    tickers: list[str] = []
    tickers.extend(u.get("benchmarks", []))
    tickers.extend(u.get("sector_etfs", []))
    tickers.extend(u.get("macro", []))
    return tickers


def _fetch_sp500() -> pd.DataFrame:
    log.info("Fetching S&P 500 constituents from Wikipedia")
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    rename = {
        "Symbol": "ticker",
        "Security": "company_name",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
    }
    df = df.rename(columns=rename)[list(rename.values())]
    # Wikipedia uses "BRK.B" but yfinance prefers "BRK-B"
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.upper().str.strip()
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    return df


def _stale(cfg: dict) -> bool:
    """True if the cached universe is older than refresh_days."""
    refresh_days = int(cfg.get("universe", {}).get("refresh_days", 7))
    with conn_ctx() as conn:
        row = conn.execute(
            "SELECT MAX(last_refreshed) AS ts FROM universe WHERE is_benchmark = 0"
        ).fetchone()
    if not row or not row["ts"]:
        return True
    last = datetime.fromisoformat(row["ts"])
    return datetime.now(timezone.utc) - last > timedelta(days=refresh_days)


def refresh(force: bool = False) -> int:
    """Refresh the universe table. Returns total tickers stored."""
    init_schema()
    cfg = _load_cfg()

    if not force and not _stale(cfg):
        log.info("Universe cache is fresh; skipping refresh")
        with conn_ctx() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM universe").fetchone()
        return int(row["n"])

    df = _fetch_sp500()
    now_iso = datetime.now(timezone.utc).isoformat()

    rows = [
        (r.ticker, r.company_name, r.gics_sector, r.gics_sub_industry, 0, now_iso)
        for r in df.itertuples(index=False)
    ]
    bench = [(t, t, "Benchmark", "Benchmark", 1, now_iso) for t in _benchmark_tickers(cfg)]

    with conn_ctx() as conn:
        upsert_many(
            conn,
            "universe",
            ["ticker", "company_name", "gics_sector", "gics_sub_industry", "is_benchmark", "last_refreshed"],
            rows + bench,
        )

    log.info("Universe refreshed: %d S&P 500 tickers + %d benchmarks", len(rows), len(bench))
    return len(rows) + len(bench)


def get_universe(include_benchmarks: bool = False) -> list[str]:
    init_schema()
    with conn_ctx() as conn:
        if include_benchmarks:
            rows = conn.execute("SELECT ticker FROM universe ORDER BY ticker").fetchall()
        else:
            rows = conn.execute(
                "SELECT ticker FROM universe WHERE is_benchmark = 0 ORDER BY ticker"
            ).fetchall()
    return [r["ticker"] for r in rows]


def get_benchmarks() -> list[str]:
    init_schema()
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT ticker FROM universe WHERE is_benchmark = 1 ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


def get_sector_map() -> dict[str, str]:
    init_schema()
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT ticker, gics_sector FROM universe WHERE is_benchmark = 0"
        ).fetchall()
    return {r["ticker"]: r["gics_sector"] for r in rows}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = refresh(force=True)
    print(f"Stored {n} tickers in universe.")
