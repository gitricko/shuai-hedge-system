"""Calendar-aware rebalance advisory warnings.

Returns advisories — never blocks trading — for:
  * Positions with earnings in <=2 days
  * FOMC meeting within 5 days (hardcoded 2026 dates from config)
  * Monthly options expiration within 3 days (third Friday of the month)
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Iterable

import yaml

from cache.db import REPO_ROOT, conn_ctx


def _cfg_dates(key: str) -> list[date]:
    """YAML auto-parses YYYY-MM-DD into date objects; accept either form."""
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("portfolio", {})
    out: list[date] = []
    for v in cfg.get(key, []):
        if isinstance(v, date):
            out.append(v)
        else:
            out.append(datetime.strptime(str(v), "%Y-%m-%d").date())
    return out


def _third_friday(year: int, month: int) -> date:
    """Third Friday of a given month — monthly options expiration."""
    cal = calendar.Calendar(firstweekday=calendar.MONDAY).itermonthdates(year, month)
    fridays = [d for d in cal if d.month == month and d.weekday() == calendar.FRIDAY]
    return fridays[2]


def upcoming_opex(today: date, lookahead_days: int = 3) -> date | None:
    """Return next monthly opex date if it's within `lookahead_days`."""
    candidates = [
        _third_friday(today.year, today.month),
        _third_friday(today.year + (today.month // 12), (today.month % 12) + 1),
    ]
    for d in candidates:
        if today <= d <= today + timedelta(days=lookahead_days):
            return d
    return None


def upcoming_fomc(today: date, lookahead_days: int = 5) -> date | None:
    for d in _cfg_dates("fomc_2026"):
        if today <= d <= today + timedelta(days=lookahead_days):
            return d
    return None


def positions_with_imminent_earnings(tickers: Iterable[str], today: date | None = None,
                                     lookahead_days: int = 2) -> list[tuple[str, str]]:
    today = today or date.today()
    cutoff = today + timedelta(days=lookahead_days)
    placeholders = ",".join(["?"] * len(list(tickers)))
    if not placeholders:
        return []
    tickers = list(tickers)
    with conn_ctx() as conn:
        rows = conn.execute(
            f"SELECT ticker, earnings_date FROM earnings_calendar "
            f"WHERE ticker IN ({placeholders}) AND earnings_date BETWEEN ? AND ?",
            (*tickers, today.isoformat(), cutoff.isoformat()),
        ).fetchall()
    return [(r["ticker"], r["earnings_date"]) for r in rows]


def advisory(tickers: Iterable[str], today: date | None = None) -> dict:
    today = today or date.today()
    return {
        "earnings": positions_with_imminent_earnings(tickers, today),
        "fomc": upcoming_fomc(today),
        "opex": upcoming_opex(today),
    }
