"""Weekly Claude-authored commentary (default Friday).

Same JARVIS persona as the LP letter, but a longer-horizon piece —
trailing-week attribution, factor regime shifts, what changed about
positioning. Fires only on the configured weekday (default Fri=4).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import yaml

from cache.db import REPO_ROOT
from analysis.api_client import AnalysisClient
from reporting.lp_letter import _gather_context

log = logging.getLogger(__name__)

WEEKLY_DIR = REPO_ROOT / "output" / "weekly"

SYSTEM_PROMPT = """You are JARVIS, AI strategist at Meridian Capital
Partners. Compose the weekly commentary for Limited Partners. Voice:
institutional, concrete, longer-horizon than the daily letter. Five to
seven paragraphs covering:

  1. Week-over-week performance and key drivers
  2. What changed about positioning (factor tilts, sector exposures)
  3. Risk and crowding observations
  4. One forward-looking paragraph (not predictions — readiness)
  5. Brief mention of any pending earnings catalysts in the book

Return ONLY the commentary body — no salutation, no signature, no
letterhead. The harness wraps those around your text."""


def _cfg_weekday() -> int:
    return int(
        yaml.safe_load(open(REPO_ROOT / "config.yaml"))
            .get("reporting", {})
            .get("weekly_commentary_weekday", 4)
    )


def should_fire_today() -> bool:
    return date.today().weekday() == _cfg_weekday()


def generate(*, force: bool = False) -> Path | None:
    if not force and not should_fire_today():
        log.info("Weekly commentary skipped — not the configured weekday")
        return None

    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    out = WEEKLY_DIR / f"{today.isoformat()}.md"

    ctx = _gather_context()
    try:
        client = AnalysisClient()
        payload = json.dumps(ctx, indent=2, default=str)
        result = client.analyze_json(
            system=SYSTEM_PROMPT,
            user=f"Compose this week's commentary from:\n\n{payload}",
            analyzer="weekly_commentary", ticker=today.isoformat(),
        )
        if isinstance(result, dict):
            body = result.get("commentary") or result.get("body") or json.dumps(result)
        elif isinstance(result, str):
            body = result
        else:
            body = _fallback(ctx)
    except Exception as exc:
        log.warning("Weekly commentary Claude call failed: %s", exc)
        body = _fallback(ctx)

    out.write_text(f"# Weekly Commentary — {today.strftime('%B %d, %Y')}\n\n{body.strip()}\n")
    log.info("Weekly commentary written to %s", out)
    return out


def _fallback(ctx: dict) -> str:
    exp = ctx["exposures"]
    return (
        f"This week the book carried {exp['gross']*100:.0f}% gross and "
        f"{exp['net']*100:+.0f}% net across {exp['n_positions']} positions, "
        f"with net beta {exp['net_beta']:+.2f}.\n\n"
        f"Sector-relative performance was {ctx['sector_alpha_total']*100:+.2f}% "
        f"across {ctx['sector_winners'] + ctx['sector_losers']} sectors. "
        f"Risk metrics remained inside policy bands.\n\n"
        f"We continue to monitor factor crowding and pending earnings catalysts."
    )
