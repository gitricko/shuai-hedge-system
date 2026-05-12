"""Daily LP letter — Claude-authored in JARVIS voice.

3-4 paragraph body covering: current positioning, key contributors,
risk posture, market context. Formal institutional tone with
letterhead, "Dear Limited Partners," opening, signature block, and a
compliance footer.

Cached by date in output/letters/{YYYY-MM-DD}.md. Regeneration
overwrites — call with force=True.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from cache.db import REPO_ROOT, conn_ctx
from analysis.api_client import AnalysisClient
from portfolio import state as port_state
from portfolio.beta import book_beta, compute_betas
from reporting import pnl_attribution, sector_alpha, win_loss
from risk import risk_state

log = logging.getLogger(__name__)

LETTERS_DIR = REPO_ROOT / "output" / "letters"
COMPLIANCE_FOOTER = (
    "Past performance is not indicative of future results. This communication is "
    "confidential and intended solely for the named recipient. Not an offer or "
    "solicitation for any security."
)


def _cfg() -> dict:
    fund = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("fund", {})
    rep = yaml.safe_load(open(REPO_ROOT / "config.yaml")).get("reporting", {})
    return {**fund, **rep}


def _gather_context() -> dict:
    """Build the data dict Claude reasons over."""
    pos = port_state.get_positions()
    weights = pd.Series(dtype=float)
    sectors = {}
    if not pos.empty:
        signs = pos["side"].map({"LONG": 1, "SHORT": -1})
        weights = pd.Series((pos["weight"] * signs).values, index=pos["ticker"])
        sectors = dict(zip(pos["ticker"], pos["sector"]))

    bb = {"long": 0.0, "short": 0.0, "net": 0.0}
    if not weights.empty:
        betas = compute_betas(weights.index.tolist())
        bb = book_beta(weights, betas)

    risk = risk_state.load()
    sector_summary = sector_alpha.summary()
    wl = win_loss.analyze()
    today_attr = pnl_attribution.attribute_today()

    return {
        "as_of": date.today().isoformat(),
        "exposures": {
            "gross": float(weights.abs().sum()),
            "net": float(weights.sum()),
            "n_positions": int(len(weights)),
            "long_beta": bb["long"], "short_beta": bb["short"], "net_beta": bb["net"],
        },
        "todays_attribution": today_attr,
        "tail_risk": risk.get("tail_risk", {}),
        "factor_alerts": risk.get("factor", {}).get("alerts", []),
        "sector_alpha_total": sector_summary["total_alpha"],
        "sector_winners": sector_summary["winners"],
        "sector_losers": sector_summary["losers"],
        "win_rate": wl["overall"].get("win_rate"),
        "trades_to_date": wl["overall"].get("n"),
    }


SYSTEM_PROMPT = """You are JARVIS, the in-house AI strategist at Meridian
Capital Partners. You write the daily letter to Limited Partners. Voice:
calm, institutional, specific, never hyperbolic. Three or four short
paragraphs. No emojis. Reference concrete numbers from the context.

Structure:
  1. Lead with today's net P&L and the principal driver (beta vs alpha).
  2. Current positioning — gross/net/beta and any sector or factor tilts.
  3. Risk posture — call out tail-risk directives, factor alerts, or
     crowding warnings if any; otherwise note risk is within bands.
  4. (Optional) One forward-looking sentence about the day ahead.

Return ONLY the letter body — no salutation, no signature, no
letterhead. The harness wraps those around your text."""


def _claude_body(ctx: dict) -> str:
    client = AnalysisClient()
    payload = json.dumps(ctx, indent=2, default=str)
    user = f"Compose today's LP letter body from the following context:\n\n{payload}"
    parsed = client.analyze_json(
        system=SYSTEM_PROMPT, user=user, analyzer="lp_letter", ticker=ctx["as_of"],
    )
    if isinstance(parsed, dict):
        # Some models wrap in {"letter": "..."} — accept that shape too
        for k in ("letter", "body", "text"):
            if k in parsed and isinstance(parsed[k], str):
                return parsed[k]
        return json.dumps(parsed, indent=2)
    if isinstance(parsed, str):
        return parsed
    # Fallback if Claude didn't return JSON — call once more in plain mode.
    return _fallback_body(ctx)


def _fallback_body(ctx: dict) -> str:
    """Deterministic fallback when Claude is unavailable (no API key etc.)."""
    exp = ctx["exposures"]
    attr = ctx["todays_attribution"] or {}
    alpha = attr.get("alpha", 0.0)
    total = attr.get("total", 0.0)
    risk = ctx.get("tail_risk") or {}
    risk_line = (f"Tail-risk directive in effect: {risk.get('directive')}."
                 if risk.get("directive") else "Risk metrics within bands.")
    return (
        f"The book closed {('up' if total >= 0 else 'down')} {total * 100:+.2f}% today, "
        f"with stock-selection alpha contributing {alpha * 100:+.2f}%. "
        f"\n\nPositioning remains balanced at {exp['gross']*100:.0f}% gross "
        f"and {exp['net']*100:+.0f}% net across {exp['n_positions']} positions; "
        f"net beta is {exp['net_beta']:+.2f}.\n\n"
        f"{risk_line} Sector-relative performance is "
        f"{ctx['sector_alpha_total']*100:+.2f}% across "
        f"{ctx['sector_winners'] + ctx['sector_losers']} sectors. "
        f"We continue to monitor the factor-spread alerts surfaced in the daily risk report."
    )


def _doc_id(today: date) -> str:
    cfg = _cfg()
    prefix = cfg.get("letter_doc_id_prefix", "MCP-IM")
    return f"{prefix}-{today.year}-{today.strftime('%m%d')}"


def _wrap(body: str, ctx: dict) -> str:
    cfg = _cfg()
    today = date.fromisoformat(ctx["as_of"])
    lines: list[str] = []
    lines.append(f"# {cfg.get('name', 'Meridian Capital Partners')}")
    lines.append(f"_{cfg.get('domicile', 'Delaware')} • Inception {cfg.get('inception', '')}_  ")
    lines.append(f"Doc ID: {_doc_id(today)} • Date: {today.strftime('%B %d, %Y')}")
    lines.append("")
    lines.append("**CONFIDENTIAL • LIMITED PARTNERS ONLY**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Dear Limited Partners,")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    lines.append("Sincerely,")
    lines.append("")
    lines.append("**JARVIS**")
    lines.append("_AI Strategist — Meridian Capital Partners_")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_{COMPLIANCE_FOOTER}_")
    return "\n".join(lines)


def generate(*, force: bool = False) -> Path:
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    out = LETTERS_DIR / f"{today.isoformat()}.md"
    if out.exists() and not force:
        log.info("LP letter already exists for %s; pass force=True to regenerate", today)
        return out
    ctx = _gather_context()
    try:
        body = _claude_body(ctx)
    except Exception as exc:
        log.warning("LP letter Claude call failed (%s) — using fallback body", exc)
        body = _fallback_body(ctx)
    out.write_text(_wrap(body, ctx))
    log.info("LP letter written to %s", out)
    return out
