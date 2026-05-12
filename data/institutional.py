"""13-F holdings for the 9 tracked hedge funds.

Pulls the latest 13-F-HR per fund from EDGAR, parses the information_table.xml,
and writes one row per (fund_cik, ticker, report_date). Detects multi-fund
opening flag (>=N funds opening a new position simultaneously).
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable

import yaml
from tqdm import tqdm

from cache.db import REPO_ROOT, conn_ctx, init_schema, upsert_many
from data.sec_data import EDGAR_ARCHIVES, EdgarClient, _fetch_submissions, _filter_recent

log = logging.getLogger(__name__)

NS_13F = "{http://www.sec.gov/edgar/document/thirteenf/informationtable}"
INDEX_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F&dateb=&owner=include&count=40&output=atom"


def _load_funds() -> list[dict]:
    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml"))
    return cfg.get("institutional", {}).get("funds", [])


def _list_filing_files(client: EdgarClient, cik_int: int, accession: str) -> list[str]:
    """List filenames in a filing's directory via the EDGAR index.json."""
    accession_nodash = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
    try:
        data = client.get(idx_url).json()
    except Exception:
        return []
    return [item["name"] for item in data.get("directory", {}).get("item", [])]


def _parse_information_table(xml_bytes: bytes, fund_name: str, fund_cik: str, report_date: str) -> list[tuple]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("13-F parse failed for %s: %s", fund_name, exc)
        return []

    rows: list[tuple] = []
    for info in root.findall(f"{NS_13F}infoTable"):
        ticker_elem = info.find(f"{NS_13F}cusip")  # 13-F doesn't carry tickers, only CUSIPs
        # We store the CUSIP in `ticker` for now; downstream code can map to tickers.
        cusip = ticker_elem.text.strip() if ticker_elem is not None and ticker_elem.text else ""
        shares_elem = info.find(f"{NS_13F}shrsOrPrnAmt/{NS_13F}sshPrnamt")
        value_elem = info.find(f"{NS_13F}value")
        try:
            shares = float(shares_elem.text) if shares_elem is not None else 0.0
            value = float(value_elem.text) if value_elem is not None else 0.0
            # Pre-2022 EDGAR reported value in thousands; modern filings report dollars.
            # We leave as-is and let downstream callers normalize if needed.
        except (TypeError, ValueError):
            continue
        rows.append((fund_name, fund_cik, cusip, report_date, shares, value))
    return rows


def refresh_13f(skip: bool = False) -> int:
    """Pull the latest 13-F per tracked fund. Returns total holding rows written."""
    init_schema()
    if skip:
        log.info("--no-13f: skipping institutional pull")
        return 0

    client = EdgarClient()
    funds = _load_funds()
    total = 0

    for fund in tqdm(funds, desc="13-F"):
        cik10 = str(fund["cik"]).zfill(10)
        cik_int = int(cik10)

        try:
            sub = _fetch_submissions(client, cik10)
        except Exception as exc:
            log.warning("submissions failed for %s: %s", fund["name"], exc)
            continue

        filings = _filter_recent(sub, {"13F-HR", "13F-HR/A"}, since=None)
        if not filings:
            continue
        latest = filings[0]
        accession = latest["accession"]
        report_date = latest["period"] or latest["filing_date"]

        files = _list_filing_files(client, cik_int, accession)
        info_files = [f for f in files if f.lower().endswith(".xml") and "info" in f.lower()]
        if not info_files:
            info_files = [f for f in files if f.lower().endswith(".xml") and f != "primary_doc.xml"]

        for fname in info_files:
            url = EDGAR_ARCHIVES.format(
                cik_int=cik_int, accession_nodash=accession.replace("-", ""), filename=fname,
            )
            try:
                resp = client.get(url)
            except Exception as exc:
                log.warning("download %s for %s failed: %s", fname, fund["name"], exc)
                continue
            rows = _parse_information_table(resp.content, fund["name"], cik10, report_date)
            if not rows:
                continue
            with conn_ctx() as conn:
                upsert_many(
                    conn,
                    "institutional_holdings",
                    ["fund_name", "fund_cik", "ticker", "report_date", "shares", "market_value"],
                    rows,
                )
            total += len(rows)

    log.info("13-F holdings rows written: %d", total)
    return total


def multi_fund_openings(threshold: int = 3) -> list[str]:
    """Tickers (CUSIPs in current schema) where >=threshold funds opened a new
    position in the latest report period vs the prior one."""
    init_schema()
    with conn_ctx() as conn:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT ticker, fund_cik, report_date,
                       ROW_NUMBER() OVER (PARTITION BY fund_cik, ticker ORDER BY report_date DESC) AS rn
                FROM institutional_holdings
            )
            SELECT ticker, COUNT(DISTINCT fund_cik) AS opens
            FROM ranked
            WHERE rn = 1
              AND ticker NOT IN (
                  SELECT ticker FROM ranked WHERE rn = 2
              )
            GROUP BY ticker
            HAVING opens >= ?
            """,
            (threshold,),
        ).fetchall()
    return [r["ticker"] for r in rows]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh_13f()
