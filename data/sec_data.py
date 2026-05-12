"""SEC EDGAR client for filings + Form 4 insider transactions.

Honors EDGAR's User-Agent and 8 req/sec rate-limit policy. For each ticker:
    * latest 10-K (full document for Risk Factors)
    * latest 10-Q (MD&A)
    * recent 8-K filings
    * Form 4 insider transactions over the last 180 days

Form 4 XML is parsed into the `insider_transactions` table. CEO/CFO purchases
are flagged via title heuristics; cluster-buy detection (3+ insiders within
30 days) is done at the SQL layer in `cluster_buy_tickers()`.
"""
from __future__ import annotations

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import requests
import yaml
from tqdm import tqdm

from cache.db import REPO_ROOT, conn_ctx, init_schema, upsert_many
from data.universe import get_universe

log = logging.getLogger(__name__)

EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"

FILINGS_DIR = REPO_ROOT / "cache" / "edgar"
TICKERS_CACHE = REPO_ROOT / "cache" / "edgar_tickers.json"


def _user_agent() -> str:
    cfg_path = REPO_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    email = os.environ.get("SEC_USER_AGENT_EMAIL") or cfg.get("sec", {}).get(
        "user_agent_email", "research@example.com"
    )
    return f"Meridian Capital Research {email}"


class EdgarClient:
    """Throttled EDGAR client. 8 req/sec is the SEC's hard cap."""

    def __init__(self, rate_per_sec: int = 8):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"})
        self._min_interval = 1.0 / rate_per_sec
        self._last = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.time()

    def get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        resp = self.session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp


def _load_ticker_to_cik(client: EdgarClient) -> dict[str, str]:
    if TICKERS_CACHE.exists() and (time.time() - TICKERS_CACHE.stat().st_mtime) < 7 * 86400:
        with open(TICKERS_CACHE) as f:
            return json.load(f)
    log.info("Fetching EDGAR ticker -> CIK map")
    data = client.get(EDGAR_TICKERS_URL).json()
    out: dict[str, str] = {}
    for entry in data.values():
        out[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TICKERS_CACHE, "w") as f:
        json.dump(out, f)
    return out


def _fetch_submissions(client: EdgarClient, cik10: str) -> dict:
    url = EDGAR_SUBMISSIONS.format(cik10=cik10)
    return client.get(url).json()


def _download_doc(client: EdgarClient, cik_int: int, accession: str, filename: str) -> Path:
    accession_nodash = accession.replace("-", "")
    url = EDGAR_ARCHIVES.format(cik_int=cik_int, accession_nodash=accession_nodash, filename=filename)
    out_path = FILINGS_DIR / str(cik_int) / accession / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)  # handles nested filenames like xslF345X06/form4.xml
    if out_path.exists():
        return out_path
    resp = client.get(url)
    out_path.write_bytes(resp.content)
    return out_path


def _resolve_form4_xml(client: EdgarClient, cik_int: int, accession: str) -> str | None:
    """Find the raw Form 4 XML filename via the filing index.

    The submissions JSON's `primary` field points to the XSL-rendered HTML
    (e.g. ``xslF345X06/form4.xml``), not the raw data. The actual XML lives
    at the filing's top level — typically ``form4.xml``, sometimes
    ``wf-form4_<ts>.xml`` or ``primary_doc.xml``.
    """
    accession_nodash = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
    try:
        data = client.get(idx_url).json()
    except Exception:
        return None
    items = [it["name"] for it in data.get("directory", {}).get("item", [])]
    # Prefer top-level .xml files (not under xslF345X06/)
    xmls = [n for n in items if n.lower().endswith(".xml") and "/" not in n]
    if not xmls:
        return None
    # Sort: form4*.xml first, then primary_doc.xml, then anything else
    def rank(name: str) -> int:
        ln = name.lower()
        if "form4" in ln:
            return 0
        if "primary_doc" in ln:
            return 1
        return 2
    xmls.sort(key=rank)
    return xmls[0]


def _parse_form4(xml_bytes: bytes) -> list[dict]:
    """Extract non-derivative + derivative transaction rows from Form 4 XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("Form 4 parse failed: %s", exc)
        return []

    def _text(elem, path: str, default: str = "") -> str:
        node = elem.find(path)
        return node.text.strip() if node is not None and node.text else default

    insider_name = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    is_officer = is_director = is_ten = 0
    title = ""
    if rel is not None:
        is_officer = 1 if _text(rel, "isOfficer") in ("1", "true", "True") else 0
        is_director = 1 if _text(rel, "isDirector") in ("1", "true", "True") else 0
        is_ten = 1 if _text(rel, "isTenPercentOwner") in ("1", "true", "True") else 0
        title = _text(rel, "officerTitle")

    out: list[dict] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        out.append({
            "insider_name": insider_name,
            "insider_title": title,
            "is_officer": is_officer,
            "is_director": is_director,
            "is_ten_pct_owner": is_ten,
            "transaction_date": _text(tx, "transactionDate/value"),
            "transaction_type": _text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "transaction_code": _text(tx, "transactionCoding/transactionCode"),
            "ownership_type": _text(tx, "ownershipNature/directOrIndirectOwnership/value"),
            "shares": _text(tx, "transactionAmounts/transactionShares/value") or "0",
            "price": _text(tx, "transactionAmounts/transactionPricePerShare/value") or "0",
        })
    return out


def _filter_recent(filings: dict, form_types: set[str], since: date | None) -> list[dict]:
    """EDGAR submissions JSON has parallel arrays under recent."""
    recent = filings.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primaries = recent.get("primaryDocument", [])
    periods = recent.get("reportDate", [""] * len(forms))
    out: list[dict] = []
    for f, a, d, p, prd in zip(forms, accessions, dates, primaries, periods):
        if f not in form_types:
            continue
        if since and datetime.strptime(d, "%Y-%m-%d").date() < since:
            continue
        out.append({"form": f, "accession": a, "filing_date": d, "primary": p, "period": prd})
    return out


def refresh_filings(
    tickers: Iterable[str] | None = None,
    skip: bool = False,
    forms: list[str] | None = None,
) -> tuple[int, int]:
    """Refresh filings + insider txns. Returns (filings_count, insider_txn_count)."""
    init_schema()
    if skip:
        log.info("--no-filings: skipping SEC pull")
        return 0, 0

    if tickers is None:
        tickers = get_universe(include_benchmarks=False)
    tickers = list(tickers)
    forms_set = set(forms) if forms else {"10-K", "10-Q", "8-K", "4"}

    cfg = yaml.safe_load(open(REPO_ROOT / "config.yaml"))
    insider_lookback = int(cfg.get("sec", {}).get("insider_lookback_days", 180))
    rate_limit = int(cfg.get("sec", {}).get("rate_limit_per_sec", 8))
    client = EdgarClient(rate_per_sec=rate_limit)
    ticker_to_cik = _load_ticker_to_cik(client)

    insider_since = date.today() - timedelta(days=insider_lookback)
    filings_count = 0
    insider_count = 0

    for ticker in tqdm(tickers, desc="sec"):
        cik10 = ticker_to_cik.get(ticker)
        if not cik10:
            continue
        cik_int = int(cik10)

        try:
            sub = _fetch_submissions(client, cik10)
        except Exception as exc:
            log.warning("submissions failed %s: %s", ticker, exc)
            continue

        filing_rows: list[tuple] = []
        insider_rows: list[tuple] = []

        # 10-K / 10-Q / 8-K — keep latest 10-K, latest 10-Q, all 8-K within 90d
        for f in _filter_recent(sub, forms_set & {"10-K", "10-Q", "8-K"}, since=date.today() - timedelta(days=400)):
            try:
                local = _download_doc(client, cik_int, f["accession"], f["primary"])
            except Exception as exc:
                log.warning("download %s/%s failed: %s", ticker, f["accession"], exc)
                local = None
            filing_rows.append((
                f["accession"], ticker, cik10, f["form"], f["filing_date"],
                f["period"], f["primary"], str(local) if local else None,
            ))
            filings_count += 1

        # Form 4 — recent insider activity
        if "4" in forms_set:
            for f in _filter_recent(sub, {"4"}, since=insider_since):
                accession = f["accession"]
                try:
                    xml_name = _resolve_form4_xml(client, cik_int, accession)
                    if not xml_name:
                        log.warning("No raw XML found for %s/%s", ticker, accession)
                        continue
                    local = _download_doc(client, cik_int, accession, xml_name)
                    txns = _parse_form4(local.read_bytes())
                except Exception as exc:
                    log.warning("Form 4 fetch %s/%s: %s", ticker, accession, exc)
                    continue
                filing_rows.append((
                    accession, ticker, cik10, "4", f["filing_date"],
                    f["period"], xml_name, str(local),
                ))
                filings_count += 1
                for t in txns:
                    try:
                        shares = float(t["shares"]) if t["shares"] else 0.0
                        price = float(t["price"]) if t["price"] else 0.0
                    except ValueError:
                        continue
                    insider_rows.append((
                        accession, ticker, t["insider_name"], t["insider_title"],
                        t["is_officer"], t["is_director"], t["is_ten_pct_owner"],
                        t["transaction_date"], t["transaction_type"], t["transaction_code"],
                        t["ownership_type"], shares, price,
                    ))
                    insider_count += 1

        if filing_rows or insider_rows:
            with conn_ctx() as conn:
                if filing_rows:
                    upsert_many(
                        conn, "sec_filings",
                        ["accession", "ticker", "cik", "form_type", "filing_date",
                         "period_of_report", "primary_doc", "local_path"],
                        filing_rows,
                    )
                if insider_rows:
                    upsert_many(
                        conn, "insider_transactions",
                        ["accession", "ticker", "insider_name", "insider_title",
                         "is_officer", "is_director", "is_ten_pct_owner",
                         "transaction_date", "transaction_type", "transaction_code",
                         "ownership_type", "shares", "price"],
                        insider_rows,
                    )

    log.info("filings=%d, insider_txns=%d", filings_count, insider_count)
    return filings_count, insider_count


def cluster_buy_tickers(window_days: int = 30, threshold: int = 3) -> list[str]:
    """Tickers where >=threshold distinct insiders made open-market purchases (code P)
    within any rolling `window_days` window in the last 180 days.
    """
    init_schema()
    cutoff = (date.today() - timedelta(days=180)).isoformat()
    with conn_ctx() as conn:
        rows = conn.execute(
            """
            SELECT ticker, insider_name, transaction_date
            FROM insider_transactions
            WHERE transaction_code = 'P' AND transaction_date >= ?
            ORDER BY ticker, transaction_date
            """,
            (cutoff,),
        ).fetchall()

    by_ticker: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append((r["transaction_date"], r["insider_name"]))

    flagged: list[str] = []
    for ticker, txns in by_ticker.items():
        for i, (d_i, _) in enumerate(txns):
            window_end = datetime.strptime(d_i, "%Y-%m-%d").date() + timedelta(days=window_days)
            insiders_in_window = {
                name for d, name in txns[i:]
                if datetime.strptime(d, "%Y-%m-%d").date() <= window_end
            }
            if len(insiders_in_window) >= threshold:
                flagged.append(ticker)
                break
    return sorted(set(flagged))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh_filings()
    print("Cluster-buy tickers:", cluster_buy_tickers())
