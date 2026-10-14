"""
SEC EDGAR data sources (all free, official, no API key):

1. 8-K "material event" filings per stock - data.sec.gov/submissions JSON.
2. 13F-HR holdings for a configurable list of prominent investors/hedge
   funds - used as a "smart money" signal. 13F filings only report CUSIP
   and issuer name (not ticker), are filed up to 45 days after quarter-end,
   and the $-value unit changed from thousands to whole dollars across the
   2023-2024 SEC amendments - so treat the value field as an approximate
   magnitude, and treat the whole 13F signal as quarterly-lag "did a big
   investor build/trim a position", not a timely trading signal.

Both endpoints are rate-limited politely (small sleep between requests,
descriptive User-Agent per SEC's access guidelines).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from common import build_name_index, get_logger, http_get, match_issuer_to_ticker, sec_headers

log = get_logger("sec_edgar")

SEC_REQUEST_SLEEP = 0.25  # be polite; SEC's fair-access guideline caps at 10 req/s


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def get_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = http_get(url, headers=sec_headers())
    time.sleep(SEC_REQUEST_SLEEP)
    return resp.json()


# ---------------------------------------------------------------------------
# 8-K material filings
# ---------------------------------------------------------------------------

def fetch_recent_8k_filings(cik: str, lookback_days: int = 30) -> list[dict]:
    data = get_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    cutoff = date.today() - timedelta(days=lookback_days)

    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        filed_date_str = recent["filingDate"][i]
        try:
            filed_date = datetime.strptime(filed_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filed_date < cutoff:
            continue
        accession = recent["accessionNumber"][i]
        accession_nodash = accession.replace("-", "")
        primary_doc = recent["primaryDocument"][i] if i < len(recent.get("primaryDocument", [])) else ""
        items = recent.get("items", [""] * len(forms))[i]
        cik_int = int(cik)
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"
        out.append(
            {
                "accession_no": accession,
                "filed_date": filed_date_str,
                "items": items,
                "primary_doc": primary_doc,
                "filing_url": filing_url,
            }
        )
    return out


def refresh_8k_for_universe(
    db, cfg: dict, universe: list[dict], on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Concurrent, but deliberately much less parallel than market_data's
    Yahoo fetches (max_workers=2, not 8) - SEC's fair-access guideline caps
    at 10 req/s and each worker already paces itself at ~4 req/s
    (SEC_REQUEST_SLEEP), so 2 workers stays safely under that ceiling while
    still being faster than one at a time for a ~368-ticker universe pass.
    """
    ttl = cfg["cache_ttl_minutes"]["sec_8k"]
    lookback_days = cfg.get("news_lookback_days", 7) * 4
    summary = {"updated": 0, "skipped_fresh": 0, "skipped_no_cik": 0, "errors": []}

    to_fetch = []
    for row in universe:
        ticker, cik = row["ticker"], row.get("cik")
        if not cik:
            summary["skipped_no_cik"] += 1
            continue
        age = db.latest_8k_fetch_age_minutes(ticker)
        if age is not None and age <= ttl:
            summary["skipped_fresh"] += 1
            continue
        to_fetch.append((ticker, cik))

    def _fetch_one(ticker: str, cik: str):
        filings = fetch_recent_8k_filings(cik, lookback_days=lookback_days)
        if filings:
            db.upsert_8k_filings(ticker, filings)

    total = len(to_fetch)
    completed = 0
    progress_every = max(10, total // 10)
    with ThreadPoolExecutor(max_workers=cfg.get("sec_edgar_max_workers", 2)) as pool:
        future_to_ticker = {pool.submit(_fetch_one, ticker, cik): ticker for ticker, cik in to_fetch}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                future.result()
                summary["updated"] += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("8-K fetch failed for %s: %s", ticker, exc)
                summary["errors"].append(f"{ticker}: {exc}")
            if on_progress and (completed % progress_every == 0 or completed == total):
                on_progress(f"Refreshing SEC 8-K filings... ({completed}/{total})")
    log.info("8-K refresh summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# 13F prominent-investor holdings
# ---------------------------------------------------------------------------

def _find_infotable_doc(cik: str, accession_no: str) -> Optional[str]:
    accession_nodash = accession_no.replace("-", "")
    cik_int = int(cik)
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
    resp = http_get(index_url, headers=sec_headers())
    time.sleep(SEC_REQUEST_SLEEP)
    items = resp.json().get("directory", {}).get("item", [])
    # Prefer anything with "infotable" in the name; fall back to any xml that
    # isn't the primary cover-page doc.
    xml_items = [it["name"] for it in items if it["name"].lower().endswith(".xml")]
    for name in xml_items:
        if "infotable" in name.lower():
            return name
    for name in xml_items:
        if name.lower() not in ("primary_doc.xml",):
            return name
    return None


def _fetch_infotable_xml(cik: str, accession_no: str, filename: str) -> str:
    accession_nodash = accession_no.replace("-", "")
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"
    resp = http_get(url, headers=sec_headers())
    time.sleep(SEC_REQUEST_SLEEP)
    return resp.text


def parse_13f_infotable(xml_text: str) -> list[dict]:
    """
    Parses a 13F information-table XML into per-CUSIP rows, aggregating
    multiple <infoTable> entries that share a CUSIP (filers sometimes split
    a position across several voting-authority rows).
    """
    root = ET.fromstring(xml_text)
    agg: dict[str, dict] = {}
    for info in root.iter():
        if _strip_ns(info.tag) != "infoTable":
            continue
        issuer, cusip, shares, value = None, None, 0, 0
        for child in info:
            local = _strip_ns(child.tag)
            if local == "nameOfIssuer":
                issuer = (child.text or "").strip()
            elif local == "cusip":
                cusip = (child.text or "").strip()
            elif local == "value":
                try:
                    value = int(float(child.text))
                except (TypeError, ValueError):
                    value = 0
            elif local == "shrsOrPrnAmt":
                for sub in child:
                    if _strip_ns(sub.tag) == "sshPrnamt":
                        try:
                            shares = int(float(sub.text))
                        except (TypeError, ValueError):
                            shares = 0
        if not cusip:
            continue
        if cusip not in agg:
            agg[cusip] = {"name_of_issuer": issuer or "", "cusip": cusip, "shares": 0, "value_usd": 0}
        agg[cusip]["shares"] += shares
        agg[cusip]["value_usd"] += value
    return list(agg.values())


def refresh_13f_for_all_investors(db, cfg: dict, universe: list[dict]) -> dict:
    """
    For each prominent investor in config, pull their most recent 13F-HR
    (and the one before it, for quarter-over-quarter diffing), match each
    holding to our ticker universe by company name, and cache it.
    """
    ttl = cfg["cache_ttl_minutes"]["sec_13f"]
    name_index = build_name_index(universe)
    summary = {"investors_updated": 0, "investors_skipped_fresh": 0, "holdings_matched": 0, "errors": []}

    for investor in cfg.get("prominent_investors", []):
        name, cik = investor["name"], investor["cik"]
        cache_key = f"13f_fetched_{cik}"
        age = db.cache_age_minutes(cache_key)
        if age is not None and age <= ttl:
            summary["investors_skipped_fresh"] += 1
            continue
        try:
            subs = get_submissions(cik)
            recent = subs.get("filings", {}).get("recent", {})
            filings_13f = [
                {
                    "accession_no": recent["accessionNumber"][i],
                    "filed_date": recent["filingDate"][i],
                    "period_of_report": recent.get("reportDate", [None] * len(recent["form"]))[i],
                }
                for i, form in enumerate(recent.get("form", []))
                if form in ("13F-HR", "13F-HR/A")
            ]
            filings_13f.sort(key=lambda f: f["filed_date"], reverse=True)
            latest_two = filings_13f[:2]  # most recent + prior, for diffing

            for filing in latest_two:
                doc = _find_infotable_doc(cik, filing["accession_no"])
                if not doc:
                    log.warning("No info table found for %s accession %s", name, filing["accession_no"])
                    continue
                xml_text = _fetch_infotable_xml(cik, filing["accession_no"], doc)
                holdings = parse_13f_infotable(xml_text)
                rows = []
                for h in holdings:
                    matched_ticker = match_issuer_to_ticker(h["name_of_issuer"], name_index)
                    if matched_ticker:
                        summary["holdings_matched"] += 1
                    rows.append(
                        {
                            "investor_cik": cik,
                            "investor_name": name,
                            "period_of_report": filing["period_of_report"] or filing["filed_date"],
                            "name_of_issuer": h["name_of_issuer"],
                            "cusip": h["cusip"],
                            "matched_ticker": matched_ticker,
                            "shares": h["shares"],
                            "value_usd": h["value_usd"],
                        }
                    )
                if rows:
                    db.upsert_13f_holdings(rows)
            db.cache_set(cache_key, {"last_checked": filings_13f[0]["filed_date"] if filings_13f else None})
            summary["investors_updated"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("13F refresh failed for %s (cik=%s): %s", name, cik, exc)
            summary["errors"].append(f"{name}: {exc}")

    log.info("13F refresh summary: %s", summary)
    return summary
