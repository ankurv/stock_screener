"""
Free news signal via Google News RSS (no API key). We don't have a
structured "corporate events" feed available for free, so headlines are
keyword-tagged into buckets that matter for a put seller:

  legal      - lawsuits, investigations, regulatory trouble
  ma_buyout  - acquisition/takeover/merger chatter (cuts both ways for a put
               seller: target stocks often jump on a deal, but a collapsed
               deal or a going-private cash deal can also strand your position
               - that's why this is surfaced as a flag, not scored directly)
  stock_split - forward/reverse split announcements (changes strike economics)
  product    - launches/announcements
  other      - everything else that still mentions the ticker

Uses only the standard library's xml.etree.ElementTree to parse the RSS feed
(Google News RSS is well-formed RSS 2.0, so no need for feedparser).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional
from urllib.parse import quote

from common import get_logger, http_get

log = get_logger("news_rss")

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def _classify(title: str, cfg: dict) -> str:
    t = title.lower()
    for kw in cfg.get("legal_keywords", []):
        if kw.lower() in t:
            return "legal"
    for kw in cfg.get("ma_buyout_keywords", []):
        if kw.lower() in t:
            return "ma_buyout"
    for kw in cfg.get("stock_split_keywords", []):
        if kw.lower() in t:
            return "stock_split"
    for kw in cfg.get("product_keywords", []):
        if kw.lower() in t:
            return "product"
    return "other"


def _parse_pubdate(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _fetch_rss_items(query_text: str, cfg: dict, limit: int) -> list[dict]:
    url = f"{GOOGLE_NEWS_RSS_URL}?q={quote(query_text)}&hl=en-US&gl=US&ceid=US:en"
    resp = http_get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; StockPutScreener/1.0)"})

    root = ET.fromstring(resp.text)
    lookback_days = cfg.get("news_lookback_days", 7)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        source_el = item.find("source")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        if not title or not link:
            continue
        published_at = _parse_pubdate(pubdate_el.text if pubdate_el is not None else None)
        if published_at:
            try:
                if datetime.fromisoformat(published_at) < cutoff:
                    continue
            except ValueError:
                pass
        items.append(
            {
                "link": link,
                "title": title,
                "source": (source_el.text or "").strip() if source_el is not None else None,
                "published_at": published_at or datetime.now(timezone.utc).isoformat(),
                "category": _classify(title, cfg),
            }
        )
        if len(items) >= limit:
            break
    return items


def fetch_news_for_ticker(ticker: str, company_name: Optional[str], cfg: dict, limit: int = 15) -> list[dict]:
    query_text = f"{company_name} stock" if company_name else f"{ticker} stock"
    return _fetch_rss_items(query_text, cfg, limit)


def fetch_market_news(cfg: dict, limit: int = 8) -> list[dict]:
    """
    Broad market-moving news - Fed rate decisions, macro data, geopolitical
    events - not tied to any single ticker. Shown next to the market mood
    indicator so the user has context for *why* the tape is behaving the way
    it is, and what's coming up that could change it.
    """
    query_text = cfg.get(
        "market_news_query",
        "stock market OR Federal Reserve OR Fed rate decision OR geopolitical",
    )
    return _fetch_rss_items(query_text, cfg, limit)


def refresh_market_news(db, cfg: dict) -> list[dict]:
    """Refresh the cached market-mood news if stale; always returns the current (possibly cached) list."""
    ttl = cfg["cache_ttl_minutes"].get("market_news", 60)
    cached = db.get_market_news(ttl_minutes=ttl)
    if cached is not None:
        return cached
    try:
        items = fetch_market_news(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Market news fetch failed: %s", exc)
        return db.get_market_news() or []
    db.set_market_news(items)
    return items


def refresh_news_for_universe(
    db, cfg: dict, universe: list[dict], on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Concurrent (no stated rate-limit guideline for Google News RSS, unlike SEC EDGAR) - same pattern as market_data."""
    ttl = cfg["cache_ttl_minutes"]["news"]
    summary = {"updated": 0, "skipped_fresh": 0, "errors": []}

    to_fetch = []
    for row in universe:
        ticker = row["ticker"]
        age = db.news_fetch_age_minutes(ticker)
        if age is not None and age <= ttl:
            summary["skipped_fresh"] += 1
            continue
        to_fetch.append(row)

    def _fetch_one(row: dict):
        items = fetch_news_for_ticker(row["ticker"], row.get("company_name"), cfg)
        if items:
            db.upsert_news(row["ticker"], items)

    total = len(to_fetch)
    completed = 0
    progress_every = max(10, total // 10)
    with ThreadPoolExecutor(max_workers=cfg.get("universe_filter", {}).get("max_workers", 8)) as pool:
        future_to_ticker = {pool.submit(_fetch_one, row): row["ticker"] for row in to_fetch}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                future.result()
                summary["updated"] += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("News fetch failed for %s: %s", ticker, exc)
                summary["errors"].append(f"{ticker}: {exc}")
            if on_progress and (completed % progress_every == 0 or completed == total):
                on_progress(f"Refreshing news... ({completed}/{total})")
    log.info("News refresh summary: %s", summary)
    return summary
