"""
Stock universe loader: either the S&P 500 (scraped from Wikipedia's
"List of S&P 500 companies" page, which conveniently already includes each
company's SEC CIK) or a custom watchlist from config.yaml, in which case we
look up CIKs via SEC's free company_tickers.json.

Everything here degrades gracefully: if the live fetch fails (no internet,
Wikipedia layout changed, etc.) we fall back to whatever is already cached
in the DB rather than crashing the whole pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import pandas as pd

from common import get_logger, http_get, sec_headers
from data_sources.market_data import fetch_market_cap_and_volume

log = get_logger("universe")

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _to_yf_ticker(ticker: str) -> str:
    """Yahoo Finance / yfinance use '-' for share classes, Wikipedia uses '.' (e.g. BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def _pad_cik(cik) -> str:
    return str(int(cik)).zfill(10)


def _fetch_sp500_from_wikipedia() -> list[dict]:
    resp = http_get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; StockPutScreener/1.0)"})
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]  # first table on the page is the constituents list
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "ticker": _to_yf_ticker(str(r["Symbol"])),
                "company_name": str(r.get("Security", "")).strip(),
                "cik": _pad_cik(r["CIK"]) if "CIK" in r and str(r["CIK"]).strip() else None,
                "sector": str(r.get("GICS Sector", "")).strip(),
            }
        )
    log.info("Fetched %d S&P 500 constituents from Wikipedia", len(rows))
    return rows


def _fetch_sec_ticker_map() -> dict[str, dict]:
    """ticker -> {cik, company_name} for every SEC-registered company."""
    resp = http_get(SEC_COMPANY_TICKERS_URL, headers=sec_headers())
    data = resp.json()
    mapping = {}
    for entry in data.values():
        t = str(entry.get("ticker", "")).upper()
        if t:
            mapping[t] = {"cik": _pad_cik(entry["cik_str"]), "company_name": entry.get("title", "")}
    return mapping


def _resolve_tickers_via_sec(tickers: list[str], sec_map: dict[str, dict], sector_label: str | None) -> list[dict]:
    """Look up company name/CIK for a flat ticker list (used for custom watchlists and ETFs)."""
    rows = []
    for raw in tickers:
        t = _to_yf_ticker(raw)
        sec_lookup_key = raw.strip().upper()  # SEC map keys use raw ticker, not yf-style dashed
        info = sec_map.get(sec_lookup_key) or sec_map.get(t)
        rows.append(
            {
                "ticker": t,
                "company_name": info["company_name"] if info else t,
                "cik": info["cik"] if info else None,
                "sector": sector_label,
            }
        )
    return rows


def load_universe(db, cfg: dict, force_refresh: bool = False) -> list[dict]:
    ttl = cfg["cache_ttl_minutes"]["universe_list"]
    age = db.universe_age_minutes()
    cached = db.get_universe()

    if not force_refresh and cached and age is not None and age <= ttl:
        log.info("Using cached universe (%d tickers, %.1f min old)", len(cached), age)
        return cached

    try:
        sec_map = {}
        try:
            sec_map = _fetch_sec_ticker_map()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch SEC ticker map for CIK lookup: %s", exc)

        if cfg.get("universe_mode") == "custom":
            rows = _resolve_tickers_via_sec(cfg.get("custom_tickers", []), sec_map, sector_label=None)
        else:
            rows = _fetch_sp500_from_wikipedia()

        etf_tickers = cfg.get("include_etfs", [])
        if etf_tickers:
            existing = {r["ticker"] for r in rows}
            etf_rows = [
                r for r in _resolve_tickers_via_sec(etf_tickers, sec_map, sector_label="ETF")
                if r["ticker"] not in existing
            ]
            rows.extend(etf_rows)

        if not rows:
            raise ValueError("Fetched an empty universe")
        db.upsert_universe(rows)
        return db.get_universe()
    except Exception as exc:  # noqa: BLE001
        log.error("Live universe fetch failed (%s); falling back to cache", exc)
        if cached:
            return cached
        raise


def build_universe_filter(db, cfg: dict, universe: list[dict], max_workers: int = 8) -> list[dict]:
    """
    Fetches/updates the market cap + avg volume reading behind the "liquid
    large caps" pre-filter and returns the resulting eligible list. This is
    the one piece of this feature that touches the network, and it's
    deliberately kept out of a normal pipeline run's hot path - call it from
    the server-startup background job or an explicit "rebuild" action
    (POST /api/universe/rebuild), not from run_pipeline.run().

    Each ticker's reading is cached independently (see config.yaml ->
    cache_ttl_minutes.universe_filter, default weekly), so most calls here
    are just a cache hit - only tickers never seen before, or whose reading
    has gone stale, trigger a fetch. Those fetches run concurrently (a
    handful of worker threads, not one request at a time) since each is an
    independent read with no shared state - this is the practical substitute
    for a single batched request: yfinance has no reliable batch endpoint
    for market cap that doesn't risk the same hidden-extra-request problem
    fetch_market_cap_and_volume's docstring warns about for fast_info.
    ETFs (include_etfs) are passed through untouched since they're opted in
    explicitly, not scanned for cap/volume.
    """
    filt_cfg = cfg.get("universe_filter", {})
    min_cap = filt_cfg.get("min_market_cap", 20_000_000_000)
    min_vol = filt_cfg.get("min_avg_volume", 1_000_000)
    ttl = cfg["cache_ttl_minutes"].get("universe_filter", 10080)
    max_workers = filt_cfg.get("max_workers", max_workers)

    eligible = []
    to_fetch = []
    for row in universe:
        if row.get("sector") == "ETF":
            eligible.append(row)
            continue
        cached = db.get_universe_filter(row["ticker"], ttl_minutes=ttl)
        if cached is None:
            to_fetch.append(row)
        elif (cached.get("market_cap") or 0) >= min_cap and (cached.get("avg_volume") or 0) >= min_vol:
            eligible.append(row)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_row = {pool.submit(fetch_market_cap_and_volume, row["ticker"]): row for row in to_fetch}
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    reading = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Market cap/volume fetch failed for %s: %s", row["ticker"], exc)
                    reading = None
                if not reading:
                    continue
                db.upsert_universe_filter(row["ticker"], reading["market_cap"], reading["avg_volume"])
                if reading["market_cap"] >= min_cap and (reading["avg_volume"] or 0) >= min_vol:
                    eligible.append(row)

    log.info(
        "Universe pre-filter built: %d/%d eligible (>=$%.0fB market cap, >=%.0fM avg volume; %d freshly fetched)",
        len(eligible), len(universe), min_cap / 1e9, min_vol / 1e6, len(to_fetch),
    )
    return eligible


def get_eligible_universe(db, cfg: dict, universe: list[dict]) -> list[dict]:
    """
    Zero-network read of the pre-built universe filter (see
    build_universe_filter, which actually fetches it) - this is what
    run_pipeline.run() calls, so a normal run never pays the market
    cap/volume lookup cost itself. A ticker build_universe_filter has never
    scored yet is simply excluded here rather than fetched inline; run the
    build step (background startup job, or POST /api/universe/rebuild) to
    pick it up.
    """
    filt_cfg = cfg.get("universe_filter", {})
    min_cap = filt_cfg.get("min_market_cap", 20_000_000_000)
    min_vol = filt_cfg.get("min_avg_volume", 1_000_000)

    eligible = []
    for row in universe:
        if row.get("sector") == "ETF":
            eligible.append(row)
            continue
        cached = db.get_universe_filter(row["ticker"])  # any age - staleness is build_universe_filter's concern
        if cached and (cached.get("market_cap") or 0) >= min_cap and (cached.get("avg_volume") or 0) >= min_vol:
            eligible.append(row)

    log.info("Universe (pre-built filter): %d/%d eligible", len(eligible), len(universe))
    return eligible
