"""
Shared utilities: config loading, HTTP helpers, logging, company-name
normalization (used to match SEC 13F filings - which only report CUSIP and
issuer name, not ticker - back to our stock universe).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_cfg_cache: Optional[dict] = None

MARKET_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN = (9, 30)
_MARKET_CLOSE = (16, 0)


def load_config(force_reload: bool = False) -> dict:
    global _cfg_cache
    if _cfg_cache is None or force_reload:
        with open(CONFIG_PATH) as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


_YAML_MAPPING_HEADER = re.compile(r"^([\w-]+):\s*(#.*)?$")


def _find_config_line(lines: list[str], dotted_key: str) -> int:
    """
    Locates the line index of a scalar `key: value` in config.yaml. Walks
    the file's indentation structure (not a full YAML parse - just enough
    to track which section headers are "open" at each line) so a dotted key
    (e.g. "strategies.cash_secured_put.max_candidates") can disambiguate a
    leaf name that repeats under different parents - every strategy has its
    own `max_candidates`, for instance. A plain, non-dotted key (e.g.
    "risk_profile") is assumed unique and matched wherever it appears, same
    as before.
    """
    parts = dotted_key.split(".")
    leaf, parents = parts[-1], parts[:-1]
    stack: list[tuple[int, str]] = []  # (indent, key) for each currently-open parent section
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        header = _YAML_MAPPING_HEADER.match(stripped)
        if header:
            stack.append((indent, header.group(1)))
        if parents and [k for _, k in stack] != parents:
            continue
        if re.match(rf"^{re.escape(leaf)}:\s*\S", stripped):
            return i
    raise KeyError(f"{dotted_key!r} not found in config.yaml")


def update_config_values(updates: dict[str, Any]) -> dict:
    """
    In-place update of `key: value` scalar lines in config.yaml (used by the
    dashboard's Settings panel). Edits the text directly rather than
    yaml.safe_dump-ing the whole file, since that would strip every comment.
    Keys may be dotted paths (see _find_config_line) to disambiguate a leaf
    name that appears under more than one parent section.
    """
    lines = CONFIG_PATH.read_text().splitlines(keepends=True)
    for dotted_key, value in updates.items():
        leaf = dotted_key.split(".")[-1]
        idx = _find_config_line(lines, dotted_key)
        line = lines[idx]
        m = re.match(rf"^(\s*{re.escape(leaf)}:)(\s*)\S+", line)
        lines[idx] = f"{m.group(1)}{m.group(2)}{value}{line[m.end():]}"
    CONFIG_PATH.write_text("".join(lines))
    return load_config(force_reload=True)


def is_market_hours(now: Optional[datetime] = None) -> bool:
    """Weekday 9:30-16:00 America/New_York. Doesn't model holidays - good enough for a cache-TTL heuristic."""
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_t, close_t = _MARKET_OPEN, _MARKET_CLOSE
    return (open_t[0], open_t[1]) <= (now.hour, now.minute) < (close_t[0], close_t[1])


def minutes_until_next_market_open(now: Optional[datetime] = None) -> float:
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    candidate_day = now
    for _ in range(8):  # at most a week out
        candidate_open = candidate_day.replace(
            hour=_MARKET_OPEN[0], minute=_MARKET_OPEN[1], second=0, microsecond=0
        )
        if candidate_day.weekday() < 5 and now < candidate_open:
            return (candidate_open - now).total_seconds() / 60.0
        candidate_day = (candidate_day + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return 7 * 24 * 60.0  # fallback, shouldn't be reached


def effective_ttl_minutes(base_ttl_minutes: float, now: Optional[datetime] = None) -> float:
    """
    Stretches a cache TTL to cover the gap until the next market open when
    the market is currently closed - prices/options chains can't have moved
    while it's shut, so there's no reason to re-fetch an identical snapshot
    every `base_ttl_minutes` overnight or across a weekend. An explicit
    "Generate recommendations" click still works any time; outside market
    hours it just reuses the already-cached closing data instead of
    re-hitting the network for a byte-identical copy, which is exactly what
    you want when planning for the next session.
    """
    if is_market_hours(now):
        return base_ttl_minutes
    return max(base_ttl_minutes, minutes_until_next_market_open(now) + 15)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger("common")


def sec_headers() -> dict:
    """
    SEC EDGAR requires a descriptive User-Agent identifying the requester
    (https://www.sec.gov/os/accessing-edgar-data). Set your contact email
    in config.yaml -> sec_user_agent before running against real SEC endpoints.
    """
    return {"User-Agent": load_config()["sec_user_agent"], "Accept-Encoding": "gzip, deflate"}


def http_get(
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 15,
    retries: int = 3,
    backoff: float = 1.5,
) -> requests.Response:
    """GET with simple retry/backoff. Raises on final failure."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise requests.HTTPError(f"429 rate-limited on {url}")
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
            last_exc = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff ** attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Company name normalization, used to match SEC 13F "nameOfIssuer" strings
# (e.g. "APPLE INC", "ALPHABET INC-CL A") to our universe's company names.
# This is inherently best-effort: 13F filings give CUSIP, not ticker, and
# there's no free, reliable CUSIP->ticker lookup, so fuzzy name matching is
# the pragmatic free-data-only option. Expect occasional misses on tricky
# names (multiple share classes, recent renames, etc.) - that's disclosed
# in the README.
# ---------------------------------------------------------------------------

_SUFFIXES = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|PLC|SA|NV|AG|HOLDINGS|HLDGS|GROUP|GRP)\b\.?",
    re.IGNORECASE,
)
_CLASS_SUFFIX = re.compile(r"[-\s]+CL\s*[A-Z]\b", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")


def normalize_company_name(name: str) -> str:
    if not name:
        return ""
    n = name.upper()
    n = _CLASS_SUFFIX.sub("", n)
    n = _SUFFIXES.sub("", n)
    n = _NON_ALNUM.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def build_name_index(universe_rows: list[dict]) -> dict[str, str]:
    """normalized company name -> ticker, for exact-after-normalization lookups."""
    index = {}
    for row in universe_rows:
        norm = normalize_company_name(row.get("company_name", ""))
        if norm:
            index[norm] = row["ticker"]
    return index


def match_issuer_to_ticker(issuer_name: str, name_index: dict[str, str]) -> Optional[str]:
    """
    Try exact normalized match first, then a loose containment match
    (handles cases like "ALPHABET INC" in universe vs "ALPHABET INC-CL C" in 13F).
    """
    norm = normalize_company_name(issuer_name)
    if not norm:
        return None
    if norm in name_index:
        return name_index[norm]
    # loose containment: shortest-common-prefix heuristic
    for candidate_norm, ticker in name_index.items():
        if not candidate_norm:
            continue
        if norm.startswith(candidate_norm) or candidate_norm.startswith(norm):
            return ticker
    return None
