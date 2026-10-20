"""
SQLite caching/storage layer for the put screener.

Design goals:
  - One file (data/screener.db) that survives restarts so the dashboard
    loads instantly from cache instead of re-hitting Yahoo/SEC every time.
  - Every cached row carries an `updated_at` timestamp. Freshness is decided
    by the *caller* (via config.yaml's cache_ttl_minutes), not baked in here -
    this module just stores/retrieves and tells you how old something is.
  - No ORM. This is a small, inspectable project; raw SQL is easier to audit.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "screener.db"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _minutes_old(iso_ts: str) -> float:
    then = datetime.fromisoformat(iso_ts)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 60.0


class Database:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # Columns added after this DB file may already exist, in a tables-with-data
    # upgrade path (ALTER TABLE ... ADD COLUMN) instead of the usual "delete
    # data/screener.db and let schema.sql recreate it fresh" - the user asked
    # that accumulated candidates/history/price_history never be wiped just
    # for a schema change. Each entry is (table, column, sqlite type).
    _LIGHT_MIGRATIONS = [
        ("analyst_snapshot", "debt_to_equity", "REAL"),
        ("put_candidate", "debt_to_equity", "REAL"),
        ("dip_candidate", "debt_to_equity", "REAL"),
        ("three_day_candidate", "debt_to_equity", "REAL"),
        ("put_candidate", "relative_strength_pct", "REAL"),
        ("put_candidate", "historical_median_depth_pct", "REAL"),
        ("put_candidate", "historical_median_duration_days", "INTEGER"),
        ("dip_candidate", "relative_strength_pct", "REAL"),
        ("dip_candidate", "historical_median_depth_pct", "REAL"),
        ("dip_candidate", "historical_median_duration_days", "INTEGER"),
        ("dip_candidate", "dip_maturity_score", "REAL"),
        ("dip_candidate", "dip_maturity_label", "TEXT"),
        ("three_day_candidate", "relative_strength_pct", "REAL"),
        ("three_day_candidate", "historical_median_depth_pct", "REAL"),
        ("three_day_candidate", "historical_median_duration_days", "INTEGER"),
        ("three_day_candidate", "dip_maturity_score", "REAL"),
        ("three_day_candidate", "dip_maturity_label", "TEXT"),
        ("analyst_snapshot", "industry", "TEXT"),
        ("put_candidate", "industry", "TEXT"),
        ("dip_candidate", "industry", "TEXT"),
        ("three_day_candidate", "industry", "TEXT"),
        ("put_candidate", "price_captured_at", "TEXT"),
        ("dip_candidate", "price_captured_at", "TEXT"),
        ("three_day_candidate", "price_captured_at", "TEXT"),
    ]

    def _init_schema(self):
        with self.conn() as c:
            c.executescript(SCHEMA_PATH.read_text())
            for table, column, sqltype in self._LIGHT_MIGRATIONS:
                try:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sqltype}")
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc):
                        raise

    # ------------------------------------------------------------------
    # Generic fetch-cache (for anything not worth its own table)
    # ------------------------------------------------------------------
    def cache_get(self, key: str, ttl_minutes: Optional[float] = None) -> Optional[Any]:
        with self.conn() as c:
            row = c.execute(
                "SELECT payload, fetched_at FROM fetch_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["fetched_at"]) > ttl_minutes:
            return None
        return json.loads(row["payload"])

    def cache_set(self, key: str, payload: Any):
        with self.conn() as c:
            c.execute(
                "INSERT INTO fetch_cache (cache_key, payload, fetched_at) VALUES (?, ?, ?)"
                " ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, fetched_at=excluded.fetched_at",
                (key, json.dumps(payload), utcnow_iso()),
            )

    def cache_age_minutes(self, key: str) -> Optional[float]:
        with self.conn() as c:
            row = c.execute("SELECT fetched_at FROM fetch_cache WHERE cache_key = ?", (key,)).fetchone()
        return _minutes_old(row["fetched_at"]) if row else None

    # ------------------------------------------------------------------
    # Market mood (broad market trend + macro news) - not ticker-specific
    # ------------------------------------------------------------------
    def get_market_mood(self, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        return self.cache_get("market_mood", ttl_minutes=ttl_minutes)

    def set_market_mood(self, data: dict):
        self.cache_set("market_mood", data)


    def get_sector_performance(self, ttl_minutes: Optional[float] = None) -> Optional[list]:
        return self.cache_get("sector_performance", ttl_minutes=ttl_minutes)

    def set_sector_performance(self, data: list[dict]):
        self.cache_set("sector_performance", data)

    def get_sparkline(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[list]:
        return self.cache_get(f"sparkline:{ticker}", ttl_minutes=ttl_minutes)

    def set_sparkline(self, ticker: str, values: list):
        self.cache_set(f"sparkline:{ticker}", values)

    # ------------------------------------------------------------------
    # Technical indicators (RSI/SMA trend/MACD) per ticker
    # ------------------------------------------------------------------
    def get_technicals(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        return self.cache_get(f"technicals:{ticker}", ttl_minutes=ttl_minutes)

    def set_technicals(self, ticker: str, data: dict):
        self.cache_set(f"technicals:{ticker}", data)

    # ------------------------------------------------------------------
    # Daily close history per ticker - the shared "common data" every
    # strategy/signal reads from instead of each re-pulling its own
    # overlapping window from Yahoo (see data_sources.market_data.
    # refresh_price_history, which does the incremental fetch-and-append).
    # ------------------------------------------------------------------
    def get_price_history(self, ticker: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT date, close FROM price_history WHERE ticker = ? ORDER BY date ASC", (ticker,)
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_price_history(self, ticker: str, rows: Iterable[dict]):
        rows = list(rows)
        if not rows:
            return
        with self.conn() as c:
            c.executemany(
                "INSERT INTO price_history (ticker, date, close) VALUES (:ticker, :date, :close) "
                "ON CONFLICT(ticker, date) DO UPDATE SET close=excluded.close",
                [{"ticker": ticker, "date": r["date"], "close": r["close"]} for r in rows],
            )

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def upsert_universe(self, rows: Iterable[dict]):
        now = utcnow_iso()
        with self.conn() as c:
            c.executemany(
                "INSERT INTO universe (ticker, company_name, cik, sector, updated_at) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(ticker) DO UPDATE SET company_name=excluded.company_name,"
                " cik=excluded.cik, sector=excluded.sector, updated_at=excluded.updated_at",
                [(r["ticker"], r.get("company_name"), r.get("cik"), r.get("sector"), now) for r in rows],
            )

    def get_universe(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM universe ORDER BY ticker").fetchall()
        return [dict(r) for r in rows]

    def universe_age_minutes(self) -> Optional[float]:
        with self.conn() as c:
            row = c.execute("SELECT MAX(updated_at) AS u FROM universe").fetchone()
        return _minutes_old(row["u"]) if row and row["u"] else None

    # ------------------------------------------------------------------
    # Universe pre-filter (market cap / avg volume) - see config.yaml -> universe_filter
    # ------------------------------------------------------------------
    def get_universe_filter(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM universe_filter_cache WHERE ticker = ?", (ticker,)).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["updated_at"]) > ttl_minutes:
            return None
        return dict(row)

    def upsert_universe_filter(self, ticker: str, market_cap: Optional[float], avg_volume: Optional[float]):
        with self.conn() as c:
            c.execute(
                "INSERT INTO universe_filter_cache (ticker, market_cap, avg_volume, updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(ticker) DO UPDATE SET market_cap=excluded.market_cap,"
                " avg_volume=excluded.avg_volume, updated_at=excluded.updated_at",
                (ticker, market_cap, avg_volume, utcnow_iso()),
            )

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------
    def upsert_price(self, ticker: str, price: float):
        with self.conn() as c:
            c.execute(
                "INSERT INTO price_snapshot (ticker, last_price, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(ticker) DO UPDATE SET last_price=excluded.last_price, updated_at=excluded.updated_at",
                (ticker, price, utcnow_iso()),
            )

    def get_price(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[float]:
        with self.conn() as c:
            row = c.execute(
                "SELECT last_price, updated_at FROM price_snapshot WHERE ticker = ?", (ticker,)
            ).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["updated_at"]) > ttl_minutes:
            return None
        return row["last_price"]

    def get_price_captured_at(self, ticker: str) -> Optional[str]:
        """
        Full ISO8601 UTC timestamp of when price_snapshot's cached quote for
        this ticker was actually fetched - the true "as of" moment behind
        whatever get_price(ticker) just returned, for strategies to stamp
        onto a freshly-flagged candidate's price_captured_at (see
        db.save_candidates / db._save_rows's carry-forward of the same).
        """
        with self.conn() as c:
            row = c.execute("SELECT updated_at FROM price_snapshot WHERE ticker = ?", (ticker,)).fetchone()
        return row["updated_at"] if row else None

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------
    def upsert_earnings(self, ticker: str, earnings_date: Optional[str], is_estimate: bool = True):
        with self.conn() as c:
            c.execute(
                "INSERT INTO earnings_calendar (ticker, earnings_date, is_estimate, updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(ticker) DO UPDATE SET earnings_date=excluded.earnings_date,"
                " is_estimate=excluded.is_estimate, updated_at=excluded.updated_at",
                (ticker, earnings_date, int(is_estimate), utcnow_iso()),
            )

    def get_earnings(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM earnings_calendar WHERE ticker = ?", (ticker,)).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["updated_at"]) > ttl_minutes:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Analyst consensus snapshot
    # ------------------------------------------------------------------
    # Named params (not positional `?`) specifically so adding a new field
    # here is a one-place change (this list) instead of three places kept
    # in sync by hand - the exact way this broke before (a column added to
    # the column list and VALUES but not the ON CONFLICT clause, so it
    # silently never updated on upsert).
    _ANALYST_SNAPSHOT_COLUMNS = [
        "recommendation_key", "recommendation_mean", "num_analysts", "target_mean_price",
        "target_high_price", "target_low_price", "pe_ratio", "week52_low", "week52_high", "debt_to_equity",
        "industry",
    ]

    def upsert_analyst_snapshot(self, ticker: str, snap: dict):
        cols = self._ANALYST_SNAPSHOT_COLUMNS
        with self.conn() as c:
            c.execute(
                f"INSERT INTO analyst_snapshot (ticker, {', '.join(cols)}, updated_at) "
                f"VALUES (:ticker, {', '.join(':' + col for col in cols)}, :updated_at) "
                f"ON CONFLICT(ticker) DO UPDATE SET {', '.join(f'{col}=excluded.{col}' for col in cols)}, "
                f"updated_at=excluded.updated_at",
                {"ticker": ticker, "updated_at": utcnow_iso(), **{col: snap.get(col) for col in cols}},
            )

    def get_analyst_snapshot(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM analyst_snapshot WHERE ticker = ?", (ticker,)).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["updated_at"]) > ttl_minutes:
            return None
        return dict(row)

    _ANALYST_SNAPSHOT_HISTORY_COLUMNS = [
        "recommendation_key", "recommendation_mean", "num_analysts",
        "target_mean_price", "target_high_price", "target_low_price",
    ]

    def append_analyst_snapshot_history(self, ticker: str, snap: dict, snapshot_date: Optional[str] = None):
        """
        One row per ticker per calendar day - idempotent re-run on the same
        day (ON CONFLICT updates in place) so calling this more than once in
        a day (e.g. re-running the pipeline) doesn't create duplicate rows.
        Called alongside upsert_analyst_snapshot, not instead of it - see
        analyst_snapshot_history's schema comment.
        """
        cols = self._ANALYST_SNAPSHOT_HISTORY_COLUMNS
        snapshot_date = snapshot_date or utcnow_iso()[:10]
        with self.conn() as c:
            c.execute(
                f"INSERT INTO analyst_snapshot_history (ticker, snapshot_date, {', '.join(cols)}, updated_at) "
                f"VALUES (:ticker, :snapshot_date, {', '.join(':' + col for col in cols)}, :updated_at) "
                f"ON CONFLICT(ticker, snapshot_date) DO UPDATE SET {', '.join(f'{col}=excluded.{col}' for col in cols)}, "
                f"updated_at=excluded.updated_at",
                {
                    "ticker": ticker, "snapshot_date": snapshot_date, "updated_at": utcnow_iso(),
                    **{col: snap.get(col) for col in cols},
                },
            )

    def get_analyst_snapshot_history(self, ticker: str, limit: int = 60) -> list[dict]:
        """Ascending by date (oldest first) - the shape engine.signals.analyst_revision_momentum_score expects."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM analyst_snapshot_history WHERE ticker = ? ORDER BY snapshot_date DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Post-earnings reaction anomaly
    # ------------------------------------------------------------------
    def upsert_post_earnings_reaction(self, ticker: str, data: dict):
        with self.conn() as c:
            c.execute(
                "INSERT INTO post_earnings_reaction (ticker, earnings_date, surprise_pct, reaction_pct, anomaly_type, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET earnings_date=excluded.earnings_date, surprise_pct=excluded.surprise_pct, "
                "reaction_pct=excluded.reaction_pct, anomaly_type=excluded.anomaly_type, updated_at=excluded.updated_at",
                (
                    ticker, data.get("earnings_date"), data.get("surprise_pct"),
                    data.get("reaction_pct"), data.get("anomaly_type"), utcnow_iso(),
                ),
            )

    def get_post_earnings_reaction(self, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM post_earnings_reaction WHERE ticker = ?", (ticker,)).fetchone()
        if row is None:
            return None
        if ttl_minutes is not None and _minutes_old(row["updated_at"]) > ttl_minutes:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Put option snapshots
    # ------------------------------------------------------------------
    def replace_put_snapshots(self, ticker: str, rows: Iterable[dict]):
        """Overwrite all cached put rows for a ticker with a fresh snapshot."""
        now = utcnow_iso()
        with self.conn() as c:
            c.execute("DELETE FROM put_option_snapshot WHERE ticker = ?", (ticker,))
            c.executemany(
                "INSERT INTO put_option_snapshot "
                "(ticker, expiry, strike, bid, ask, last_price, implied_vol, open_interest, volume, in_the_money, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        ticker, r["expiry"], r["strike"], r.get("bid"), r.get("ask"),
                        r.get("last_price"), r.get("implied_vol"), r.get("open_interest"),
                        r.get("volume"), int(bool(r.get("in_the_money"))), now,
                    )
                    for r in rows
                ],
            )

    def get_put_snapshots(self, ticker: str, ttl_minutes: Optional[float] = None) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM put_option_snapshot WHERE ticker = ? ORDER BY expiry, strike", (ticker,)
            ).fetchall()
        if not rows:
            return []
        if ttl_minutes is not None and _minutes_old(rows[0]["updated_at"]) > ttl_minutes:
            return []
        return [dict(r) for r in rows]

    def replace_call_snapshots(self, ticker: str, rows: Iterable[dict]):
        """See replace_put_snapshots - identical shape, call_option_snapshot table."""
        now = utcnow_iso()
        with self.conn() as c:
            c.execute("DELETE FROM call_option_snapshot WHERE ticker = ?", (ticker,))
            c.executemany(
                "INSERT INTO call_option_snapshot "
                "(ticker, expiry, strike, bid, ask, last_price, implied_vol, open_interest, volume, in_the_money, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        ticker, r["expiry"], r["strike"], r.get("bid"), r.get("ask"),
                        r.get("last_price"), r.get("implied_vol"), r.get("open_interest"),
                        r.get("volume"), int(bool(r.get("in_the_money"))), now,
                    )
                    for r in rows
                ],
            )

    def get_call_snapshots(self, ticker: str, ttl_minutes: Optional[float] = None) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM call_option_snapshot WHERE ticker = ? ORDER BY expiry, strike", (ticker,)
            ).fetchall()
        if not rows:
            return []
        if ttl_minutes is not None and _minutes_old(rows[0]["updated_at"]) > ttl_minutes:
            return []
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # SEC 8-K filings
    # ------------------------------------------------------------------
    def upsert_8k_filings(self, ticker: str, rows: Iterable[dict]):
        now = utcnow_iso()
        with self.conn() as c:
            c.executemany(
                "INSERT INTO sec_8k_filing (ticker, accession_no, filed_date, items, primary_doc, filing_url, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, accession_no) DO UPDATE SET filed_date=excluded.filed_date, "
                "items=excluded.items, primary_doc=excluded.primary_doc, filing_url=excluded.filing_url, updated_at=excluded.updated_at",
                [
                    (ticker, r["accession_no"], r["filed_date"], r.get("items", ""), r.get("primary_doc"), r.get("filing_url"), now)
                    for r in rows
                ],
            )

    def get_8k_filings(self, ticker: str, limit: int = 5) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM sec_8k_filing WHERE ticker = ? ORDER BY filed_date DESC LIMIT ?", (ticker, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_8k_fetch_age_minutes(self, ticker: str) -> Optional[float]:
        with self.conn() as c:
            row = c.execute(
                "SELECT MAX(updated_at) AS u FROM sec_8k_filing WHERE ticker = ?", (ticker,)
            ).fetchone()
        return _minutes_old(row["u"]) if row and row["u"] else None

    # ------------------------------------------------------------------
    # 13F holdings (prominent investors)
    # ------------------------------------------------------------------
    def upsert_13f_holdings(self, rows: Iterable[dict]):
        now = utcnow_iso()
        with self.conn() as c:
            c.executemany(
                "INSERT INTO thirteenf_holding "
                "(investor_cik, investor_name, period_of_report, name_of_issuer, cusip, matched_ticker, shares, value_usd, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(investor_cik, period_of_report, cusip) DO UPDATE SET "
                "matched_ticker=excluded.matched_ticker, shares=excluded.shares, value_usd=excluded.value_usd, updated_at=excluded.updated_at",
                [
                    (
                        r["investor_cik"], r["investor_name"], r["period_of_report"], r["name_of_issuer"],
                        r["cusip"], r.get("matched_ticker"), r.get("shares"), r.get("value_usd"), now,
                    )
                    for r in rows
                ],
            )

    def get_13f_periods_for_investor(self, investor_cik: str) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT DISTINCT period_of_report FROM thirteenf_holding WHERE investor_cik = ? ORDER BY period_of_report DESC",
                (investor_cik,),
            ).fetchall()
        return [r["period_of_report"] for r in rows]

    def get_13f_changes_for_ticker(self, ticker: str) -> list[dict]:
        """
        For a given matched ticker, return one row per investor showing the
        two most recent periods we have on file and the share-count delta.
        """
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM thirteenf_holding WHERE matched_ticker = ? ORDER BY investor_cik, period_of_report DESC",
                (ticker,),
            ).fetchall()
        by_investor: dict[str, list[dict]] = {}
        for r in rows:
            by_investor.setdefault(r["investor_cik"], []).append(dict(r))

        results = []
        for cik, holdings in by_investor.items():
            holdings.sort(key=lambda h: h["period_of_report"], reverse=True)
            latest = holdings[0]
            prior = holdings[1] if len(holdings) > 1 else None
            prior_shares = prior["shares"] if prior else 0
            delta = (latest["shares"] or 0) - (prior_shares or 0)
            pct_change = (delta / prior_shares * 100) if prior_shares else None
            results.append(
                {
                    "investor_name": latest["investor_name"],
                    "period_of_report": latest["period_of_report"],
                    "shares": latest["shares"],
                    "value_usd": latest["value_usd"],
                    "prior_period": prior["period_of_report"] if prior else None,
                    "prior_shares": prior_shares if prior else None,
                    "share_delta": delta,
                    "pct_change": pct_change,
                    "action": _classify_13f_action(delta, prior_shares, latest["shares"]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    def upsert_news(self, ticker: str, rows: Iterable[dict]):
        now = utcnow_iso()
        with self.conn() as c:
            c.executemany(
                "INSERT INTO news_item (ticker, link, title, source, published_at, category, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, link) DO UPDATE SET category=excluded.category, updated_at=excluded.updated_at",
                [
                    (ticker, r["link"], r.get("title"), r.get("source"), r.get("published_at"), r.get("category"), now)
                    for r in rows
                ],
            )

    def get_news(self, ticker: str, limit: int = 10) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM news_item WHERE ticker = ? ORDER BY published_at DESC LIMIT ?", (ticker, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def news_fetch_age_minutes(self, ticker: str) -> Optional[float]:
        with self.conn() as c:
            row = c.execute("SELECT MAX(updated_at) AS u FROM news_item WHERE ticker = ?", (ticker,)).fetchone()
        return _minutes_old(row["u"]) if row and row["u"] else None

    def get_cached_news_links(self, ticker: str) -> set[str]:
        """Every link already archived for this ticker - lets a fresh fetch tell new items from ones already seen."""
        with self.conn() as c:
            rows = c.execute("SELECT link FROM news_item WHERE ticker = ?", (ticker,)).fetchall()
        return {r["link"] for r in rows}

    # ------------------------------------------------------------------
    # Pipeline runs + final candidates
    # ------------------------------------------------------------------
    def start_run(self) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO pipeline_run (started_at, status) VALUES (?, 'running')", (utcnow_iso(),)
            )
            return cur.lastrowid

    def finish_run(self, run_id: int, status: str, tickers_scanned: int = 0,
                    candidates_found: int = 0, error_message: Optional[str] = None):
        with self.conn() as c:
            c.execute(
                "UPDATE pipeline_run SET finished_at=?, status=?, tickers_scanned=?, candidates_found=?, error_message=? "
                "WHERE run_id=?",
                (utcnow_iso(), status, tickers_scanned, candidates_found, error_message, run_id),
            )

    def get_latest_run(self) -> Optional[dict]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM pipeline_run ORDER BY run_id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def mark_interrupted_runs(self):
        """
        Any run still 'running' at server startup didn't crash gracefully -
        nothing in-memory survived the restart to ever call finish_run() on
        it, so it would otherwise show as a permanently-stuck "(running)"
        with null tickers_scanned/candidates_found in the dashboard header.
        Called once on startup to clean those up.
        """
        with self.conn() as c:
            c.execute(
                "UPDATE pipeline_run SET status='error', finished_at=?, error_message='Interrupted (server restarted)' "
                "WHERE status='running'",
                (utcnow_iso(),),
            )

    def save_candidates(self, run_id: int, candidates: Iterable[dict]):
        candidates = list(candidates)
        with self.conn() as c:
            tickers = list({cand["ticker"] for cand in candidates})
            # Carry forward first_seen_at - AND the underlying price as of
            # that first-seen date - from any pending copy this run
            # supersedes. A ticker that's been pending for a few runs in a
            # row should keep its ORIGINAL "first flagged" date and entry
            # price, not reset either to today's every time a fresh scan
            # re-surfaces it (price especially: it's documented elsewhere as
            # needing to "stay fixed for return_pct to be computed correctly
            # once resolved" - silently drifting it forward on every
            # regeneration broke that invariant).
            first_seen_by_ticker: dict[str, str] = {}
            entry_price_by_ticker: dict[str, float] = {}
            price_captured_at_by_ticker: dict[str, Optional[str]] = {}
            if tickers:
                existing = c.execute(
                    f"SELECT ticker, first_seen_at, price, price_captured_at FROM put_candidate WHERE decision IS NULL AND outcome IS NULL "
                    f"AND ticker IN ({', '.join('?' * len(tickers))})",
                    tickers,
                ).fetchall()
                for row in existing:
                    ticker = row["ticker"]
                    if row["first_seen_at"] and (ticker not in first_seen_by_ticker or row["first_seen_at"] < first_seen_by_ticker[ticker]):
                        first_seen_by_ticker[ticker] = row["first_seen_at"]
                        entry_price_by_ticker[ticker] = row["price"]
                        price_captured_at_by_ticker[ticker] = row["price_captured_at"]
                # A ticker that was already pending from an earlier run (never
                # decided, never resolved) is superseded the moment a fresh scan
                # re-surfaces it - delete that stale copy, so it doesn't
                # silently sit there and later resolve into a phantom duplicate
                # history row once its expiry passes with nobody ever having
                # decided on it. Doesn't touch anything already decided/resolved.
                c.execute(
                    f"DELETE FROM put_candidate WHERE decision IS NULL AND outcome IS NULL "
                    f"AND ticker IN ({', '.join('?' * len(tickers))})",
                    tickers,
                )
            for cand in candidates:
                if cand["ticker"] in first_seen_by_ticker:
                    cand["first_seen_at"] = first_seen_by_ticker[cand["ticker"]]
                    if entry_price_by_ticker.get(cand["ticker"]) is not None:
                        cand["price"] = entry_price_by_ticker[cand["ticker"]]
                    cand["price_captured_at"] = price_captured_at_by_ticker.get(cand["ticker"])
            c.executemany(
                "INSERT INTO put_candidate (run_id, ticker, company_name, price, sector, first_seen_at, price_captured_at, expiry, strike, "
                "bid, ask, mid_price, collateral, premium_total, annualized_yield_pct, prob_otm_pct, delta, open_interest, "
                "spread_pct, days_to_expiry, earnings_before_expiry, analyst_recommendation, analyst_target_mean_price, "
                "pe_ratio, week52_low, week52_high, debt_to_equity, industry, "
                "relative_strength_pct, historical_median_depth_pct, historical_median_duration_days, "
                "earnings_anomaly_type, composite_score, risk_score, risk_label, rsi_14, trend, macd_bullish, rank) VALUES "
                "(:run_id, :ticker, :company_name, :price, :sector, :first_seen_at, :price_captured_at, :expiry, :strike, :bid, :ask, "
                ":mid_price, :collateral, :premium_total, :annualized_yield_pct, :prob_otm_pct, :delta, :open_interest, "
                ":spread_pct, :days_to_expiry, :earnings_before_expiry, :analyst_recommendation, :analyst_target_mean_price, "
                ":pe_ratio, :week52_low, :week52_high, :debt_to_equity, :industry, "
                ":relative_strength_pct, :historical_median_depth_pct, :historical_median_duration_days, "
                ":earnings_anomaly_type, :composite_score, :risk_score, :risk_label, :rsi_14, :trend, :macd_bullish, :rank)",
                [{**cand, "run_id": run_id} for cand in candidates],
            )

    def get_latest_candidates(self, active_only: bool = False, as_of: Optional[str] = None) -> list[dict]:
        """
        active_only=True returns every candidate that's still live - not
        decided (accept/ignore), not resolved (outcome), and not yet
        expired - from ANY run, not just the latest. This is what lets a
        fresh "Generate" ADD to what's still outstanding from earlier runs
        instead of replacing it: nothing here is scoped to one run_id, and
        a ticker surfaced again just updates (dedups to) its most recent row.
        """
        if not active_only:
            return self._latest_rows_for_run("put_candidate")
        as_of = as_of or utcnow_iso()[:10]
        return self._active_rows("put_candidate", "expiry >= ?", (as_of,))

    # ------------------------------------------------------------------
    # Recommendation outcomes - every row in put_candidate is a recommendation
    # that was made at some point, so this table doubles as the history;
    # these just query/update the outcome-tracking columns on it.
    # ------------------------------------------------------------------
    def get_unevaluated_recommendations(self, as_of: Optional[str] = None) -> list[dict]:
        """Past recommendations whose expiry has passed and haven't been scored yet - across every run."""
        as_of = as_of or utcnow_iso()[:10]
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM put_candidate WHERE outcome IS NULL AND expiry <= ? ORDER BY expiry", (as_of,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float) -> Optional[dict]:
        """One exact put_candidate row - see _get_row."""
        return self._get_row("put_candidate", run_id, ticker, "expiry=? AND strike=?", (expiry, strike))

    def record_recommendation_outcome(
        self, run_id: int, ticker: str, expiry: str, strike: float, outcome: str,
        expiry_close_price: Optional[float], return_pct: Optional[float] = None,
    ):
        with self.conn() as c:
            c.execute(
                "UPDATE put_candidate SET outcome=?, expiry_close_price=?, evaluated_at=?, return_pct=? "
                "WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (outcome, expiry_close_price, utcnow_iso(), return_pct, run_id, ticker, expiry, strike),
            )

    def record_candidate_decision(self, run_id: int, ticker: str, expiry: str, strike: float, decision: str):
        """decision: 'accepted' | 'ignored' - moves this row out of the active list and into the track record immediately."""
        with self.conn() as c:
            c.execute(
                "UPDATE put_candidate SET decision=?, decided_at=? WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (decision, utcnow_iso(), run_id, ticker, expiry, strike),
            )

    def delete_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float):
        """Permanently removes one row from the track record - 'stop tracking this one', not an outcome of any kind."""
        with self.conn() as c:
            c.execute(
                "DELETE FROM put_candidate WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (run_id, ticker, expiry, strike),
            )

    def get_recommendation_history(self, limit: int = 200) -> list[dict]:
        """Evaluated and/or decided recommendations, most recent first - for the dashboard's track record."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM put_candidate WHERE outcome IS NOT NULL OR decision IS NOT NULL "
                "ORDER BY expiry DESC, ticker LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Covered calls (see engine/strategies/covered_call.py) - same natural
    # key shape as put_candidate (run_id, ticker, expiry, strike), built on
    # the generic _save_rows/_latest_rows_for_run/_active_rows helpers
    # below instead of a hand-rolled save method - unlike put_candidate,
    # there's only ever one contract-shaped table per ticker at a time
    # pending, so the generic carry-forward logic applies directly.
    # ------------------------------------------------------------------
    _CALL_CANDIDATE_COLUMNS = [
        "run_id", "ticker", "company_name", "price", "sector", "first_seen_at", "price_captured_at",
        "expiry", "strike", "bid", "ask", "mid_price", "shares_value", "premium_total",
        "annualized_yield_pct", "prob_otm_pct", "delta", "open_interest", "spread_pct",
        "days_to_expiry", "earnings_before_expiry", "analyst_recommendation", "analyst_target_mean_price",
        "pe_ratio", "week52_low", "week52_high", "debt_to_equity", "industry",
        "relative_strength_pct", "historical_median_depth_pct", "historical_median_duration_days",
        "earnings_anomaly_type", "composite_score", "risk_score", "risk_label",
        "rsi_14", "trend", "macd_bullish", "rank",
    ]

    def save_call_candidates(self, run_id: int, candidates: Iterable[dict]):
        self._save_rows("call_candidate", self._CALL_CANDIDATE_COLUMNS, run_id, candidates, date_col="first_seen_at")

    def get_latest_call_candidates(self, active_only: bool = False, as_of: Optional[str] = None) -> list[dict]:
        if not active_only:
            return self._latest_rows_for_run("call_candidate")
        as_of = as_of or utcnow_iso()[:10]
        return self._active_rows("call_candidate", "expiry >= ?", (as_of,))

    def get_unevaluated_call_candidates(self, as_of: Optional[str] = None) -> list[dict]:
        as_of = as_of or utcnow_iso()[:10]
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM call_candidate WHERE outcome IS NULL AND expiry <= ? ORDER BY expiry", (as_of,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_call_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float) -> Optional[dict]:
        return self._get_row("call_candidate", run_id, ticker, "expiry=? AND strike=?", (expiry, strike))

    def record_call_outcome(
        self, run_id: int, ticker: str, expiry: str, strike: float, outcome: str,
        expiry_close_price: Optional[float], return_pct: Optional[float] = None,
    ):
        with self.conn() as c:
            c.execute(
                "UPDATE call_candidate SET outcome=?, expiry_close_price=?, evaluated_at=?, return_pct=? "
                "WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (outcome, expiry_close_price, utcnow_iso(), return_pct, run_id, ticker, expiry, strike),
            )

    def record_call_decision(self, run_id: int, ticker: str, expiry: str, strike: float, decision: str):
        with self.conn() as c:
            c.execute(
                "UPDATE call_candidate SET decision=?, decided_at=? WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (decision, utcnow_iso(), run_id, ticker, expiry, strike),
            )

    def delete_call_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float):
        with self.conn() as c:
            c.execute(
                "DELETE FROM call_candidate WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (run_id, ticker, expiry, strike),
            )

    def get_call_history(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM call_candidate WHERE outcome IS NOT NULL OR decision IS NOT NULL "
                "ORDER BY expiry DESC, ticker LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Weekly call buys (see engine/strategies/weekly_call_buy.py) - same
    # natural key/generic-helper pattern as call_candidate above.
    # ------------------------------------------------------------------
    _WEEKLY_CALL_BUY_COLUMNS = [
        "run_id", "ticker", "company_name", "price", "sector", "first_seen_at", "price_captured_at",
        "expiry", "strike", "bid", "ask", "mid_price", "premium_total", "breakeven_price",
        "required_move_pct", "prob_profit_pct", "delta", "implied_vol", "open_interest", "spread_pct",
        "days_to_expiry", "analyst_recommendation", "analyst_target_mean_price",
        "pe_ratio", "week52_low", "week52_high", "debt_to_equity", "industry",
        "relative_strength_pct", "rsi_14", "trend", "macd_bullish",
        "composite_score", "risk_score", "risk_label", "rank",
    ]

    def save_weekly_call_buy_candidates(self, run_id: int, candidates: Iterable[dict]):
        self._save_rows(
            "weekly_call_buy_candidate", self._WEEKLY_CALL_BUY_COLUMNS, run_id, candidates, date_col="first_seen_at"
        )

    def get_latest_weekly_call_buy_candidates(self, active_only: bool = False, as_of: Optional[str] = None) -> list[dict]:
        if not active_only:
            return self._latest_rows_for_run("weekly_call_buy_candidate")
        as_of = as_of or utcnow_iso()[:10]
        return self._active_rows("weekly_call_buy_candidate", "expiry >= ?", (as_of,))

    def get_unevaluated_weekly_call_buy_candidates(self, as_of: Optional[str] = None) -> list[dict]:
        as_of = as_of or utcnow_iso()[:10]
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM weekly_call_buy_candidate WHERE outcome IS NULL AND expiry <= ? ORDER BY expiry", (as_of,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_weekly_call_buy_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float) -> Optional[dict]:
        return self._get_row("weekly_call_buy_candidate", run_id, ticker, "expiry=? AND strike=?", (expiry, strike))

    def record_weekly_call_buy_outcome(
        self, run_id: int, ticker: str, expiry: str, strike: float, outcome: str,
        expiry_close_price: Optional[float], return_pct: Optional[float] = None,
    ):
        with self.conn() as c:
            c.execute(
                "UPDATE weekly_call_buy_candidate SET outcome=?, expiry_close_price=?, evaluated_at=?, return_pct=? "
                "WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (outcome, expiry_close_price, utcnow_iso(), return_pct, run_id, ticker, expiry, strike),
            )

    def record_weekly_call_buy_decision(self, run_id: int, ticker: str, expiry: str, strike: float, decision: str):
        with self.conn() as c:
            c.execute(
                "UPDATE weekly_call_buy_candidate SET decision=?, decided_at=? WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (decision, utcnow_iso(), run_id, ticker, expiry, strike),
            )

    def delete_weekly_call_buy_candidate_row(self, run_id: int, ticker: str, expiry: str, strike: float):
        with self.conn() as c:
            c.execute(
                "DELETE FROM weekly_call_buy_candidate WHERE run_id=? AND ticker=? AND expiry=? AND strike=?",
                (run_id, ticker, expiry, strike),
            )

    def get_weekly_call_buy_history(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM weekly_call_buy_candidate WHERE outcome IS NOT NULL OR decision IS NOT NULL "
                "ORDER BY expiry DESC, ticker LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Generic helpers for per-strategy candidate tables (one table per
    # strategy, see db/schema.sql - put_candidate above, dip_candidate below).
    # These just factor out the executemany/select boilerplate; each
    # strategy still gets its own public methods below since the tables'
    # columns and natural keys genuinely differ (option contract vs. a
    # dated flag).
    # ------------------------------------------------------------------
    def _save_rows(self, table: str, columns: list[str], run_id: int, rows: Iterable[dict], date_col: Optional[str] = None):
        rows = list(rows)
        placeholders = ", ".join(f":{col}" for col in columns)
        with self.conn() as c:
            tickers = list({row["ticker"] for row in rows})
            # See save_candidates' identical carry-forward/delete-stale-pending
            # steps - same reasoning applies to every per-strategy candidate
            # table. date_col names whichever column is this table's "first
            # seen" tracking-start date (e.g. dip_candidate's flagged_date).
            # `price` (and the exact timestamp it was captured at) is carried
            # forward right alongside it, for the same reason: it's the
            # entry price AT that first-seen date, and must stay fixed for
            # return_pct to mean anything once resolved - a still-pending
            # ticker that keeps re-qualifying on later runs should never have
            # its "entry" quietly drift forward to whatever the stock
            # happens to cost on the day it's re-surfaced.
            carried_date: dict[str, str] = {}
            carried_price: dict[str, float] = {}
            carried_captured_at: dict[str, Optional[str]] = {}
            if tickers and date_col:
                existing = c.execute(
                    f"SELECT ticker, {date_col}, price, price_captured_at FROM {table} WHERE decision IS NULL AND outcome IS NULL "
                    f"AND ticker IN ({', '.join('?' * len(tickers))})",
                    tickers,
                ).fetchall()
                for row in existing:
                    val = row[date_col]
                    ticker = row["ticker"]
                    if val and (ticker not in carried_date or val < carried_date[ticker]):
                        carried_date[ticker] = val
                        carried_price[ticker] = row["price"]
                        carried_captured_at[ticker] = row["price_captured_at"]
            if tickers:
                c.execute(
                    f"DELETE FROM {table} WHERE decision IS NULL AND outcome IS NULL "
                    f"AND ticker IN ({', '.join('?' * len(tickers))})",
                    tickers,
                )
            if date_col:
                for row in rows:
                    ticker = row["ticker"]
                    if ticker in carried_date:
                        row[date_col] = carried_date[ticker]
                        if carried_price.get(ticker) is not None:
                            row["price"] = carried_price[ticker]
                        row["price_captured_at"] = carried_captured_at.get(ticker)
            c.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                [{**row, "run_id": run_id} for row in rows],
            )

    def _latest_rows_for_run(
        self, table: str, order_by: str = "rank ASC", where_extra: str = "", params: tuple = ()
    ) -> list[dict]:
        with self.conn() as c:
            run_row = c.execute("SELECT run_id FROM pipeline_run WHERE status='ok' ORDER BY run_id DESC LIMIT 1").fetchone()
            if not run_row:
                return []
            query = f"SELECT * FROM {table} WHERE run_id = ?"
            query_params = [run_row["run_id"], *params]
            if where_extra:
                query += f" AND {where_extra}"
            query += f" ORDER BY {order_by}"
            rows = c.execute(query, query_params).fetchall()
        return [dict(r) for r in rows]

    def _active_rows(self, table: str, extra_where: str = "", params: tuple = ()) -> list[dict]:
        """
        Every row across ALL runs that's still "live" - decision IS NULL
        (not accepted/ignored) and outcome IS NULL (not resolved yet), plus
        whatever extra_where adds (e.g. "expiry >= ?" for put_candidate) -
        deduped to just the most recent run per ticker, since the same
        ticker can legitimately still be pending from several past runs in
        a row (each "Generate" only ADDS to this set, see get_latest_candidates).
        Re-ranks the merged result by composite_score, since the stored
        `rank` column was only ever assigned within a single run.
        """
        cond = "decision IS NULL AND outcome IS NULL"
        if extra_where:
            cond += f" AND {extra_where}"
        query = (
            f"SELECT t1.* FROM {table} t1 WHERE {cond} AND t1.run_id = ("
            f"SELECT MAX(t2.run_id) FROM {table} t2 WHERE t2.ticker = t1.ticker AND {cond}"
            f")"
        )
        with self.conn() as c:
            rows = c.execute(query, [*params, *params]).fetchall()
        results = [dict(r) for r in rows]
        results.sort(key=lambda r: r.get("composite_score") or 0, reverse=True)
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        return results

    def _get_row(self, table: str, run_id: int, ticker: str, extra_where: str = "", params: tuple = ()) -> Optional[dict]:
        """
        One exact row by its natural key - used by the track record's
        manual "check now" refresh button to re-fetch a row's full data
        (every column, not just the identifying key) before handing it to
        a strategy's resolve_outcome/record_outcome, regardless of its
        current decision/outcome state (unlike _active_rows above, which
        deliberately excludes anything already decided or resolved).
        """
        query = f"SELECT * FROM {table} WHERE run_id=? AND ticker=?"
        if extra_where:
            query += f" AND {extra_where}"
        with self.conn() as c:
            row = c.execute(query, (run_id, ticker, *params)).fetchone()
        return dict(row) if row else None

    def get_recent_decided_tickers(self, table: str, cutoff_iso: str) -> set[str]:
        """Tickers decided (accepted/ignored) on/after cutoff_iso in `table` - used to keep a cooldown from re-surfacing them."""
        with self.conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT ticker FROM {table} WHERE decision IS NOT NULL AND decided_at >= ?", (cutoff_iso,)
            ).fetchall()
        return {r["ticker"] for r in rows}

    # ------------------------------------------------------------------
    # Dip-buy candidates (see engine/strategies/dip_buy.py) - mirrors the
    # put_candidate methods above, but keyed on (run_id, ticker, flagged_date)
    # instead of (run_id, ticker, expiry, strike), since there's no option
    # contract here.
    # ------------------------------------------------------------------
    _DIP_CANDIDATE_COLUMNS = [
        "run_id", "ticker", "company_name", "price", "sector", "flagged_date", "price_captured_at",
        "drawdown_pct", "drawdown_lookback_days", "rsi_14", "analyst_recommendation",
        "analyst_target_mean_price", "analyst_upside_pct", "pe_ratio", "week52_low", "week52_high", "debt_to_equity",
        "industry",
        "relative_strength_pct", "historical_median_depth_pct", "historical_median_duration_days",
        "dip_maturity_score", "dip_maturity_label",
        "earnings_date", "earnings_before_horizon",
        "composite_score", "risk_score", "risk_label", "rank",
    ]

    def save_dip_candidates(self, run_id: int, candidates: Iterable[dict]):
        self._save_rows("dip_candidate", self._DIP_CANDIDATE_COLUMNS, run_id, candidates, date_col="flagged_date")

    def get_latest_dip_candidates(self, active_only: bool = False) -> list[dict]:
        """active_only=True returns every still-live candidate across all runs, deduped by ticker - see get_latest_candidates."""
        if not active_only:
            return self._latest_rows_for_run("dip_candidate")
        return self._active_rows("dip_candidate")

    def get_unevaluated_dip_candidates(self, as_of: Optional[str] = None) -> list[dict]:
        """Flagged dip-buy candidates whose evaluation_horizon_days has passed and haven't been scored yet."""
        as_of = as_of or utcnow_iso()[:10]
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM dip_candidate WHERE outcome IS NULL AND flagged_date <= ? ORDER BY flagged_date", (as_of,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dip_candidate_row(self, run_id: int, ticker: str, flagged_date: str) -> Optional[dict]:
        """One exact dip_candidate row - see _get_row."""
        return self._get_row("dip_candidate", run_id, ticker, "flagged_date=?", (flagged_date,))

    def record_dip_outcome(
        self, run_id: int, ticker: str, flagged_date: str, outcome: str,
        resolved_price: Optional[float], return_pct: Optional[float] = None,
    ):
        with self.conn() as c:
            c.execute(
                "UPDATE dip_candidate SET outcome=?, resolved_price=?, evaluated_at=?, return_pct=? "
                "WHERE run_id=? AND ticker=? AND flagged_date=?",
                (outcome, resolved_price, utcnow_iso(), return_pct, run_id, ticker, flagged_date),
            )

    def record_dip_decision(self, run_id: int, ticker: str, flagged_date: str, decision: str):
        """decision: 'accepted' | 'ignored' - see record_candidate_decision."""
        with self.conn() as c:
            c.execute(
                "UPDATE dip_candidate SET decision=?, decided_at=? WHERE run_id=? AND ticker=? AND flagged_date=?",
                (decision, utcnow_iso(), run_id, ticker, flagged_date),
            )

    def delete_dip_candidate_row(self, run_id: int, ticker: str, flagged_date: str):
        """See delete_candidate_row."""
        with self.conn() as c:
            c.execute(
                "DELETE FROM dip_candidate WHERE run_id=? AND ticker=? AND flagged_date=?",
                (run_id, ticker, flagged_date),
            )

    def get_dip_history(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM dip_candidate WHERE outcome IS NOT NULL OR decision IS NOT NULL "
                "ORDER BY flagged_date DESC, ticker LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 3-Day Rule candidates (see engine/strategies/three_day_rule.py) -
    # mirrors the dip_candidate methods above exactly; same natural key
    # shape (run_id, ticker, flagged_date), just a different table.
    # ------------------------------------------------------------------
    _THREE_DAY_CANDIDATE_COLUMNS = [
        "run_id", "ticker", "company_name", "price", "sector", "flagged_date", "price_captured_at",
        "last_earnings_date", "days_since_earnings", "window_start_date", "window_end_date",
        "anchor_close", "window_last_close", "drawdown_pct", "down_day_count", "surprise_pct",
        "rsi_14", "analyst_recommendation", "analyst_target_mean_price", "analyst_upside_pct",
        "pe_ratio", "week52_low", "week52_high", "debt_to_equity",
        "industry",
        "relative_strength_pct", "historical_median_depth_pct", "historical_median_duration_days",
        "dip_maturity_score", "dip_maturity_label",
        "next_earnings_date", "earnings_before_horizon",
        "composite_score", "risk_score", "risk_label", "rank",
    ]

    def save_three_day_candidates(self, run_id: int, candidates: Iterable[dict]):
        self._save_rows("three_day_candidate", self._THREE_DAY_CANDIDATE_COLUMNS, run_id, candidates, date_col="flagged_date")

    def get_latest_three_day_candidates(self, active_only: bool = False) -> list[dict]:
        """active_only=True returns every still-live candidate across all runs, deduped by ticker - see get_latest_candidates."""
        if not active_only:
            return self._latest_rows_for_run("three_day_candidate")
        return self._active_rows("three_day_candidate")

    def get_unevaluated_three_day_candidates(self, as_of: Optional[str] = None) -> list[dict]:
        """Flagged 3-day-rule candidates whose evaluation_horizon_days has passed and haven't been scored yet."""
        as_of = as_of or utcnow_iso()[:10]
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM three_day_candidate WHERE outcome IS NULL AND flagged_date <= ? ORDER BY flagged_date", (as_of,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_three_day_candidate_row(self, run_id: int, ticker: str, flagged_date: str) -> Optional[dict]:
        """One exact three_day_candidate row - see _get_row."""
        return self._get_row("three_day_candidate", run_id, ticker, "flagged_date=?", (flagged_date,))

    def record_three_day_outcome(
        self, run_id: int, ticker: str, flagged_date: str, outcome: str,
        resolved_price: Optional[float], return_pct: Optional[float] = None,
    ):
        with self.conn() as c:
            c.execute(
                "UPDATE three_day_candidate SET outcome=?, resolved_price=?, evaluated_at=?, return_pct=? "
                "WHERE run_id=? AND ticker=? AND flagged_date=?",
                (outcome, resolved_price, utcnow_iso(), return_pct, run_id, ticker, flagged_date),
            )

    def record_three_day_decision(self, run_id: int, ticker: str, flagged_date: str, decision: str):
        """decision: 'accepted' | 'ignored' - see record_candidate_decision."""
        with self.conn() as c:
            c.execute(
                "UPDATE three_day_candidate SET decision=?, decided_at=? WHERE run_id=? AND ticker=? AND flagged_date=?",
                (decision, utcnow_iso(), run_id, ticker, flagged_date),
            )

    def delete_three_day_candidate_row(self, run_id: int, ticker: str, flagged_date: str):
        """See delete_candidate_row."""
        with self.conn() as c:
            c.execute(
                "DELETE FROM three_day_candidate WHERE run_id=? AND ticker=? AND flagged_date=?",
                (run_id, ticker, flagged_date),
            )

    def get_three_day_history(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM three_day_candidate WHERE outcome IS NOT NULL OR decision IS NOT NULL "
                "ORDER BY flagged_date DESC, ticker LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def _classify_13f_action(delta: int, prior_shares: Optional[int], latest_shares: Optional[int]) -> str:
    if not prior_shares:
        return "new position"
    if not latest_shares:
        return "sold out"
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "unchanged"
