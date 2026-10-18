-- ---------------------------------------------------------------------------
-- Cash-Secured Put Screener - SQLite schema
-- Pure cache/store. Nothing here is "the truth" - it's a local mirror of
-- whatever free public source we last pulled, with a timestamp so the app
-- can decide when data is stale and needs a refresh.
-- ---------------------------------------------------------------------------

PRAGMA journal_mode = WAL;

-- Generic key/value fetch-cache, used by cache_get/cache_set in database.py
-- for anything that doesn't deserve its own table (e.g. raw API responses).
CREATE TABLE IF NOT EXISTS fetch_cache (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,       -- JSON blob
    fetched_at  TEXT NOT NULL        -- ISO8601 UTC timestamp
);

-- The stock universe we screen (S&P 500 or custom watchlist)
CREATE TABLE IF NOT EXISTS universe (
    ticker       TEXT PRIMARY KEY,
    company_name TEXT,
    cik          TEXT,               -- SEC CIK, zero-padded 10 digits, for EDGAR lookups
    sector       TEXT,
    updated_at   TEXT NOT NULL
);

-- Market cap / average share volume per ticker, used only to pre-filter the
-- universe above down to liquid large caps before a pipeline run - refreshed
-- on its own (much longer) cadence than the universe listing itself, see
-- config.yaml -> universe_filter and cache_ttl_minutes.universe_filter.
CREATE TABLE IF NOT EXISTS universe_filter_cache (
    ticker       TEXT PRIMARY KEY,
    market_cap   REAL,
    avg_volume   REAL,
    updated_at   TEXT NOT NULL
);

-- Latest known price / basic quote snapshot per ticker
CREATE TABLE IF NOT EXISTS price_snapshot (
    ticker       TEXT PRIMARY KEY,
    last_price   REAL,
    updated_at   TEXT NOT NULL
);

-- Shared local cache of daily closes per ticker - the common "basic data"
-- every strategy and signal (technicals, drawdown/trend, post-earnings
-- windows, outcome resolution, market mood, sector performance) reads from
-- instead of each independently re-pulling overlapping history from Yahoo.
-- Populated incrementally (see data_sources.market_data.refresh_price_history) -
-- a cold ticker gets one full HISTORY_WINDOW_DAYS pull, every refresh after
-- that only fetches and appends the days since the last cached date.
CREATE TABLE IF NOT EXISTS price_history (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,   -- ISO trading date
    close  REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- Earnings dates per ticker (next known earnings date; yfinance estimate)
CREATE TABLE IF NOT EXISTS earnings_calendar (
    ticker        TEXT PRIMARY KEY,
    earnings_date TEXT,              -- ISO date, nullable if unknown
    is_estimate   INTEGER DEFAULT 1, -- 1 = estimated date, 0 = confirmed
    updated_at    TEXT NOT NULL
);

-- Reaction to the most recently *completed* earnings report: did the stock
-- move in the direction you'd expect given the EPS beat/miss? A stock that
-- dropped hard despite a beat (or rallied despite a miss) is a flag worth
-- seeing before you sell a put into the next report.
CREATE TABLE IF NOT EXISTS post_earnings_reaction (
    ticker         TEXT PRIMARY KEY,
    earnings_date  TEXT,             -- ISO date of the last completed report
    surprise_pct   REAL,             -- EPS surprise %, positive = beat
    reaction_pct   REAL,             -- price change around that report, %
    anomaly_type   TEXT,             -- 'beat_but_sold_off' | 'miss_but_rallied' | 'normal' | NULL (insufficient data)
    updated_at     TEXT NOT NULL
);

-- Raw option contract snapshots (puts only - this tool is CSP-focused)
-- One row per (ticker, expiry, strike) as of the last refresh.
CREATE TABLE IF NOT EXISTS put_option_snapshot (
    ticker          TEXT NOT NULL,
    expiry          TEXT NOT NULL,     -- ISO date
    strike          REAL NOT NULL,
    bid             REAL,
    ask             REAL,
    last_price      REAL,
    implied_vol     REAL,
    open_interest    INTEGER,
    volume          INTEGER,
    in_the_money     INTEGER,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (ticker, expiry, strike)
);

-- Analyst consensus snapshot (recommendation + price target) per ticker
CREATE TABLE IF NOT EXISTS analyst_snapshot (
    ticker              TEXT PRIMARY KEY,
    recommendation_key  TEXT,       -- e.g. 'buy', 'hold', 'strong_buy', 'sell'
    recommendation_mean REAL,       -- 1.0 (Strong Buy) .. 5.0 (Sell)
    num_analysts        INTEGER,
    target_mean_price   REAL,
    target_high_price   REAL,
    target_low_price    REAL,
    pe_ratio            REAL,       -- trailing P/E
    week52_low          REAL,
    week52_high         REAL,
    debt_to_equity      REAL,       -- total debt / shareholder equity, as a % (yfinance's raw convention - 150 = 150%, i.e. 1.5x)
    industry            TEXT,       -- yfinance's finer-grained GICS sub-industry (e.g. "Semiconductors"), distinct from
                                     -- the broad `universe.sector` (e.g. "Information Technology") - used to diversify
                                     -- recommendations within a sector that actually spans very different businesses
    updated_at          TEXT NOT NULL
);

-- SEC 8-K ("material event") filings per ticker
CREATE TABLE IF NOT EXISTS sec_8k_filing (
    ticker        TEXT NOT NULL,
    accession_no  TEXT NOT NULL,
    filed_date    TEXT NOT NULL,
    items         TEXT,              -- comma-separated 8-K item codes, e.g. "1.01,2.02"
    primary_doc   TEXT,
    filing_url    TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, accession_no)
);

-- Raw 13F holdings lines pulled from each tracked prominent investor's
-- latest filings (we keep 2 periods so we can diff quarter over quarter).
CREATE TABLE IF NOT EXISTS thirteenf_holding (
    investor_cik    TEXT NOT NULL,
    investor_name   TEXT NOT NULL,
    period_of_report TEXT NOT NULL,   -- quarter end date, ISO
    name_of_issuer  TEXT NOT NULL,
    cusip           TEXT NOT NULL,
    matched_ticker  TEXT,             -- best-effort match into `universe`, nullable
    shares          INTEGER,
    value_usd       INTEGER,          -- reported in $ (SEC reports in thousands; we store actual $)
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (investor_cik, period_of_report, cusip)
);

-- News / RSS items, keyword-tagged
CREATE TABLE IF NOT EXISTS news_item (
    ticker       TEXT NOT NULL,
    link         TEXT NOT NULL,
    title        TEXT,
    source       TEXT,
    published_at TEXT,
    category     TEXT,              -- 'legal' | 'product' | 'other'
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (ticker, link)
);

-- One row per pipeline run, for the dashboard's "last updated" + history
CREATE TABLE IF NOT EXISTS pipeline_run (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT,              -- 'running' | 'ok' | 'error'
    tickers_scanned INTEGER,
    candidates_found INTEGER,
    error_message TEXT
);

-- Final ranked candidates for the most recent run (what the dashboard reads)
CREATE TABLE IF NOT EXISTS put_candidate (
    run_id            INTEGER NOT NULL,
    ticker            TEXT NOT NULL,
    company_name      TEXT,
    price             REAL,              -- underlying price at the time of this recommendation
    sector            TEXT,              -- GICS sector (from the universe table), for the sector-performance tag
    first_seen_at     TEXT,              -- ISO date first flagged - the tracking "start date". Preserved across
                                          -- re-generations of the same still-pending ticker (see db.save_candidates),
                                          -- not reset to today every time a run re-surfaces it.
    expiry            TEXT NOT NULL,
    strike            REAL NOT NULL,
    bid               REAL,
    ask               REAL,
    mid_price         REAL,
    collateral        REAL,
    premium_total     REAL,
    annualized_yield_pct REAL,
    prob_otm_pct      REAL,
    delta             REAL,
    open_interest     INTEGER,
    spread_pct        REAL,
    days_to_expiry    INTEGER,
    earnings_before_expiry INTEGER,
    analyst_recommendation TEXT,
    analyst_target_mean_price REAL,
    pe_ratio          REAL,
    week52_low        REAL,
    week52_high       REAL,
    debt_to_equity    REAL,       -- total debt / shareholder equity, as a % (yfinance's raw convention)
    industry          TEXT,       -- GICS sub-industry (e.g. "Semiconductors"), finer than `sector` above - see analyst_snapshot.industry
    relative_strength_pct REAL,   -- this stock's trend minus the benchmark's, same window - negative means real stock-specific weakness, not just beta
    historical_median_depth_pct REAL,     -- median depth of this stock's own past qualifying corrections (see drawdown_episode_stats)
    historical_median_duration_days INTEGER, -- median full time-underwater (peak-to-recovery) of those same past corrections
    earnings_anomaly_type TEXT,
    composite_score   REAL,
    risk_score        REAL,
    risk_label        TEXT,
    rsi_14            REAL,
    trend             TEXT,
    macd_bullish      INTEGER,
    rank              INTEGER,
    -- Outcome tracking: every row here is a recommendation that was made at some
    -- point, so this table doubles as the recommendation history. Filled in by
    -- engine.outcomes once the contract's expiry has passed (checked on server
    -- startup) - NULL until then.
    outcome              TEXT,    -- NULL (pending) | 'won' | 'assigned_profitable' | 'assigned_loss'
    expiry_close_price   REAL,    -- underlying close price on/after expiry, once evaluated
    evaluated_at         TEXT,    -- when outcome was scored - the tracking "end date"
    return_pct           REAL,    -- realized return on collateral once resolved, locked in and never recomputed:
                                   -- (premium_total + min(0, close-strike)*100) / collateral * 100 - positive if the
                                   -- premium alone (or premium net of being assigned below strike) still nets you
                                   -- ahead, negative if the assignment loss outweighs the premium collected.
    -- Manual accept/ignore - independent of outcome above (that's the automatic
    -- win/loss scored at expiry; this is the user's own decision, made any time
    -- before then). Either value moves the row out of the "active" list into
    -- the track record immediately; outcome still gets filled in later either
    -- way, so an ignored pick's eventual result is tracked too.
    decision             TEXT,    -- NULL (pending) | 'accepted' | 'ignored'
    decided_at           TEXT,
    PRIMARY KEY (run_id, ticker, expiry, strike)
);

-- "Dip buy" strategy: stocks flagged for a sharp recent correction that still
-- look buyable (not a falling knife) - see engine/strategies/dip_buy.py. A
-- watchlist, not a sized portfolio, so no strike/collateral/premium columns -
-- those are cash-secured-put concepts. Keyed on flagged_date rather than an
-- option expiry since there's no contract here; outcome is scored a fixed
-- number of days later (cfg -> strategies.dip_buy.evaluation_horizon_days).
CREATE TABLE IF NOT EXISTS dip_candidate (
    run_id            INTEGER NOT NULL,
    ticker            TEXT NOT NULL,
    company_name      TEXT,
    price             REAL,              -- entry price: underlying price at the time flagged
    sector            TEXT,
    flagged_date      TEXT NOT NULL,     -- ISO date this candidate was surfaced
    drawdown_pct      REAL,              -- % change over drawdown_lookback_days (negative)
    drawdown_lookback_days INTEGER,
    rsi_14            REAL,
    analyst_recommendation TEXT,
    analyst_target_mean_price REAL,
    analyst_upside_pct REAL,
    pe_ratio          REAL,
    week52_low        REAL,
    week52_high       REAL,
    debt_to_equity    REAL,       -- total debt / shareholder equity, as a % (yfinance's raw convention)
    industry          TEXT,       -- GICS sub-industry (e.g. "Semiconductors"), finer than `sector` above - see analyst_snapshot.industry
    relative_strength_pct REAL,   -- this stock's trend minus the benchmark's, same window - negative means real stock-specific weakness, not just beta
    historical_median_depth_pct REAL,     -- median depth of this stock's own past qualifying corrections (see drawdown_episode_stats)
    historical_median_duration_days INTEGER, -- median full time-underwater (peak-to-recovery) of those same past corrections
    dip_maturity_score REAL,      -- 0-1: how deep/long the CURRENT dip already is vs. that same history (see engine.signals.dip_maturity_score)
    dip_maturity_label TEXT,      -- 'early' | 'typical' | 'mature' | 'unknown' - classification of dip_maturity_score
    earnings_date         TEXT,     -- next known earnings date, nullable
    earnings_before_horizon INTEGER, -- 1 if earnings_date falls within earnings_alert_days of flagged_date - always
                                      -- shown as a warning badge, not a hard exclusion (see config.yaml)
    composite_score   REAL,
    risk_score        REAL,
    risk_label        TEXT,
    rank              INTEGER,
    -- Outcome tracking, same pattern as put_candidate: every row here is a
    -- recommendation made at some point, filled in by engine.outcomes once
    -- evaluation_horizon_days has passed - NULL until then.
    outcome              TEXT,    -- NULL (pending) | 'rebounded' | 'flat' | 'further_decline'
    resolved_price        REAL,    -- underlying price at evaluation time, once evaluated
    evaluated_at         TEXT,    -- when outcome was scored - the tracking "end date" (flagged_date is the start date,
                                   -- preserved across re-generations of the same still-pending ticker - see db.save_dip_candidates)
    return_pct           REAL,    -- (resolved_price - price) / price * 100, locked in once resolved and never recomputed
    -- Manual accept/ignore - see put_candidate's identical columns above.
    decision             TEXT,    -- NULL (pending) | 'accepted' | 'ignored'
    decided_at           TEXT,
    PRIMARY KEY (run_id, ticker, flagged_date)
);

-- "3-Day Rule" strategy: stocks that corrected for ~3 trading days right
-- after reporting earnings and now look like an overreaction worth buying,
-- not a real decline - see engine/strategies/three_day_rule.py. Same
-- watchlist shape as dip_candidate; the trigger here is specifically the
-- post-earnings correction window rather than a generic rolling drawdown.
CREATE TABLE IF NOT EXISTS three_day_candidate (
    run_id            INTEGER NOT NULL,
    ticker            TEXT NOT NULL,
    company_name      TEXT,
    price             REAL,              -- entry price: underlying price at the time flagged (end of the correction window)
    sector            TEXT,
    flagged_date      TEXT NOT NULL,     -- ISO date this candidate was surfaced
    last_earnings_date    TEXT,          -- the past, already-reported quarter this correction is measured from
    days_since_earnings   INTEGER,
    window_start_date     TEXT,          -- first of the 3 correction-window trading days
    window_end_date       TEXT,          -- last of the 3 (closest to today)
    anchor_close          REAL,          -- close right before the correction window
    window_last_close     REAL,          -- close at the end of the correction window (== price, at flag time)
    drawdown_pct          REAL,          -- % change anchor_close -> window_last_close (negative)
    down_day_count        INTEGER,       -- how many of the 3 window days closed lower than the prior day (0-3)
    surprise_pct          REAL,          -- EPS surprise % for last_earnings_date - a real miss (very negative) is hard-excluded
    rsi_14            REAL,
    analyst_recommendation TEXT,
    analyst_target_mean_price REAL,
    analyst_upside_pct REAL,
    pe_ratio          REAL,
    week52_low        REAL,
    week52_high       REAL,
    debt_to_equity    REAL,       -- total debt / shareholder equity, as a % (yfinance's raw convention)
    industry          TEXT,       -- GICS sub-industry (e.g. "Semiconductors"), finer than `sector` above - see analyst_snapshot.industry
    relative_strength_pct REAL,   -- this stock's trend minus the benchmark's, same window - negative means real stock-specific weakness, not just beta
    historical_median_depth_pct REAL,     -- median depth of this stock's own past qualifying corrections (see drawdown_episode_stats)
    historical_median_duration_days INTEGER, -- median full time-underwater (peak-to-recovery) of those same past corrections
    dip_maturity_score REAL,      -- 0-1: how deep/long the CURRENT dip already is vs. that same history (see engine.signals.dip_maturity_score)
    dip_maturity_label TEXT,      -- 'early' | 'typical' | 'mature' | 'unknown' - classification of dip_maturity_score
    next_earnings_date      TEXT,     -- next (upcoming) known earnings date, nullable
    earnings_before_horizon INTEGER,  -- 1 if next_earnings_date falls within earnings_alert_days of flagged_date - always
                                       -- shown as a warning badge, not a hard exclusion (see config.yaml)
    composite_score   REAL,
    risk_score        REAL,
    risk_label        TEXT,
    rank              INTEGER,
    -- Outcome tracking, same pattern as dip_candidate.
    outcome              TEXT,    -- NULL (pending) | 'rebounded' | 'flat' | 'further_decline'
    resolved_price        REAL,    -- underlying price at evaluation time, once evaluated
    evaluated_at         TEXT,    -- when outcome was scored - the tracking "end date" (flagged_date is the start date,
                                   -- preserved across re-generations of the same still-pending ticker - see db._save_rows)
    return_pct           REAL,    -- (resolved_price - price) / price * 100, locked in once resolved and never recomputed
    -- Manual accept/ignore - see put_candidate's identical columns above.
    decision             TEXT,    -- NULL (pending) | 'accepted' | 'ignored'
    decided_at           TEXT,
    PRIMARY KEY (run_id, ticker, flagged_date)
);
