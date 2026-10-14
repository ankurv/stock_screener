# Stock Screener

A local tool that scans a stock/ETF universe each day and surfaces the
events (earnings, analyst sentiment, post-earnings anomalies, SEC 8-K
filings, 13F "smart money" moves, M&A/split/legal news) that might affect
each name - then runs one or more independent **strategies** over that
same shared, cached data, each with its own ranked candidates on the
dashboard. Everything runs on your own machine against free, public data
sources, with a local SQLite cache so restarts are instant and you're not
re-fetching the world every time you open the dashboard.

## Strategies

The pipeline (`run_pipeline.py`) is a shared data layer feeding independent
strategy modules under `engine/strategies/` (see that package's
`__init__.py` -> `STRATEGIES`). Enable/disable strategies and tune each
one's settings via `config.yaml` -> `strategies_enabled` / `strategies`.
Adding a new strategy means dropping a module there with a small, fixed
interface (`generate`, `save`, `get_latest`, `get_history`,
`due_for_evaluation`, `resolve_outcome`, `record_outcome`) and a key in
config - nothing in `run_pipeline.py`, `app.py`, or `engine/outcomes.py`
needs to change.

- **Cash-secured puts** (`cash_secured_put`) - the original strategy; see
  "How the ranking works" below.
- **Dip buy watch** (`dip_buy`) - an "overnight sweep" that flags stocks
  which just saw a sharp correction but still look buyable, not a falling
  knife. A candidate must clear all four: a drawdown over the last
  `drawdown_lookback_days` beyond `drawdown_threshold_pct`, an oversold
  RSI(14) at/below `rsi_oversold_max`, analyst support (price target/rating
  still implies real upside - `min_analyst_score`), and no 'legal'-category
  news (lawsuit, SEC investigation, etc.) in the last
  `legal_news_lookback_days`. It's a ranked watchlist, not a sized
  portfolio - no capital allocation, just the top `max_candidates` by
  composite score. Outcomes (`rebounded` / `flat` / `further_decline`) are
  scored `evaluation_horizon_days` after being flagged, same pattern as the
  cash-secured-put track record.

**This is a screening tool, not investment advice.** It surfaces candidates
that match a configurable rule set so you can do your own diligence - it
does not know your full portfolio, risk tolerance, or tax situation, and
free data sources (described below) have real limitations. Read the
"Data sources & honest limitations" section before trusting any number here.

## Quick start

```bash
cd stock-put-screener
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Open `config.yaml` and replace `sec_user_agent`'s placeholder email with
   your real contact email. SEC EDGAR requires a descriptive User-Agent and
   will reject requests from an obviously-fake one.
2. Run the pipeline once by hand to populate the cache and sanity-check your
   network/SEC access:
   ```bash
   python run_pipeline.py --tickers AAPL,MSFT,QQQ -v
   ```
   You should see a summary with `candidates_found` and a short ranked list.
   If this fails, see Troubleshooting below before doing a full run.
3. Run a full universe pass:
   ```bash
   python run_pipeline.py
   ```
4. Start the dashboard:
   ```bash
   uvicorn app:app --reload --port 8787
   ```
   Open http://127.0.0.1:8787 in your browser. It opens straight to a live
   market pulse (a handful of ETF quotes, auto-refreshed every 30s - see
   below) with no recommendations yet; each strategy section has its own
   "Generate" button, since each run hits options chains and SEC filings
   for the whole universe and isn't something that should happen silently
   on load.

## How the ranking works

For every ticker, every put strike expiring within "this week" (today
through the upcoming Friday) is pulled, filtered, and scored. This is a
**ranked watchlist, not a sized portfolio** - there's no capital allocation
here (see `max_candidates` below); size each position's collateral yourself.

**Hard filters** (a contract must pass all of these to be considered):
- Must have a real bid (no bid = no premium to collect), and that bid must clear `min_bid_price`
  (default $0.50) - below that, brokerage commissions eat most or all of the premium
- Annualized yield (`bid / strike`, annualized) must clear `min_annualized_yield_pct`
- Open interest must clear `min_open_interest`
- Bid/ask spread must be within `max_bid_ask_spread_pct` (default 20%) - this is a hard
  filter, not just a scoring penalty: anything wider is dropped entirely

**Composite score** (weights in `config.yaml` -> `weights`), blending:
- **Annualized yield** - premium collected, annualized against your collateral
- **Probability OTM** - a Black-Scholes estimate (from Yahoo's implied vol) of
  the contract expiring worthless, i.e. you keep the premium and avoid assignment
- **Liquidity** - rewards tight spreads and reasonable open interest
- **Analyst sentiment** - consensus rating + price-target upside (neutral score
  for names with no analyst coverage, e.g. ETFs, rather than penalizing them)
- **Earnings penalty** - subtracted if earnings fall before the contract expires
  (flagged either way; set `exclude_if_earnings_before_expiry: true` to hard-filter instead)
- **Anomaly penalty** - subtracted if the *last* earnings report was a
  "beat but the stock sold off anyway" anomaly (see below) - a loose proxy
  for "the market may know something the headline number doesn't"

**Risk score** (1 = least risky, 10 = most risky, shown per candidate,
separate from the composite rank): a heuristic read on how risky the
*underlying/contract* is to actually hold if assigned, dominated by two
roughly-equal factors:
- **implied volatility** - the market's own read on expected swings; a
  thinly-traded, volatile small-cap will run noticeably higher than a
  mega-cap like GOOG even at a similar annualized yield
- **how far the strike sits from the current price** - a strike right at
  the money is a real assignment risk even on an otherwise "safe" stock,
  while one sold comfortably below the market (>=15% OTM by default) adds
  nothing from this factor; this is on top of implied vol, not instead of it,
  since a calm stock sold too close to the money is still risky

Smaller contributions:
- having no analyst coverage, or earnings falling before expiry, or a wide bid/ask spread
- **technical indicators** - RSI(14), the SMA20/SMA50 trend (golden/death
  cross), and MACD(12,26,9), computed from ~6 months of daily closes.
  Overbought/oversold RSI, a death cross, or a bearish MACD each nudge risk
  up a little. The raw RSI/trend/MACD values are also shown per candidate.
- **market mood** (see below) - a broad selloff raises risk across every
  position, not just volatile ones

`Low` (1-3) / `Medium` (4-6) / `High` (7-10) bands.

**Market mood** (`market_mood` in `config.yaml`): a simple broad-market
regime read - bearish/neutral/bullish based on a benchmark's (default SPY)
% change over the last few trading days (default 5) - shown clearly at the
top of the dashboard along with the latest market-moving headlines (Fed
decisions, macro data, geopolitical events, via `market_news_query`) so you
have context for *why* the tape is behaving the way it is and what's coming
up that could change it. Also folds into every candidate's risk score above.
Both the mood and the news refresh automatically (see "Live market pulse"
below) - no need to generate recommendations just to see them.

### Live market pulse

Shown on the dashboard with no action needed, independent of the
recommendation pipeline: live quotes for a handful of benchmark ETFs
(`market_pulse.tickers` in `config.yaml` - default SPY/IWM/QQQ/USO/GLD/TLT,
covering broad market, small-caps, tech, oil, gold, and long bonds), polled
by the browser every `market_pulse.refresh_seconds` (default 30) via
`GET /api/market-pulse`. Each tick also opportunistically refreshes the
market mood and news above (both still respect their own `cache_ttl_minutes`,
so this doesn't actually hit the network every 30 seconds). This endpoint
only pulls lightweight quotes - it never touches options chains or SEC
filings, so it's safe to leave running continuously.

**Risk profile** (`risk_profile` in `config.yaml`, or the dashboard's
Settings panel): `conservative` / `balanced` / `aggressive`, capping which
candidates are even eligible to be picked at risk score <= 3 / 6 / 10
respectively. Change this to see different "best returns" results that stay
within your risk tolerance - a conservative profile only ever considers
lower-volatility names, even if a riskier one would have scored higher.

The screener then keeps only the **best contract per ticker** among those
eligible under the risk profile, and surfaces the top `max_candidates`
(`strategies.cash_secured_put.max_candidates`, default 10) by composite
score - one contract per ticker, so you get that many *different* names to
look at, not several strikes stacked on one stock. There's no capital
sizing baked in; `collateral` (`strike x 100`, i.e. for a single 1-lot
contract - never multiplied by any quantity) is still reported on every
candidate so you have the number, but you decide how many lots to actually
sell against your own available collateral. `max_candidates` and
`risk_profile` can also be edited from the dashboard's Settings panel, which writes straight back to
`config.yaml`.

### Track record

Every recommendation ever made is kept (the `put_candidate` table is never
pruned, so it doubles as a history). Each time the dashboard server
*starts*, it checks for any past recommendation whose expiry has now passed
and scores it against the underlying's actual close price on/after that
date:
- **won** - closed at or above the strike: never assigned, full premium kept
- **assigned, profitable** - closed below the strike (assigned), but still
  above breakeven (`strike - premium`), so the trade nets positive overall
- **assigned, loss** - closed below breakeven - worse off than just holding cash

This is a startup check rather than a scheduled job, since "periodically" in
practice just means "whenever the server comes back up," and a past
recommendation's outcome never changes once computed. The dashboard's
"Track record" panel shows the running win rate and a table of scored past
recommendations.

### Post-earnings anomaly detection

For the most recently *completed* earnings report, the pipeline compares
the EPS surprise direction (beat/miss, from Yahoo's data) to the stock's
actual price reaction in the days around that report. If the stock dropped
hard despite a beat, or rallied despite a miss, it's flagged as an anomaly
(`beat_but_sold_off` / `miss_but_rallied`) and shown on the dashboard. This
is a simple heuristic (closing prices a few days apart, a configurable %
threshold in `post_earnings_anomaly_threshold_pct`) - it's meant to catch
"guidance spooked the market" type situations, not to be a precise signal.

### M&A / buyout / stock-split news

These are **flagged, not scored**. Deal/split news genuinely cuts both ways
for a put seller (an acquisition target's stock often jumps, which is good
for you - but a collapsed deal, or being forced out of a position in an
all-cash buyout, is a real risk) - rather than force a confident numeric
adjustment onto something that ambiguous, the dashboard just surfaces the
headline so you can judge it yourself.

## Data sources & honest limitations

Everything here is free and requires no API key, which comes with real
trade-offs you should know about:

| Source | Used for | Limitation |
|---|---|---|
| Yahoo Finance (via `yfinance`) | price, option chains, IV, earnings dates, analyst consensus | Unofficial/scraped - Yahoo can change its backend and break `yfinance` without notice; implied vol on thin weekly contracts can be noisy |
| SEC EDGAR (`data.sec.gov`, official) | 8-K material filings, 13F holdings | 13F filings land up to 45 days after quarter-end (not timely), only report CUSIP/issuer name (not ticker - we fuzzy-match company names, which can miss edge cases like very recent renames), and the $-value unit changed from thousands to whole dollars across SEC's 2023-2024 rule change (treat `value_usd` as an approximate magnitude) |
| Wikipedia "List of S&P 500 companies" | default stock universe | Could lag an index change by a few days; falls back to cached data if the page layout changes |
| Google News RSS | legal/M&A/split/product signal | Keyword-matched headlines, not a structured events feed - can miss stories with unusual phrasing or occasionally mis-tag one |

This project was built and logic-tested with synthetic/mocked data (unit
tests for the date math, Black-Scholes math, 13F XML parsing, news
classification, and the full pipeline wiring all pass offline). The actual
live network calls to Yahoo/SEC/Google News have **not** been exercised
end-to-end against the real internet by the assistant that built this -
do the `--tickers AAPL,MSFT,QQQ -v` smoke test in Quick Start step 2 and
read its log output before trusting a full run.

## Automation ("run every day the market is open")

Two options, pick one (or both):

**A. Keep the dashboard running.** `config.yaml` -> `schedule` controls a
background job inside `app.py` (via APScheduler) that refreshes automatically
on weekdays at `hour:minute` in the given timezone. Off (`enabled: false`) by
default, since recommendations are meant to be generated on demand via the
dashboard button - turn it on if you'd rather it just happen every morning.
Only fires while `uvicorn app:app` is actually running.

**B. OS-level scheduler (recommended for "set and forget").** Use cron or
launchd to run `run_pipeline.py` directly - the dashboard just reads
whatever's in the cache whenever you open it, so it doesn't need to be
running all the time:

- **macOS (launchd):** see `scripts/com.user.putscreener.refresh.plist` -
  edit the `/REPLACE/ME/` paths, copy it to `~/Library/LaunchAgents/`, then
  `launchctl load ~/Library/LaunchAgents/com.user.putscreener.refresh.plist`.
- **cron (macOS/Linux):**
  ```
  35 9 * * 1-5 cd /path/to/stock-put-screener && .venv/bin/python run_pipeline.py >> logs/pipeline.log 2>&1
  ```

Neither option is market-holiday-aware - on a holiday the run will just
either skip tickers gracefully or produce slightly stale-looking data.

## Configuration

Everything tunable lives in `config.yaml` with inline comments: universe
(S&P 500 / custom watchlist / ETFs), per-strategy settings under
`strategies.cash_secured_put` / `strategies.dip_buy` (collateral cap,
yield/liquidity filters, scoring weights, drawdown/RSI/analyst thresholds),
the prominent-investor list for the 13F signal, news keyword lists, cache
TTLs per data source, and the refresh schedule. `risk_free_rate` and
`post_earnings_anomaly_threshold_pct` stay global - they're consumed by the
shared data-refresh step, not a specific strategy.

**Caching notes:**
- Price and options-chain TTLs stretch automatically to cover the gap until
  the next market open on evenings/weekends (see `common.effective_ttl_minutes`) -
  that data can't change while the market's shut, so there's no reason to
  re-fetch an identical snapshot every TTL window. An explicit "Generate
  recommendations" click still works any time, including after hours and on
  weekends to plan for the next session - it just reuses the cached closing
  data instead of re-hitting the network for a byte-identical copy.
- Building/rebuilding the universe filter (market cap + avg volume per
  ticker) fetches concurrently (`universe_filter.max_workers`, default 8)
  rather than one request at a time - the only piece of that feature that
  touches the network, and it's kept out of a normal pipeline run entirely
  (see "Live market pulse" and the Settings panel's "Rebuild universe" button).

## Project structure

```
stock-put-screener/
  config.yaml              All tunable settings
  common.py                 Shared config/HTTP/logging/name-matching helpers
  run_pipeline.py           CLI: refresh everything, rank, cache results
  app.py                     FastAPI dashboard (reads from cache only)
  db/
    schema.sql               SQLite schema
    database.py              Caching/storage layer
  data_sources/
    universe.py               S&P 500 / watchlist / ETF loader
    market_data.py            yfinance: price, options, earnings, analyst, anomaly
    sec_edgar.py               SEC EDGAR: 8-K filings, 13F holdings
    news_rss.py                Google News RSS, keyword-tagged
  engine/
    greeks.py                  Black-Scholes probability-OTM / delta
    signals.py                   Shared ticker-level risk/sentiment signals (used by every strategy)
    scorer.py                   Cash-secured-put filtering + composite scoring + ranking
    outcomes.py                  Generic past-recommendation scoring loop (per-strategy due/resolve)
    strategies/
      __init__.py                 STRATEGIES registry
      cash_secured_put.py         Cash-secured put strategy
      dip_buy.py                   Dip-buy ("overnight sweep") strategy
  web/
    templates/dashboard.html    Dashboard page
    static/style.css             Dashboard styling
  scripts/
    com.user.putscreener.refresh.plist   macOS launchd template
  data/                       SQLite DB lives here (gitignored, created on first run)
```

## Troubleshooting

- **SEC requests failing / 403:** make sure you edited `sec_user_agent` in
  `config.yaml` to a real contact email - SEC blocks generic/placeholder agents.
- **yfinance errors or empty option chains:** Yahoo occasionally rate-limits;
  rerun later, or increase `sleep_between` in `refresh_market_data` calls
  (edit `run_pipeline.py` if you want this exposed in config). Some small-cap
  or newly-listed tickers simply don't have weekly options.
- **No candidates found:** your filters may be too strict for current market
  conditions (e.g. `min_annualized_yield_pct` too high, or it's a low-IV
  week). Loosen `config.yaml` and rerun with `--force-refresh`.
- **13F holdings_matched stays low:** this is expected - CUSIP-to-ticker
  matching is inherently best-effort on free data; check `thirteenf_holding`
  rows in the DB directly if you want to debug a specific name_of_issuer match.
