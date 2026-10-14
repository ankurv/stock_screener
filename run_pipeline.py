#!/usr/bin/env python3
"""
CLI entrypoint that refreshes everything (universe, prices, option chains,
earnings, analyst consensus, post-earnings-reaction, SEC 8-K/13F, news) and
then runs every enabled strategy (see engine/strategies/ and config.yaml ->
strategies_enabled) over that same shared, cached data - each strategy
writes its own ranked candidates into the SQLite cache for the dashboard
(app.py) to serve.

Usage:
    python run_pipeline.py                  # normal run, respects cache TTLs
    python run_pipeline.py --force-refresh    # ignore universe cache age
    python run_pipeline.py --tickers AAPL,MSFT  # quick test on a few names
    python run_pipeline.py -v                 # verbose logging

Intended to be run once each weekday near market open (see README for a
macOS launchd/cron example), but it's also just a normal script you can run
by hand any time - every step is TTL-cached so re-running it seconds later
does almost no network work.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from typing import Callable, Optional

from common import load_config, get_logger
from db.database import Database
from data_sources.universe import load_universe, get_eligible_universe
from data_sources.market_data import refresh_market_data, refresh_market_mood, refresh_sector_performance
from data_sources.sec_edgar import refresh_8k_for_universe, refresh_13f_for_all_investors
from data_sources.news_rss import refresh_news_for_universe, refresh_market_news
from engine.strategies import STRATEGIES

log = get_logger("run_pipeline")


def run(
    cfg: dict,
    db: Database,
    tickers_override: list[str] | None = None,
    force_refresh: bool = False,
    strategies_override: list[str] | None = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    on_progress, if given, is called with a short human-readable step name
    as each stage starts (e.g. "Refreshing option chains...") - app.py's
    background wrapper uses this to drive the dashboard's "Generate" button
    status text, so a multi-minute full-universe run doesn't look hung.
    """
    def report(step: str):
        log.info(step)
        if on_progress:
            on_progress(step)

    run_id = db.start_run()
    t0 = time.time()
    try:
        report("Loading universe...")
        universe = load_universe(db, cfg, force_refresh=force_refresh)
        if tickers_override:
            wanted = {t.upper() for t in tickers_override}
            universe = [r for r in universe if r["ticker"] in wanted]
        else:
            # Explicit --tickers bypasses the cap/volume pre-filter - you asked for those names.
            # Zero-network read of whatever the universe filter was last built as (see
            # data_sources.universe.build_universe_filter) - a normal run never pays that cost itself.
            universe = get_eligible_universe(db, cfg, universe)
        log.info("Universe: %d tickers", len(universe))

        # Resolved once, up front, so the shared refresh step below can skip
        # work (option chains, by far its slowest part) that nothing in this
        # run's scope will actually read - e.g. a dip-buy-only "Generate"
        # click doesn't need to pay for cash-secured-put's option chains.
        strategies_in_scope = [
            name for name in (strategies_override or cfg.get("strategies_enabled", list(STRATEGIES)))
            if name in STRATEGIES
        ]
        needs_chains = any(STRATEGIES[name].NEEDS_OPTION_CHAINS for name in strategies_in_scope)
        needs_events = any(getattr(STRATEGIES[name], "NEEDS_EVENTS_DATA", True) for name in strategies_in_scope)

        report(
            f"Refreshing prices & option chains for {len(universe)} tickers..." if needs_chains
            else f"Refreshing prices for {len(universe)} tickers..."
        )
        md_summary = refresh_market_data(db, cfg, universe, needs_chains=needs_chains, on_progress=report)
        report("Refreshing market mood...")
        market_mood = refresh_market_mood(db, cfg)
        report("Refreshing market news...")
        refresh_market_news(db, cfg)
        report("Refreshing sector performance...")
        sector_performance = refresh_sector_performance(db, cfg)
        if needs_events:
            report("Refreshing SEC 8-K filings...")
            k8_summary = refresh_8k_for_universe(db, cfg, universe, on_progress=report)
            report("Refreshing SEC 13F filings...")
            f13_summary = refresh_13f_for_all_investors(db, cfg, universe)
        else:
            k8_summary = {"skipped": True}
            f13_summary = {"skipped": True}
        report("Refreshing news...")
        news_summary = refresh_news_for_universe(db, cfg, universe, on_progress=report)

        # Everything above is shared, strategy-agnostic data; from here each
        # enabled strategy (see engine/strategies/) scans that same cached
        # data independently and writes its own ranked candidates.
        # strategies_override narrows this to one (or a few) strategies - the
        # dashboard's per-strategy "Generate" buttons use it rather than
        # re-running every strategy just because one was clicked; the
        # scheduler/CLI default (None) still means "all enabled strategies".
        candidates_by_strategy = {}
        for name in strategies_in_scope:
            strategy = STRATEGIES[name]
            report(f"Scoring {getattr(strategy, 'DISPLAY_NAME', name)}...")
            candidates = strategy.generate(db, cfg, universe)
            strategy.save(db, run_id, candidates)
            candidates_by_strategy[name] = len(candidates)

        report("Done")

        total_candidates = sum(candidates_by_strategy.values())
        db.finish_run(run_id, "ok", tickers_scanned=len(universe), candidates_found=total_candidates)

        elapsed = time.time() - t0
        summary = {
            "run_id": run_id,
            "elapsed_sec": round(elapsed, 1),
            "universe_size": len(universe),
            "candidates_by_strategy": candidates_by_strategy,
            "candidates_found": total_candidates,
            "market_mood": market_mood,
            "sector_performance": sector_performance,
            "market_data": md_summary,
            "sec_8k": k8_summary,
            "sec_13f": f13_summary,
            "news": news_summary,
        }
        log.info("Pipeline run #%d complete in %.1fs: %s", run_id, elapsed, candidates_by_strategy)
        return summary
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", error_message=str(exc))
        log.exception("Pipeline run #%d failed", run_id)
        raise


def main():
    parser = argparse.ArgumentParser(description="Refresh the cash-secured put screener's data and rankings.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore the universe cache age and re-pull it.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker list to limit this run to (for quick testing).")
    parser.add_argument("--strategy", type=str, default=None, help="Comma-separated strategy key(s) to run (default: all of strategies_enabled).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for name in ("universe", "market_data", "sec_edgar", "news_rss", "run_pipeline", "common"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    cfg = load_config()
    db = Database()
    tickers_override = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    strategies_override = [s.strip() for s in args.strategy.split(",")] if args.strategy else None

    try:
        summary = run(
            cfg, db, tickers_override=tickers_override, force_refresh=args.force_refresh,
            strategies_override=strategies_override,
        )
    except Exception:  # noqa: BLE001
        sys.exit(1)

    print("\n=== Pipeline run summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
