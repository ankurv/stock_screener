#!/usr/bin/env python3
"""
Local dashboard for the stock screener. Cash-secured puts and dip-buy are
two independent strategies (see engine/strategies/) scanning the same
cached data, each with its own candidates section below. Run with:

    uvicorn app:app --reload --port 8787

then open http://127.0.0.1:8787 in your browser. Recommendations are
generated on demand - each strategy section has its own "Generate" button
(runs run_pipeline.py in the background, scoped to just that strategy)
rather than something that happens automatically on load; the dashboard
only ever reads from the local SQLite cache otherwise (fast, works even if
you're offline). A lightweight "live market pulse" (a handful of ETF
quotes, refreshed every config.yaml -> market_pulse.refresh_seconds by the
browser) runs independently of that and needs no button - see
/api/market-pulse.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from common import load_config, update_config_values, get_logger, effective_ttl_minutes
from db.database import Database
from data_sources.universe import load_universe, build_universe_filter
from data_sources.market_data import (
    fetch_price,
    fetch_quote,
    refresh_market_mood,
    blend_live_mood,
    refresh_sparkline,
    refresh_sector_performance,
)
from data_sources.news_rss import refresh_market_news, MARKET_NEWS_KEY
from engine.scorer import RISK_PROFILE_MAX_SCORE
from engine.outcomes import evaluate_past_recommendations
from engine.strategies import STRATEGIES
import run_pipeline as pipeline

log = get_logger("app")

BASE_DIR = Path(__file__).parent
cfg = load_config()
db = Database()
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="Stock Screener")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

_refresh_lock = threading.Lock()
# Keyed per strategy, not one shared dict - two DIFFERENT strategies'
# "Generate" clicks should run truly concurrently (they write to separate
# tables and only incidentally share the TTL-cached data-refresh step), so
# only the SAME strategy being triggered twice should ever block. A single
# shared flag here was the bug: it serialized every click regardless of
# which strategy, so the other button always showed "waiting" even though
# nothing about it actually required that.
_refresh_states = {
    name: {"running": False, "step": None, "started_at": None, "last_error": None} for name in STRATEGIES
}
_universe_rebuild_lock = threading.Lock()
_universe_rebuild_state = {"running": False, "started_at": None, "finished_at": None, "eligible_count": None, "last_error": None}
# Last successful /api/market-pulse result, so a page reload shows the most
# recent live data immediately instead of flashing back to placeholders.
_pulse_cache: dict = {"pulse": [], "failed_tickers": [], "market_mood": {}, "market_news": [], "sector_performance": []}


def _try_claim_refresh_slot(strategy: Optional[str]) -> bool:
    """
    Atomically checks-and-sets _refresh_states[name]["running"] under the
    lock for every name this call touches - just [strategy] for an
    individual "Generate" click, or every strategy for the scheduler's
    full run (strategy=None) - so two near-simultaneous callers for the
    SAME strategy can't both slip through, while two DIFFERENT strategies
    claim independent slots and can run concurrently. The scheduler's "all
    strategies" claim locks every slot at once, so it can't start mid-way
    through an individual click, and an individual click can't start
    mid-way through the scheduler's run either. Returns True if this
    caller claimed every slot it needed (and must now actually run the
    pipeline), False if any of them was already held.
    """
    names = [strategy] if strategy else list(STRATEGIES)
    with _refresh_lock:
        if any(_refresh_states[name]["running"] for name in names):
            return False
        now = datetime.now(timezone.utc).isoformat()
        for name in names:
            _refresh_states[name].update({"running": True, "step": "Starting...", "started_at": now, "last_error": None})
        return True


def _run_pipeline_background(strategy: Optional[str] = None):
    """
    strategy=None (the scheduler/startup default) runs every strategy in
    strategies_enabled, sharing one data refresh. strategy="dip_buy" (etc,
    from a dashboard "Generate" button) only scores that one strategy, and
    the shared data-refresh step itself skips whatever that strategy
    doesn't need (option chains, 8-K/13F - see run_pipeline.run).

    Caller must have already claimed the slot(s) via _try_claim_refresh_slot -
    this function just runs the pipeline and releases them.
    """
    names = [strategy] if strategy else list(STRATEGIES)

    def on_progress(step: str):
        for name in names:
            _refresh_states[name]["step"] = step

    try:
        pipeline.run(cfg, db, strategies_override=[strategy] if strategy else None, on_progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("Background pipeline run failed")
        for name in names:
            _refresh_states[name]["last_error"] = str(exc)
    finally:
        for name in names:
            _refresh_states[name]["running"] = False


def _scheduled_refresh():
    """Entry point for the cron scheduler (see _maybe_start_scheduler) - claims every strategy's slot since it doesn't go through POST /refresh."""
    if _try_claim_refresh_slot(None):
        _run_pipeline_background(None)


def _build_events_for_ticker(ticker: str) -> dict:
    earnings = db.get_earnings(ticker)
    analyst = db.get_analyst_snapshot(ticker)
    reaction = db.get_post_earnings_reaction(ticker)
    thirteenf = db.get_13f_changes_for_ticker(ticker)
    filings_8k = db.get_8k_filings(ticker, limit=5)
    news = db.get_news(ticker, limit=8)
    return {
        "earnings": earnings,
        "analyst": analyst,
        "reaction": reaction,
        "thirteenf": thirteenf,
        "filings_8k": filings_8k,
        "news": news,
    }


def _history_summary(history: list[dict]) -> dict:
    total = len(history)
    won = sum(1 for h in history if h["outcome"] == "won")
    assigned_profitable = sum(1 for h in history if h["outcome"] == "assigned_profitable")
    assigned_loss = sum(1 for h in history if h["outcome"] == "assigned_loss")
    return {
        "total": total,
        "won": won,
        "assigned_profitable": assigned_profitable,
        "assigned_loss": assigned_loss,
        "win_rate_pct": round(won / total * 100.0, 1) if total else None,
    }


def _dip_history_summary(history: list[dict]) -> dict:
    """Generic over any strategy using the rebounded/flat/further_decline outcome scheme - also used by three_day_rule."""
    total = len(history)
    rebounded = sum(1 for h in history if h["outcome"] == "rebounded")
    flat = sum(1 for h in history if h["outcome"] == "flat")
    further_decline = sum(1 for h in history if h["outcome"] == "further_decline")
    return {
        "total": total,
        "rebounded": rebounded,
        "flat": flat,
        "further_decline": further_decline,
        "win_rate_pct": round(rebounded / total * 100.0, 1) if total else None,
    }


def _fetch_live_prices(tickers: set) -> dict:
    """
    Concurrent live fetch (ThreadPoolExecutor, same pattern as
    data_sources.market_data.refresh_market_data) for every ticker with a
    still-pending track-record row across all strategies. A real network
    hit, not the cached price_snapshot - that cache only updates when
    something else happens to touch that specific ticker (a pipeline run
    that includes it, a manual refresh-price click), so a position that's
    been pending for a few days with nothing else refreshing it would
    otherwise show a stale, misleadingly-flat "live" return. Bounded to
    just the handful of tickers that actually need one, not the whole
    universe, so this stays fast even though it's a real fetch.
    """
    prices = {}
    if not tickers:
        return prices
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_price, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                price = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("Live price fetch failed for %s: %s", ticker, exc)
                price = None
            if price:
                prices[ticker] = price
    return prices


def _attach_live_returns(strat, history: list[dict], live_prices: dict):
    """
    Fills in `live_return_pct` on every still-pending (outcome not yet set)
    track-record row, so the dashboard shows an unrealized mark-to-market
    estimate by default instead of a blank "n/a" until someone clicks the
    row's manual refresh (⟳) button. `live_prices` comes from
    _fetch_live_prices - a real quote, not the cached price_snapshot, so
    this reflects actual price movement since the row was flagged. Never
    touches the locked-in `return_pct` column - this is display-only.
    """
    for h in history:
        if h.get("outcome"):
            continue
        price = live_prices.get(h["ticker"])
        h["live_return_pct"] = round(strat.live_return_pct(h, price), 2) if price else None


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    # "Active" = from the latest run and not yet expired/resolved - so a
    # plain page reload (or revisiting the bare URL with no query string)
    # keeps showing what's still outstanding instead of needing a fresh
    # "Generate" click just to see it again. Once a candidate expires
    # (cash-secured puts) or gets resolved (dip-buy), it naturally drops out
    # of this view and shows up in its track record below instead.
    candidates = db.get_latest_candidates(active_only=True)
    dip_candidates = db.get_latest_dip_candidates(active_only=True)
    three_day_candidates = db.get_latest_three_day_candidates(active_only=True)
    last_run = db.get_latest_run()
    events_by_ticker = {c["ticker"]: _build_events_for_ticker(c["ticker"]) for c in candidates}
    all_active_tickers = {c["ticker"] for c in candidates + dip_candidates + three_day_candidates}
    # The 'stock_split' news category (data_sources.news_rss) was being
    # classified and stored but never actually shown anywhere - checked
    # here, at render time, rather than baked into each candidate row,
    # since it's "FYI as of right now" context, not a locked-in fact from
    # whenever the row was generated.
    recent_split_tickers = {
        t for t in all_active_tickers if any(n.get("category") == "stock_split" for n in db.get_news(t, limit=10))
    }
    history = db.get_recommendation_history(limit=50)
    dip_history = db.get_dip_history(limit=50)
    three_day_history = db.get_three_day_history(limit=50)
    pending_tickers = {
        h["ticker"] for h in (history + dip_history + three_day_history) if not h.get("outcome")
    }
    live_prices = _fetch_live_prices(pending_tickers)
    _attach_live_returns(STRATEGIES["cash_secured_put"], history, live_prices)
    _attach_live_returns(STRATEGIES["dip_buy"], dip_history, live_prices)
    _attach_live_returns(STRATEGIES["three_day_rule"], three_day_history, live_prices)
    market_mood = _pulse_cache.get("market_mood") or db.get_market_mood() or {}
    market_news = _pulse_cache.get("market_news") or db.get_news(MARKET_NEWS_KEY, limit=8) or []
    pulse_by_ticker = {q["ticker"]: q for q in _pulse_cache.get("pulse", [])}
    sector_performance = _pulse_cache.get("sector_performance") or db.get_sector_performance() or []
    sector_trend_by_name = {s["sector"]: s["trend_pct"] for s in sector_performance}
    csp_cfg = cfg.get("strategies", {}).get("cash_secured_put", {})
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "candidates": candidates,
            "dip_candidates": dip_candidates,
            "three_day_candidates": three_day_candidates,
            "events_by_ticker": events_by_ticker,
            "recent_split_tickers": recent_split_tickers,
            "last_run": last_run,
            "max_candidates": csp_cfg.get("max_candidates", 10),
            "risk_profile": cfg.get("risk_profile", "balanced"),
            "market_mood": market_mood,
            "market_news": market_news,
            "market_pulse_tickers": cfg.get("market_pulse", {}).get("tickers", []),
            "market_pulse_quotes": pulse_by_ticker,
            "market_pulse_refresh_seconds": cfg.get("market_pulse", {}).get("refresh_seconds", 30),
            "sector_performance": sector_performance,
            "sector_trend_by_name": sector_trend_by_name,
            "sector_lookback_days": cfg.get("sector_performance", {}).get("lookback_days", 15),
            "history": history,
            "history_summary": _history_summary(history),
            "dip_history": dip_history,
            "dip_history_summary": _dip_history_summary(dip_history),
            "three_day_history": three_day_history,
            "three_day_history_summary": _dip_history_summary(three_day_history),
            "refresh_states": _refresh_states,
            "universe_rebuild": _universe_rebuild_state,
        },
    )


@app.get("/api/market-pulse")
def market_pulse():
    """
    Lightweight, independent of the recommendation pipeline: live quotes for
    a handful of benchmark ETFs, plus an opportunistic refresh of market
    mood/news (both still respect their own cache_ttl_minutes, so polling
    this every 30s doesn't actually hit the network that often). The mood
    returned here is live-adjusted (see blend_live_mood) using these same
    quotes - no extra network cost - while the structural, SPY-trend-only
    mood used for risk scoring during a pipeline run is left untouched.
    """
    pulse_cfg = cfg.get("market_pulse", {})
    mood_cfg = cfg.get("market_mood", {})
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    quotes = []
    failed_tickers = []
    for ticker in pulse_cfg.get("tickers", []):
        try:
            quote = fetch_quote(db, ticker, ttl_minutes=history_ttl)
        except Exception as exc:  # noqa: BLE001
            log.warning("Live quote fetch failed for %s: %s", ticker, exc)
            quote = None
        if quote:
            try:
                quote["sparkline"] = refresh_sparkline(db, cfg, ticker)
            except Exception as exc:  # noqa: BLE001
                log.warning("Sparkline fetch failed for %s: %s", ticker, exc)
                quote["sparkline"] = db.get_sparkline(ticker) or []
            quotes.append(quote)
        else:
            failed_tickers.append(ticker)

    try:
        market_mood = refresh_market_mood(db, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Market mood refresh failed: %s", exc)
        market_mood = db.get_market_mood() or {}
    market_mood = blend_live_mood(
        market_mood,
        quotes,
        bearish_threshold_pct=mood_cfg.get("bearish_threshold_pct", -1.5),
        bullish_threshold_pct=mood_cfg.get("bullish_threshold_pct", 1.5),
    )

    try:
        market_news = refresh_market_news(db, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Market news refresh failed: %s", exc)
        market_news = db.get_news(MARKET_NEWS_KEY, limit=8) or []

    try:
        sector_performance = refresh_sector_performance(db, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Sector performance refresh failed: %s", exc)
        sector_performance = db.get_sector_performance() or []

    result = {
        "pulse": quotes,
        "failed_tickers": failed_tickers,
        "market_mood": market_mood,
        "market_news": market_news,
        "sector_performance": sector_performance,
    }
    _pulse_cache.update(result)
    return JSONResponse(result)


@app.post("/refresh")
def trigger_refresh(strategy: Optional[str] = Form(None)):
    if strategy and strategy not in STRATEGIES:
        return JSONResponse({"error": f"Unknown strategy: {strategy}"}, status_code=400)
    if not _try_claim_refresh_slot(strategy):
        return JSONResponse({"status": "already_running"})
    thread = threading.Thread(target=_run_pipeline_background, args=(strategy,), daemon=True)
    thread.start()
    return JSONResponse({"status": "started"})


@app.get("/api/status")
def status():
    last_run = db.get_latest_run()
    return JSONResponse({"refresh": _refresh_states, "last_run": last_run, "universe_rebuild": _universe_rebuild_state})


@app.post("/api/universe/rebuild")
def rebuild_universe():
    """
    Manually refreshes the liquid-large-cap universe filter (see
    data_sources.universe.build_universe_filter) - the "update as needed"
    counterpart to the automatic build on server startup. Normal pipeline
    runs never trigger this themselves.
    """
    if _universe_rebuild_state["running"]:
        return JSONResponse({"status": "already_running"})
    threading.Thread(target=_rebuild_universe_filter_background, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/api/candidates")
def api_candidates():
    return JSONResponse(db.get_latest_candidates(active_only=True))


@app.get("/api/dip-candidates")
def api_dip_candidates():
    return JSONResponse(db.get_latest_dip_candidates(active_only=True))


@app.get("/api/three-day-candidates")
def api_three_day_candidates():
    return JSONResponse(db.get_latest_three_day_candidates(active_only=True))


@app.post("/api/refresh-price")
def refresh_price(ticker: str = Form(...)):
    """
    The per-card refresh button - forces a live price fetch for this one
    ticker, bypassing cache_ttl_minutes.price, instead of waiting for it to
    naturally go stale. Only updates the shared live price_snapshot (see
    db.upsert_price) for display - never a candidate row's own `price`
    column, which is the entry price at flag time and must stay fixed for
    return_pct to be computed correctly once resolved.
    """
    try:
        price = fetch_price(ticker)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Couldn't refresh {ticker}: {exc}"}, status_code=502)
    if not price:
        return JSONResponse({"error": f"No live price available for {ticker} right now."}, status_code=502)
    db.upsert_price(ticker, price)
    return JSONResponse({"ticker": ticker, "price": round(price, 2)})


@app.post("/api/decide")
def decide_candidate(
    strategy: str = Form(...),
    decision: str = Form(...),
    run_id: int = Form(...),
    ticker: str = Form(...),
    expiry: Optional[str] = Form(None),
    strike: Optional[float] = Form(None),
    flagged_date: Optional[str] = Form(None),
):
    """
    Accept/ignore a candidate - moves it out of the active list (see
    db.get_latest_candidates/get_latest_dip_candidates's active_only query,
    which excludes anything with decision set) and into the track record
    immediately, regardless of whether its outcome has resolved yet. Also
    starts the decision_cooldown_days clock (see engine.strategies.*.generate),
    so this ticker won't be re-surfaced by "Generate" right away.
    """
    if decision not in ("accepted", "ignored"):
        return JSONResponse({"error": "decision must be 'accepted' or 'ignored'"}, status_code=400)
    strat = STRATEGIES.get(strategy)
    if strat is None:
        return JSONResponse({"error": f"Unknown strategy: {strategy}"}, status_code=400)
    rec = {"run_id": run_id, "ticker": ticker, "expiry": expiry, "strike": strike, "flagged_date": flagged_date}
    strat.record_decision(db, rec, decision)
    return JSONResponse({"status": "ok"})


@app.post("/api/history/delete")
def delete_history_entry(
    strategy: str = Form(...),
    run_id: int = Form(...),
    ticker: str = Form(...),
    expiry: Optional[str] = Form(None),
    strike: Optional[float] = Form(None),
    flagged_date: Optional[str] = Form(None),
):
    """'Stop tracking' a track-record row - permanently deletes it, not an outcome of any kind."""
    strat = STRATEGIES.get(strategy)
    if strat is None:
        return JSONResponse({"error": f"Unknown strategy: {strategy}"}, status_code=400)
    rec = {"run_id": run_id, "ticker": ticker, "expiry": expiry, "strike": strike, "flagged_date": flagged_date}
    strat.delete_history_row(db, rec)
    return JSONResponse({"status": "ok"})


@app.post("/api/history/refresh")
def refresh_history_outcome(
    strategy: str = Form(...),
    run_id: int = Form(...),
    ticker: str = Form(...),
    expiry: Optional[str] = Form(None),
    strike: Optional[float] = Form(None),
    flagged_date: Optional[str] = Form(None),
):
    """
    The track record's manual "check now" button on a still-pending
    (decided but unresolved) row - tries to resolve it on demand instead
    of waiting for the next server-startup sweep (see engine.outcomes.
    evaluate_past_recommendations), using the exact same per-strategy
    resolve_outcome/record_outcome. Returns status 'resolved' with the
    locked-in fresh values if it's officially due and the data exists for
    it; otherwise 'not_ready', but still with a live mark-to-market
    live_return_pct (see each strategy's live_return_pct) - the user wants
    a current return on click either way, not just once it's resolved.
    That preview is NEVER written to the return_pct column - only
    record_outcome ever locks that in, exactly once.
    """
    strat = STRATEGIES.get(strategy)
    if strat is None:
        return JSONResponse({"error": f"Unknown strategy: {strategy}"}, status_code=400)
    rec = {"run_id": run_id, "ticker": ticker, "expiry": expiry, "strike": strike, "flagged_date": flagged_date}
    full_rec = strat.get_row(db, rec)
    if full_rec is None:
        return JSONResponse({"error": "Row not found."}, status_code=404)
    if full_rec.get("outcome") is not None:
        return JSONResponse({"status": "already_resolved"})
    try:
        result = strat.resolve_outcome(db, full_rec, cfg)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Couldn't check outcome: {exc}"}, status_code=502)
    if result is not None:
        outcome, resolved_price = result
        strat.record_outcome(db, full_rec, outcome, resolved_price)
        updated = strat.get_row(db, rec)
        return JSONResponse({
            "status": "resolved",
            "outcome": updated["outcome"],
            "resolved_price": updated.get("resolved_price", updated.get("expiry_close_price")),
            "return_pct": updated.get("return_pct"),
            "evaluated_at": updated.get("evaluated_at"),
        })

    try:
        live_price = fetch_price(ticker)
    except Exception:  # noqa: BLE001
        live_price = None
    live_return = strat.live_return_pct(full_rec, live_price) if live_price else None
    return JSONResponse({
        "status": "not_ready",
        "live_price": round(live_price, 2) if live_price else None,
        "live_return_pct": round(live_return, 2) if live_return is not None else None,
    })


@app.post("/settings")
def update_settings(
    max_candidates: int = Form(...),
    risk_profile: str = Form(...),
):
    if max_candidates < 1:
        return JSONResponse({"error": "Max recommendations must be at least 1."}, status_code=400)
    if risk_profile not in RISK_PROFILE_MAX_SCORE:
        return JSONResponse(
            {"error": f"Risk profile must be one of: {', '.join(RISK_PROFILE_MAX_SCORE)}."}, status_code=400
        )
    global cfg
    updates = {
        # Dotted path: "max_candidates" exists under every strategy, so the
        # plain leaf name alone would be ambiguous (see common.
        # update_config_values) - applies to all of them, not just CSP.
        f"strategies.{name}.max_candidates": max_candidates for name in STRATEGIES
    }
    updates["risk_profile"] = f'"{risk_profile}"'
    cfg = update_config_values(updates)
    return JSONResponse({"status": "ok"})


def _maybe_start_scheduler():
    sched_cfg = cfg.get("schedule", {})
    if not sched_cfg.get("enabled"):
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(sched_cfg.get("timezone", "America/New_York"))
        trigger = CronTrigger(
            day_of_week=sched_cfg.get("days_of_week", "mon-fri"),
            hour=sched_cfg.get("hour", 9),
            minute=sched_cfg.get("minute", 35),
            timezone=tz,
        )
        scheduler = BackgroundScheduler()
        scheduler.add_job(_scheduled_refresh, trigger, id="daily_refresh", replace_existing=True)
        scheduler.start()
        log.info(
            "Background scheduler started: refresh at %02d:%02d %s on %s",
            sched_cfg.get("hour", 9), sched_cfg.get("minute", 35),
            sched_cfg.get("timezone"), sched_cfg.get("days_of_week"),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not start background scheduler (%s) - the dashboard will still work, "
                    "just use the Refresh button or run_pipeline.py via cron instead.", exc)


def _rebuild_universe_filter_background():
    with _universe_rebuild_lock:
        if _universe_rebuild_state["running"]:
            return
        _universe_rebuild_state["running"] = True
        _universe_rebuild_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _universe_rebuild_state["last_error"] = None
    try:
        universe = load_universe(db, cfg)
        eligible = build_universe_filter(db, cfg, universe)
        _universe_rebuild_state["eligible_count"] = len(eligible)
    except Exception as exc:  # noqa: BLE001
        log.exception("Universe pre-filter build failed")
        _universe_rebuild_state["last_error"] = str(exc)
    finally:
        _universe_rebuild_state["running"] = False
        _universe_rebuild_state["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.on_event("startup")
def on_startup():
    db.mark_interrupted_runs()
    _maybe_start_scheduler()
    # Score any past recommendations whose expiry has now passed, so the
    # track record stays current without needing a dedicated scheduled job -
    # "periodically" here just means "each time the server comes back up".
    threading.Thread(target=evaluate_past_recommendations, args=(db, cfg), daemon=True).start()
    # Build the liquid-large-cap universe filter now, in the background, so
    # it's already warm by the time someone clicks "Generate recommendations" -
    # this is the one place that pays the per-ticker market-cap/volume lookup
    # cost; run_pipeline.run() itself just reads whatever this already built
    # (see data_sources.universe.get_eligible_universe). Once built it's
    # reused indefinitely (cache_ttl_minutes.universe_filter, default weekly)
    # until this runs again or someone hits "Rebuild universe" in Settings.
    threading.Thread(target=_rebuild_universe_filter_background, daemon=True).start()
    # Recommendations themselves are generated on demand (the "Generate
    # recommendations" button / POST /refresh) rather than automatically -
    # the full pipeline hits options chains and SEC filings for the whole
    # universe, which isn't something that should kick off silently on load.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=False)
