"""
Covered call strategy. Mirror of cash_secured_put.py for the opposite
direction - selling calls against 100 shares you already hold (or are
willing to acquire), capping upside instead of protecting downside. Shares
the same signal/risk-score machinery (engine.scorer._risk_score,
engine.signals.*) via engine.scorer.evaluate_call_candidates, the
call-direction sibling of evaluate_candidates - the shared ranking helpers
(select_top_candidates, rank_top_candidates, diversify_by_sector) are
direction-agnostic and reused as-is.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from common import effective_ttl_minutes
from data_sources.market_data import fetch_close_price_on_or_after, fetch_recent_trend_pct, drawdown_episode_stats
from engine.scorer import evaluate_call_candidates, select_top_candidates

KEY = "covered_call"
DISPLAY_NAME = "Covered Calls"
NEEDS_OPTION_CHAINS = True
NEEDS_EVENTS_DATA = True


def _flat_cfg(cfg: dict) -> dict:
    """See cash_secured_put._flat_cfg - identical pattern, own config section."""
    flat = dict(cfg.get("strategies", {}).get("covered_call", {}))
    flat["risk_free_rate"] = cfg.get("risk_free_rate", 0.045)
    flat["risk_profile"] = cfg.get("risk_profile", "balanced")
    return flat


def generate(db, cfg: dict, universe: list[dict]) -> list[dict]:
    strat_cfg = _flat_cfg(cfg)
    market_mood = db.get_market_mood() or {}
    cooldown_days = cfg.get("decision_cooldown_days", 7)
    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    snoozed = db.get_recent_decided_tickers("call_candidate", cooldown_cutoff)
    relative_strength_lookback_days = strat_cfg.get("relative_strength_lookback_days", 10)
    historical_drawdown_min_depth_pct = strat_cfg.get("historical_drawdown_min_depth_pct", -15.0)
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    spy_trend_pct = fetch_recent_trend_pct(db, "SPY", relative_strength_lookback_days, ttl_minutes=history_ttl)

    all_candidates = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in snoozed:
            continue
        price = db.get_price(ticker)
        if not price:
            continue
        call_rows = db.get_call_snapshots(ticker)
        if not call_rows:
            continue
        earnings = db.get_earnings(ticker)
        earnings_date = earnings["earnings_date"] if earnings else None
        analyst_snapshot = db.get_analyst_snapshot(ticker)
        earnings_reaction = db.get_post_earnings_reaction(ticker)
        technicals = db.get_technicals(ticker)

        stock_trend_pct = fetch_recent_trend_pct(db, ticker, relative_strength_lookback_days, ttl_minutes=history_ttl)
        relative_strength_pct = (stock_trend_pct - spy_trend_pct) if (stock_trend_pct is not None and spy_trend_pct is not None) else None
        historical_drawdown_stats = drawdown_episode_stats(db.get_price_history(ticker), historical_drawdown_min_depth_pct)

        cands = evaluate_call_candidates(
            ticker=ticker,
            company_name=row.get("company_name"),
            price=price,
            call_rows=call_rows,
            earnings_date=earnings_date,
            cfg=strat_cfg,
            analyst_snapshot=analyst_snapshot,
            earnings_reaction=earnings_reaction,
            market_mood=market_mood,
            technicals=technicals,
            sector=row.get("sector"),
            relative_strength_pct=relative_strength_pct,
            historical_drawdown_stats=historical_drawdown_stats,
        )
        price_captured_at = db.get_price_captured_at(ticker)
        for cand in cands:
            cand["price_captured_at"] = price_captured_at
        all_candidates.extend(cands)

    return select_top_candidates(all_candidates, strat_cfg)


def save(db, run_id: int, candidates: list[dict]):
    db.save_call_candidates(run_id, candidates)


def get_latest(db) -> list[dict]:
    return db.get_latest_call_candidates()


def get_history(db, limit: int = 200) -> list[dict]:
    return db.get_call_history(limit=limit)


def due_for_evaluation(db, cfg: dict) -> list[dict]:
    return db.get_unevaluated_call_candidates()


def classify_outcome(strike: float, close_price: float) -> str:
    """
    'won'      - expired OTM (close < strike): never called away, keep the
                 shares and the full premium.
    'assigned' - expired at/above strike: shares called away at the strike.
                 Since this screener only sells calls OTM (strike above the
                 entry price at the time it was sold), being assigned here
                 can't realize a loss vs. that entry price the way a put's
                 assignment can - it only caps how much further upside you
                 capture, so there's no 'assigned_loss' state to mirror
                 cash_secured_put's.
    """
    return "won" if close_price < strike else "assigned"


def resolve_outcome(db, rec: dict, cfg: dict) -> Optional[tuple[str, float]]:
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    close_price = fetch_close_price_on_or_after(db, rec["ticker"], rec["expiry"], ttl_minutes=history_ttl)
    if close_price is None:
        return None
    return classify_outcome(rec["strike"], close_price), close_price


def realized_return_pct(strike: float, entry_price: float, premium_per_share: float, close_price: float, shares_value: float) -> Optional[float]:
    """
    Locked in once resolved, never recomputed. 'won' (close < strike) ->
    just the premium, same as cash_secured_put - you still hold the shares,
    their own unrealized move isn't counted here. 'assigned' (close >=
    strike) -> premium plus the realized gain on the shares themselves,
    capped at the strike (the capital gain you actually got to keep before
    being called away), relative to your entry price.
    """
    if not shares_value:
        return None
    premium_total = premium_per_share * 100.0
    if close_price < strike:
        return premium_total / shares_value * 100.0
    share_gain = (strike - entry_price) * 100.0
    return (premium_total + share_gain) / shares_value * 100.0


def record_outcome(db, rec: dict, outcome: str, resolved_price: float):
    return_pct = realized_return_pct(rec["strike"], rec["price"], rec["bid"], resolved_price, rec["shares_value"])
    db.record_call_outcome(
        rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], outcome, resolved_price, return_pct
    )


def record_decision(db, rec: dict, decision: str):
    db.record_call_decision(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], decision)


def live_return_pct(rec: dict, current_price: float) -> Optional[float]:
    """Unrealized mark-to-market read for a still-pending row - see realized_return_pct."""
    return realized_return_pct(rec["strike"], rec["price"], rec["bid"], current_price, rec["shares_value"])


def get_row(db, rec: dict) -> Optional[dict]:
    return db.get_call_candidate_row(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"])


def delete_history_row(db, rec: dict):
    db.delete_call_candidate_row(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"])
