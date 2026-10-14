"""
Cash-secured put strategy. Thin wrapper around engine.scorer (the
pre-existing pure scoring module) - generate() is the per-ticker loop that
used to live directly in run_pipeline.run() before the pipeline became
strategy-agnostic (see engine/strategies/__init__.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from common import effective_ttl_minutes
from data_sources.market_data import fetch_close_price_on_or_after
from engine.scorer import evaluate_candidates, select_top_candidates

KEY = "cash_secured_put"
DISPLAY_NAME = "Cash-Secured Puts"
# Tells run_pipeline.run() whether the shared data-refresh step needs to
# bother fetching option chains (by far its slowest part) for a run scoped
# to this strategy - see data_sources.market_data.refresh_market_data.
NEEDS_OPTION_CHAINS = True
# SEC 8-K/13F aren't read by generate() below either - they only feed
# app.py's "Events & filings" panel shown on this strategy's candidate
# cards, so a run scoped to this strategy still needs them for that display.
NEEDS_EVENTS_DATA = True


def _flat_cfg(cfg: dict) -> dict:
    """
    engine.scorer's functions predate the multi-strategy config nesting and
    just expect one flat dict of CSP settings - this builds that from
    cfg["strategies"]["cash_secured_put"] plus the global risk_free_rate
    (a market constant, not CSP-specific, so it stays top-level in config.yaml).
    """
    flat = dict(cfg.get("strategies", {}).get("cash_secured_put", {}))
    flat["risk_free_rate"] = cfg.get("risk_free_rate", 0.045)
    return flat


def generate(db, cfg: dict, universe: list[dict]) -> list[dict]:
    strat_cfg = _flat_cfg(cfg)
    market_mood = db.get_market_mood() or {}
    cooldown_days = cfg.get("decision_cooldown_days", 7)
    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    # Don't re-surface a ticker you already accepted or ignored recently -
    # you made a call on it, no need to see it again right away.
    snoozed = db.get_recent_decided_tickers("put_candidate", cooldown_cutoff)

    all_candidates = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in snoozed:
            continue
        price = db.get_price(ticker)
        if not price:
            continue
        put_rows = db.get_put_snapshots(ticker)
        if not put_rows:
            continue
        earnings = db.get_earnings(ticker)
        earnings_date = earnings["earnings_date"] if earnings else None
        analyst_snapshot = db.get_analyst_snapshot(ticker)
        earnings_reaction = db.get_post_earnings_reaction(ticker)
        technicals = db.get_technicals(ticker)

        cands = evaluate_candidates(
            ticker=ticker,
            company_name=row.get("company_name"),
            price=price,
            put_rows=put_rows,
            earnings_date=earnings_date,
            cfg=strat_cfg,
            analyst_snapshot=analyst_snapshot,
            earnings_reaction=earnings_reaction,
            market_mood=market_mood,
            technicals=technicals,
            sector=row.get("sector"),
        )
        all_candidates.extend(cands)

    return select_top_candidates(all_candidates, strat_cfg)


def save(db, run_id: int, candidates: list[dict]):
    db.save_candidates(run_id, candidates)


def get_latest(db) -> list[dict]:
    return db.get_latest_candidates()


def get_history(db, limit: int = 200) -> list[dict]:
    return db.get_recommendation_history(limit=limit)


def due_for_evaluation(db, cfg: dict) -> list[dict]:
    return db.get_unevaluated_recommendations()


def classify_outcome(strike: float, premium_per_share: float, close_price: float) -> str:
    """
    'won'                 - expired at/above strike: never assigned, premium kept in full.
    'assigned_profitable' - expired below strike (assigned), but close price was still
                             above breakeven (strike - premium), so the trade nets positive.
    'assigned_loss'       - expired below breakeven: assigned, and worse off than if you'd
                             just held cash.
    """
    if close_price >= strike:
        return "won"
    breakeven = strike - premium_per_share
    return "assigned_profitable" if close_price >= breakeven else "assigned_loss"


def resolve_outcome(db, rec: dict, cfg: dict) -> Optional[tuple[str, float]]:
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    close_price = fetch_close_price_on_or_after(db, rec["ticker"], rec["expiry"], ttl_minutes=history_ttl)
    if close_price is None:
        return None  # no trading data yet for that date - try again next startup
    return classify_outcome(rec["strike"], rec["bid"], close_price), close_price


def realized_return_pct(strike: float, premium_per_share: float, close_price: float, collateral: float) -> Optional[float]:
    """
    Locked in once resolved, never recomputed: premium collected, net of
    being assigned below strike if that happened, as a % of the collateral
    you put up. close_price >= strike (never assigned) -> just the premium;
    below strike -> premium minus the mark-to-market loss on the 100 shares,
    which can still net positive ('assigned_profitable') or not
    ('assigned_loss') - this single number captures either case correctly.
    """
    if not collateral:
        return None
    premium_total = premium_per_share * 100.0
    mark_to_market = min(0.0, close_price - strike) * 100.0
    return (premium_total + mark_to_market) / collateral * 100.0


def record_outcome(db, rec: dict, outcome: str, resolved_price: float):
    return_pct = realized_return_pct(rec["strike"], rec["bid"], resolved_price, rec["collateral"])
    db.record_recommendation_outcome(
        rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], outcome, resolved_price, return_pct
    )


def record_decision(db, rec: dict, decision: str):
    db.record_candidate_decision(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], decision)


def delete_history_row(db, rec: dict):
    """'Stop tracking' - permanently removes this row, not an outcome of any kind."""
    db.delete_candidate_row(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"])
