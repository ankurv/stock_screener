"""
Dip-buy strategy ("overnight sweep"): flags stocks that just saw a sharp
correction but still look buyable rather than a falling knife. Combines
(see config.yaml -> strategies.dip_buy):
  - a sharp recent drawdown (the core "major correction" signal)
  - an oversold RSI(14) reading, confirming it's not just drifting down
  - analyst support (price target still implies real upside / non-bearish
    rating) - screens out names correctly being re-rated down for a real
    fundamental problem
  - no adverse legal news behind the drop (data_sources.news_rss already
    categorizes news items as 'legal' via config.yaml -> legal_keywords)
  - no "beat but sold off anyway" earnings-reaction anomaly - the same
    signal cash_secured_put uses as a score penalty, but a hard exclusion
    here since it's a sign something's off beneath the headline number
  - risk score also factors in whether the stock's own *sector* is also
    correcting (less of a stock-specific overreaction, more just beta) and
    whether earnings falls within strategies.dip_buy.earnings_alert_days -
    always shown as a warning badge regardless of risk score, since that's
    real gap risk the technical/analyst signals above say nothing about

A ranked watchlist, not a sized portfolio - there's no capital allocation
here, just the top `max_candidates` by composite score. Reuses the same
cached per-ticker data (price, technicals, analyst snapshot, news, earnings
date, post-earnings reaction, sector performance) the cash-secured-put
strategy and the dashboard already read, plus the drawdown % derived from
the shared local price-history cache (data_sources.market_data.
fetch_recent_trend_pct / refresh_price_history), the same cache every
other daily-close-based signal in this app reads from.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from common import effective_ttl_minutes
from data_sources.market_data import fetch_recent_trend_pct, fetch_close_price_on_or_after, drawdown_episode_stats
from engine.scorer import diversify_by_sector, RISK_PROFILE_MAX_SCORE
from engine.signals import (
    market_mood_risk_component,
    technical_risk_component,
    analyst_sentiment_score,
    analyst_upside_pct,
    leverage_risk_component,
    relative_weakness_risk_component,
    dip_maturity_score,
    classify_dip_maturity,
)

KEY = "dip_buy"
DISPLAY_NAME = "Dip Buy Watch"
# generate() below never reads an option chain - see run_pipeline.run() /
# data_sources.market_data.refresh_market_data's needs_chains param.
NEEDS_OPTION_CHAINS = False
# Dip-buy candidate cards have no "Events & filings" panel, and generate()
# never reads 8-K/13F data - see run_pipeline.run().
NEEDS_EVENTS_DATA = False


def _parse_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _risk_score(
    technical_component: float, market_mood_component: float, sector_component: float,
    earnings_component: float, leverage_component: float, relative_weakness_component: float, weights: dict,
) -> tuple[int, str]:
    """
    Same 1-10 scale/labels as the cash-secured-put strategy's risk score (see
    engine.scorer._risk_score) so badges read consistently across strategies.
    There's no option contract here (no IV/spread/coverage), so this blends
    six 0-1 risk contributions instead: technical overextension, broad
    market mood, the stock's own sector trend (a sector also correcting
    means less of a stock-specific overreaction to bet on), whether
    earnings falls within the alert window (real gap risk the other
    signals can't see), balance-sheet leverage, and how much this stock's
    own decline outpaced the broad market's over the same window (see
    engine.signals.leverage_risk_component / relative_weakness_risk_component) -
    fundamental and relative-performance signals, instead of this being
    purely a price/technicals screen.
    """
    raw = (
        weights.get("technical", 0.2) * technical_component
        + weights.get("market_mood", 0.1) * market_mood_component
        + weights.get("sector", 0.1) * sector_component
        + weights.get("earnings", 0.2) * earnings_component
        + weights.get("leverage", 0.2) * leverage_component
        + weights.get("relative_weakness", 0.2) * relative_weakness_component
    )
    score = max(1, min(10, round(1 + raw * 9)))
    label = "Low" if score <= 3 else "Medium" if score <= 6 else "High"
    return score, label


def generate(db, cfg: dict, universe: list[dict]) -> list[dict]:
    strat_cfg = cfg.get("strategies", {}).get("dip_buy", {})
    lookback_days = strat_cfg.get("drawdown_lookback_days", 10)
    drawdown_threshold_pct = strat_cfg.get("drawdown_threshold_pct", -8.0)
    rsi_oversold_max = strat_cfg.get("rsi_oversold_max", 32)
    min_analyst_score = strat_cfg.get("min_analyst_score", 0.5)
    legal_lookback_days = strat_cfg.get("legal_news_lookback_days", 10)
    exclude_beat_but_sold_off = strat_cfg.get("exclude_beat_but_sold_off", True)
    earnings_alert_days = strat_cfg.get("earnings_alert_days", 15)
    max_candidates = strat_cfg.get("max_candidates", 10)
    min_composite_score = strat_cfg.get("min_composite_score", 0.0)
    max_per_sector = strat_cfg.get("max_per_sector", 2)
    weights = strat_cfg.get("weights", {})
    risk_weights = strat_cfg.get("risk_weights", {})
    # Shared across every strategy (see config.yaml's top-level risk_profile) -
    # caps which candidates are eligible by their own 1-10 risk_score, same
    # scale/meaning as cash_secured_put's identical gate.
    max_risk_score = RISK_PROFILE_MAX_SCORE.get(cfg.get("risk_profile", "balanced"), 6)

    market_mood = db.get_market_mood() or {}
    mood_component = market_mood_risk_component(market_mood.get("trend_pct"))
    sector_trend_by_name = {s["sector"]: s["trend_pct"] for s in (db.get_sector_performance() or [])}
    legal_cutoff = (date.today() - timedelta(days=legal_lookback_days)).isoformat()
    today = date.today()
    flagged_date = today.isoformat()
    earnings_alert_cutoff = today + timedelta(days=earnings_alert_days)
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    # Same window as each candidate's own drawdown, so "relative_strength_pct"
    # below is a fair apples-to-apples comparison - fetched once here, not
    # per-ticker, since it's the same value for everyone this run.
    spy_trend_pct = fetch_recent_trend_pct(db, "SPY", lookback_days, ttl_minutes=history_ttl)
    cooldown_days = cfg.get("decision_cooldown_days", 7)
    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    # Don't re-surface a ticker you already accepted or ignored recently.
    snoozed = db.get_recent_decided_tickers("dip_candidate", cooldown_cutoff)

    results = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in snoozed:
            continue
        price = db.get_price(ticker)
        if not price or price <= 0:
            continue

        # Backed by the shared price_history cache (see data_sources.
        # market_data.refresh_price_history) - a "Generate" click within
        # the TTL window reuses it instead of re-hitting yfinance for every
        # ticker in the universe every single time.
        drawdown_pct = fetch_recent_trend_pct(db, ticker, lookback_days, ttl_minutes=history_ttl)
        if drawdown_pct is None or drawdown_pct > drawdown_threshold_pct:
            continue  # hasn't dropped enough (or no data) - not a "major correction"

        technicals = db.get_technicals(ticker)
        rsi = (technicals or {}).get("rsi_14")
        if rsi is None or rsi > rsi_oversold_max:
            continue  # not confirmed oversold

        analyst = db.get_analyst_snapshot(ticker)
        a_score = analyst_sentiment_score(analyst, price)
        if a_score < min_analyst_score:
            continue  # analysts aren't backing a bounce - possibly a real fundamental problem, not an overreaction

        recent_legal_news = [
            n for n in db.get_news(ticker, limit=8)
            if n.get("category") == "legal" and (n.get("published_at") or "") >= legal_cutoff
        ]
        if recent_legal_news:
            continue  # drop looks driven by a real legal/regulatory issue, not a market overreaction - skip, falling-knife risk

        earnings_reaction = db.get_post_earnings_reaction(ticker)
        if exclude_beat_but_sold_off and (earnings_reaction or {}).get("anomaly_type") == "beat_but_sold_off":
            continue  # beat and still sold off last report - something's off beneath the headline number, falling-knife risk

        earnings = db.get_earnings(ticker)
        earnings_date = _parse_date((earnings or {}).get("earnings_date"))
        earnings_before_horizon = bool(earnings_date and today <= earnings_date <= earnings_alert_cutoff)

        # How much of this drop is just the broad market, vs. a real
        # stock-specific problem - and how deep/long this dip already is
        # relative to this stock's *own* past corrections (is it too
        # early to call this a bottom, by its own history?).
        relative_strength_pct = (drawdown_pct - spy_trend_pct) if spy_trend_pct is not None else None
        episode_stats = drawdown_episode_stats(db.get_price_history(ticker), drawdown_threshold_pct) or {}
        maturity_score = dip_maturity_score(episode_stats.get("depth_ratio"), episode_stats.get("duration_ratio"))
        maturity_label = classify_dip_maturity(maturity_score)

        tech_component = technical_risk_component(technicals)
        sector_component = market_mood_risk_component(sector_trend_by_name.get(row.get("sector")), max_swing_pct=10.0)
        earnings_component = 1.0 if earnings_before_horizon else 0.0
        leverage_component = leverage_risk_component((analyst or {}).get("debt_to_equity"), market_mood.get("rate_trend_pct"))
        relative_weakness_component = relative_weakness_risk_component(relative_strength_pct)
        # Normalized 0-1: right at the threshold -> 0.5, twice the threshold -> 1.0 (capped).
        drawdown_component = min(abs(drawdown_pct) / abs(drawdown_threshold_pct) / 2.0, 1.0)
        oversold_component = max(0.0, (rsi_oversold_max - rsi) / rsi_oversold_max)

        composite = (
            weights.get("drawdown", 0.35) * drawdown_component
            + weights.get("oversold", 0.25) * oversold_component
            + weights.get("analyst_sentiment", 0.25) * a_score
            + weights.get("maturity", 0.15) * (maturity_score or 0.0)
        )
        risk_score, risk_label = _risk_score(
            tech_component, mood_component, sector_component, earnings_component,
            leverage_component, relative_weakness_component, risk_weights,
        )
        upside_pct = analyst_upside_pct(analyst, price)

        results.append(
            {
                "ticker": ticker,
                "company_name": row.get("company_name"),
                "price": round(price, 2),
                "sector": row.get("sector"),
                "flagged_date": flagged_date,
                "drawdown_pct": round(drawdown_pct, 2),
                "drawdown_lookback_days": lookback_days,
                "rsi_14": round(rsi, 2),
                "analyst_recommendation": (analyst or {}).get("recommendation_key"),
                "analyst_target_mean_price": (analyst or {}).get("target_mean_price"),
                "analyst_upside_pct": round(upside_pct, 2) if upside_pct is not None else None,
                "pe_ratio": (analyst or {}).get("pe_ratio"),
                "week52_low": (analyst or {}).get("week52_low"),
                "week52_high": (analyst or {}).get("week52_high"),
                "debt_to_equity": (analyst or {}).get("debt_to_equity"),
                "industry": (analyst or {}).get("industry"),
                "relative_strength_pct": round(relative_strength_pct, 2) if relative_strength_pct is not None else None,
                "historical_median_depth_pct": episode_stats.get("median_depth_pct"),
                "historical_median_duration_days": episode_stats.get("median_duration_days"),
                "dip_maturity_score": round(maturity_score, 2) if maturity_score is not None else None,
                "dip_maturity_label": maturity_label,
                "earnings_date": earnings_date.isoformat() if earnings_date else None,
                "earnings_before_horizon": int(earnings_before_horizon),
                "composite_score": round(composite, 4),
                "risk_score": risk_score,
                "risk_label": risk_label,
                "rank": None,
            }
        )

    # Soft quality floor on top of the hard filters above - returning fewer
    # than max_candidates (including zero) is expected and correct when
    # nothing clears this bar, not a bug to work around.
    qualified = [
        c for c in results
        if c["composite_score"] >= min_composite_score and c["risk_score"] <= max_risk_score
    ]
    qualified.sort(key=lambda c: c["composite_score"], reverse=True)
    # Sector-capped (see engine.scorer.diversify_by_sector) so a strong week
    # doesn't hand back max_candidates names that are all the same sector -
    # they'd tend to move together, which defeats the point of a watchlist
    # of "different" picks.
    ranked = diversify_by_sector(qualified, max_candidates, max_per_sector)
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked


def save(db, run_id: int, candidates: list[dict]):
    db.save_dip_candidates(run_id, candidates)


def get_latest(db) -> list[dict]:
    return db.get_latest_dip_candidates()


def get_history(db, limit: int = 200) -> list[dict]:
    return db.get_dip_history(limit=limit)


def due_for_evaluation(db, cfg: dict) -> list[dict]:
    strat_cfg = cfg.get("strategies", {}).get("dip_buy", {})
    horizon_days = strat_cfg.get("evaluation_horizon_days", 10)
    cutoff = (date.today() - timedelta(days=horizon_days)).isoformat()
    return db.get_unevaluated_dip_candidates(as_of=cutoff)


def classify_outcome(entry_price: float, resolved_price: float, flat_band_pct: float) -> str:
    change_pct = (resolved_price - entry_price) / entry_price * 100.0
    if change_pct > flat_band_pct:
        return "rebounded"
    if change_pct < -flat_band_pct:
        return "further_decline"
    return "flat"


def resolve_outcome(db, rec: dict, cfg: dict) -> Optional[tuple[str, float]]:
    strat_cfg = cfg.get("strategies", {}).get("dip_buy", {})
    horizon_days = strat_cfg.get("evaluation_horizon_days", 10)
    flat_band_pct = strat_cfg.get("flat_band_pct", 2.0)
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    flagged = date.fromisoformat(rec["flagged_date"])
    resolution_date = (flagged + timedelta(days=horizon_days)).isoformat()
    resolved_price = fetch_close_price_on_or_after(db, rec["ticker"], resolution_date, ttl_minutes=history_ttl)
    if resolved_price is None:
        return None  # no trading data yet for that date - try again next startup
    return classify_outcome(rec["price"], resolved_price, flat_band_pct), resolved_price


def record_outcome(db, rec: dict, outcome: str, resolved_price: float):
    return_pct = (resolved_price - rec["price"]) / rec["price"] * 100.0 if rec.get("price") else None
    db.record_dip_outcome(rec["run_id"], rec["ticker"], rec["flagged_date"], outcome, resolved_price, return_pct)


def record_decision(db, rec: dict, decision: str):
    db.record_dip_decision(rec["run_id"], rec["ticker"], rec["flagged_date"], decision)


def live_return_pct(rec: dict, current_price: float) -> Optional[float]:
    """
    Unrealized mark-to-market read for a still-pending row - same formula
    as record_outcome's, just fed the current live price instead of the
    resolved price. Never stored (see app.py's /api/history/refresh) - the
    locked-in return_pct column is only ever written once by record_outcome.
    """
    return (current_price - rec["price"]) / rec["price"] * 100.0 if rec.get("price") else None


def get_row(db, rec: dict) -> Optional[dict]:
    """Full row by natural key - see the track record's manual 'check now' outcome refresh in app.py."""
    return db.get_dip_candidate_row(rec["run_id"], rec["ticker"], rec["flagged_date"])


def delete_history_row(db, rec: dict):
    """'Stop tracking' - permanently removes this row, not an outcome of any kind."""
    db.delete_dip_candidate_row(rec["run_id"], rec["ticker"], rec["flagged_date"])
