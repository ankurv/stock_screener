"""
"3-Day Rule" strategy: a stock often keeps correcting for a few trading
days right after it reports earnings, before stabilizing - the day the
correction window ends is often an overreaction worth buying, not the start
of a real decline (see config.yaml -> strategies.three_day_rule). Combines:
  - a post-earnings correction window: the 3 most recent trading days since
    the ticker's last reported quarter show a meaningful cumulative decline
    (data_sources.market_data.fetch_post_earnings_correction)
  - today falling within the "day 4-ish" entry window after that report
  - the EPS surprise NOT being a real, sizeable miss - deliberately the
    opposite framing from dip_buy's "beat but sold off" exclusion: a beat
    (or in-line) that still sold off for 3 days *is* the target setup here
    (classic overreaction), so only a genuine miss is screened out
  - the same "good stock" quality gates dip_buy uses so this isn't a
    falling knife: oversold RSI confirmation, analyst support, no adverse
    legal news behind the drop, and a risk score that also factors in
    broad market mood, the stock's own sector trend, and whether its NEXT
    earnings falls within strategies.three_day_rule.earnings_alert_days -
    always shown as a warning badge regardless of risk score

A ranked watchlist, not a sized portfolio - same shape as dip_buy. Reuses
the same cached per-ticker data (price, technicals, analyst snapshot, news,
next earnings date, sector performance) the other two strategies already
read, plus the correction window derived from the shared local
price-history cache (data_sources.market_data.fetch_post_earnings_correction
/ refresh_price_history), the same cache every other daily-close-based
signal in this app reads from.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from common import effective_ttl_minutes
from data_sources.market_data import fetch_post_earnings_correction, fetch_close_price_on_or_after
from engine.scorer import diversify_by_sector
from engine.signals import (
    market_mood_risk_component,
    technical_risk_component,
    analyst_sentiment_score,
    analyst_upside_pct,
)

KEY = "three_day_rule"
DISPLAY_NAME = "3-Day Rule"
# generate() below never reads an option chain - see run_pipeline.run() /
# data_sources.market_data.refresh_market_data's needs_chains param.
NEEDS_OPTION_CHAINS = False
# 3-Day Rule candidate cards have no "Events & filings" panel, and generate()
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
    earnings_component: float, weights: dict,
) -> tuple[int, str]:
    """Same 1-10 scale/labels and blend shape as dip_buy._risk_score, for consistent badges across strategies."""
    raw = (
        weights.get("technical", 0.3) * technical_component
        + weights.get("market_mood", 0.2) * market_mood_component
        + weights.get("sector", 0.2) * sector_component
        + weights.get("earnings", 0.3) * earnings_component
    )
    score = max(1, min(10, round(1 + raw * 9)))
    label = "Low" if score <= 3 else "Medium" if score <= 6 else "High"
    return score, label


def generate(db, cfg: dict, universe: list[dict]) -> list[dict]:
    strat_cfg = cfg.get("strategies", {}).get("three_day_rule", {})
    min_days_since_earnings = strat_cfg.get("min_days_since_earnings", 4)
    max_days_since_earnings = strat_cfg.get("max_days_since_earnings", 8)
    drawdown_threshold_pct = strat_cfg.get("drawdown_threshold_pct", -6.0)
    min_earnings_surprise_pct = strat_cfg.get("min_earnings_surprise_pct", -5.0)
    rsi_oversold_max = strat_cfg.get("rsi_oversold_max", 35)
    min_analyst_score = strat_cfg.get("min_analyst_score", 0.5)
    legal_lookback_days = strat_cfg.get("legal_news_lookback_days", 10)
    earnings_alert_days = strat_cfg.get("earnings_alert_days", 15)
    max_candidates = strat_cfg.get("max_candidates", 10)
    min_composite_score = strat_cfg.get("min_composite_score", 0.0)
    max_per_sector = strat_cfg.get("max_per_sector", 2)
    weights = strat_cfg.get("weights", {})
    risk_weights = strat_cfg.get("risk_weights", {})

    market_mood = db.get_market_mood() or {}
    mood_component = market_mood_risk_component(market_mood.get("trend_pct"))
    sector_trend_by_name = {s["sector"]: s["trend_pct"] for s in (db.get_sector_performance() or [])}
    legal_cutoff = (date.today() - timedelta(days=legal_lookback_days)).isoformat()
    today = date.today()
    flagged_date = today.isoformat()
    earnings_alert_cutoff = today + timedelta(days=earnings_alert_days)
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    cooldown_days = cfg.get("decision_cooldown_days", 7)
    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    # Don't re-surface a ticker you already accepted or ignored recently.
    snoozed = db.get_recent_decided_tickers("three_day_candidate", cooldown_cutoff)

    results = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in snoozed:
            continue
        price = db.get_price(ticker)
        if not price or price <= 0:
            continue

        reaction = db.get_post_earnings_reaction(ticker)
        last_earnings_date = _parse_date((reaction or {}).get("earnings_date"))
        if last_earnings_date is None:
            continue  # no reported quarter on file yet (e.g. recent IPO)
        days_since_earnings = (today - last_earnings_date).days
        if not (min_days_since_earnings <= days_since_earnings <= max_days_since_earnings):
            continue  # not in the "day 4-ish" entry window right now

        # Backed by the shared price_history cache (see data_sources.
        # market_data.refresh_price_history) - a "Generate" click within
        # the TTL window reuses it instead of re-hitting yfinance for every
        # ticker in the universe every single time.
        window = fetch_post_earnings_correction(db, ticker, last_earnings_date, ttl_minutes=history_ttl)
        if window is None or window["drawdown_pct"] > drawdown_threshold_pct:
            continue  # hasn't corrected enough since earnings (or no data) - not a "3-day rule" setup

        surprise_pct = window.get("surprise_pct")
        if surprise_pct is not None and surprise_pct < min_earnings_surprise_pct:
            continue  # a real, sizeable miss - more likely a genuine falling knife than an overreaction

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
            continue  # correction looks driven by a real legal/regulatory issue, not an earnings overreaction

        next_earnings = db.get_earnings(ticker)
        next_earnings_date = _parse_date((next_earnings or {}).get("earnings_date"))
        earnings_before_horizon = bool(next_earnings_date and today <= next_earnings_date <= earnings_alert_cutoff)

        tech_component = technical_risk_component(technicals)
        sector_component = market_mood_risk_component(sector_trend_by_name.get(row.get("sector")), max_swing_pct=10.0)
        earnings_component = 1.0 if earnings_before_horizon else 0.0
        # Normalized 0-1: right at the threshold -> 0.5, twice the threshold -> 1.0 (capped).
        drawdown_component = min(abs(window["drawdown_pct"]) / abs(drawdown_threshold_pct) / 2.0, 1.0)
        down_days_component = window["down_day_count"] / 3.0
        oversold_component = max(0.0, (rsi_oversold_max - rsi) / rsi_oversold_max)

        composite = (
            weights.get("drawdown", 0.35) * drawdown_component
            + weights.get("down_days", 0.15) * down_days_component
            + weights.get("oversold", 0.2) * oversold_component
            + weights.get("analyst_sentiment", 0.3) * a_score
        )
        risk_score, risk_label = _risk_score(tech_component, mood_component, sector_component, earnings_component, risk_weights)
        upside_pct = analyst_upside_pct(analyst, price)

        results.append(
            {
                "ticker": ticker,
                "company_name": row.get("company_name"),
                "price": round(price, 2),
                "sector": row.get("sector"),
                "flagged_date": flagged_date,
                "last_earnings_date": last_earnings_date.isoformat(),
                "days_since_earnings": days_since_earnings,
                "window_start_date": window["window_start_date"],
                "window_end_date": window["window_end_date"],
                "anchor_close": window["anchor_close"],
                "window_last_close": window["window_last_close"],
                "drawdown_pct": window["drawdown_pct"],
                "down_day_count": window["down_day_count"],
                "surprise_pct": surprise_pct,
                "rsi_14": round(rsi, 2),
                "analyst_recommendation": (analyst or {}).get("recommendation_key"),
                "analyst_target_mean_price": (analyst or {}).get("target_mean_price"),
                "analyst_upside_pct": round(upside_pct, 2) if upside_pct is not None else None,
                "pe_ratio": (analyst or {}).get("pe_ratio"),
                "week52_low": (analyst or {}).get("week52_low"),
                "week52_high": (analyst or {}).get("week52_high"),
                "next_earnings_date": next_earnings_date.isoformat() if next_earnings_date else None,
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
    qualified = [c for c in results if c["composite_score"] >= min_composite_score]
    qualified.sort(key=lambda c: c["composite_score"], reverse=True)
    # Sector-capped (see engine.scorer.diversify_by_sector) so a strong week
    # doesn't hand back max_candidates names that are all the same sector.
    ranked = diversify_by_sector(qualified, max_candidates, max_per_sector)
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked


def save(db, run_id: int, candidates: list[dict]):
    db.save_three_day_candidates(run_id, candidates)


def get_latest(db) -> list[dict]:
    return db.get_latest_three_day_candidates()


def get_history(db, limit: int = 200) -> list[dict]:
    return db.get_three_day_history(limit=limit)


def due_for_evaluation(db, cfg: dict) -> list[dict]:
    strat_cfg = cfg.get("strategies", {}).get("three_day_rule", {})
    horizon_days = strat_cfg.get("evaluation_horizon_days", 10)
    cutoff = (date.today() - timedelta(days=horizon_days)).isoformat()
    return db.get_unevaluated_three_day_candidates(as_of=cutoff)


def classify_outcome(entry_price: float, resolved_price: float, flat_band_pct: float) -> str:
    change_pct = (resolved_price - entry_price) / entry_price * 100.0
    if change_pct > flat_band_pct:
        return "rebounded"
    if change_pct < -flat_band_pct:
        return "further_decline"
    return "flat"


def resolve_outcome(db, rec: dict, cfg: dict) -> Optional[tuple[str, float]]:
    strat_cfg = cfg.get("strategies", {}).get("three_day_rule", {})
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
    db.record_three_day_outcome(rec["run_id"], rec["ticker"], rec["flagged_date"], outcome, resolved_price, return_pct)


def record_decision(db, rec: dict, decision: str):
    db.record_three_day_decision(rec["run_id"], rec["ticker"], rec["flagged_date"], decision)


def delete_history_row(db, rec: dict):
    """'Stop tracking' - permanently removes this row, not an outcome of any kind."""
    db.delete_three_day_candidate_row(rec["run_id"], rec["ticker"], rec["flagged_date"])
