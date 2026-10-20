"""
"Weekly call buy" strategy - a directional, leveraged, defined-risk bet
(buy a short-dated call outright) instead of selling premium for income
like cash_secured_put/covered_call. Genuinely different objective from
every other strategy here: a bought option wants a *cheap* premium
relative to a strong bullish signal, not a rich one - high IV helps a
seller (more income) but hurts a buyer (more paid for the same thesis, and
exposed to IV crush even if the direction call is right). Reuses the same
call_option_snapshot chain data covered_call already caches (same ticker,
same nearest weekly expiry - see data_sources.market_data.
fetch_weekly_option_chain) - no new network calls.

Hard filters are deliberately stricter than the other dip-buying strategies
given the short window and decaying premium: real outperformance vs. the
benchmark (not just "not falling"), no earnings before this week's expiry
(IV-crush/gap risk on top of an already-decaying position is a bad
combination), no recent adverse legal news, and a capped required-move -
past a certain point a strike needs too big a rally to be worth calling a
"recommendation" instead of a lottery ticket.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from common import effective_ttl_minutes
from data_sources.market_data import fetch_close_price_on_or_after, fetch_recent_trend_pct
from engine.greeks import call_greeks
from engine.scorer import RISK_PROFILE_MAX_SCORE, diversify_by_sector
from engine.signals import (
    market_mood_risk_component,
    technical_risk_component,
    analyst_sentiment_score,
    leverage_risk_component,
)

KEY = "weekly_call_buy"
DISPLAY_NAME = "Weekly Call Buys"
NEEDS_OPTION_CHAINS = True
NEEDS_EVENTS_DATA = True


def _parse_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _risk_score(
    iv: Optional[float], required_move_pct: float, days_to_expiry: int,
    market_mood_component: float, technical_component: float, leverage_component: float, weights: dict,
) -> tuple[int, str]:
    """
    Own shape, not shared with the selling strategies' _risk_score (same
    "every strategy owns its own risk_score" pattern as dip_buy/
    three_day_rule) - high IV is risk here too, just via a different
    mechanism (rich premium + IV-crush exposure instead of a bigger
    expected adverse swing), so it's weighted similarly, plus how big a
    rally is actually required and how little time is left to make it.
    """
    iv_component = min((iv or 0.0) / 1.5, 1.0)
    move_component = min(max(required_move_pct, 0.0) / 10.0, 1.0)
    time_component = max(0.0, (7 - days_to_expiry) / 7.0)  # fewer days left -> more theta risk
    raw = (
        weights.get("iv", 0.30) * iv_component
        + weights.get("required_move", 0.25) * move_component
        + weights.get("time_decay", 0.20) * time_component
        + weights.get("market_mood", 0.10) * market_mood_component
        + weights.get("technical", 0.10) * technical_component
        + weights.get("leverage", 0.05) * leverage_component
    )
    score = max(1, min(10, round(1 + raw * 9)))
    label = "Low" if score <= 3 else "Medium" if score <= 6 else "High"
    return score, label


def generate(db, cfg: dict, universe: list[dict]) -> list[dict]:
    strat_cfg = cfg.get("strategies", {}).get("weekly_call_buy", {})
    min_relative_strength_pct = strat_cfg.get("min_relative_strength_pct", 2.0)
    relative_strength_lookback_days = strat_cfg.get("relative_strength_lookback_days", 10)
    min_analyst_score = strat_cfg.get("min_analyst_score", 0.5)
    legal_news_lookback_days = strat_cfg.get("legal_news_lookback_days", 10)
    strike_otm_pct_min = strat_cfg.get("strike_otm_pct_min", -2.0)   # allows slightly ITM
    strike_otm_pct_max = strat_cfg.get("strike_otm_pct_max", 8.0)
    max_required_move_pct = strat_cfg.get("max_required_move_pct", 10.0)
    min_bid_price = strat_cfg.get("min_bid_price", 0.20)
    min_open_interest = strat_cfg.get("min_open_interest", 50)
    max_bid_ask_spread_pct = strat_cfg.get("max_bid_ask_spread_pct", 25.0)
    max_candidates = strat_cfg.get("max_candidates", 10)
    min_composite_score = strat_cfg.get("min_composite_score", 0.0)
    max_per_sector = strat_cfg.get("max_per_sector", 5)
    weights = strat_cfg.get("weights", {})
    risk_weights = strat_cfg.get("risk_weights", {})
    risk_free_rate = cfg.get("risk_free_rate", 0.045)
    max_risk_score = RISK_PROFILE_MAX_SCORE.get(cfg.get("risk_profile", "balanced"), 6)

    market_mood = db.get_market_mood() or {}
    mood_component = market_mood_risk_component(market_mood.get("trend_pct"))
    today = date.today()
    cooldown_days = cfg.get("decision_cooldown_days", 7)
    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    snoozed = db.get_recent_decided_tickers("weekly_call_buy_candidate", cooldown_cutoff)
    legal_cutoff = (today - timedelta(days=legal_news_lookback_days)).isoformat()
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    spy_trend_pct = fetch_recent_trend_pct(db, "SPY", relative_strength_lookback_days, ttl_minutes=history_ttl)

    results = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in snoozed:
            continue
        price = db.get_price(ticker)
        if not price or price <= 0:
            continue
        call_rows = db.get_call_snapshots(ticker)
        if not call_rows:
            continue

        stock_trend_pct = fetch_recent_trend_pct(db, ticker, relative_strength_lookback_days, ttl_minutes=history_ttl)
        if stock_trend_pct is None or spy_trend_pct is None:
            continue
        relative_strength_pct = stock_trend_pct - spy_trend_pct
        if relative_strength_pct < min_relative_strength_pct:
            continue  # the whole thesis is real outperformance, not just "not falling"

        technicals = db.get_technicals(ticker)
        if (technicals or {}).get("trend") == "death_cross":
            continue  # a clear downtrend contradicts the bullish bet regardless of recent relative strength

        analyst_snapshot = db.get_analyst_snapshot(ticker)
        a_score = analyst_sentiment_score(analyst_snapshot, price)
        if a_score < min_analyst_score:
            continue

        recent_legal_news = [
            n for n in db.get_news(ticker, limit=8)
            if n.get("category") == "legal" and (n.get("published_at") or "") >= legal_cutoff
        ]
        if recent_legal_news:
            continue

        # Earnings before this week's expiry is a hard exclude here (not just
        # a flag like the selling strategies) - IV crush plus an already-
        # decaying bought option is a particularly bad combination, and this
        # strategy's whole premise is a short, clean momentum window, not an
        # earnings bet.
        earnings = db.get_earnings(ticker)
        next_earnings_date = _parse_date((earnings or {}).get("earnings_date"))

        tech_component = technical_risk_component(technicals)
        leverage_component = leverage_risk_component(
            (analyst_snapshot or {}).get("debt_to_equity"), market_mood.get("rate_trend_pct")
        )

        for opt in call_rows:
            strike = opt.get("strike")
            if not strike or strike <= 0:
                continue
            expiry_date = _parse_date(opt.get("expiry"))
            if expiry_date is None:
                continue
            if next_earnings_date and today <= next_earnings_date <= expiry_date:
                continue
            days_to_expiry = max((expiry_date - today).days, 0)
            if days_to_expiry > 7:
                continue  # "expiring within the week" is the whole point of this strategy

            ask = opt.get("ask") or 0.0
            bid = opt.get("bid") or 0.0
            if ask <= 0:
                continue  # no real ask means you can't actually buy this at a known price
            if ask < min_bid_price:
                continue
            mid = (bid + ask) / 2.0 if bid > 0 else ask

            open_interest = opt.get("open_interest") or 0
            if open_interest < min_open_interest:
                continue
            spread_pct = ((ask - bid) / mid * 100.0) if (bid > 0 and mid > 0) else None
            if spread_pct is not None and spread_pct > max_bid_ask_spread_pct:
                continue

            pct_otm = (strike - price) / price * 100.0
            if pct_otm < strike_otm_pct_min or pct_otm > strike_otm_pct_max:
                continue

            # Conservative: assume you pay the ask, not the mid - same
            # "worse realistic fill" convention the selling strategies use
            # for the opposite side (assume you're filled at the bid).
            premium_per_share = ask
            premium_total = premium_per_share * 100.0
            breakeven_price = strike + premium_per_share
            required_move_pct = (breakeven_price - price) / price * 100.0
            if required_move_pct > max_required_move_pct:
                continue  # too big a rally needed to be a real "recommendation" rather than a lottery ticket

            implied_vol = opt.get("implied_vol")
            greeks = call_greeks(
                spot=price, strike=strike, days_to_expiry=days_to_expiry,
                implied_vol=implied_vol, risk_free_rate=risk_free_rate,
            )
            # P(close > breakeven), not just P(close > strike) - the real bar
            # for this position to actually profit, reusing the same
            # Black-Scholes machinery against a different reference price.
            breakeven_greeks = call_greeks(
                spot=price, strike=breakeven_price, days_to_expiry=days_to_expiry,
                implied_vol=implied_vol, risk_free_rate=risk_free_rate,
            )
            prob_profit_pct = breakeven_greeks.prob_itm * 100.0

            risk_score, risk_label = _risk_score(
                iv=implied_vol, required_move_pct=required_move_pct, days_to_expiry=days_to_expiry,
                market_mood_component=mood_component, technical_component=tech_component,
                leverage_component=leverage_component, weights=risk_weights,
            )
            if risk_score > max_risk_score:
                continue

            bullish_strength_score = min(max(relative_strength_pct, 0.0) / 10.0, 1.0)
            cheap_move_score = max(0.0, 1.0 - min(required_move_pct, max_required_move_pct) / max_required_move_pct)
            oi_component = min(open_interest / 500.0, 1.0)
            spread_component = 1.0 if spread_pct is None else max(0.0, 1.0 - spread_pct / max_bid_ask_spread_pct)
            liquidity_score = 0.5 * oi_component + 0.5 * spread_component

            composite = (
                weights.get("bullish_strength", 0.40) * bullish_strength_score
                + weights.get("cheap_move", 0.30) * cheap_move_score
                + weights.get("liquidity", 0.15) * liquidity_score
                + weights.get("analyst_sentiment", 0.15) * a_score
            )
            if composite < min_composite_score:
                continue

            results.append(
                {
                    "ticker": ticker,
                    "company_name": row.get("company_name"),
                    "price": round(price, 2),
                    "sector": row.get("sector"),
                    "first_seen_at": today.isoformat(),
                    "price_captured_at": db.get_price_captured_at(ticker),
                    "expiry": opt["expiry"],
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid_price": round(mid, 4),
                    "premium_total": round(premium_total, 2),
                    "breakeven_price": round(breakeven_price, 2),
                    "required_move_pct": round(required_move_pct, 2),
                    "prob_profit_pct": round(prob_profit_pct, 2),
                    "delta": round(greeks.delta, 4),
                    "implied_vol": implied_vol,
                    "open_interest": open_interest,
                    "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
                    "days_to_expiry": days_to_expiry,
                    "analyst_recommendation": (analyst_snapshot or {}).get("recommendation_key"),
                    "analyst_target_mean_price": (analyst_snapshot or {}).get("target_mean_price"),
                    "pe_ratio": (analyst_snapshot or {}).get("pe_ratio"),
                    "week52_low": (analyst_snapshot or {}).get("week52_low"),
                    "week52_high": (analyst_snapshot or {}).get("week52_high"),
                    "debt_to_equity": (analyst_snapshot or {}).get("debt_to_equity"),
                    "industry": (analyst_snapshot or {}).get("industry"),
                    "relative_strength_pct": round(relative_strength_pct, 2),
                    "rsi_14": (technicals or {}).get("rsi_14"),
                    "trend": (technicals or {}).get("trend"),
                    "macd_bullish": None if (technicals or {}).get("macd_bullish") is None else int(bool(technicals["macd_bullish"])),
                    "composite_score": round(composite, 4),
                    "risk_score": risk_score,
                    "risk_label": risk_label,
                    "rank": None,
                }
            )

    # One contract per ticker (the best composite score), same pattern as
    # engine.scorer.rank_top_candidates, then sector-capped.
    best_per_ticker: dict[str, dict] = {}
    for c in results:
        t = c["ticker"]
        if t not in best_per_ticker or c["composite_score"] > best_per_ticker[t]["composite_score"]:
            best_per_ticker[t] = c
    ranked = sorted(best_per_ticker.values(), key=lambda c: c["composite_score"], reverse=True)
    top = diversify_by_sector(ranked, max_candidates, max_per_sector)
    for i, c in enumerate(top, start=1):
        c["rank"] = i
    return top


def save(db, run_id: int, candidates: list[dict]):
    db.save_weekly_call_buy_candidates(run_id, candidates)


def get_latest(db) -> list[dict]:
    return db.get_latest_weekly_call_buy_candidates()


def get_history(db, limit: int = 200) -> list[dict]:
    return db.get_weekly_call_buy_history(limit=limit)


def due_for_evaluation(db, cfg: dict) -> list[dict]:
    return db.get_unevaluated_weekly_call_buy_candidates()


def classify_outcome(strike: float, premium_per_share: float, close_price: float) -> str:
    """
    'win'          - close at/above breakeven: the trade nets a real profit.
    'partial_loss' - finished in-the-money but short of breakeven: some of
                     the premium was recovered, but not all of it.
    'max_loss'     - expired at/below strike: worthless, the full premium
                     paid is lost - the defined, capped downside of buying
                     an option instead of selling one.
    """
    breakeven = strike + premium_per_share
    if close_price >= breakeven:
        return "win"
    if close_price > strike:
        return "partial_loss"
    return "max_loss"


def resolve_outcome(db, rec: dict, cfg: dict) -> Optional[tuple[str, float]]:
    history_ttl = effective_ttl_minutes(cfg.get("cache_ttl_minutes", {}).get("price_history", 240))
    close_price = fetch_close_price_on_or_after(db, rec["ticker"], rec["expiry"], ttl_minutes=history_ttl)
    if close_price is None:
        return None
    return classify_outcome(rec["strike"], rec["ask"], close_price), close_price


def realized_return_pct(strike: float, premium_per_share: float, close_price: float) -> Optional[float]:
    """
    % return on the premium paid - the real capital base for a bought
    option (there's no collateral/shares_value here). payoff is capped
    below at -100% (premium_per_share > 0 always, by the min_bid_price
    hard filter, so this never divides by zero).
    """
    if not premium_per_share:
        return None
    payoff_per_share = max(close_price - strike, 0.0)
    pnl_per_share = payoff_per_share - premium_per_share
    return pnl_per_share / premium_per_share * 100.0


def record_outcome(db, rec: dict, outcome: str, resolved_price: float):
    return_pct = realized_return_pct(rec["strike"], rec["ask"], resolved_price)
    db.record_weekly_call_buy_outcome(
        rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], outcome, resolved_price, return_pct
    )


def record_decision(db, rec: dict, decision: str):
    db.record_weekly_call_buy_decision(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"], decision)


def live_return_pct(rec: dict, current_price: float) -> Optional[float]:
    """Unrealized mark-to-market read for a still-pending row - see realized_return_pct."""
    return realized_return_pct(rec["strike"], rec["ask"], current_price)


def get_row(db, rec: dict) -> Optional[dict]:
    return db.get_weekly_call_buy_candidate_row(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"])


def delete_history_row(db, rec: dict):
    db.delete_weekly_call_buy_candidate_row(rec["run_id"], rec["ticker"], rec["expiry"], rec["strike"])
