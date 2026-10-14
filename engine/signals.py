"""
Ticker-level risk/sentiment signals shared across strategies (see
engine/strategies/). Deliberately has no I/O - callers fetch the underlying
data (market mood, technicals, analyst snapshot) from the cache and pass it
in here, same convention as engine/scorer.py.
"""

from __future__ import annotations

from typing import Optional


def market_mood_risk_component(trend_pct: Optional[float], max_swing_pct: float = 5.0) -> float:
    """
    0-1 extra-risk contribution from the overall market trend (see
    data_sources.market_data.fetch_market_mood): a broad selloff raises the
    odds of assignment/drawdown across *every* position, not just the risky
    ones, so it nudges every candidate's risk score up a bit. A flat or
    rising market contributes nothing (we don't reward candidates for a calm
    tape - the rest of the score already covers the contract-specific risk).
    """
    if trend_pct is None or trend_pct >= 0:
        return 0.0
    return min(-trend_pct / max_swing_pct, 1.0)


def technical_risk_component(technicals: Optional[dict]) -> float:
    """
    0-1 extra-risk contribution from popular technical indicators (RSI(14),
    SMA20/SMA50 trend, MACD(12,26,9) - see
    data_sources.market_data.fetch_technical_indicators). Overbought RSI, a
    death cross, and a bearish MACD histogram each nudge risk up a little;
    none of them are hard signals on their own, so they're averaged rather
    than stacked. Missing/insufficient history (e.g. a recent IPO) -> 0,
    the same "don't penalize for missing data" stance taken elsewhere.
    """
    if not technicals:
        return 0.0
    rsi = technicals.get("rsi_14")
    # RSI 30-70 is the "normal" band; treat 0 or 100 as max risk, linear outside 30-70.
    rsi_component = 0.0 if rsi is None else max(0.0, (abs(rsi - 50.0) - 20.0) / 30.0)
    rsi_component = min(rsi_component, 1.0)
    trend_component = 1.0 if technicals.get("trend") == "death_cross" else 0.0
    macd_component = 0.0 if technicals.get("macd_bullish") else 1.0
    return (rsi_component + trend_component + macd_component) / 3.0


def analyst_sentiment_score(analyst_snapshot: Optional[dict], price: float) -> float:
    """
    0-1 score blending consensus rating (1=Strong Buy..5=Sell) with price-target
    upside. Returns a neutral 0.5 when there's no analyst coverage (e.g. ETFs)
    rather than penalizing names that simply aren't covered.
    """
    if not analyst_snapshot:
        return 0.5
    rec_mean = analyst_snapshot.get("recommendation_mean")
    rating_component = 0.5
    if rec_mean:
        rating_component = max(0.0, min(1.0, (5.0 - rec_mean) / 4.0))

    upside_component = 0.5
    target = analyst_snapshot.get("target_mean_price")
    if target and price:
        upside_pct = (target - price) / price * 100.0
        upside_component = max(0.0, min(1.0, (upside_pct + 10.0) / 30.0))  # -10% upside -> 0, +20% -> 1

    return 0.6 * rating_component + 0.4 * upside_component


def analyst_upside_pct(analyst_snapshot: Optional[dict], price: float) -> Optional[float]:
    """% upside of analyst target_mean_price over the current price, or None if no target/price."""
    if not analyst_snapshot or not price:
        return None
    target = analyst_snapshot.get("target_mean_price")
    if not target:
        return None
    return (target - price) / price * 100.0
