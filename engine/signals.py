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


def leverage_risk_component(
    debt_to_equity: Optional[float], rate_trend_pct: Optional[float] = None,
    max_de: float = 200.0, rate_tightening_threshold_pct: float = -2.0,
) -> float:
    """
    0-1 extra-risk contribution from balance-sheet leverage - something no
    other signal in this screener looks at (everything else is price,
    technicals, analyst sentiment, or event timing). debt_to_equity is
    yfinance's raw convention (150 = 150%, i.e. 1.5x debt-to-equity);
    ~200%+ is treated as max risk on its own.

    Amplified up to 1.5x (capped at 1.0) when rate_trend_pct - TLT's
    recent trend, a market-based proxy for rate direction (see
    data_sources.market_data.fetch_market_mood) - shows the long end
    selling off past rate_tightening_threshold_pct: a levered balance
    sheet is a materially bigger risk specifically when financing/
    refinancing costs are rising, not in isolation from the rate
    environment. Missing/zero/negative debt_to_equity -> 0, the same
    "don't penalize for missing data" stance every other signal here takes.
    """
    if debt_to_equity is None or debt_to_equity <= 0:
        return 0.0
    base = min(debt_to_equity / max_de, 1.0)
    if rate_trend_pct is not None and rate_trend_pct <= rate_tightening_threshold_pct:
        base = min(base * 1.5, 1.0)
    return base


def relative_weakness_risk_component(
    relative_strength_pct: Optional[float], max_underperformance_pct: float = 15.0,
) -> float:
    """
    0-1 extra-risk contribution from how much a stock's own decline
    outpaced the broad market's over the same window (relative_strength_pct
    = stock's drawdown minus the benchmark's trend - see data_sources.
    market_data.fetch_recent_trend_pct, called with the same lookback for
    both). A dip that's mostly just beta (the stock fell about as much as
    the market) contributes nothing here - that's closer to the "overreaction,
    not a falling knife" case these strategies want; a stock falling far
    more than the market has a real, idiosyncratic problem the other
    signals (technical/analyst/earnings) might not catch. Missing data -> 0,
    the same stance every other signal here takes.
    """
    if relative_strength_pct is None or relative_strength_pct >= 0:
        return 0.0
    return min(-relative_strength_pct / max_underperformance_pct, 1.0)


def dip_maturity_score(depth_ratio: Optional[float], duration_ratio: Optional[float]) -> Optional[float]:
    """
    0-1 read on how far along the current drawdown looks relative to this
    stock's *own* history of past corrections (see data_sources.
    market_data.drawdown_episode_stats - depth_ratio/duration_ratio are
    current-vs-historical-median, so 1.0 means "right at the stock's
    typical past correction"). Higher means the dip already looks as
    deep/long as this stock's typical past corrections (closer to a likely
    bottom by its own history); lower means it's still shallow/short by
    that same standard - a real answer to "are we too early to buy this
    dip," grounded in the stock's actual past behavior instead of a guess.
    None when there isn't enough history to judge (not penalized either
    way - see drawdown_episode_stats).
    """
    if depth_ratio is None or duration_ratio is None:
        return None
    return max(0.0, min(1.0, (min(depth_ratio, 1.5) + min(duration_ratio, 1.5)) / 3.0))


def classify_dip_maturity(score: Optional[float]) -> str:
    """Pure classification, split out for easy unit testing - see dip_maturity_score."""
    if score is None:
        return "unknown"
    if score < 0.4:
        return "early"
    if score < 0.7:
        return "typical"
    return "mature"


def historical_drawdown_risk_component(
    median_depth_pct: Optional[float], median_duration_days: Optional[float],
    severe_depth_pct: float = -25.0, long_duration_days: float = 60.0,
) -> float:
    """
    0-1 extra-risk contribution from how rough this stock's past
    corrections have typically run (see data_sources.market_data.
    drawdown_episode_stats) - independent of whether it's in one right
    now. Unlike dip_maturity_score (which asks "is the *current* dip
    already as deep/long as usual"), this is a plain standing risk read:
    a cash-secured put should be sold on names you're fine holding if
    assigned, and a stock whose typical past dip runs deep and long is a
    worse name to get stuck holding even when nothing looks wrong about
    it today. No qualifying past episodes -> 0, the same "don't penalize
    missing data" stance everything else here takes.
    """
    if median_depth_pct is None or median_duration_days is None:
        return 0.0
    depth_component = min(abs(median_depth_pct) / abs(severe_depth_pct), 1.0)
    duration_component = min(median_duration_days / long_duration_days, 1.0)
    return (depth_component + duration_component) / 2.0


def analyst_upside_pct(analyst_snapshot: Optional[dict], price: float) -> Optional[float]:
    """% upside of analyst target_mean_price over the current price, or None if no target/price."""
    if not analyst_snapshot or not price:
        return None
    target = analyst_snapshot.get("target_mean_price")
    if not target:
        return None
    return (target - price) / price * 100.0
