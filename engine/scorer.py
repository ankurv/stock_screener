"""
Pure scoring/ranking logic for cash-secured put candidates. Deliberately
has no I/O (no DB, no network) so it can be unit-tested with synthetic data -
the pipeline is responsible for fetching rows and feeding them in here.

Scoring approach (see config.yaml -> weights to tune):
  - annualized_yield: premium collected, annualized against collateral
  - probability_otm: Black-Scholes estimate (from engine.greeks) of the
    contract expiring worthless, i.e. you keep the premium and are not assigned
  - liquidity: rewards tight bid/ask spread and reasonable open interest
  - earnings_penalty: subtracted if the stock reports earnings before this
    contract expires (gap risk) - flagged either way, hard-excluded only if
    config.exclude_if_earnings_before_expiry is true

This is a heuristic screener, not investment advice: it surfaces candidates
that match a configurable rule set so you can do your own diligence, not a
recommendation to execute any specific trade.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from engine.greeks import put_greeks
from engine.signals import (
    market_mood_risk_component,
    technical_risk_component,
    analyst_sentiment_score,
)


def _nullable_int(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(bool(value))


def _parse_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def strike_distance_risk_component(price: float, strike: float, comfortable_otm_pct: float = 15.0) -> float:
    """
    0-1 extra-risk contribution from how close the strike is to the current
    price - independent of volatility: a strike sitting right at the money
    is a coin flip on assignment regardless of how calm the stock has been,
    while a strike sold comfortably below the market has to fall further
    before you're on the hook. 0% OTM (at or in the money) -> max risk;
    >= comfortable_otm_pct% OTM -> no extra risk from this factor.
    """
    if not price or price <= 0:
        return 0.0
    pct_otm = (price - strike) / price * 100.0  # negative => strike is above price (in the money)
    return max(0.0, min(1.0, 1.0 - pct_otm / comfortable_otm_pct))


def _risk_score(
    implied_vol: Optional[float],
    has_analyst_coverage: bool,
    earnings_before_expiry: bool,
    spread_pct: Optional[float],
    max_spread_pct: float,
    market_mood_component: float = 0.0,
    technical_component: float = 0.0,
    strike_distance_component: float = 0.0,
) -> tuple[int, str]:
    """
    1 (least risky) to 10 (most risky) read on how risky the *underlying/
    contract* is - separate from the composite ranking score, which is about
    how *attractive* the trade's terms are. A mega-cap like GOOG and a
    thinly-traded small-cap can offer similarly attractive annualized yields,
    but the small-cap is riskier to actually hold if assigned - implied vol
    is the clearest signal of that the pipeline already has (the market
    pricing in bigger expected swings), so it dominates the score, with how
    far the strike sits from the current price (see
    strike_distance_risk_component) a close second - a near-the-money strike
    on an otherwise "safe" stock is still a real assignment risk.
    coverage/earnings-gap/spread/market-mood/technicals are secondary.
    """
    iv = implied_vol or 0.0
    iv_component = min(iv / 1.5, 1.0)  # ~150%+ annualized IV treated as max risk
    coverage_component = 0.0 if has_analyst_coverage else 1.0
    earnings_component = 1.0 if earnings_before_expiry else 0.0
    spread_component = 0.0 if spread_pct is None else min(spread_pct / max_spread_pct, 1.0)

    raw = (
        0.35 * iv_component
        + 0.20 * strike_distance_component
        + 0.10 * coverage_component
        + 0.10 * earnings_component
        + 0.05 * spread_component
        + 0.10 * market_mood_component
        + 0.10 * technical_component
    )
    score = max(1, min(10, round(1 + raw * 9)))
    label = "Low" if score <= 3 else "Medium" if score <= 6 else "High"
    return score, label


def evaluate_candidates(
    ticker: str,
    company_name: Optional[str],
    price: float,
    put_rows: list[dict],
    earnings_date: Optional[str],
    cfg: dict,
    today: Optional[date] = None,
    analyst_snapshot: Optional[dict] = None,
    earnings_reaction: Optional[dict] = None,
    market_mood: Optional[dict] = None,
    technicals: Optional[dict] = None,
    sector: Optional[str] = None,
) -> list[dict]:
    """
    Apply hard filters (min yield, min OI, max spread) and compute a
    composite score for every surviving put row for this ticker. Returns
    one dict per surviving contract (ticker can have 0+ rows). No capital/
    collateral cap here - this is a ranked watchlist (see
    select_top_candidates); size each position's collateral yourself.
    """
    today = today or date.today()
    if not price or price <= 0 or not put_rows:
        return []

    w = cfg["weights"]
    results = []

    # Ticker-level (not contract-level) signals, computed once per ticker.
    analyst_score = analyst_sentiment_score(analyst_snapshot, price)
    anomaly_type = (earnings_reaction or {}).get("anomaly_type")
    mood_component = market_mood_risk_component((market_mood or {}).get("trend_pct"))
    tech_component = technical_risk_component(technicals)
    is_beat_but_sold_off = anomaly_type == "beat_but_sold_off"

    for row in put_rows:
        strike = row.get("strike")
        if not strike or strike <= 0:
            continue
        if row.get("in_the_money"):
            # Never sell an in-the-money put, full stop - the point of a cash-secured
            # put is to collect the premium and *not* get assigned at expiry. This is
            # the authoritative check (yfinance's own real-time flag from the chain);
            # the pct_otm check below is a second, price-derived guard against the
            # same thing in case our cached `price` has drifted slightly stale.
            continue
        collateral = strike * 100.0  # reported, not filtered - you size positions yourself

        bid = row.get("bid") or 0.0
        ask = row.get("ask") or 0.0
        if bid <= 0:
            continue  # no real bid means you can't actually collect a premium here
        if bid < cfg.get("min_bid_price", 0.50):
            continue  # too cheap - commissions eat most/all of the premium
        mid = (bid + ask) / 2.0 if ask > 0 else bid

        expiry_date = _parse_date(row.get("expiry"))
        if expiry_date is None:
            continue
        days_to_expiry = max((expiry_date - today).days, 0)
        # Same-day (0DTE) expiry: still sellable in the morning: treat as a
        # minimum of a few hours of time value by flooring at 1 day for the
        # annualization math so it isn't divide-by-zero / infinite yield.
        annualize_days = max(days_to_expiry, 1)

        premium_total = bid * 100.0  # conservative: assume fill at the bid, not mid
        annualized_yield_pct = (bid / strike) * (365.0 / annualize_days) * 100.0
        if annualized_yield_pct < cfg["min_annualized_yield_pct"]:
            continue

        open_interest = row.get("open_interest") or 0
        if open_interest < cfg["min_open_interest"]:
            continue

        spread_pct = ((ask - bid) / mid * 100.0) if (ask > 0 and mid > 0) else None
        if spread_pct is not None and spread_pct > cfg["max_bid_ask_spread_pct"]:
            continue

        greeks = put_greeks(
            spot=price, strike=strike, days_to_expiry=days_to_expiry,
            implied_vol=row.get("implied_vol"), risk_free_rate=cfg.get("risk_free_rate", 0.045),
        )

        earnings_dt = _parse_date(earnings_date)
        earnings_before_expiry = bool(earnings_dt and today <= earnings_dt <= expiry_date)
        if earnings_before_expiry and cfg.get("exclude_if_earnings_before_expiry"):
            continue

        # Strike must be OTM and within a bounded distance of the current price -
        # too close is a coin-flip on assignment, too far is barely worth the
        # premium. Earnings before expiry gets a wider band: the extra event
        # risk is worth paying for with a bit more distance from the money.
        pct_otm = (price - strike) / price * 100.0
        max_otm_pct = (
            cfg.get("strike_otm_pct_max_with_earnings", 15.0)
            if earnings_before_expiry
            else cfg.get("strike_otm_pct_max", 5.0)
        )
        if pct_otm < 0 or pct_otm > max_otm_pct:
            continue

        # --- sub-scores normalized to roughly 0-1 for the composite blend ---
        yield_score = min(annualized_yield_pct / 50.0, 1.0)
        prob_score = greeks.prob_otm
        oi_component = min(open_interest / 500.0, 1.0)
        spread_component = 1.0 if spread_pct is None else max(0.0, 1.0 - spread_pct / cfg["max_bid_ask_spread_pct"])
        liquidity_score = 0.5 * oi_component + 0.5 * spread_component
        earnings_flag_score = 1.0 if earnings_before_expiry else 0.0
        anomaly_flag_score = 1.0 if is_beat_but_sold_off else 0.0

        composite = (
            w["annualized_yield"] * yield_score
            + w["probability_otm"] * prob_score
            + w["liquidity"] * liquidity_score
            + w.get("analyst_sentiment", 0.0) * analyst_score
            - w["earnings_penalty"] * earnings_flag_score
            - w.get("anomaly_penalty", 0.0) * anomaly_flag_score
        )

        risk_score, risk_label = _risk_score(
            implied_vol=row.get("implied_vol"),
            has_analyst_coverage=bool(analyst_snapshot),
            earnings_before_expiry=earnings_before_expiry,
            spread_pct=spread_pct,
            max_spread_pct=cfg["max_bid_ask_spread_pct"],
            market_mood_component=mood_component,
            technical_component=tech_component,
            strike_distance_component=strike_distance_risk_component(price, strike),
        )

        results.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "price": round(price, 2),
                "sector": sector,
                # Default to today - db.save_candidates overrides this with the
                # ORIGINAL first-seen date if this ticker was already pending
                # from an earlier, still-undecided run (see its docstring).
                "first_seen_at": today.isoformat(),
                "expiry": row["expiry"],
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid_price": round(mid, 4),
                "collateral": collateral,
                "premium_total": round(premium_total, 2),
                "annualized_yield_pct": round(annualized_yield_pct, 2),
                "prob_otm_pct": round(greeks.prob_otm * 100.0, 2),
                "delta": round(greeks.delta, 4),
                "open_interest": open_interest,
                "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
                "days_to_expiry": days_to_expiry,
                "earnings_before_expiry": int(earnings_before_expiry),
                "analyst_recommendation": (analyst_snapshot or {}).get("recommendation_key"),
                "analyst_target_mean_price": (analyst_snapshot or {}).get("target_mean_price"),
                "pe_ratio": (analyst_snapshot or {}).get("pe_ratio"),
                "week52_low": (analyst_snapshot or {}).get("week52_low"),
                "week52_high": (analyst_snapshot or {}).get("week52_high"),
                "earnings_anomaly_type": anomaly_type,
                "composite_score": round(composite, 4),
                "risk_score": risk_score,
                "risk_label": risk_label,
                "rsi_14": (technicals or {}).get("rsi_14"),
                "trend": (technicals or {}).get("trend"),
                "macd_bullish": _nullable_int((technicals or {}).get("macd_bullish")),
                "rank": None,
            }
        )

    return results


def diversify_by_sector(ranked_candidates: list[dict], top_n: int, max_per_sector: int = 2) -> list[dict]:
    """
    Greedy sector cap, shared by every strategy's final ranking step (see
    rank_top_candidates below and engine.strategies.dip_buy.generate) -
    without this, "top N by composite_score" can easily hand back N names
    that are all the same sector (e.g. all semiconductors), which isn't
    actually diversified: they'll tend to move together, so N "different"
    picks don't reduce risk the way N independent ones would.

    `ranked_candidates` must already be sorted best-first. The cap is
    strict, with no backfill into an over-represented sector: if the
    qualifying pool is thin on sector variety (in the extreme, all one
    sector), this deliberately returns fewer than top_n rather than
    diluting the result with more of the same - exactly the "all 5 will
    move together" complaint this exists to fix. Same philosophy as
    min_composite_score elsewhere here: returning less than asked for is
    the correct behavior when there isn't enough of what you actually want,
    not something to paper over.
    """
    sector_counts: dict[str, int] = {}
    picked: list[dict] = []
    for c in ranked_candidates:
        sector = c.get("sector")
        if sector and sector_counts.get(sector, 0) >= max_per_sector:
            continue
        picked.append(c)
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picked) >= top_n:
            break
    return picked


def rank_top_candidates(all_candidates: list[dict], top_n: int = 5, max_per_sector: int = 2) -> list[dict]:
    """
    Pick one (the best) contract per ticker, then rank tickers by composite
    score, sector-capped (see diversify_by_sector) so the result is a
    genuinely diversified set of names, not just the top N scores which
    could all be clustered in one hot sector.
    """
    best_per_ticker: dict[str, dict] = {}
    for c in all_candidates:
        t = c["ticker"]
        if t not in best_per_ticker or c["composite_score"] > best_per_ticker[t]["composite_score"]:
            best_per_ticker[t] = c

    ranked = sorted(best_per_ticker.values(), key=lambda c: c["composite_score"], reverse=True)
    top = diversify_by_sector(ranked, top_n, max_per_sector)
    for i, c in enumerate(top, start=1):
        c["rank"] = i
    return top


# Risk profile selected in the dashboard -> max acceptable risk_score (1-10 scale, see _risk_score).
RISK_PROFILE_MAX_SCORE = {"conservative": 3, "balanced": 6, "aggressive": 10}


def select_top_candidates(all_candidates: list[dict], cfg: dict) -> list[dict]:
    """
    A ranked watchlist, not a sized portfolio: drops any candidate riskier
    than the selected risk_profile allows or scoring below min_composite_score
    (a soft quality floor on top of evaluate_candidates' hard filters - a
    contract that only barely clears those isn't necessarily worth
    surfacing), then takes the top `max_candidates` by composite_score (one
    contract per ticker - see rank_top_candidates). No capital/collateral
    allocation here - size each position's collateral yourself; `collateral`
    is still reported on every candidate so you have the number to work with.
    Returning fewer than max_candidates - including zero - is expected and
    correct when nothing clears the bar; this never pads the list.
    """
    max_risk_score = RISK_PROFILE_MAX_SCORE.get(cfg.get("risk_profile", "balanced"), 6)
    max_candidates = cfg.get("max_candidates", 10)
    min_composite_score = cfg.get("min_composite_score", 0.0)
    max_per_sector = cfg.get("max_per_sector", 2)

    eligible = [
        c for c in all_candidates
        if c["risk_score"] <= max_risk_score and c["composite_score"] >= min_composite_score
    ]
    return rank_top_candidates(eligible, top_n=max_candidates, max_per_sector=max_per_sector)
