"""
Black-Scholes helpers for estimating a put's delta and probability of
expiring out-of-the-money (OTM), using only the Python standard library
(`math.erf` for the normal CDF - no scipy dependency).

These are *risk-neutral, model-based estimates* driven by Yahoo Finance's
reported implied volatility. They are a standard, widely-used approximation
in retail options tools, not a guarantee - IV quotes for thin weekly
contracts can be noisy, and this ignores dividends and American-style early
assignment. Treat probability-OTM as "roughly how the market is pricing
this," not a precise probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class PutGreeks:
    delta: float          # negative, standard put delta convention
    prob_otm: float        # risk-neutral probability stock stays >= strike at expiry (0-1)
    prob_itm: float         # 1 - prob_otm


def put_greeks(
    spot: float,
    strike: float,
    days_to_expiry: float,
    implied_vol: float,
    risk_free_rate: float = 0.045,
) -> PutGreeks:
    """
    spot: current underlying price
    strike: option strike
    days_to_expiry: calendar days until expiration (fractional ok)
    implied_vol: annualized IV as a decimal (e.g. 0.35 for 35%)
    risk_free_rate: annualized, as a decimal
    """
    if spot <= 0 or strike <= 0:
        return PutGreeks(delta=0.0, prob_otm=0.5, prob_itm=0.5)

    t_years = max(days_to_expiry, 0.0) / 365.0

    # Degenerate cases: no time left or no vol quoted -> fall back to a hard
    # ITM/OTM call based on spot vs strike rather than dividing by zero.
    if t_years <= 0 or implied_vol is None or implied_vol <= 0:
        is_otm = spot >= strike
        return PutGreeks(delta=(0.0 if is_otm else -1.0), prob_otm=(1.0 if is_otm else 0.0),
                          prob_itm=(0.0 if is_otm else 1.0))

    sigma = implied_vol
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    prob_otm = norm_cdf(d2)          # P(S_T > K) under risk-neutral measure -> put expires worthless
    delta = norm_cdf(d1) - 1.0        # standard Black-Scholes put delta (<=0)

    prob_otm = min(max(prob_otm, 0.0), 1.0)
    return PutGreeks(delta=delta, prob_otm=prob_otm, prob_itm=1.0 - prob_otm)
