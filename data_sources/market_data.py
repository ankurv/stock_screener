"""
Price, option chain, and earnings-date fetching via yfinance (an unofficial
but actively maintained wrapper around Yahoo Finance's public endpoints -
the most practical free source for options data; no official free API with
Greeks/IV exists). Everything here writes through to the SQLite cache so a
transient Yahoo hiccup on one ticker doesn't take down the whole run, and
so the dashboard can serve the last-known-good snapshot.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from common import get_logger, effective_ttl_minutes

log = get_logger("market_data")

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - allows the rest of the codebase to be imported/tested without yfinance installed
    yf = None


def _week_end_friday(today: date) -> date:
    """Friday of the current trading week; if today is Sat/Sun, next week's Friday."""
    weekday = today.weekday()  # Mon=0 .. Sun=6
    if weekday <= 4:
        return today + timedelta(days=(4 - weekday))
    return today + timedelta(days=(4 - weekday + 7))


def pick_nearest_weekly_expiry(available_expiries: list[str], today: Optional[date] = None) -> Optional[str]:
    """
    From a ticker's listed option expirations, pick the one that falls
    within "this week" (today through the upcoming Friday inclusive).
    Returns the ISO date string, or None if nothing qualifies.
    """
    today = today or date.today()
    week_end = _week_end_friday(today)
    candidates = []
    for e in available_expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= d <= week_end:
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates).isoformat()  # the furthest-out date in-window, i.e. that week's Friday if listed


def fetch_price(ticker: str) -> Optional[float]:
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        price = t.fast_info.get("lastPrice") if hasattr(t.fast_info, "get") else t.fast_info["lastPrice"]
        if price:
            return float(price)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: fast_info price failed (%s), falling back to history()", ticker, exc)
    try:
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: price fetch failed entirely: %s", ticker, exc)
    return None


def _fast_info_get(info, key: str, default=None):
    """fast_info is dict-like in some yfinance versions, attribute-only FastInfo in others."""
    try:
        return info.get(key, default) if hasattr(info, "get") else info[key]
    except (KeyError, TypeError):
        return default


def fetch_quote(ticker: str) -> Optional[dict]:
    """
    Lightweight last-price + day-over-day % change via fast_info only (no
    options chains, no multi-month history) - cheap enough to call for a
    handful of tickers every 30s for the dashboard's live market pulse.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info
        price = _fast_info_get(info, "lastPrice")
        prev_close = _fast_info_get(info, "previousClose")
        if not price or not prev_close:
            return None
        return {
            "ticker": ticker,
            "price": round(float(price), 2),
            "change_pct": round((float(price) - float(prev_close)) / float(prev_close) * 100.0, 2),
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: live quote fetch failed (%s)", ticker, exc)
        return None


def fetch_market_cap_and_volume(ticker: str) -> Optional[dict]:
    """
    Market cap + average daily share volume via yfinance's .info (the same
    source already used for the analyst snapshot) - used to build the
    "liquid large caps" pre-filter. Deliberately NOT fast_info here: its
    `marketCap` *looks* cheap but isn't - in current yfinance it lazily
    triggers a separate shares-outstanding history request per ticker,
    which is slower and far less predictable (it can hang) than one bounded
    .info call.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: market cap/volume .info fetch failed (%s)", ticker, exc)
        return None
    market_cap = info.get("marketCap")
    avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")
    if not market_cap:
        return None
    return {
        "ticker": ticker,
        "market_cap": float(market_cap),
        "avg_volume": float(avg_volume) if avg_volume else None,
    }


def fetch_sparkline(ticker: str, period: str = "1d", interval: str = "15m") -> list[float]:
    """
    A short series of recent closes for the simple trend line shown next to
    each live market-pulse tile - intentionally coarse (a handful of bars,
    not a real chart) since it's just "which way has it been drifting
    today", not a trading chart. Returns [] on any failure (missing ticker,
    market closed with no bars yet, etc.) rather than raising, since this is
    a nice-to-have next to the quote, not something worth failing the tile over.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: sparkline history fetch failed (%s)", ticker, exc)
        return []
    if hist is None or hist.empty:
        return []
    closes = hist["Close"].dropna()
    return [round(float(c), 2) for c in closes.tolist()]


def refresh_sparkline(db, cfg: dict, ticker: str) -> list[float]:
    """Refresh the cached sparkline for one ticker if stale; always returns the current (possibly cached) series."""
    pulse_cfg = cfg.get("market_pulse", {})
    ttl = cfg["cache_ttl_minutes"].get("sparkline", 10)
    cached = db.get_sparkline(ticker, ttl_minutes=ttl)
    if cached is not None:
        return cached
    try:
        values = fetch_sparkline(
            ticker,
            period=pulse_cfg.get("sparkline_period", "1d"),
            interval=pulse_cfg.get("sparkline_interval", "15m"),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Sparkline refresh failed for %s: %s", ticker, exc)
        return db.get_sparkline(ticker) or []
    db.set_sparkline(ticker, values)
    return values


# Comfortably covers every current consumer's lookback (6mo technicals is
# the largest) - the one-time "cold start" window for a ticker that's never
# had its history cached before. Every refresh after that is incremental
# (see refresh_price_history), so this size only matters once per ticker.
HISTORY_WINDOW_DAYS = 400


def fetch_daily_history(ticker: str, start: date, end: date) -> Optional[list[dict]]:
    """Pure yfinance pull - ascending [{"date": iso, "close": float}, ...], or None on failure/empty."""
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        hist = t.history(start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: daily history fetch failed: %s", ticker, exc)
        return None
    if hist is None or hist.empty:
        return None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None
    return [{"date": d.date().isoformat(), "close": float(c)} for d, c in zip(closes.index, closes)]


def refresh_price_history(
    db, ticker: str, ttl_minutes: Optional[float] = None, required_through: Optional[date] = None
) -> list[dict]:
    """
    The shared "common data" every signal below reads from instead of each
    independently re-pulling overlapping history from Yahoo (technicals,
    trend/drawdown %, post-earnings windows, outcome resolution, market
    mood, sector performance all call this with the same underlying
    ticker). Returns the full cached daily-close series (ascending), after
    refreshing it first if stale.

    The *first* fetch for a ticker pulls the full HISTORY_WINDOW_DAYS
    window; every fetch after that only pulls from (last cached date + 1)
    through today and appends - fetching "the newer data" instead of
    re-pulling the same ~400 days on every TTL expiry.

    `required_through`, if given, forces an incremental fetch (regardless
    of ttl_minutes) when the cached series doesn't yet reach that date -
    needed by outcome resolution, which asks for a specific past date that
    must be present, not just "whatever's fresh enough." A failed/empty
    fetch leaves the freshness marker unset so the next call retries
    rather than silently waiting out the TTL on bad data.
    """
    cached = db.get_price_history(ticker)
    last_cached_date = date.fromisoformat(cached[-1]["date"]) if cached else None
    cache_key = f"price_history_fetched_at:{ticker}"
    is_stale = db.cache_get(cache_key, ttl_minutes=ttl_minutes) is None
    needs_more = required_through is not None and (last_cached_date is None or last_cached_date < required_through)

    if is_stale or needs_more:
        start = (last_cached_date + timedelta(days=1)) if last_cached_date else (date.today() - timedelta(days=HISTORY_WINDOW_DAYS))
        end = date.today() + timedelta(days=1)
        fetched_ok = start >= end  # nothing new to ask for (already up to date for today) counts as a successful check
        if start < end:
            new_rows = fetch_daily_history(ticker, start, end)
            if new_rows:
                db.upsert_price_history(ticker, new_rows)
                cached = db.get_price_history(ticker)
                fetched_ok = True
        if fetched_ok:
            db.cache_set(cache_key, True)
    return cached


def fetch_close_price_on_or_after(db, ticker: str, target_date: str, ttl_minutes: Optional[float] = None) -> Optional[float]:
    """
    Underlying close price on the given date, or the next trading day if it
    fell on a weekend/holiday - used by engine.outcomes to score a past
    recommendation once it's due. Returns None if that date isn't in the
    market's history yet, or the ticker has no data for it.
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    closes = refresh_price_history(db, ticker, ttl_minutes=ttl_minutes, required_through=target)
    for row in closes:
        if date.fromisoformat(row["date"]) >= target:
            return row["close"]
    return None


def fetch_recent_trend_pct(db, ticker: str, lookback_days: int = 5, ttl_minutes: Optional[float] = None) -> Optional[float]:
    """% price change over the last `lookback_days` *trading* days - the
    broad-market trend signal behind fetch_market_mood() below."""
    closes = refresh_price_history(db, ticker, ttl_minutes=ttl_minutes)
    if len(closes) < 2:
        return None
    window = closes[-(lookback_days + 1):] if len(closes) > lookback_days else closes
    start, end = window[0]["close"], window[-1]["close"]
    if start <= 0:
        return None
    return (end - start) / start * 100.0


def classify_market_mood(
    trend_pct: Optional[float], bearish_threshold_pct: float = -1.5, bullish_threshold_pct: float = 1.5
) -> str:
    if trend_pct is None:
        return "unknown"
    if trend_pct <= bearish_threshold_pct:
        return "bearish"
    if trend_pct >= bullish_threshold_pct:
        return "bullish"
    return "neutral"


def blend_live_mood(
    structural_mood: dict,
    live_quotes: list[dict],
    bearish_threshold_pct: float = -1.5,
    bullish_threshold_pct: float = 1.5,
) -> dict:
    """
    Folds the live market-pulse ETF quotes into the structural (multi-day
    SPY trend) mood read, for display only - this is pure/no I/O, reusing
    whatever was already fetched for the pulse tiles plus the already-cached
    structural trend, so calling it every 30s costs nothing extra.

    Equity breadth (SPY/QQQ/IWM's live day-over-day change) carries most of
    the weight; gold/long-bond moves (GLD/TLT) are a safe-haven confirmation
    nudge - equities selling off while havens rally is a stronger risk-off
    tell than either alone. Oil (USO) is left out of the math on purpose: it
    cuts both ways (demand-driven vs. inflation-driven) and isn't a reliable
    risk-on/off signal by itself, though it's still shown in the pulse tiles.

    The structural trend_pct (used for risk_score during a pipeline run) is
    left untouched by this - only the dict returned here, for the dashboard,
    gets the live-adjusted mood/combined_pct.
    """
    result = dict(structural_mood)
    quotes_by_ticker = {q["ticker"]: q["change_pct"] for q in live_quotes if q}
    equity_pcts = [quotes_by_ticker[t] for t in ("SPY", "QQQ", "IWM") if t in quotes_by_ticker]
    if not equity_pcts:
        return result

    live_breadth_pct = sum(equity_pcts) / len(equity_pcts)
    trend_pct = structural_mood.get("trend_pct")
    combined_pct = (0.6 * trend_pct + 0.4 * live_breadth_pct) if trend_pct is not None else live_breadth_pct

    haven_pcts = [quotes_by_ticker[t] for t in ("GLD", "TLT") if t in quotes_by_ticker]
    if haven_pcts:
        haven_avg = sum(haven_pcts) / len(haven_pcts)
        if live_breadth_pct < 0 and haven_avg > 0:
            combined_pct -= min(haven_avg, 1.0) * 0.3
        elif live_breadth_pct > 0 and haven_avg < 0:
            combined_pct += min(-haven_avg, 1.0) * 0.3

    result["live_breadth_pct"] = round(live_breadth_pct, 2)
    result["combined_pct"] = round(combined_pct, 2)
    result["mood"] = classify_market_mood(combined_pct, bearish_threshold_pct, bullish_threshold_pct)
    return result


def fetch_market_mood(
    db,
    benchmark: str = "SPY",
    lookback_days: int = 5,
    bearish_threshold_pct: float = -1.5,
    bullish_threshold_pct: float = 1.5,
    ttl_minutes: Optional[float] = None,
) -> dict:
    """
    Simple broad-market regime read, used both to show "is the market
    bearish/bullish right now" clearly in the dashboard, and to nudge every
    candidate's risk_score (see engine.signals.market_mood_risk_component) -
    a broad selloff raises assignment/drawdown risk even for low-vol names.
    """
    trend_pct = fetch_recent_trend_pct(db, benchmark, lookback_days, ttl_minutes=ttl_minutes)
    mood = classify_market_mood(trend_pct, bearish_threshold_pct, bullish_threshold_pct)
    return {
        "benchmark": benchmark,
        "lookback_days": lookback_days,
        "trend_pct": round(trend_pct, 2) if trend_pct is not None else None,
        "mood": mood,
    }


def refresh_market_mood(db, cfg: dict) -> dict:
    """Refresh the cached broad-market mood if stale; always returns the current (possibly cached) value."""
    mood_cfg = cfg.get("market_mood", {})
    ttl = cfg["cache_ttl_minutes"].get("market_mood", 60)
    history_ttl = effective_ttl_minutes(cfg["cache_ttl_minutes"].get("price_history", 240))
    cached = db.get_market_mood(ttl_minutes=ttl)
    if cached is not None:
        return cached
    try:
        data = fetch_market_mood(
            db,
            benchmark=mood_cfg.get("benchmark", "SPY"),
            lookback_days=mood_cfg.get("lookback_days", 5),
            bearish_threshold_pct=mood_cfg.get("bearish_threshold_pct", -1.5),
            bullish_threshold_pct=mood_cfg.get("bullish_threshold_pct", 1.5),
            ttl_minutes=history_ttl,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Market mood refresh failed: %s", exc)
        return db.get_market_mood() or {"benchmark": mood_cfg.get("benchmark", "SPY"), "mood": "unknown", "trend_pct": None}
    db.set_market_mood(data)
    log.info("Market mood: %s", data)
    return data


def fetch_sector_performance(db, sector_tickers: dict[str, str], lookback_days: int = 15, ttl_minutes: Optional[float] = None) -> list[dict]:
    """
    % price change over the last `lookback_days` trading days for each SPDR
    Select Sector ETF - the standard, cheap proxy for "how is tech/financials/
    energy etc. doing" (one ticker per sector, not an average over every S&P
    500 constituent). Reuses fetch_recent_trend_pct, the same shared
    price-history cache already used for the market mood's benchmark trend.
    Sectors whose fetch fails are just omitted, not retried inline.
    """
    results = []
    for ticker, label in sector_tickers.items():
        trend_pct = fetch_recent_trend_pct(db, ticker, lookback_days, ttl_minutes=ttl_minutes)
        if trend_pct is not None:
            results.append({"ticker": ticker, "sector": label, "trend_pct": round(trend_pct, 2)})
    results.sort(key=lambda r: r["trend_pct"], reverse=True)
    return results


def refresh_sector_performance(db, cfg: dict) -> list[dict]:
    """Refresh the cached sector-performance ranking if stale; always returns the current (possibly cached) value."""
    sect_cfg = cfg.get("sector_performance", {})
    ttl = cfg["cache_ttl_minutes"].get("sector_performance", 240)
    history_ttl = effective_ttl_minutes(cfg["cache_ttl_minutes"].get("price_history", 240))
    cached = db.get_sector_performance(ttl_minutes=ttl)
    if cached is not None:
        return cached
    try:
        data = fetch_sector_performance(
            db, sect_cfg.get("tickers", {}), sect_cfg.get("lookback_days", 15), ttl_minutes=history_ttl
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Sector performance refresh failed: %s", exc)
        return db.get_sector_performance() or []
    db.set_sector_performance(data)
    log.info("Sector performance: %s", [(r["sector"], r["trend_pct"]) for r in data])
    return data


def fetch_technical_indicators(db, ticker: str, ttl_minutes: Optional[float] = None) -> Optional[dict]:
    """
    A handful of the most commonly-watched technical indicators, computed
    from the shared daily-close history (see refresh_price_history):
    RSI(14), the SMA20/SMA50 trend (golden/death cross), and MACD(12,26,9).
    These feed into risk_score (see engine.signals.technical_risk_component -
    overbought RSI and a death cross/bearish MACD nudge risk up) and are
    shown on the dashboard for context; they're a secondary signal, not a
    replacement for the yield/liquidity/analyst-driven composite score.
    """
    rows = refresh_price_history(db, ticker, ttl_minutes=ttl_minutes)
    if len(rows) < 50:
        return None  # not enough history for a meaningful SMA50/RSI14
    closes = pd.Series([r["close"] for r in rows], index=pd.to_datetime([r["date"] for r in rows]))

    delta = closes.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi = rsi_series.iloc[-1]

    sma20 = closes.rolling(20).mean().iloc[-1]
    sma50 = closes.rolling(50).mean().iloc[-1]
    trend = None
    if pd.notna(sma20) and pd.notna(sma50):
        if sma20 > sma50:
            trend = "golden_cross"
        elif sma20 < sma50:
            trend = "death_cross"
        else:
            trend = "flat"

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line).iloc[-1]

    return {
        "rsi_14": round(float(rsi), 1) if pd.notna(rsi) else None,
        "sma20": round(float(sma20), 2) if pd.notna(sma20) else None,
        "sma50": round(float(sma50), 2) if pd.notna(sma50) else None,
        "trend": trend,
        "macd_hist": round(float(macd_hist), 4) if pd.notna(macd_hist) else None,
        "macd_bullish": bool(pd.notna(macd_hist) and macd_hist > 0),
    }


def fetch_next_earnings_date(ticker: str) -> tuple[Optional[str], bool]:
    """Returns (iso_date_or_None, is_estimate)."""
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        edf = t.get_earnings_dates(limit=8)
        if edf is not None and not edf.empty:
            idx = edf.index
            now_ts = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
            future = edf[idx >= now_ts]
            if not future.empty:
                return future.index.min().date().isoformat(), True
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: get_earnings_dates failed (%s)", ticker, exc)
    # Fallback: some tickers only expose t.calendar
    try:
        cal = t.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            d = cal["Earnings Date"][0]
            return (d.isoformat() if hasattr(d, "isoformat") else str(d)), True
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: calendar fallback failed (%s)", ticker, exc)
    return None, True


def fetch_analyst_snapshot(ticker: str) -> Optional[dict]:
    """
    Analyst consensus rating + price target, plus a couple of basic
    valuation/range fundamentals (P/E, 52-week high/low) read from the same
    `.info` call rather than paying for a second network round-trip just
    for those - all three come back for most tickers even without analyst
    coverage (ETFs have no P/E but do have a 52-week range), so this only
    returns None when there's truly nothing usable at all.
    recommendation_mean convention: 1.0 = Strong Buy ... 5.0 = Sell.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: .info fetch failed (%s)", ticker, exc)
        return None

    rec_mean = info.get("recommendationMean")
    pe_ratio = info.get("trailingPE")
    week52_low = info.get("fiftyTwoWeekLow")
    week52_high = info.get("fiftyTwoWeekHigh")
    if rec_mean is None and not info.get("targetMeanPrice") and pe_ratio is None and week52_low is None:
        return None  # nothing usable at all

    return {
        "recommendation_key": info.get("recommendationKey"),
        "recommendation_mean": rec_mean,
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "pe_ratio": pe_ratio,
        "week52_low": week52_low,
        "week52_high": week52_high,
    }


def classify_earnings_anomaly(surprise_pct: float, reaction_pct: float, threshold_pct: float = 3.0) -> str:
    """Pure classification, split out from fetch_post_earnings_reaction for easy unit testing."""
    if surprise_pct > 0 and reaction_pct <= -threshold_pct:
        return "beat_but_sold_off"
    if surprise_pct < 0 and reaction_pct >= threshold_pct:
        return "miss_but_rallied"
    return "normal"


def fetch_post_earnings_reaction(db, ticker: str, anomaly_threshold_pct: float = 3.0, ttl_minutes: Optional[float] = None) -> Optional[dict]:
    """
    Looks at the most recently *completed* earnings report (not the
    upcoming estimated one) and compares the EPS surprise direction to the
    stock's price reaction. Flags it when they disagree in a notable way:
    beat-but-sold-off, or miss-but-rallied.

    Returns None if we can't find a past reported quarter (e.g. recent IPO)
    or can't get clean price history around it. The EPS surprise still
    needs its own small yfinance call (not price history); the price side
    comes from the shared cache (see refresh_price_history).
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        edf = t.get_earnings_dates(limit=12)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: get_earnings_dates failed (%s)", ticker, exc)
        return None
    if edf is None or edf.empty or "Reported EPS" not in edf.columns:
        return None

    past = edf[edf["Reported EPS"].notna()]
    if past.empty:
        return None
    past = past.sort_index(ascending=False)
    last_row = past.iloc[0]
    earnings_ts = past.index[0]
    earnings_date = earnings_ts.date()

    surprise_pct = last_row.get("Surprise(%)")
    try:
        surprise_pct = float(surprise_pct) * 100 if abs(float(surprise_pct)) < 1 else float(surprise_pct)
    except (TypeError, ValueError):
        surprise_pct = None
    if surprise_pct is None:
        return None

    dated_closes = [(date.fromisoformat(r["date"]), r["close"]) for r in refresh_price_history(db, ticker, ttl_minutes=ttl_minutes)]
    pre_candidates = [dc for dc in dated_closes if dc[0] < earnings_date]
    post_candidates = [dc for dc in dated_closes if dc[0] > earnings_date]
    if not pre_candidates or not post_candidates:
        return None
    pre_close = pre_candidates[-1][1]
    # take the close ~2 trading days after, to let the market fully digest a
    # before-open or after-close report either way
    post_close = post_candidates[min(1, len(post_candidates) - 1)][1]

    if pre_close <= 0:
        return None
    reaction_pct = (post_close - pre_close) / pre_close * 100.0
    anomaly_type = classify_earnings_anomaly(surprise_pct, reaction_pct, anomaly_threshold_pct)

    return {
        "earnings_date": earnings_date.isoformat(),
        "surprise_pct": round(surprise_pct, 2),
        "reaction_pct": round(reaction_pct, 2),
        "anomaly_type": anomaly_type,
    }


def fetch_post_earnings_correction(db, ticker: str, earnings_date: date, ttl_minutes: Optional[float] = None) -> Optional[dict]:
    """
    Behind engine.strategies.three_day_rule's "3-day rule": a stock often
    keeps correcting for a few trading days right after it reports, before
    stabilizing - the day the correction window ends is often a buyable
    overreaction rather than the start of a real decline. Looks at the
    trading days between `earnings_date` (a past, already-reported quarter -
    same date fetch_post_earnings_reaction above resolves) and today, and
    returns the 3 most recent post-earnings closes as the "correction
    window", the close right before them (`anchor_close`), and the EPS
    surprise for that same report. Returns None if fewer than 3 trading
    days have passed since earnings yet, or there's no usable history.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    try:
        edf = t.get_earnings_dates(limit=12)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: get_earnings_dates failed (%s)", ticker, exc)
        return None
    surprise_pct = None
    if edf is not None and not edf.empty and "Reported EPS" in edf.columns:
        matching = edf[edf.index.date == earnings_date]
        if not matching.empty:
            raw = matching.iloc[0].get("Surprise(%)")
            try:
                surprise_pct = float(raw) * 100 if abs(float(raw)) < 1 else float(raw)
            except (TypeError, ValueError):
                surprise_pct = None

    dated_closes = [(date.fromisoformat(r["date"]), r["close"]) for r in refresh_price_history(db, ticker, ttl_minutes=ttl_minutes)]
    pre = [dc for dc in dated_closes if dc[0] <= earnings_date]
    post = [dc for dc in dated_closes if dc[0] > earnings_date]
    if len(post) < 3 or not pre:
        return None  # not enough post-earnings trading days yet (or no pre-earnings close)

    window = post[-3:]
    anchor_close = post[-4][1] if len(post) >= 4 else pre[-1][1]
    if anchor_close <= 0:
        return None
    down_day_count = sum(
        1 for i, (_, close) in enumerate(window)
        if close < (window[i - 1][1] if i > 0 else anchor_close)
    )
    window_last_close = window[-1][1]
    drawdown_pct = (window_last_close - anchor_close) / anchor_close * 100.0

    return {
        "days_since_earnings": (date.today() - earnings_date).days,
        "window_start_date": window[0][0].isoformat(),
        "window_end_date": window[-1][0].isoformat(),
        "anchor_close": round(anchor_close, 2),
        "window_last_close": round(window_last_close, 2),
        "down_day_count": down_day_count,
        "drawdown_pct": round(drawdown_pct, 2),
        "surprise_pct": round(surprise_pct, 2) if surprise_pct is not None else None,
    }


def fetch_weekly_put_chain(ticker: str) -> tuple[Optional[str], list[dict]]:
    """Returns (expiry_iso_or_None, list of put-contract dicts) for this week's expiry."""
    if yf is None:
        raise RuntimeError("yfinance is not installed - run: pip install -r requirements.txt")
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        return None, []
    expiry = pick_nearest_weekly_expiry(list(expiries))
    if not expiry:
        return None, []
    chain = t.option_chain(expiry)
    puts = chain.puts
    rows = []
    for _, r in puts.iterrows():
        rows.append(
            {
                "expiry": expiry,
                "strike": float(r["strike"]),
                "bid": float(r["bid"]) if pd.notna(r.get("bid")) else None,
                "ask": float(r["ask"]) if pd.notna(r.get("ask")) else None,
                "last_price": float(r["lastPrice"]) if pd.notna(r.get("lastPrice")) else None,
                "implied_vol": float(r["impliedVolatility"]) if pd.notna(r.get("impliedVolatility")) else None,
                "open_interest": int(r["openInterest"]) if pd.notna(r.get("openInterest")) else 0,
                "volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
                "in_the_money": bool(r.get("inTheMoney", False)),
            }
        )
    return expiry, rows


def _refresh_one_ticker(
    db, ticker: str, is_etf: bool, price_ttl: float, chain_ttl: float, history_ttl: float, ttl: dict,
    anomaly_threshold: float, needs_chains: bool,
) -> dict:
    """
    Everything one ticker needs from this module, run inside a single
    worker thread (see refresh_market_data's ThreadPoolExecutor below - same
    concurrency pattern already proven for the universe-filter build:
    db.conn() opens its own sqlite3 connection per call, so concurrent
    writes from multiple threads are safe under WAL mode). Each sub-fetch
    is still independently TTL-gated, so a warm cache does zero network
    work here regardless of how many threads are in flight.
    """
    result = {
        "priced": 0, "earnings_updated": 0, "chains_updated": 0,
        "analyst_updated": 0, "reaction_updated": 0, "technicals_updated": 0, "error": None,
    }
    try:
        if db.get_price(ticker, ttl_minutes=price_ttl) is None:
            price = fetch_price(ticker)
            if price:
                db.upsert_price(ticker, price)
                result["priced"] = 1

        if not is_etf and db.get_earnings(ticker, ttl_minutes=ttl["earnings_date"]) is None:
            edate, is_est = fetch_next_earnings_date(ticker)
            db.upsert_earnings(ticker, edate, is_est)
            result["earnings_updated"] = 1

        if not is_etf and db.get_analyst_snapshot(ticker, ttl_minutes=ttl.get("analyst_snapshot", 1440)) is None:
            snap = fetch_analyst_snapshot(ticker)
            if snap:
                db.upsert_analyst_snapshot(ticker, snap)
                result["analyst_updated"] = 1

        if not is_etf and db.get_post_earnings_reaction(ticker, ttl_minutes=ttl["earnings_date"]) is None:
            reaction = fetch_post_earnings_reaction(db, ticker, anomaly_threshold_pct=anomaly_threshold, ttl_minutes=history_ttl)
            if reaction:
                db.upsert_post_earnings_reaction(ticker, reaction)
                result["reaction_updated"] = 1

        if db.get_technicals(ticker, ttl_minutes=ttl.get("technicals", 240)) is None:
            tech = fetch_technical_indicators(db, ticker, ttl_minutes=history_ttl)
            if tech:
                db.set_technicals(ticker, tech)
                result["technicals_updated"] = 1

        if needs_chains:
            cached_chain = db.get_put_snapshots(ticker, ttl_minutes=chain_ttl)
            if not cached_chain:
                expiry, rows = fetch_weekly_put_chain(ticker)
                if expiry and rows:
                    db.replace_put_snapshots(ticker, rows)
                    result["chains_updated"] = 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{ticker}: {exc}"
    return result


def refresh_market_data(
    db, cfg: dict, universe: list[dict], needs_chains: bool = True,
    max_workers: Optional[int] = None, on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    For each ticker in the universe, refresh price / earnings / weekly put
    chain if the cached copy is older than its configured TTL. Returns a
    summary dict with counts, for logging/pipeline_run bookkeeping.

    Fetches tickers concurrently (same ThreadPoolExecutor pattern already
    used by data_sources.universe.build_universe_filter, which cut a
    503-ticker pass from minutes to ~27s) instead of one at a time with a
    fixed sleep between each - a cold cache for the full universe used to
    take several minutes serialized; concurrency is what actually fixes
    that, not just skipping steps a given run doesn't need. on_progress, if
    given, is called periodically with how many tickers are done so a
    multi-minute cold-cache run still shows visible movement instead of
    sitting on one static message.

    needs_chains=False skips the option-chain fetch entirely - by far the
    slowest of the six steps here, and only cash-secured-put actually reads
    it (see engine/strategies/cash_secured_put.py's NEEDS_OPTION_CHAINS).
    run_pipeline.run() passes False when no strategy in scope for this run
    needs it (e.g. a dip-buy-only "Generate" click), so that run isn't
    wasted fetching data nothing in it will use.
    """
    ttl = cfg["cache_ttl_minutes"]
    anomaly_threshold = cfg.get("post_earnings_anomaly_threshold_pct", 3.0)
    # Price/options chain can't move while the market's shut, so stretch
    # those two TTLs to cover the gap until next open on evenings/weekends -
    # computed once per call, not per ticker, since "now" barely moves
    # across a single refresh pass.
    price_ttl = effective_ttl_minutes(ttl["price"])
    chain_ttl = effective_ttl_minutes(ttl["options_chain"])
    history_ttl = effective_ttl_minutes(ttl.get("price_history", 240))
    max_workers = max_workers or cfg.get("universe_filter", {}).get("max_workers", 8)
    summary = {
        "priced": 0, "earnings_updated": 0, "chains_updated": 0, "chains_skipped": not needs_chains,
        "analyst_updated": 0, "reaction_updated": 0, "technicals_updated": 0, "errors": [],
    }

    total = len(universe)
    completed = 0
    progress_every = max(10, total // 20)  # ~20 progress updates across the run, never more chatty than every 10
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {
            pool.submit(
                _refresh_one_ticker, db, row["ticker"], row.get("sector") == "ETF",
                price_ttl, chain_ttl, history_ttl, ttl, anomaly_threshold, needs_chains,
            ): row["ticker"]
            for row in universe
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                r = future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("Market data refresh failed for %s: %s", ticker, exc)
                summary["errors"].append(f"{ticker}: {exc}")
                continue
            if r["error"]:
                log.warning("Market data refresh failed for %s: %s", ticker, r["error"])
                summary["errors"].append(r["error"])
            summary["priced"] += r["priced"]
            summary["earnings_updated"] += r["earnings_updated"]
            summary["analyst_updated"] += r["analyst_updated"]
            summary["reaction_updated"] += r["reaction_updated"]
            summary["technicals_updated"] += r["technicals_updated"]
            summary["chains_updated"] += r["chains_updated"]
            if on_progress and (completed % progress_every == 0 or completed == total):
                on_progress(f"Refreshing prices{' & option chains' if needs_chains else ''}... ({completed}/{total} tickers)")

    log.info("Market data refresh summary: %s", summary)
    return summary
