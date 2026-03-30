"""
data/indicators.py
------------------
Pure-python technical indicator calculations from OHLCV data.
Uses only pandas for calculations (no ta-lib required).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calculate_rsi(closes: list[float], period: int = 14) -> float:
    """Compute the Relative Strength Index for the most recent candle.

    Returns 50.0 when there is insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0
    try:
        s = pd.Series(closes, dtype=float)
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        return val if not pd.isna(val) else 50.0
    except Exception:
        logger.debug("calculate_rsi failed", exc_info=True)
        return 50.0


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def calculate_ema(closes: list[float], period: int) -> float:
    """Return the most recent EMA value.  Returns 0.0 on error."""
    if len(closes) < period:
        return 0.0
    try:
        s = pd.Series(closes, dtype=float)
        ema = s.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    except Exception:
        logger.debug("calculate_ema failed", exc_info=True)
        return 0.0


def calculate_ema_series(closes: list[float], period: int) -> list[float]:
    """Return the full EMA series.  Returns empty list on error."""
    if len(closes) < period:
        return []
    try:
        s = pd.Series(closes, dtype=float)
        ema = s.ewm(span=period, adjust=False).mean()
        return ema.tolist()
    except Exception:
        logger.debug("calculate_ema_series failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def calculate_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """Return ``{macd, signal, histogram}`` for the most recent candle.

    Returns all-zero dict on error.
    """
    empty = {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    if len(closes) < slow + signal:
        return empty
    try:
        s = pd.Series(closes, dtype=float)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
        }
    except Exception:
        logger.debug("calculate_macd failed", exc_info=True)
        return empty


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """Return the Average True Range.  Returns 0.0 on error."""
    if len(closes) < period + 1:
        return 0.0
    try:
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, dtype=float)
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        val = float(atr.iloc[-1])
        return val if not pd.isna(val) else 0.0
    except Exception:
        logger.debug("calculate_atr failed", exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, float]:
    """Return ``{upper, middle, lower, bandwidth}``.  Returns all-zero on error."""
    empty = {"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0}
    if len(closes) < period:
        return empty
    try:
        s = pd.Series(closes, dtype=float)
        middle = s.rolling(period).mean()
        std = s.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        m = float(middle.iloc[-1])
        u = float(upper.iloc[-1])
        lo = float(lower.iloc[-1])
        bw = (u - lo) / m if m != 0 else 0.0
        return {"upper": u, "middle": m, "lower": lo, "bandwidth": bw}
    except Exception:
        logger.debug("calculate_bollinger_bands failed", exc_info=True)
        return empty


# ---------------------------------------------------------------------------
# Volume SMA
# ---------------------------------------------------------------------------

def calculate_volume_sma(volumes: list[float], period: int = 20) -> float:
    """Return the simple moving average of volume.  Returns 0.0 on error."""
    if len(volumes) < period:
        return 0.0
    try:
        s = pd.Series(volumes, dtype=float)
        return float(s.rolling(period).mean().iloc[-1])
    except Exception:
        logger.debug("calculate_volume_sma failed", exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def calculate_stochastic(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, float]:
    """Return ``{k, d}`` stochastic values.  Returns ``{k: 50.0, d: 50.0}`` on error."""
    empty = {"k": 50.0, "d": 50.0}
    if len(closes) < k_period + d_period:
        return empty
    try:
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, dtype=float)
        lowest_low = df["low"].rolling(k_period).min()
        highest_high = df["high"].rolling(k_period).max()
        denom = highest_high - lowest_low
        k = 100 * (df["close"] - lowest_low) / denom.replace(0, float("nan"))
        d = k.rolling(d_period).mean()
        k_val = float(k.iloc[-1])
        d_val = float(d.iloc[-1])
        return {
            "k": k_val if not pd.isna(k_val) else 50.0,
            "d": d_val if not pd.isna(d_val) else 50.0,
        }
    except Exception:
        logger.debug("calculate_stochastic failed", exc_info=True)
        return empty


# ---------------------------------------------------------------------------
# EMA Crossover Detection
# ---------------------------------------------------------------------------

def detect_ema_crossover(
    closes: list[float],
    fast_period: int,
    slow_period: int,
) -> str:
    """Return ``"BULLISH"``, ``"BEARISH"``, or ``"NONE"``.

    A crossover is detected when the fast EMA crosses the slow EMA
    between the previous candle and the most recent candle.
    """
    if len(closes) < slow_period + 2:
        return "NONE"
    try:
        fast_series = calculate_ema_series(closes, fast_period)
        slow_series = calculate_ema_series(closes, slow_period)
        if len(fast_series) < 2 or len(slow_series) < 2:
            return "NONE"
        # Align to same length
        min_len = min(len(fast_series), len(slow_series))
        fast_series = fast_series[-min_len:]
        slow_series = slow_series[-min_len:]

        prev_fast = fast_series[-2]
        prev_slow = slow_series[-2]
        curr_fast = fast_series[-1]
        curr_slow = slow_series[-1]

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return "BULLISH"
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return "BEARISH"
        return "NONE"
    except Exception:
        logger.debug("detect_ema_crossover failed", exc_info=True)
        return "NONE"


# ---------------------------------------------------------------------------
# RSI Divergence Detection
# ---------------------------------------------------------------------------

def detect_rsi_divergence(
    closes: list[float],
    rsi_values: list[float],
    lookback: int = 20,
) -> str:
    """Return ``"BULLISH_DIV"``, ``"BEARISH_DIV"``, or ``"NONE"``.

    Bullish divergence: price makes a lower low but RSI makes a higher low.
    Bearish divergence: price makes a higher high but RSI makes a lower high.
    """
    if len(closes) < lookback or len(rsi_values) < lookback:
        return "NONE"
    try:
        price_window = closes[-lookback:]
        rsi_window = rsi_values[-lookback:]

        # Use the second-to-last element as the "prior" window (exclude current candle)
        prior_prices = price_window[:-1]
        current_price = price_window[-1]
        current_rsi = rsi_window[-1]

        # Bullish divergence: latest price low lower than prior low, RSI low higher
        prior_price_low = min(prior_prices)
        prior_low_idx = prior_prices.index(prior_price_low)
        prior_rsi_at_low = rsi_window[prior_low_idx]

        if current_price < prior_price_low and current_rsi > prior_rsi_at_low:
            return "BULLISH_DIV"

        # Bearish divergence: latest price high higher than prior high, RSI high lower
        prior_price_high = max(prior_prices)
        prior_high_idx = prior_prices.index(prior_price_high)
        prior_rsi_at_high = rsi_window[prior_high_idx]
        if current_price > prior_price_high and current_rsi < prior_rsi_at_high:
            return "BEARISH_DIV"

        return "NONE"
    except Exception:
        logger.debug("detect_rsi_divergence failed", exc_info=True)
        return "NONE"


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def calculate_adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> dict[str, float]:
    """Return ``{adx, plus_di, minus_di}`` for the most recent candle.

    Uses Wilder's smoothing (EMA with ``com=period-1``).
    Returns all-zero dict when there is insufficient data (need
    ``period * 2 + 1`` candles minimum) or on any error.
    """
    empty: dict[str, float] = {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}
    if len(closes) < period * 2 + 1:
        return empty
    try:
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, dtype=float)
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)

        # True Range
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional movement
        up_move = df["high"] - prev_high
        down_move = prev_low - df["low"]
        plus_dm = pd.Series(0.0, index=df.index, dtype=float)
        minus_dm = pd.Series(0.0, index=df.index, dtype=float)
        plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
        minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)

        # Wilder smoothing
        smoothed_tr = tr.ewm(com=period - 1, min_periods=period).mean()
        smoothed_plus_dm = plus_dm.ewm(com=period - 1, min_periods=period).mean()
        smoothed_minus_dm = minus_dm.ewm(com=period - 1, min_periods=period).mean()

        plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, float("nan"))
        minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, float("nan"))

        dx_denom = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / dx_denom.replace(0, float("nan"))
        dx = dx.fillna(0)

        adx = dx.ewm(com=period - 1, min_periods=period).mean()

        adx_val = float(adx.iloc[-1])
        plus_di_val = float(plus_di.iloc[-1])
        minus_di_val = float(minus_di.iloc[-1])

        return {
            "adx": adx_val if not pd.isna(adx_val) else 0.0,
            "plus_di": plus_di_val if not pd.isna(plus_di_val) else 0.0,
            "minus_di": minus_di_val if not pd.isna(minus_di_val) else 0.0,
        }
    except Exception:
        logger.debug("calculate_adx failed", exc_info=True)
        return empty


# ---------------------------------------------------------------------------
# Support / Resistance Detection
# ---------------------------------------------------------------------------

def detect_support_resistance(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    lookback: int = 50,
    min_touches: int = 2,
) -> dict[str, Any]:
    """Identify the nearest support and resistance levels via swing-point clustering.

    Returns ``{support, resistance, support_touches, resistance_touches}``.
    Returns zeroed dict on error or when no swing points are found.
    """
    empty: dict[str, Any] = {
        "support": 0.0,
        "resistance": 0.0,
        "support_touches": 0,
        "resistance_touches": 0,
    }
    try:
        n = min(lookback, len(lows))
        h = highs[-n:]
        lo = lows[-n:]
        current_price = closes[-1]

        # Find swing lows (exclude first and last)
        swing_lows: list[float] = []
        for i in range(1, n - 1):
            if lo[i] < lo[i - 1] and lo[i] < lo[i + 1]:
                swing_lows.append(lo[i])

        # Find swing highs
        swing_highs: list[float] = []
        for i in range(1, n - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1]:
                swing_highs.append(h[i])

        def _cluster(levels: list[float], tolerance: float = 0.005) -> list[tuple[float, int]]:
            """Group nearby levels within *tolerance* fraction of each other.

            Returns a list of ``(representative_level, touch_count)`` tuples.
            """
            if not levels:
                return []
            sorted_levels = sorted(levels)
            clusters: list[tuple[float, int]] = []
            group = [sorted_levels[0]]
            for lvl in sorted_levels[1:]:
                ref = group[0]
                if ref != 0 and abs(lvl - ref) / ref <= tolerance:
                    group.append(lvl)
                else:
                    clusters.append((sum(group) / len(group), len(group)))
                    group = [lvl]
            clusters.append((sum(group) / len(group), len(group)))
            return clusters

        # Cluster swing lows → support candidates
        support_clusters = _cluster(swing_lows)
        valid_supports = [(lvl, cnt) for lvl, cnt in support_clusters if cnt >= min_touches]

        if valid_supports:
            # Nearest support below current price
            below = [(lvl, cnt) for lvl, cnt in valid_supports if lvl < current_price]
            if below:
                support_lvl, support_touches = max(below, key=lambda x: x[0])
            else:
                support_lvl, support_touches = max(valid_supports, key=lambda x: x[0])
        elif swing_lows:
            support_lvl = min(swing_lows)
            support_touches = 1
        else:
            support_lvl = 0.0
            support_touches = 0

        # Cluster swing highs → resistance candidates
        resistance_clusters = _cluster(swing_highs)
        valid_resistances = [(lvl, cnt) for lvl, cnt in resistance_clusters if cnt >= min_touches]

        if valid_resistances:
            # Nearest resistance above current price
            above = [(lvl, cnt) for lvl, cnt in valid_resistances if lvl > current_price]
            if above:
                resistance_lvl, resistance_touches = min(above, key=lambda x: x[0])
            else:
                resistance_lvl, resistance_touches = min(valid_resistances, key=lambda x: x[0])
        elif swing_highs:
            resistance_lvl = max(swing_highs)
            resistance_touches = 1
        else:
            resistance_lvl = 0.0
            resistance_touches = 0

        return {
            "support": float(support_lvl),
            "resistance": float(resistance_lvl),
            "support_touches": int(support_touches),
            "resistance_touches": int(resistance_touches),
        }
    except Exception:
        logger.debug("detect_support_resistance failed", exc_info=True)
        return empty


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def calculate_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> float:
    """Return the Volume-Weighted Average Price for the provided candles.

    Returns ``closes[-1]`` on error or zero total volume.
    """
    try:
        tp = [(h + lo + c) / 3.0 for h, lo, c in zip(highs, lows, closes)]
        total_vol = sum(volumes)
        if total_vol == 0:
            return closes[-1]
        vwap = sum(t * v for t, v in zip(tp, volumes)) / total_vol
        return float(vwap)
    except Exception:
        logger.debug("calculate_vwap failed", exc_info=True)
        return closes[-1] if closes else 0.0


# ---------------------------------------------------------------------------
# Market Regime Classification
# ---------------------------------------------------------------------------

def classify_market_regime(adx: float, plus_di: float, minus_di: float) -> str:
    """Return a market regime string based on ADX and DI values.

    Returns:
        ``"STRONG_TREND"`` when ADX >= 25 and ``|+DI - -DI| > 10``
        ``"WEAK_TREND"``   when ADX >= 20
        ``"RANGING"``      otherwise
    """
    if adx >= 25 and abs(plus_di - minus_di) > 10:
        return "STRONG_TREND"
    if adx >= 20:
        return "WEAK_TREND"
    return "RANGING"
