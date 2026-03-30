"""
tests/test_indicators.py
------------------------
Unit tests for every public function in data/indicators.py.
"""
from __future__ import annotations

import pytest

from data.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_ema_series,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    calculate_vwap,
    calculate_volume_sma,
    classify_market_regime,
    detect_ema_crossover,
    detect_rsi_divergence,
    detect_support_resistance,
)


# ---------------------------------------------------------------------------
# calculate_rsi
# ---------------------------------------------------------------------------

class TestCalculateRsi:
    def test_returns_50_when_too_short(self):
        # RSI needs period+1 = 15 items; 14 is too short
        result = calculate_rsi([100.0] * 14)
        assert result == 50.0

    def test_returns_float_in_range(self):
        closes = list(range(1, 51))  # 50 ascending values
        result = calculate_rsi(closes)
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    def test_returns_below_50_for_downtrend(self):
        closes = [float(x) for x in range(100, 50, -1)]  # declining prices
        result = calculate_rsi(closes)
        assert result < 50.0

    def test_returns_above_50_for_uptrend(self):
        # 3 up then 1 small down — net bullish with both gains and losses so avg_loss > 0
        prices = []
        p = 100.0
        for i in range(60):
            p += [1.0, 1.0, 1.0, -0.5][i % 4]
            prices.append(p)
        result = calculate_rsi(prices)
        assert result > 50.0


# ---------------------------------------------------------------------------
# calculate_ema
# ---------------------------------------------------------------------------

class TestCalculateEma:
    def test_returns_zero_when_too_short(self):
        result = calculate_ema([100.0] * 5, period=9)
        assert result == 0.0

    def test_returns_float_for_valid_input(self):
        closes = [100.0] * 30
        result = calculate_ema(closes, period=9)
        assert isinstance(result, float)
        assert result > 0.0

    def test_fast_ema_above_slow_in_uptrend(self):
        closes = [float(x) for x in range(1, 51)]  # 50 rising values
        ema_fast = calculate_ema(closes, period=9)
        ema_slow = calculate_ema(closes, period=21)
        assert ema_fast > ema_slow


# ---------------------------------------------------------------------------
# calculate_ema_series
# ---------------------------------------------------------------------------

class TestCalculateEmaSeries:
    def test_returns_empty_when_too_short(self):
        result = calculate_ema_series([100.0] * 5, period=9)
        assert result == []

    def test_returns_correct_length(self):
        closes = [100.0] * 30
        result = calculate_ema_series(closes, period=9)
        assert len(result) == 30

    def test_last_value_matches_calculate_ema(self):
        closes = [float(x) for x in range(1, 51)]
        period = 9
        series = calculate_ema_series(closes, period)
        single = calculate_ema(closes, period)
        assert abs(series[-1] - single) < 1e-9


# ---------------------------------------------------------------------------
# calculate_macd
# ---------------------------------------------------------------------------

class TestCalculateMacd:
    def test_returns_zeros_when_too_short(self):
        # Need slow(26) + signal(9) = 35 items; 34 is too short
        result = calculate_macd([100.0] * 34)
        assert result == {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    def test_returns_dict_with_all_keys(self):
        closes = [float(x) for x in range(1, 51)]
        result = calculate_macd(closes)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_equals_macd_minus_signal(self):
        closes = [float(x) for x in range(1, 51)]
        result = calculate_macd(closes)
        expected = round(result["macd"] - result["signal"], 9)
        assert abs(result["histogram"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# calculate_atr
# ---------------------------------------------------------------------------

class TestCalculateAtr:
    def _make_ohlcv(self, n: int, spread: float = 1.0, base: float = 100.0):
        highs = [base + spread / 2] * n
        lows = [base - spread / 2] * n
        closes = [base] * n
        return highs, lows, closes

    def test_returns_zero_when_too_short(self):
        h, lo, c = self._make_ohlcv(14)
        assert calculate_atr(h, lo, c) == 0.0

    def test_returns_positive_float_for_valid_input(self):
        h, lo, c = self._make_ohlcv(20)
        result = calculate_atr(h, lo, c)
        assert isinstance(result, float)
        assert result > 0.0

    def test_volatile_greater_than_calm(self):
        h_volatile, lo_volatile, c_volatile = self._make_ohlcv(30, spread=10.0)
        h_calm, lo_calm, c_calm = self._make_ohlcv(30, spread=0.5)
        atr_volatile = calculate_atr(h_volatile, lo_volatile, c_volatile)
        atr_calm = calculate_atr(h_calm, lo_calm, c_calm)
        assert atr_volatile > atr_calm


# ---------------------------------------------------------------------------
# calculate_bollinger_bands
# ---------------------------------------------------------------------------

class TestCalculateBollingerBands:
    def test_returns_zeros_when_too_short(self):
        result = calculate_bollinger_bands([100.0] * 19, period=20)
        assert result == {"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0}

    def test_returns_dict_with_correct_keys(self):
        closes = [100.0 + i * 0.1 for i in range(30)]
        result = calculate_bollinger_bands(closes)
        assert set(result.keys()) == {"upper", "middle", "lower", "bandwidth"}

    def test_upper_above_middle_above_lower(self):
        closes = [100.0 + i * 0.5 for i in range(30)]
        result = calculate_bollinger_bands(closes)
        assert result["upper"] > result["middle"] > result["lower"]

    def test_bandwidth_positive_for_volatile_data(self):
        closes = [100.0 + (i % 5) * 2.0 for i in range(30)]
        result = calculate_bollinger_bands(closes)
        assert result["bandwidth"] > 0.0


# ---------------------------------------------------------------------------
# calculate_volume_sma
# ---------------------------------------------------------------------------

class TestCalculateVolumeSma:
    def test_returns_zero_when_too_short(self):
        result = calculate_volume_sma([1000.0] * 19, period=20)
        assert result == 0.0

    def test_returns_correct_average(self):
        volumes = [1000.0] * 30
        result = calculate_volume_sma(volumes, period=20)
        assert abs(result - 1000.0) < 1e-9


# ---------------------------------------------------------------------------
# calculate_stochastic
# ---------------------------------------------------------------------------

class TestCalculateStochastic:
    def test_returns_default_when_too_short(self):
        # Need k_period(14) + d_period(3) = 17; 16 is too short
        h = [101.0] * 16
        lo = [99.0] * 16
        c = [100.0] * 16
        result = calculate_stochastic(h, lo, c)
        assert result == {"k": 50.0, "d": 50.0}

    def test_returns_dict_with_k_d_in_range(self):
        n = 25
        h = [101.0] * n
        lo = [99.0] * n
        c = [100.0] * n
        result = calculate_stochastic(h, lo, c)
        assert set(result.keys()) == {"k", "d"}
        assert 0.0 <= result["k"] <= 100.0
        assert 0.0 <= result["d"] <= 100.0

    def test_k_above_70_in_strong_uptrend(self):
        # Close near high → k should be high
        n = 25
        highs = [100.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [100.0 + i for i in range(n)]  # close == high - 0 spread
        result = calculate_stochastic(highs, lows, closes)
        assert result["k"] > 70.0


# ---------------------------------------------------------------------------
# detect_ema_crossover
# ---------------------------------------------------------------------------

class TestDetectEmaCrossover:
    def test_returns_none_when_too_short(self):
        # Need slow_period + 2 = 23; 22 is too short
        result = detect_ema_crossover([100.0] * 22, fast_period=9, slow_period=21)
        assert result == "NONE"

    def test_returns_bullish_on_upward_spike(self):
        # 40 flat candles bring both EMAs to equal level; spike tips fast above slow
        closes = [100.0] * 40 + [200.0]
        result = detect_ema_crossover(closes, fast_period=9, slow_period=21)
        assert result == "BULLISH"

    def test_returns_bearish_on_downward_spike(self):
        closes = [100.0] * 40 + [0.0]
        result = detect_ema_crossover(closes, fast_period=9, slow_period=21)
        assert result == "BEARISH"

    def test_returns_none_for_flat_parallel_emas(self):
        closes = [100.0] * 100
        result = detect_ema_crossover(closes, fast_period=9, slow_period=21)
        assert result == "NONE"


# ---------------------------------------------------------------------------
# detect_rsi_divergence
# ---------------------------------------------------------------------------

class TestDetectRsiDivergence:
    def test_returns_none_when_too_short(self):
        result = detect_rsi_divergence([100.0] * 10, [50.0] * 10)
        assert result == "NONE"

    def test_returns_bullish_div(self):
        # Price makes a lower low (50 < 100), RSI makes a higher low (40 > 20)
        closes = [100.0] * 19 + [50.0]
        rsi_values = [20.0] + [30.0] * 18 + [40.0]
        result = detect_rsi_divergence(closes, rsi_values)
        assert result == "BULLISH_DIV"

    def test_returns_bearish_div(self):
        # Price makes a higher high (150 > 100), RSI makes a lower high (60 < 80)
        closes = [100.0] * 19 + [150.0]
        rsi_values = [80.0] + [70.0] * 18 + [60.0]
        result = detect_rsi_divergence(closes, rsi_values)
        assert result == "BEARISH_DIV"

    def test_returns_none_when_no_divergence(self):
        # Current price (102) is within the prior range [100, 105] → neither condition
        closes = [100.0, 105.0] * 9 + [100.0, 102.0]
        rsi_values = [50.0] * 20
        result = detect_rsi_divergence(closes, rsi_values)
        assert result == "NONE"


# ---------------------------------------------------------------------------
# calculate_adx
# ---------------------------------------------------------------------------

class TestCalculateAdx:
    def test_returns_zeros_when_too_short(self):
        # Need period*2+1 = 29; 28 is too short
        h = [101.0] * 28
        lo = [99.0] * 28
        c = [100.0] * 28
        result = calculate_adx(h, lo, c)
        assert result == {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    def test_returns_dict_with_correct_keys(self):
        h = [101.0 + i * 0.5 for i in range(50)]
        lo = [99.0 + i * 0.5 for i in range(50)]
        c = [100.0 + i * 0.5 for i in range(50)]
        result = calculate_adx(h, lo, c)
        assert set(result.keys()) == {"adx", "plus_di", "minus_di"}

    def test_adx_above_20_for_strong_trend(self):
        # 200 candles with +1.0/candle trend
        n = 200
        h = [100.0 + i * 1.0 + 0.5 for i in range(n)]
        lo = [100.0 + i * 1.0 - 0.5 for i in range(n)]
        c = [100.0 + i * 1.0 for i in range(n)]
        result = calculate_adx(h, lo, c)
        assert result["adx"] > 20.0

    def test_plus_di_above_minus_di_in_uptrend(self):
        n = 200
        h = [100.0 + i * 1.0 + 0.5 for i in range(n)]
        lo = [100.0 + i * 1.0 - 0.5 for i in range(n)]
        c = [100.0 + i * 1.0 for i in range(n)]
        result = calculate_adx(h, lo, c)
        assert result["plus_di"] > result["minus_di"]


# ---------------------------------------------------------------------------
# detect_support_resistance
# ---------------------------------------------------------------------------

class TestDetectSupportResistance:
    def test_returns_correct_keys(self):
        h = [101.0] * 10
        lo = [99.0] * 10
        c = [100.0] * 10
        result = detect_support_resistance(h, lo, c)
        assert set(result.keys()) == {"support", "resistance", "support_touches", "resistance_touches"}

    def test_returns_zero_for_fewer_than_3_candles(self):
        h = [101.0, 102.0]
        lo = [99.0, 98.0]
        c = [100.0, 100.0]
        result = detect_support_resistance(h, lo, c)
        assert result["support"] == 0.0
        assert result["resistance"] == 0.0

    def test_support_below_price_resistance_above(self):
        # Zigzag pattern with clear swing lows at 95 and swing highs at 105
        n = 21
        highs = [(105.0 if i % 2 == 1 else 100.0) for i in range(n)]
        lows = [(95.0 if i % 2 == 1 else 100.0) for i in range(n)]
        closes = [100.0] * n
        result = detect_support_resistance(highs, lows, closes)
        current_price = closes[-1]
        assert result["support"] < current_price
        assert result["resistance"] > current_price

    def test_does_not_raise_on_empty_input(self):
        result = detect_support_resistance([], [], [])
        assert result["support"] == 0.0
        assert result["resistance"] == 0.0


# ---------------------------------------------------------------------------
# calculate_vwap
# ---------------------------------------------------------------------------

class TestCalculateVwap:
    def test_returns_close_when_volumes_zero(self):
        highs = [101.0] * 5
        lows = [99.0] * 5
        closes = [100.0] * 5
        volumes = [0.0] * 5
        result = calculate_vwap(highs, lows, closes, volumes)
        assert result == 100.0

    def test_returns_typical_price_for_uniform_data(self):
        h, lo, c, v = 110.0, 90.0, 100.0, 100.0
        typical = (h + lo + c) / 3.0
        result = calculate_vwap([h], [lo], [c], [v])
        assert abs(result - typical) < 1e-9

    def test_vwap_between_low_and_high(self):
        highs = [110.0, 112.0, 108.0]
        lows = [90.0, 92.0, 88.0]
        closes = [100.0, 102.0, 99.0]
        volumes = [1000.0, 1200.0, 800.0]
        result = calculate_vwap(highs, lows, closes, volumes)
        assert min(lows) <= result <= max(highs)


# ---------------------------------------------------------------------------
# classify_market_regime
# ---------------------------------------------------------------------------

class TestClassifyMarketRegime:
    def test_strong_trend(self):
        # adx=30 >= 25, |di_diff|=15 > 10
        result = classify_market_regime(adx=30.0, plus_di=25.0, minus_di=10.0)
        assert result == "STRONG_TREND"

    def test_weak_trend_from_moderate_adx(self):
        # adx=22 >= 20 but |di_diff|=5 <= 10 → not STRONG_TREND, but WEAK_TREND
        result = classify_market_regime(adx=22.0, plus_di=15.0, minus_di=10.0)
        assert result == "WEAK_TREND"

    def test_ranging_when_adx_below_20(self):
        result = classify_market_regime(adx=15.0, plus_di=12.0, minus_di=8.0)
        assert result == "RANGING"

    def test_weak_trend_when_adx_24_small_di_spread(self):
        # adx=24 >= 20 but < 25 → WEAK_TREND (not RANGING, not STRONG_TREND)
        result = classify_market_regime(adx=24.0, plus_di=14.0, minus_di=10.0)
        assert result == "WEAK_TREND"
