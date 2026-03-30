"""
tests/test_swing_engine.py
--------------------------
Unit tests for engines/swing_engine.py.
All network/exchange calls are replaced with mocks.
"""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engines.swing_engine import SwingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_client(candles: list[dict], futures_data: dict | None = None) -> MagicMock:
    """Return a mock exchange client."""
    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(return_value=candles)
    client.fetch_aggregated_futures_data = AsyncMock(
        return_value=futures_data or {
            "funding_rate_avg": 0.0001,
            "oi_change_pct": 2.0,
            "ls_ratio_avg": 1.2,
        }
    )
    return client


def _make_candles(n: int, price: float = 100.0, trend: float = 0.0) -> list[dict]:
    candles = []
    p = price
    for _ in range(n):
        p += trend
        candles.append(
            {"open": p - 0.1, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 1000.0}
        )
    return candles


# Fixed datetime that satisfies the candle-age gate for 4h candles:
#   UTC hour = 11 (not in any blackout)
#   timestamp % 14400 = 12240 → completion = 85 % (≥ 80 %)
_MOCK_DT_4H = datetime.datetime(2024, 1, 15, 11, 24, 0, tzinfo=datetime.timezone.utc)


def _make_dt_patch(fixed_dt: datetime.datetime):
    """Return a context manager that patches datetime in swing_engine."""
    mock_module = MagicMock()
    mock_module.datetime.now.return_value = fixed_dt
    mock_module.timezone.utc = datetime.timezone.utc
    return patch("engines.swing_engine.datetime", mock_module)


# ---------------------------------------------------------------------------
# scan() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scan_returns_empty_when_no_client():
    engine = SwingEngine(pairs=["BTCUSDT"], timeframes=["4h"])
    result = await engine.scan()
    assert result == []


@pytest.mark.asyncio
async def test_scan_returns_empty_on_insufficient_candles():
    candles = _make_candles(50)  # fewer than _MIN_CANDLES (100)
    client = make_mock_client(candles)
    engine = SwingEngine(pairs=["BTCUSDT"], timeframes=["4h"], exchange_client=client)
    result = await engine.scan()
    assert result == []


# ---------------------------------------------------------------------------
# _fetch_market_data tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_market_data_returns_correct_keys():
    candles = _make_candles(300)
    client = make_mock_client(candles)
    engine = SwingEngine(pairs=["BTCUSDT"], timeframes=["4h"], exchange_client=client)

    data = await engine._fetch_market_data("BTCUSDT", "4h")
    assert data is not None

    indicators = data["indicators"]
    for key in ("rsi", "ema50", "ema200", "ema_crossover", "macd", "atr", "rsi_divergence"):
        assert key in indicators, f"Missing indicator key: {key}"

    macd = indicators["macd"]
    assert set(macd.keys()) == {"macd", "signal", "histogram"}


# ---------------------------------------------------------------------------
# _has_setup tests
# ---------------------------------------------------------------------------

def test_has_setup_returns_false_when_all_conditions_miss():
    """All three conditions False + RANGING ADX gate → False."""
    market_data = {
        "indicators": {
            "regime": "WEAK_TREND",    # pass the ADX gate
            "adx": 22.0,
            "rsi_divergence": "NONE",  # condition 1: False
            "ema_crossover": "NONE",   # condition 2 part 1: False
            "ema50": 100.0,
            "ema200": 100.0,           # |separation| = 0 ≤ 0.005 → condition 2: False
            "macd": {"histogram": 0.0},  # condition 3: False
        }
    }
    assert SwingEngine._has_setup(market_data) is False


def test_has_setup_returns_true_with_two_conditions():
    """RSI divergence + EMA separation ≥ 2 conditions → True."""
    market_data = {
        "indicators": {
            "regime": "WEAK_TREND",
            "adx": 22.0,
            "rsi_divergence": "BULLISH_DIV",  # condition 1: True
            "ema_crossover": "NONE",
            "ema50": 105.0,
            "ema200": 100.0,  # |5/100| = 0.05 > 0.005 → condition 2: True
            "macd": {"histogram": 0.0},         # condition 3: False
        }
    }
    assert SwingEngine._has_setup(market_data) is True


def test_has_setup_returns_false_for_ranging_market():
    """ADX gate: RANGING + adx < 20 immediately returns False."""
    market_data = {
        "indicators": {
            "regime": "RANGING",
            "adx": 10.0,
            "rsi_divergence": "BULLISH_DIV",  # would be True…
            "ema_crossover": "BULLISH",        # …and True…
            "ema50": 105.0,
            "ema200": 100.0,
            "macd": {"histogram": 1.0},        # …and True, but ADX gate fires first
        }
    }
    assert SwingEngine._has_setup(market_data) is False


# ---------------------------------------------------------------------------
# _build_signal tests
# ---------------------------------------------------------------------------

def _bullish_market_data(close: float = 100.0, atr: float = 2.0) -> dict:
    return {
        "pair": "BTCUSDT",
        "timeframe": "4h",
        "close": close,
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "volume": 2000.0,
        "candles": [],
        "closes": [],
        "highs": [],
        "lows": [],
        "indicators": {
            "rsi": 45.0,
            "ema50": 101.0,
            "ema200": 100.0,
            "ema_crossover": "BULLISH",  # triggers LONG direction
            "macd": {"macd": 0.5, "signal": 0.3, "histogram": 0.2},
            "atr": atr,
            "rsi_divergence": "NONE",
            "bb": {"upper": 110.0, "middle": 100.0, "lower": 90.0, "bandwidth": 0.20},
            "stoch": {"k": 50.0, "d": 50.0},
            "adx": 30.0,
            "plus_di": 25.0,
            "minus_di": 10.0,
            "support": 99.0,
            "resistance": 110.0,
            "vwap": 99.0,
            "regime": "WEAK_TREND",
            "volume_ratio": 2.0,
        },
    }


def _bearish_market_data(close: float = 100.0, atr: float = 2.0) -> dict:
    return {
        "pair": "BTCUSDT",
        "timeframe": "4h",
        "close": close,
        "open": close + 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "volume": 2000.0,
        "candles": [],
        "closes": [],
        "highs": [],
        "lows": [],
        "indicators": {
            "rsi": 55.0,
            "ema50": 99.0,
            "ema200": 100.0,
            "ema_crossover": "BEARISH",  # triggers SHORT direction
            "macd": {"macd": -0.5, "signal": -0.3, "histogram": -0.2},
            "atr": atr,
            "rsi_divergence": "NONE",
            "bb": {"upper": 110.0, "middle": 100.0, "lower": 90.0, "bandwidth": 0.20},
            "stoch": {"k": 50.0, "d": 50.0},
            "adx": 30.0,
            "plus_di": 10.0,
            "minus_di": 25.0,
            "support": 90.0,
            "resistance": 101.0,
            "vwap": 101.0,
            "regime": "WEAK_TREND",
            "volume_ratio": 2.0,
        },
    }


@pytest.mark.asyncio
async def test_build_signal_long_direction():
    market_data = _bullish_market_data()
    client = make_mock_client([])
    engine = SwingEngine(pairs=["BTCUSDT"], timeframes=["4h"], exchange_client=client)

    with _make_dt_patch(_MOCK_DT_4H):
        signal = await engine._build_signal("BTCUSDT", market_data)

    assert signal is not None, "Expected a LONG signal but got None"
    assert signal["direction"] == "LONG"
    assert signal["take_profit"] > signal["entry"] > signal["stop_loss"]


@pytest.mark.asyncio
async def test_build_signal_short_direction():
    market_data = _bearish_market_data()
    client = make_mock_client(
        [],
        futures_data={
            "funding_rate_avg": 0.0001,
            "oi_change_pct": -2.0,  # negative OI → SHORT confirmation
            "ls_ratio_avg": 1.2,
        },
    )
    engine = SwingEngine(pairs=["BTCUSDT"], timeframes=["4h"], exchange_client=client)

    with _make_dt_patch(_MOCK_DT_4H):
        signal = await engine._build_signal("BTCUSDT", market_data)

    assert signal is not None, "Expected a SHORT signal but got None"
    assert signal["direction"] == "SHORT"
    assert signal["stop_loss"] > signal["entry"] > signal["take_profit"]
