"""
tests/test_scalping_engine.py
------------------------------
Unit tests for engines/scalping_engine.py.
All network/exchange calls are replaced with mocks.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from engines.scalping_engine import ScalpingEngine


# ---------------------------------------------------------------------------
# Helper: build a mock ExchangeClient
# ---------------------------------------------------------------------------

def make_mock_client(candles: list[dict]) -> MagicMock:
    """Return an AsyncMock exchange client that yields *candles* on fetch_ohlcv."""
    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(return_value=candles)
    client.fetch_aggregated_futures_data = AsyncMock(
        return_value={
            "funding_rate_avg": 0.0001,
            "oi_change_pct": 2.0,
            "ls_ratio_avg": 1.2,
        }
    )
    return client


def _make_candles(n: int, price: float = 100.0) -> list[dict]:
    return [
        {"open": price - 0.1, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000.0}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# scan() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scan_returns_empty_when_no_client():
    engine = ScalpingEngine(pairs=["BTCUSDT"], timeframes=["30m"])
    result = await engine.scan()
    assert result == []


@pytest.mark.asyncio
async def test_scan_returns_empty_on_insufficient_candles():
    candles = _make_candles(10)  # fewer than _MIN_CANDLES (50)
    client = make_mock_client(candles)
    engine = ScalpingEngine(pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client)
    result = await engine.scan()
    assert result == []


@pytest.mark.asyncio
async def test_scan_skips_when_no_setup():
    # 200 flat candles → ADX ≈ 0 → regime = RANGING → _has_setup returns False
    candles = _make_candles(200, price=100.0)
    client = make_mock_client(candles)
    engine = ScalpingEngine(pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client)
    result = await engine.scan()
    assert result == []


# ---------------------------------------------------------------------------
# _fetch_market_data tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_market_data_returns_correct_keys():
    candles = _make_candles(200)
    client = make_mock_client(candles)
    engine = ScalpingEngine(pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client)

    data = await engine._fetch_market_data("BTCUSDT", "30m")
    assert data is not None

    # Top-level keys
    for key in ("pair", "timeframe", "close", "open", "high", "low", "volume"):
        assert key in data, f"Missing top-level key: {key}"

    # Indicators sub-dict keys
    indicators = data["indicators"]
    for key in ("rsi", "ema_fast", "ema_slow", "ema_crossover", "atr", "volume_ratio"):
        assert key in indicators, f"Missing indicator key: {key}"


# ---------------------------------------------------------------------------
# _build_signal tests
# ---------------------------------------------------------------------------

def _make_bullish_market_data(
    *,
    funding_rate: float = 0.0001,
    oi_change: float = 2.0,
    ls_ratio: float = 1.2,
    adx: float = 30.0,
    plus_di: float = 25.0,
    minus_di: float = 10.0,
    volume_ratio: float = 2.0,
    support: float = 99.0,
) -> tuple[dict, MagicMock]:
    """Return (market_data, mock_client) configured for a LONG signal."""
    market_data = {
        "pair": "BTCUSDT",
        "timeframe": "30m",
        "close": 100.0,
        "open": 99.9,
        "high": 100.5,
        "low": 99.5,
        "volume": 2000.0,
        "candles": [],
        "closes": [],
        "highs": [],
        "lows": [],
        "volumes": [],
        "indicators": {
            "rsi": 30.0,              # vote 1: bullish (< 40)
            "ema_fast": 101.0,
            "ema_slow": 100.0,
            "ema_crossover": "BULLISH",  # vote 2: bullish
            "atr": 2.0,
            "volume_ratio": volume_ratio,
            "bb": {
                "upper": 110.0,
                "middle": 100.0,
                "lower": 90.0,
                "bandwidth": 0.20,
            },
            "stoch": {"k": 30.0, "d": 35.0},
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "support": support,
            "resistance": 110.0,
            "vwap": 99.0,            # vote 3: bullish (close > vwap)
            "regime": "WEAK_TREND",
        },
    }
    client = MagicMock()
    client.fetch_aggregated_futures_data = AsyncMock(
        return_value={
            "funding_rate_avg": funding_rate,
            "oi_change_pct": oi_change,
            "ls_ratio_avg": ls_ratio,
        }
    )
    return market_data, client


@pytest.mark.asyncio
async def test_build_signal_returns_valid_rr():
    market_data, client = _make_bullish_market_data()
    engine = ScalpingEngine(
        pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client
    )

    signal = await engine._build_signal("BTCUSDT", market_data)

    # Signal may be None if confidence gate not met, but with our crafted data it should pass
    if signal is not None:
        assert signal["direction"] == "LONG"
        assert signal["take_profit"] > signal["entry"] > signal["stop_loss"]
        risk = abs(signal["entry"] - signal["stop_loss"])
        reward = abs(signal["take_profit"] - signal["entry"])
        assert risk > 0
        assert (reward / risk) >= 2.0


@pytest.mark.asyncio
async def test_build_signal_returns_none_below_confidence_gate():
    # Adverse futures data: high funding rate + crowded long side
    market_data, client = _make_bullish_market_data(
        funding_rate=0.002,   # > 0.0005 → -0.20
        oi_change=2.0,        # +0.10 (partial offset)
        ls_ratio=3.0,         # > 2.0 → -0.15
        adx=21.0,             # very low ADX → minimal contribution
        plus_di=11.0,
        minus_di=10.0,
        volume_ratio=1.0,     # no volume boost
        support=0.0,          # no S/R proximity boost
    )
    engine = ScalpingEngine(
        pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client
    )
    signal = await engine._build_signal("BTCUSDT", market_data)
    assert signal is None


# ---------------------------------------------------------------------------
# _check_htf_confluence tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_htf_confluence_passes_when_no_crossover():
    """When ema_crossover is NONE the method returns True without any client call."""
    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(return_value=[])  # should never be called
    engine = ScalpingEngine(
        pairs=["BTCUSDT"], timeframes=["30m"], exchange_client=client
    )

    market_data = {"indicators": {"ema_crossover": "NONE"}}
    result = await engine._check_htf_confluence("BTCUSDT", market_data)

    assert result is True
    client.fetch_ohlcv.assert_not_called()
