"""Shared pytest fixtures for GuyGuyBot tests."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Synthetic OHLCV helpers
# ---------------------------------------------------------------------------

def make_candles(n: int = 200, base_price: float = 100.0, trend: float = 0.0) -> list[dict]:
    """Generate *n* synthetic OHLCV candles.

    Parameters
    ----------
    n:
        Number of candles.
    base_price:
        Starting close price.
    trend:
        Per-candle price increment (positive = uptrend, negative = downtrend).
    """
    candles = []
    price = base_price
    for i in range(n):
        price += trend
        candles.append({
            "open": price - 0.1,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0 + i * 10,
        })
    return candles


@pytest.fixture
def flat_candles():
    """200 flat candles at $100."""
    return make_candles(200, base_price=100.0, trend=0.0)


@pytest.fixture
def uptrend_candles():
    """200 candles in a clear uptrend (+0.5/candle)."""
    return make_candles(200, base_price=100.0, trend=0.5)


@pytest.fixture
def downtrend_candles():
    """200 candles in a clear downtrend (-0.5/candle)."""
    return make_candles(200, base_price=200.0, trend=-0.5)
