"""
data/exchanges/base.py
----------------------
Abstract base class that every exchange adapter must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExchange(ABC):
    """Abstract exchange adapter.

    All methods return ``None`` (or safe defaults) on failure — they must
    never raise so callers can proceed without crashing on transient API
    errors.
    """

    name: str

    @abstractmethod
    async def fetch_ohlcv(
        self, pair: str, timeframe: str, limit: int = 200
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def fetch_current_price(self, pair: str) -> float | None: ...

    async def fetch_funding_rate(self, pair: str) -> float | None:
        """Spot exchanges return ``None``."""
        return None

    async def fetch_open_interest(self, pair: str) -> dict[str, float] | None:
        return None

    async def fetch_long_short_ratio(self, pair: str) -> dict[str, float] | None:
        return None

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        return None

    async def close(self) -> None:
        pass
