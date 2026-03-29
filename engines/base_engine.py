"""
engines/base_engine.py
----------------------
Abstract base class for all trade engines (scalping, swing, etc.).

Every concrete engine must implement :meth:`scan` and :meth:`get_timeframes`,
and expose an :attr:`engine_name` property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Abstract trade engine.

    Parameters
    ----------
    pairs:
        List of trading pair symbols to scan, e.g. ``["BTCUSDT", "ETHUSDT"]``.
    timeframes:
        List of timeframe strings relevant to this engine, e.g. ``["1m", "5m"]``.
    """

    def __init__(self, pairs: list[str], timeframes: list[str]) -> None:
        self._pairs = pairs
        self._timeframes = timeframes

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the engine identifier string (e.g. ``'scalping'``)."""

    @abstractmethod
    async def scan(self) -> list[dict[str, Any]]:
        """Scan all configured pairs/timeframes and return candidate signals.

        Returns
        -------
        list of signal dicts, each containing at minimum::

            {
                "pair":        str,
                "engine":      str,    # self.engine_name
                "timeframe":   str,
                "direction":   "LONG" | "SHORT",
                "entry":       float,
                "stop_loss":   float,
                "take_profit": float,
                "confidence":  float,  # 0.0 – 1.0
                "reasoning":   str,
                "market_data": dict,   # raw market snapshot for agents
            }

        Return an empty list when no actionable setups are found.
        """

    def get_timeframes(self) -> list[str]:
        """Return the list of timeframes this engine monitors."""
        return list(self._timeframes)

    def get_pairs(self) -> list[str]:
        """Return the list of pairs this engine monitors."""
        return list(self._pairs)
