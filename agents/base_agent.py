"""
agents/base_agent.py
--------------------
Abstract base class for all LLM agents.

Every concrete agent (Claude, GPT-5, Grok) must inherit from BaseAgent
and implement :meth:`analyze` and :meth:`summarize`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract LLM agent.

    Concrete subclasses are responsible for calling their respective LLM
    API and returning structured responses.

    Parameters
    ----------
    api_key:
        Authentication key for the LLM provider.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name, e.g. ``'claude'``."""

    @abstractmethod
    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Analyse *market_data* and return a signal recommendation.

        Parameters
        ----------
        market_data:
            Snapshot of market data for a specific pair/timeframe.
            Expected keys: ``pair``, ``timeframe``, ``ohlcv``,
            ``indicators`` (optional).
        context:
            Additional context such as recent signals or account state.

        Returns
        -------
        dict with at minimum:

        .. code-block:: python

            {
                "direction": "LONG" | "SHORT" | "NEUTRAL",
                "confidence": 0.0 – 1.0,
                "entry":      float,
                "stop_loss":  float,
                "take_profit": float,
                "reasoning":  str,
            }

        Return ``{"direction": "NEUTRAL", "confidence": 0.0, ...}`` when
        the agent sees no actionable opportunity.
        """

    @abstractmethod
    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Produce a human-readable summary of *signals*.

        Used by the arbitration step and for periodic stat reports.

        Parameters
        ----------
        signals:
            List of signal recommendation dicts (as returned by
            :meth:`analyze`).

        Returns
        -------
        A plain-text summary string.
        """
