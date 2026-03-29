"""
agents/grok_agent.py
--------------------
Grok (xAI) LLM agent stub.

Replace the ``# TODO`` sections with your actual xAI / Grok SDK calls
once you have a valid API key.  The stub returns a NEUTRAL signal so
the pipeline runs end-to-end without real credentials.

Grok is compatible with the OpenAI API format — simply point the
``base_url`` at xAI's endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uncomment once you have a real API key:
#
# from openai import AsyncOpenAI
#
# XAI_BASE_URL = "https://api.x.ai/v1"
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert crypto-futures trader.
Analyse the provided market data and return a JSON signal with fields:
direction (LONG|SHORT|NEUTRAL), confidence (0–1), entry, stop_loss,
take_profit, reasoning."""


class GrokAgent(BaseAgent):
    """LLM agent backed by xAI's Grok API (OpenAI-compatible).

    Replace the stub body of :meth:`analyze` with a real ``AsyncOpenAI``
    client call (pointed at xAI's base URL) once you have a valid
    ``grok_api_key`` in ``config.yaml``.
    """

    @property
    def name(self) -> str:
        return "grok"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call Grok to analyse *market_data*.

        **Stub implementation** — returns NEUTRAL until wired up.

        To implement:
        1. Build a prompt from ``market_data`` and ``context``.
        2. Call ``AsyncOpenAI(api_key=self.api_key, base_url=XAI_BASE_URL)
           .chat.completions.create(...)``.
        3. Parse the JSON response and return it.
        """
        # TODO: Replace with real xAI/Grok API call, e.g.:
        #
        # client = AsyncOpenAI(api_key=self.api_key, base_url=XAI_BASE_URL)
        # response = await client.chat.completions.create(
        #     model="grok-3",
        #     messages=[
        #         {"role": "system", "content": _SYSTEM_PROMPT},
        #         {"role": "user", "content": str(market_data)},
        #     ],
        #     response_format={"type": "json_object"},
        # )
        # return json.loads(response.choices[0].message.content)

        logger.debug("[GrokAgent] analyze called for %s — returning stub NEUTRAL", market_data.get("pair"))
        return {
            "agent": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "entry": float(market_data.get("close", 0.0)),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": "Stub: Grok API not yet configured.",
        }

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call Grok to summarise *signals*.

        **Stub implementation** — returns a placeholder string.

        To implement, send the signals list as a prompt to Grok and
        return its text response.
        """
        # TODO: Replace with real xAI/Grok API call.
        logger.debug("[GrokAgent] summarize called — returning stub summary")
        count = len(signals)
        return f"[Grok stub] Received {count} signal(s) for summarisation. Configure your API key to enable."
