"""
agents/gpt_agent.py
-------------------
GPT-5 (OpenAI) LLM agent stub.

Replace the ``# TODO`` sections with your actual OpenAI SDK calls once
you have a valid API key. The stub returns a NEUTRAL signal so the
pipeline runs end-to-end without real credentials.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uncomment and populate once you have a real API key:
#
# from openai import AsyncOpenAI
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert crypto-futures trader.
Analyse the provided market data and return a JSON signal with fields:
direction (LONG|SHORT|NEUTRAL), confidence (0–1), entry, stop_loss,
take_profit, reasoning."""


class GPTAgent(BaseAgent):
    """LLM agent backed by OpenAI's GPT-5 API.

    Replace the stub body of :meth:`analyze` with a real ``AsyncOpenAI``
    client call once you have a valid ``gpt5_api_key`` in ``config.yaml``.
    """

    @property
    def name(self) -> str:
        return "gpt5"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call GPT-5 to analyse *market_data*.

        **Stub implementation** — returns NEUTRAL until wired up.

        To implement:
        1. Build a prompt from ``market_data`` and ``context``.
        2. Call ``AsyncOpenAI(api_key=self.api_key).chat.completions.create(...)``.
        3. Parse the JSON response and return it.
        """
        # TODO: Replace with real OpenAI API call, e.g.:
        #
        # client = AsyncOpenAI(api_key=self.api_key)
        # response = await client.chat.completions.create(
        #     model="gpt-5",
        #     messages=[
        #         {"role": "system", "content": _SYSTEM_PROMPT},
        #         {"role": "user", "content": str(market_data)},
        #     ],
        #     response_format={"type": "json_object"},
        # )
        # return json.loads(response.choices[0].message.content)

        logger.debug("[GPTAgent] analyze called for %s — returning stub NEUTRAL", market_data.get("pair"))
        return {
            "agent": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "entry": float(market_data.get("close", 0.0)),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": "Stub: GPT-5 API not yet configured.",
        }

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call GPT-5 to summarise *signals*.

        **Stub implementation** — returns a placeholder string.

        To implement, send the signals list as a prompt to GPT-5 and
        return its text response.
        """
        # TODO: Replace with real OpenAI API call.
        logger.debug("[GPTAgent] summarize called — returning stub summary")
        count = len(signals)
        return f"[GPT-5 stub] Received {count} signal(s) for summarisation. Configure your API key to enable."
