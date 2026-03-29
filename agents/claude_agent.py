"""
agents/claude_agent.py
----------------------
Claude (Anthropic) LLM agent stub.

Replace the ``# TODO`` sections with your actual Anthropic SDK calls once
you have a valid API key. The stub returns a NEUTRAL signal so the pipeline
runs end-to-end without real credentials.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uncomment and populate once you have a real API key:
#
# import anthropic
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert crypto-futures trader.
Analyse the provided market data and return a JSON signal with fields:
direction (LONG|SHORT|NEUTRAL), confidence (0–1), entry, stop_loss,
take_profit, reasoning."""


class ClaudeAgent(BaseAgent):
    """LLM agent backed by Anthropic's Claude API.

    Replace the stub body of :meth:`analyze` with a real ``anthropic.Anthropic``
    client call once you have a valid ``claude_api_key`` in ``config.yaml``.
    """

    @property
    def name(self) -> str:
        return "claude"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call Claude to analyse *market_data*.

        **Stub implementation** — returns NEUTRAL until wired up.

        To implement:
        1. Build a prompt from ``market_data`` and ``context``.
        2. Call ``anthropic.Anthropic(api_key=self.api_key).messages.create(...)``.
        3. Parse the JSON response and return it.
        """
        # TODO: Replace with real Anthropic API call, e.g.:
        #
        # client = anthropic.Anthropic(api_key=self.api_key)
        # message = client.messages.create(
        #     model="claude-opus-4-5",
        #     max_tokens=1024,
        #     system=_SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": str(market_data)}],
        # )
        # return json.loads(message.content[0].text)

        logger.debug("[ClaudeAgent] analyze called for %s — returning stub NEUTRAL", market_data.get("pair"))
        return {
            "agent": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "entry": float(market_data.get("close", 0.0)),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": "Stub: Claude API not yet configured.",
        }

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call Claude to summarise *signals*.

        **Stub implementation** — returns a placeholder string.

        To implement, send the signals list as a prompt to Claude and
        return its text response.
        """
        # TODO: Replace with real Anthropic API call.
        logger.debug("[ClaudeAgent] summarize called — returning stub summary")
        count = len(signals)
        return f"[Claude stub] Received {count} signal(s) for summarisation. Configure your API key to enable."
