"""
agents/gpt_agent.py
-------------------
GPT-4o-mini (OpenAI) LLM agent — TA screener (Stage 1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI, APIError

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """You are a fast crypto futures technical analysis screener.
Your ONLY job is to analyse the technical indicators provided and determine
if a trade setup exists.

Focus EXCLUSIVELY on:
- RSI levels and extremes (oversold < 35, overbought > 65)
- EMA crossovers and trend direction
- MACD histogram direction and momentum
- ATR-based volatility context
- Volume confirmation (ratio vs average)

You MUST respond with ONLY a valid JSON object:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "reasoning": "<max 100 chars, TA factors only>"
}

Be strict: only emit LONG or SHORT if confidence >= 0.65 based on TA alone.
Do NOT consider news, sentiment, or macro — that is handled separately.
This is a pre-filter. When in doubt, return NEUTRAL."""

_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 1.0


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, handling markdown code blocks."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _build_prompt(market_data: dict[str, Any], context: dict[str, Any]) -> str:  # noqa: ARG001
    """Build a TA-only prompt from market data (excludes sentiment/futures data)."""
    parts = [f"Pair: {market_data.get('pair', '?')}"]
    parts.append(f"Timeframe: {market_data.get('timeframe', '?')}")
    parts.append(f"Current Price: {market_data.get('close', 0)}")

    indicators = market_data.get("indicators", {})
    if indicators:
        parts.append(f"RSI: {indicators.get('rsi', 'N/A')}")
        parts.append(f"EMA Fast: {indicators.get('ema_fast', 'N/A')}")
        parts.append(f"EMA Slow: {indicators.get('ema_slow', 'N/A')}")
        parts.append(f"EMA Crossover: {indicators.get('ema_crossover', 'N/A')}")
        macd = indicators.get("macd", {})
        if macd:
            parts.append(f"MACD Histogram: {macd.get('histogram', 'N/A')}")
        parts.append(f"ATR: {indicators.get('atr', 'N/A')}")
        parts.append(f"Volume vs Avg: {indicators.get('volume_ratio', 'N/A')}")

    return "\n".join(parts)


class GPTAgent(BaseAgent):
    """LLM agent backed by OpenAI's GPT-4o-mini API (TA screener)."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        super().__init__(api_key)
        self.model = model

    @property
    def name(self) -> str:
        return "gpt4o"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call GPT-4o-mini to screen *market_data* for TA setups."""
        prompt = _build_prompt(market_data, context)
        pair = market_data.get("pair", "?")
        close = float(market_data.get("close", 0.0))

        neutral = {
            "agent": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "entry": close,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "reasoning": "GPT: no actionable TA signal.",
        }

        if not self.api_key:
            logger.warning("[GPTAgent] No API key configured — returning NEUTRAL")
            return neutral

        client = AsyncOpenAI(api_key=self.api_key)

        for attempt in range(_RETRY_ATTEMPTS + 1):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=256,
                )
                raw_text = response.choices[0].message.content or ""
                parsed = _parse_llm_json(raw_text)
                return {
                    "agent": self.name,
                    "direction": parsed.get("direction", "NEUTRAL"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "entry": close,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "reasoning": str(parsed.get("reasoning", ""))[:100],
                }
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("[GPTAgent] JSON parse error for %s (attempt %d): %s", pair, attempt, exc)
                return {**neutral, "reasoning": f"GPT: JSON parse error — {exc}"}
            except APIError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("[GPTAgent] API error for %s (attempt %d): %s", pair, attempt, exc)
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    logger.error("[GPTAgent] API error after retries for %s: %s", pair, exc)
                    return {**neutral, "reasoning": f"GPT: API error — {exc}"}
            except Exception as exc:
                logger.error("[GPTAgent] Unexpected error for %s: %s", pair, exc)
                return {**neutral, "reasoning": f"GPT: unexpected error — {exc}"}

        return neutral

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call GPT-4o-mini to summarise a list of signals."""
        if not self.api_key:
            return f"[GPT] {len(signals)} signal(s) — API key not configured."

        prompt = "Summarise the following trading signals in 2-3 sentences:\n" + json.dumps(signals, indent=2)
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise crypto trading analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("[GPTAgent] summarize failed: %s", exc)
            return f"[GPT] Summary unavailable — {exc}"
