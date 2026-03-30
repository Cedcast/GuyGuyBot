"""
agents/grok_agent.py
--------------------
Grok-3-mini (xAI) LLM agent — sentiment & futures data screener (Stage 1).

Grok is compatible with the OpenAI API format — the client is pointed
at xAI's endpoint via ``base_url``.
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

_XAI_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-3-mini"

_SYSTEM_PROMPT = """You are a crypto futures sentiment and market structure screener.
Your ONLY job is to analyse the futures market data and sentiment signals.

Focus EXCLUSIVELY on:
- Funding rate direction and magnitude (positive = longs paying = squeeze risk)
- Open interest change percentage (rising OI confirms trend)
- Long/short ratio (extreme ratios = crowded trade risk)
- Fear & Greed Index (>80 extreme greed = caution longs, <20 extreme fear = caution shorts)
- News sentiment score and recent headlines

You MUST respond with ONLY a valid JSON object:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "reasoning": "<max 100 chars, sentiment/futures factors only>"
}

Be strict: only emit LONG or SHORT if confidence >= 0.65 based on sentiment/futures data alone.
Do NOT consider price action or technical indicators — that is handled separately.
This is a pre-filter. When in doubt, return NEUTRAL."""

_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 1.0


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, handling markdown code blocks."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _build_prompt(market_data: dict[str, Any], context: dict[str, Any]) -> str:
    """Build a sentiment/futures-only prompt (excludes TA indicators)."""
    parts = [f"Pair: {market_data.get('pair', '?')}"]
    parts.append(f"Timeframe: {market_data.get('timeframe', '?')}")
    parts.append(f"Current Price: {market_data.get('close', 0)}")

    futures = market_data.get("futures", {})
    if futures:
        parts.append(f"Funding Rate: {futures.get('funding_rate', 'N/A')}")
        oi = futures.get("open_interest") or {}
        if oi:
            parts.append(f"Open Interest Change: {oi.get('change_pct', 0):.2f}%")
        ls = futures.get("long_short_ratio") or {}
        if ls:
            parts.append(f"Long/Short Ratio: {ls.get('long_short_ratio', 'N/A')}")

    news_ctx = context.get("news_context", {})
    if news_ctx:
        parts.append(f"Fear & Greed Index: {news_ctx.get('fear_greed_index', 50)} ({news_ctx.get('fear_greed_label', 'Neutral')})")
        parts.append(f"Market Sentiment Score: {news_ctx.get('market_sentiment_score', 0):.2f}")
        headlines = news_ctx.get("recent_headlines", [])
        if headlines:
            parts.append(f"Recent Headlines: {'; '.join(headlines[:3])}")

    return "\n".join(parts)


class GrokAgent(BaseAgent):
    """LLM agent backed by xAI's Grok API (sentiment & futures screener)."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        super().__init__(api_key)
        self.model = model

    @property
    def name(self) -> str:
        return "grok"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call Grok to screen *market_data* for sentiment/futures setups."""
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
            "reasoning": "Grok: no actionable sentiment signal.",
        }

        if not self.api_key:
            logger.warning("[GrokAgent] No API key configured — returning NEUTRAL")
            return neutral

        client = AsyncOpenAI(api_key=self.api_key, base_url=_XAI_BASE_URL)

        for attempt in range(_RETRY_ATTEMPTS + 1):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
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
                logger.warning("[GrokAgent] JSON parse error for %s (attempt %d): %s", pair, attempt, exc)
                return {**neutral, "reasoning": f"Grok: JSON parse error — {exc}"}
            except APIError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("[GrokAgent] API error for %s (attempt %d): %s", pair, attempt, exc)
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    logger.error("[GrokAgent] API error after retries for %s: %s", pair, exc)
                    return {**neutral, "reasoning": f"Grok: API error — {exc}"}
            except Exception as exc:
                logger.error("[GrokAgent] Unexpected error for %s: %s", pair, exc)
                return {**neutral, "reasoning": f"Grok: unexpected error — {exc}"}

        return neutral

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call Grok to summarise a list of signals."""
        if not self.api_key:
            return f"[Grok] {len(signals)} signal(s) — API key not configured."

        prompt = "Summarise the following trading signals in 2-3 sentences:\n" + json.dumps(signals, indent=2)
        try:
            client = AsyncOpenAI(api_key=self.api_key, base_url=_XAI_BASE_URL)
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
            logger.error("[GrokAgent] summarize failed: %s", exc)
            return f"[Grok] Summary unavailable — {exc}"
