"""
agents/gpt_agent.py
-------------------
GPT-4o (OpenAI) LLM agent — real OpenAI API implementation.
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

_SYSTEM_PROMPT = """You are an expert crypto derivatives trader specialising in perpetual futures.
You have deep knowledge of technical analysis, funding rates, open interest, and market microstructure.

When analysing market data, you MUST consider:
1. Price action and trend direction
2. RSI, EMA crossovers, MACD for momentum
3. Funding rate direction (positive = longs paying = potential short squeeze risk)
4. Open interest trend (rising OI confirms trend, falling OI = weakening)
5. Fear & Greed Index (extreme greed > 80 = caution on longs, extreme fear < 20 = caution on shorts)
6. News sentiment (negative news = avoid longs, positive news = confirms bullish bias)
7. Volume confirmation (high volume breakouts are more reliable)

You MUST respond with ONLY a valid JSON object, no markdown, no explanation outside the JSON:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "entry": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "leverage": <int between 3 and 20>,
  "reasoning": "<concise explanation max 200 chars>",
  "key_factors": ["factor1", "factor2", "factor3"]
}

Only emit LONG or SHORT if confidence >= 0.60. Otherwise emit NEUTRAL.
For futures, always calculate TP/SL based on ATR if provided, maintaining minimum 2:1 R:R."""

_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 1.0


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, handling markdown code blocks."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _build_prompt(market_data: dict[str, Any], context: dict[str, Any]) -> str:
    """Build a structured prompt from market data and context."""
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


class GPTAgent(BaseAgent):
    """LLM agent backed by OpenAI's GPT-4o API."""

    @property
    def name(self) -> str:
        return "gpt4o"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call GPT-4o to analyse *market_data* and return a structured signal."""
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
            "reasoning": "GPT-4o: no actionable signal.",
        }

        if not self.api_key:
            logger.warning("[GPTAgent] No API key configured — returning NEUTRAL")
            return neutral

        client = AsyncOpenAI(api_key=self.api_key)

        for attempt in range(_RETRY_ATTEMPTS + 1):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
                raw_text = response.choices[0].message.content or ""
                parsed = _parse_llm_json(raw_text)
                return {
                    "agent": self.name,
                    "direction": parsed.get("direction", "NEUTRAL"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "entry": float(parsed.get("entry", close)),
                    "stop_loss": float(parsed.get("stop_loss", 0.0)),
                    "take_profit": float(parsed.get("take_profit", 0.0)),
                    "leverage": int(parsed.get("leverage", 10)),
                    "reasoning": str(parsed.get("reasoning", ""))[:200],
                    "key_factors": parsed.get("key_factors", []),
                }
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("[GPTAgent] JSON parse error for %s (attempt %d): %s", pair, attempt, exc)
                return {**neutral, "reasoning": f"GPT-4o: JSON parse error — {exc}"}
            except APIError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("[GPTAgent] API error for %s (attempt %d): %s", pair, attempt, exc)
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    logger.error("[GPTAgent] API error after retries for %s: %s", pair, exc)
                    return {**neutral, "reasoning": f"GPT-4o: API error — {exc}"}
            except Exception as exc:
                logger.error("[GPTAgent] Unexpected error for %s: %s", pair, exc)
                return {**neutral, "reasoning": f"GPT-4o: unexpected error — {exc}"}

        return neutral

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call GPT-4o to summarise a list of signals."""
        if not self.api_key:
            return f"[GPT-4o] {len(signals)} signal(s) — API key not configured."

        prompt = "Summarise the following trading signals in 2-3 sentences:\n" + json.dumps(signals, indent=2)
        try:
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a concise crypto trading analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("[GPTAgent] summarize failed: %s", exc)
            return f"[GPT-4o] Summary unavailable — {exc}"
