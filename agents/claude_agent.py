"""
agents/claude_agent.py
----------------------
Claude Opus (Anthropic) LLM agent — final decision validator (Stage 2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import anthropic

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-opus-4-5"

_SYSTEM_PROMPT = """You are the final decision authority for a crypto futures trading bot.
Two specialist screeners have already pre-filtered this signal:
- A technical analysis screener
- A sentiment and futures data screener

Both screeners agreed on a direction with sufficient confidence.
Your job is to make the FINAL go/no-go decision with full context.

Consider EVERYTHING:
1. Technical setup quality (price action, indicators, trend strength)
2. Futures market structure (funding rate, open interest, long/short ratio)
3. Sentiment context (fear/greed, news headlines, market mood)
4. Risk assessment (is the reward worth the risk right now?)
5. Macro context (are there red flags the screeners may have missed?)
6. Leverage recommendation (conservative 3-10x based on conviction)

The screener pre-analysis will be provided in the context.

You MUST respond with ONLY a valid JSON object:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "entry": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "leverage": <int between 3 and 15>,
  "reasoning": "<concise explanation max 200 chars>",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_notes": "<any risk warnings, max 100 chars>"
}

Only emit LONG or SHORT if confidence >= 0.65.
You have the final veto — if anything feels wrong, return NEUTRAL.
Maintain minimum 2:1 R:R on all trades."""

_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 1.0


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _build_prompt(market_data: dict[str, Any], context: dict[str, Any]) -> str:
    """Build a full-context prompt including screener consensus when available."""
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

    # Include screener pre-analysis when available (tiered pipeline context)
    screener_consensus = context.get("screener_consensus")
    if screener_consensus:
        parts.append("")
        parts.append("Screener Pre-Analysis:")
        gpt_screen = screener_consensus.get("gpt4o", {})
        grok_screen = screener_consensus.get("grok", {})
        if gpt_screen:
            parts.append(
                f"- TA Screener (GPT): {gpt_screen.get('direction')} "
                f"@ {gpt_screen.get('confidence', 0):.2f} — {gpt_screen.get('reasoning', '')}"
            )
        if grok_screen:
            parts.append(
                f"- Sentiment Screener (Grok): {grok_screen.get('direction')} "
                f"@ {grok_screen.get('confidence', 0):.2f} — {grok_screen.get('reasoning', '')}"
            )

    return "\n".join(parts)


class ClaudeAgent(BaseAgent):
    """LLM agent backed by Anthropic's Claude API (final decision validator)."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        super().__init__(api_key)
        self.model = model

    @property
    def name(self) -> str:
        return "claude"

    async def analyze(self, market_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Call Claude to make the final go/no-go decision on *market_data*."""
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
            "reasoning": "Claude: no actionable signal.",
        }

        if not self.api_key:
            logger.warning("[ClaudeAgent] No API key configured — returning NEUTRAL")
            return neutral

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        for attempt in range(_RETRY_ATTEMPTS + 1):
            try:
                message = await client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = message.content[0].text
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
                    "risk_notes": str(parsed.get("risk_notes", ""))[:100],
                }
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("[ClaudeAgent] JSON parse error for %s (attempt %d): %s", pair, attempt, exc)
                return {**neutral, "reasoning": f"Claude: JSON parse error — {exc}"}
            except anthropic.APIError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("[ClaudeAgent] API error for %s (attempt %d): %s", pair, attempt, exc)
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    logger.error("[ClaudeAgent] API error after retries for %s: %s", pair, exc)
                    return {**neutral, "reasoning": f"Claude: API error — {exc}"}
            except Exception as exc:
                logger.error("[ClaudeAgent] Unexpected error for %s: %s", pair, exc)
                return {**neutral, "reasoning": f"Claude: unexpected error — {exc}"}

        return neutral

    async def summarize(self, signals: list[dict[str, Any]]) -> str:
        """Call Claude to summarise a list of signals."""
        if not self.api_key:
            return f"[Claude] {len(signals)} signal(s) — API key not configured."

        prompt = "Summarise the following trading signals in 2-3 sentences:\n" + json.dumps(signals, indent=2)
        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            message = await client.messages.create(
                model=self.model,
                max_tokens=256,
                system="You are a concise crypto trading analyst.",
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            logger.error("[ClaudeAgent] summarize failed: %s", exc)
            return f"[Claude] Summary unavailable — {exc}"
