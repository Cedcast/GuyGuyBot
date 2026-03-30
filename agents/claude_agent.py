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
2. ADX/regime context — only trade STRONG_TREND or WEAK_TREND; reject RANGING markets
3. BB squeeze warnings — low bandwidth (< 0.02) signals a potential volatility breakout
4. S/R distances — validate that SL sits behind a key level and TP has room to breathe
5. DI bias — +DI vs -DI alignment should agree with the intended direction
6. Futures market structure (funding rate, open interest, long/short ratio)
7. Sentiment context (fear/greed, news headlines, market mood)
8. Risk assessment (is the reward worth the risk right now?)
9. Macro context (are there red flags the screeners may have missed?)
10. Leverage recommendation (conservative 3-10x based on conviction)

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

Only emit LONG or SHORT if confidence >= 0.70.
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
    """Build a full-context structured prompt for Claude including screener consensus."""
    pair = market_data.get("pair", "?")
    timeframe = market_data.get("timeframe", "?")
    engine = market_data.get("engine", "?")
    direction = market_data.get("direction", "?")
    entry = market_data.get("entry", market_data.get("close", 0))
    stop_loss = market_data.get("stop_loss", 0)
    take_profit = market_data.get("take_profit", 0)
    tp1 = market_data.get("take_profit_1", "N/A")
    tp2 = market_data.get("take_profit_2", "N/A")
    candle_pct = market_data.get("candle_completion_pct")
    candle_pct_str = f"{candle_pct:.0f}%" if candle_pct is not None else "N/A"
    confidence = float(market_data.get("confidence", 0.0))
    reasoning = market_data.get("reasoning", "N/A")
    close = float(market_data.get("close", entry or 0))

    indicators = market_data.get("indicators", {})

    # Regime / ADX — prefer top-level keys set by the engine, fall back to indicators dict
    regime = market_data.get("regime") or indicators.get("regime", "N/A")
    adx = float(market_data.get("adx") if market_data.get("adx") is not None else indicators.get("adx", 0))
    plus_di = float(market_data.get("plus_di") if market_data.get("plus_di") is not None else indicators.get("plus_di", 0))
    minus_di = float(market_data.get("minus_di") if market_data.get("minus_di") is not None else indicators.get("minus_di", 0))

    # S/R / VWAP
    support = float(market_data.get("support") if market_data.get("support") is not None else indicators.get("support", 0))
    resistance = float(market_data.get("resistance") if market_data.get("resistance") is not None else indicators.get("resistance", 0))
    vwap = float(market_data.get("vwap") if market_data.get("vwap") is not None else indicators.get("vwap", close))

    support_dist = f"{abs(close - support) / close * 100:.2f}%" if support > 0 and close > 0 else "N/A"
    resistance_dist = f"{abs(resistance - close) / close * 100:.2f}%" if resistance > 0 and close > 0 else "N/A"

    rsi = indicators.get("rsi", "N/A")
    ema_crossover = indicators.get("ema_crossover", "N/A")
    macd_hist = indicators.get("macd", {}).get("histogram", "N/A") if isinstance(indicators.get("macd"), dict) else "N/A"
    stoch = indicators.get("stoch", {})
    stoch_k = stoch.get("k", "N/A") if isinstance(stoch, dict) else "N/A"
    stoch_d = stoch.get("d", "N/A") if isinstance(stoch, dict) else "N/A"
    bb = indicators.get("bb", {})
    bb_bw = float(bb.get("bandwidth", 0)) if isinstance(bb, dict) else 0.0
    atr = indicators.get("atr", "N/A")
    volume_ratio = float(indicators.get("volume_ratio", 1.0))

    # Futures data
    futures = market_data.get("futures", {})
    funding_rate = futures.get("funding_rate_avg", "N/A")
    oi_change = float(futures.get("oi_change_pct", 0)) if futures.get("oi_change_pct") is not None else 0.0
    ls_ratio = futures.get("ls_ratio_avg", "N/A")

    # News / sentiment context
    news_ctx = context.get("news_context", {})
    fear_greed = news_ctx.get("fear_greed_index", "N/A")
    fear_greed_label = news_ctx.get("fear_greed_label", "N/A")
    sentiment = float(news_ctx.get("market_sentiment_score", 0.0))
    lunarcrush = news_ctx.get("lunarcrush_score", "N/A")
    headlines = news_ctx.get("recent_headlines", [])[:3]
    headline_parts = [f"  {i + 1}. {h}" for i, h in enumerate(headlines)] if headlines else ["  N/A"]

    # Screener consensus
    screener_consensus = context.get("screener_consensus", {})
    gpt_screen = screener_consensus.get("gpt4o", {}) if screener_consensus else {}
    grok_screen = screener_consensus.get("grok", {}) if screener_consensus else {}
    gpt_line = (
        f"  TA Screener (GPT): {gpt_screen.get('direction', 'N/A')} "
        f"@ {float(gpt_screen.get('confidence', 0)):.2f} — {gpt_screen.get('reasoning', 'N/A')}"
        if gpt_screen else "  TA Screener (GPT): N/A"
    )
    grok_line = (
        f"  Sentiment Screener (Grok): {grok_screen.get('direction', 'N/A')} "
        f"@ {float(grok_screen.get('confidence', 0)):.2f} — {grok_screen.get('reasoning', 'N/A')}"
        if grok_screen else "  Sentiment Screener (Grok): N/A"
    )

    bb_squeeze_msg = "YES — low volatility breakout risk" if bb_bw < 0.02 else "No"
    vwap_pos = "ABOVE" if close > vwap else "BELOW"

    lines = [
        "=== SIGNAL BRIEF ===",
        f"Pair: {pair}  |  Timeframe: {timeframe}  |  Engine: {engine}",
        f"Direction: {direction}  |  Entry: {entry}  |  SL: {stop_loss}  |  TP: {take_profit}",
        f"TP1 (50% at 1:1 RR): {tp1}  |  TP2 (50% at 3:1 RR): {tp2}",
        f"Candle completion: {candle_pct_str}",
        f"Engine Confidence: {confidence:.2f}  |  Reasoning: {reasoning}",
        "",
        "=== MARKET REGIME ===",
        f"Regime: {regime}  |  ADX: {adx:.1f}  |  +DI: {plus_di:.1f}  |  -DI: {minus_di:.1f}",
        "(Only trade STRONG_TREND or WEAK_TREND regimes)",
        "",
        "=== TECHNICAL INDICATORS ===",
        f"RSI: {rsi}  |  EMA Crossover: {ema_crossover}  |  MACD Histogram: {macd_hist}",
        f"Stoch K/D: {stoch_k}/{stoch_d}  |  BB Bandwidth: {bb_bw:.4f}",
        f"BB Squeeze: {bb_squeeze_msg}",
        f"ATR: {atr}  |  Volume vs Avg: {volume_ratio:.1f}x",
        "",
        "=== STRUCTURE ===",
        f"Support: {support}  |  Distance from support: {support_dist}",
        f"Resistance: {resistance}  |  Distance to resistance: {resistance_dist}",
        f"VWAP: {vwap}  |  Price vs VWAP: {vwap_pos}",
        "",
        "=== FUTURES DATA ===",
        f"Funding Rate (avg): {funding_rate}",
        f"OI Change: {oi_change:.2f}%",
        f"Long/Short Ratio: {ls_ratio}",
        "",
        "=== SENTIMENT ===",
        f"Fear & Greed: {fear_greed} ({fear_greed_label})",
        f"Sentiment Score: {sentiment:.2f}",
        f"LunarCrush Score: {lunarcrush}",
        "Headlines:",
    ] + headline_parts + [
        "",
        "=== SCREENER PRE-ANALYSIS ===",
        gpt_line,
        grok_line,
    ]

    return "\n".join(lines)


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
