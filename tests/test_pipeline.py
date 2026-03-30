"""
tests/test_pipeline.py
----------------------
Unit tests for agents/pipeline.py (AgentPipeline.run_pipeline).
All LLM agent calls are replaced with AsyncMock objects.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.pipeline import AgentPipeline


# ---------------------------------------------------------------------------
# Helper: create a mock BaseAgent
# ---------------------------------------------------------------------------

def make_agent(name: str, analyze_return: dict) -> MagicMock:
    """Return a MagicMock that quacks like a BaseAgent."""
    agent = MagicMock()
    agent.name = name
    agent.analyze = AsyncMock(return_value=analyze_return)
    return agent


# Minimal signal dict used as the market_data input to run_pipeline
_SAMPLE_SIGNAL = {
    "pair": "BTCUSDT",
    "timeframe": "30m",
    "close": 100.0,
    "indicators": {},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_skips_claude_when_screeners_disagree():
    """Opposing directions → Claude never called, returns None."""
    gpt = make_agent("gpt", {"direction": "LONG", "confidence": 0.70, "reasoning": "TA bullish"})
    grok = make_agent("grok", {"direction": "SHORT", "confidence": 0.70, "reasoning": "Sentiment bearish"})
    claude = make_agent("claude", {"direction": "LONG", "confidence": 0.80, "reasoning": "ok"})

    pipeline = AgentPipeline(
        screener_agents=[gpt, grok],
        decision_agent=claude,
        screener_confidence_gate=0.65,
    )

    result = await pipeline.run_pipeline(_SAMPLE_SIGNAL, context={})

    assert result is None
    claude.analyze.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_skips_claude_when_confidence_below_gate():
    """Both screeners agree direction but confidence below gate → Claude not called."""
    gpt = make_agent("gpt", {"direction": "LONG", "confidence": 0.55, "reasoning": "weak"})
    grok = make_agent("grok", {"direction": "LONG", "confidence": 0.55, "reasoning": "weak"})
    claude = make_agent("claude", {"direction": "LONG", "confidence": 0.80, "reasoning": "ok"})

    pipeline = AgentPipeline(
        screener_agents=[gpt, grok],
        decision_agent=claude,
        screener_confidence_gate=0.65,
    )

    result = await pipeline.run_pipeline(_SAMPLE_SIGNAL, context={})

    assert result is None
    claude.analyze.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_calls_claude_when_screeners_agree():
    """Both screeners agree LONG above gate → Claude called, result returned."""
    gpt = make_agent("gpt", {"direction": "LONG", "confidence": 0.72, "reasoning": "bullish"})
    grok = make_agent("grok", {"direction": "LONG", "confidence": 0.72, "reasoning": "bullish"})
    claude = make_agent(
        "claude",
        {
            "direction": "LONG",
            "confidence": 0.80,
            "reasoning": "confirmed",
            "entry": 100.0,
            "stop_loss": 97.0,
            "take_profit": 106.0,
            "leverage": 5,
        },
    )

    pipeline = AgentPipeline(
        screener_agents=[gpt, grok],
        decision_agent=claude,
        screener_confidence_gate=0.65,
    )

    result = await pipeline.run_pipeline(_SAMPLE_SIGNAL, context={})

    claude.analyze.assert_called_once()
    assert result is not None
    assert result["direction"] == "LONG"


@pytest.mark.asyncio
async def test_pipeline_returns_none_when_claude_neutral():
    """Both screeners agree LONG, but Claude returns NEUTRAL → None."""
    gpt = make_agent("gpt", {"direction": "LONG", "confidence": 0.72, "reasoning": "bullish"})
    grok = make_agent("grok", {"direction": "LONG", "confidence": 0.72, "reasoning": "bullish"})
    claude = make_agent(
        "claude",
        {"direction": "NEUTRAL", "confidence": 0.40, "reasoning": "no trade"},
    )

    pipeline = AgentPipeline(
        screener_agents=[gpt, grok],
        decision_agent=claude,
        screener_confidence_gate=0.65,
    )

    result = await pipeline.run_pipeline(_SAMPLE_SIGNAL, context={})

    claude.analyze.assert_called_once()
    assert result is None


@pytest.mark.asyncio
async def test_pipeline_returns_none_when_screeners_neutral():
    """Both screeners return NEUTRAL → Claude not called, returns None."""
    gpt = make_agent("gpt", {"direction": "NEUTRAL", "confidence": 0.50, "reasoning": "unclear"})
    grok = make_agent("grok", {"direction": "NEUTRAL", "confidence": 0.50, "reasoning": "unclear"})
    claude = make_agent("claude", {"direction": "LONG", "confidence": 0.80, "reasoning": "ok"})

    pipeline = AgentPipeline(
        screener_agents=[gpt, grok],
        decision_agent=claude,
        screener_confidence_gate=0.65,
    )

    result = await pipeline.run_pipeline(_SAMPLE_SIGNAL, context={})

    assert result is None
    claude.analyze.assert_not_called()
