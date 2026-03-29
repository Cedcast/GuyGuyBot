"""
agents/pipeline.py
------------------
AgentPipeline orchestrates the multi-LLM signal generation workflow:

1. Run the two *primary* agents in parallel.
2. If both agree on direction → combine their recommendations.
3. If they disagree → invoke the *arbitration* agent as a tie-breaker.
4. Return ``None`` if no actionable signal emerges (both NEUTRAL, or
   arbitration is also NEUTRAL).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Minimum confidence threshold to emit a signal.
_CONFIDENCE_THRESHOLD = 0.5


def _agents_agree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return ``True`` when two agent responses share the same non-NEUTRAL direction."""
    return (
        a.get("direction") == b.get("direction")
        and a.get("direction") not in (None, "NEUTRAL")
    )


def _average_recommendation(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge a list of agent responses into a single averaged recommendation."""
    direction = responses[0]["direction"]
    avg_confidence = sum(r.get("confidence", 0.0) for r in responses) / len(responses)
    avg_entry = sum(r.get("entry", 0.0) for r in responses) / len(responses)
    avg_sl = sum(r.get("stop_loss", 0.0) for r in responses) / len(responses)
    avg_tp = sum(r.get("take_profit", 0.0) for r in responses) / len(responses)
    reasonings = " | ".join(r.get("reasoning", "") for r in responses)
    return {
        "direction": direction,
        "confidence": round(avg_confidence, 4),
        "entry": round(avg_entry, 6),
        "stop_loss": round(avg_sl, 6),
        "take_profit": round(avg_tp, 6),
        "reasoning": reasonings,
        "agents": [r.get("agent", "unknown") for r in responses],
    }


class AgentPipeline:
    """Orchestrates primary and arbitration LLM agents for signal generation.

    Parameters
    ----------
    primary_agents:
        Exactly two :class:`~agents.base_agent.BaseAgent` instances that
        run in parallel for every signal scan.
    arbitration_agent:
        A third :class:`~agents.base_agent.BaseAgent` used when the
        primaries disagree.
    """

    def __init__(
        self,
        primary_agents: list[BaseAgent],
        arbitration_agent: BaseAgent,
    ) -> None:
        if len(primary_agents) != 2:
            raise ValueError("AgentPipeline requires exactly 2 primary agents")
        self._primaries = primary_agents
        self._arbitrator = arbitration_agent

    async def run_pipeline(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run the full agent pipeline for *market_data*.

        Parameters
        ----------
        market_data:
            Market snapshot dict (pair, timeframe, ohlcv, indicators …).
        context:
            Optional extra context forwarded to each agent.

        Returns
        -------
        A merged signal dict, or ``None`` if no actionable signal is found.
        """
        ctx = context or {}
        pair = market_data.get("pair", "?")

        # Step 1: Run both primaries concurrently.
        logger.debug("Pipeline: running primary agents for %s", pair)
        results = await asyncio.gather(
            self._primaries[0].analyze(market_data, ctx),
            self._primaries[1].analyze(market_data, ctx),
            return_exceptions=True,
        )

        # Separate successful responses from errors.
        valid: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Primary agent[%d] raised: %s", i, result)
            else:
                valid.append(result)

        if not valid:
            logger.warning("Pipeline: all primary agents failed for %s", pair)
            return None

        if len(valid) == 1:
            # Only one agent responded; use it if above threshold.
            return self._emit_if_confident(valid[0])

        # Step 2: Check agreement.
        if _agents_agree(valid[0], valid[1]):
            logger.info("Pipeline: primary agents AGREE on %s for %s", valid[0]["direction"], pair)
            merged = _average_recommendation(valid)
            return self._emit_if_confident(merged)

        # Step 3: Agents disagree → run arbitration.
        logger.info("Pipeline: primaries DISAGREE for %s — invoking arbitrator", pair)
        arb_context = {**ctx, "primary_responses": valid}
        try:
            arb_result = await self._arbitrator.analyze(market_data, arb_context)
        except Exception as exc:
            logger.error("Arbitration agent raised: %s", exc)
            return None

        if arb_result.get("direction", "NEUTRAL") == "NEUTRAL":
            logger.info("Pipeline: arbitrator returned NEUTRAL for %s — no signal", pair)
            return None

        # The arbitrator breaks the tie.
        logger.info("Pipeline: arbitrator chose %s for %s", arb_result["direction"], pair)
        arb_result["agents"] = [
            v.get("agent", "unknown") for v in valid
        ] + [self._arbitrator.name]
        return self._emit_if_confident(arb_result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_if_confident(signal: dict[str, Any]) -> dict[str, Any] | None:
        """Return *signal* if its confidence meets the threshold, else ``None``."""
        if signal.get("direction", "NEUTRAL") == "NEUTRAL":
            return None
        if float(signal.get("confidence", 0.0)) < _CONFIDENCE_THRESHOLD:
            logger.debug("Signal confidence %.2f below threshold — suppressed", signal.get("confidence", 0.0))
            return None
        return signal

    def build_agent_factory(config_llm: Any) -> "AgentPipeline":  # noqa: N805
        """Convenience factory — not a method; see :func:`build_pipeline`."""
        ...  # pragma: no cover


def build_pipeline(config_llm: Any) -> AgentPipeline:
    """Construct an :class:`AgentPipeline` from the ``llm`` config section.

    Imports agent classes lazily so that missing optional dependencies do
    not crash the entire application.

    Parameters
    ----------
    config_llm:
        The :class:`~core.config.LLMConfig` dataclass from the loaded config.
    """
    from agents.claude_agent import ClaudeAgent
    from agents.gpt_agent import GPTAgent
    from agents.grok_agent import GrokAgent

    _agent_classes = {
        "claude": (ClaudeAgent, config_llm.claude_api_key),
        "gpt5": (GPTAgent, config_llm.gpt5_api_key),
        "grok": (GrokAgent, config_llm.grok_api_key),
    }

    def _make(name: str) -> BaseAgent:
        cls, key = _agent_classes[name]
        return cls(api_key=key)

    primaries = [_make(n) for n in config_llm.primary_agents[:2]]
    arbitrator = _make(config_llm.arbitration_agent)
    return AgentPipeline(primary_agents=primaries, arbitration_agent=arbitrator)
