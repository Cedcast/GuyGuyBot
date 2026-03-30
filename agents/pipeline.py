"""
agents/pipeline.py
------------------
AgentPipeline implements the 3-stage tiered LLM pipeline:

Stage 1 — Screeners (always called, cheap):
    GPT-4o-mini (TA) + Grok-3-mini (sentiment) run concurrently.
    Both must agree on the same non-NEUTRAL direction AND both must
    reach the ``screener_confidence_gate`` (default 0.65).
    If either condition fails, ``None`` is returned and Claude is never
    called.

Stage 2 — Final validator (only called on screener consensus):
    Claude Opus receives the full market context plus the screener
    pre-analysis and makes the final go/no-go decision.
    Returns ``None`` when NEUTRAL or below threshold.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Final signal confidence threshold (Claude's output must clear this)
_CONFIDENCE_THRESHOLD = 0.65


class AgentPipeline:
    """Orchestrates the tiered screener → decision-maker LLM pipeline.

    Parameters
    ----------
    screener_agents:
        Exactly two :class:`~agents.base_agent.BaseAgent` instances that
        run in parallel on every candidate (cheap, fast models).
    decision_agent:
        A single :class:`~agents.base_agent.BaseAgent` (Claude Opus) that
        only activates when both screeners reach consensus.
    screener_confidence_gate:
        Minimum confidence that *both* screeners must report before the
        decision agent is invoked.  Defaults to ``0.65``.
    """

    def __init__(
        self,
        screener_agents: list[BaseAgent],
        decision_agent: BaseAgent,
        screener_confidence_gate: float = 0.65,
    ) -> None:
        if len(screener_agents) != 2:
            raise ValueError("AgentPipeline requires exactly 2 screener agents")
        self._screeners = screener_agents
        self._decision = decision_agent
        self._gate = screener_confidence_gate

    async def run_pipeline(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run the full tiered pipeline for *market_data*.

        Parameters
        ----------
        market_data:
            Market snapshot dict (pair, timeframe, ohlcv, indicators …).
        context:
            Optional extra context forwarded to each agent.  Include a
            ``news_context`` key to inject real-time sentiment data.

        Returns
        -------
        A final signal dict from Claude, or ``None`` if the screeners
        did not reach consensus or Claude vetoed the trade.
        """
        ctx = context or {}
        pair = market_data.get("pair", "?")

        # ------------------------------------------------------------------
        # Stage 1: Run both screeners concurrently (always, every candidate)
        # ------------------------------------------------------------------
        logger.debug("Pipeline Stage 1: running screeners for %s", pair)
        results = await asyncio.gather(
            self._screeners[0].analyze(market_data, ctx),
            self._screeners[1].analyze(market_data, ctx),
            return_exceptions=True,
        )

        valid: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Screener[%d] raised: %s", i, result)
            else:
                valid.append(result)

        if len(valid) < 2:
            logger.warning("Pipeline: screener(s) failed for %s — skipping", pair)
            return None

        screener_a, screener_b = valid[0], valid[1]
        dir_a = screener_a.get("direction", "NEUTRAL")
        dir_b = screener_b.get("direction", "NEUTRAL")
        conf_a = float(screener_a.get("confidence", 0.0))
        conf_b = float(screener_b.get("confidence", 0.0))

        # Both screeners must agree on the same non-NEUTRAL direction
        if dir_a == "NEUTRAL" or dir_b == "NEUTRAL" or dir_a != dir_b:
            logger.info(
                "Pipeline: screeners disagree or NEUTRAL for %s (%s@%.2f vs %s@%.2f) — no Claude call",
                pair, dir_a, conf_a, dir_b, conf_b,
            )
            return None

        # Both screeners must clear the confidence gate
        if conf_a < self._gate or conf_b < self._gate:
            logger.info(
                "Pipeline: screener confidence below gate (%.2f) for %s (%.2f / %.2f) — no Claude call",
                self._gate, pair, conf_a, conf_b,
            )
            return None

        logger.info(
            "Pipeline: screener consensus %s for %s (%.2f / %.2f) — invoking Claude",
            dir_a, pair, conf_a, conf_b,
        )

        # ------------------------------------------------------------------
        # Stage 2: Final validation by Claude (only reached on consensus)
        # ------------------------------------------------------------------
        screener_consensus = {
            self._screeners[0].name: {
                "direction": dir_a,
                "confidence": conf_a,
                "reasoning": screener_a.get("reasoning", ""),
            },
            self._screeners[1].name: {
                "direction": dir_b,
                "confidence": conf_b,
                "reasoning": screener_b.get("reasoning", ""),
            },
        }
        decision_ctx = {**ctx, "screener_consensus": screener_consensus}

        try:
            decision = await self._decision.analyze(market_data, decision_ctx)
        except Exception as exc:
            logger.error("Decision agent raised for %s: %s", pair, exc)
            return None

        logger.info(
            "Pipeline: Claude returned %s@%.2f for %s",
            decision.get("direction"), decision.get("confidence", 0.0), pair,
        )
        return self._emit_if_confident(decision)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_if_confident(signal: dict[str, Any]) -> dict[str, Any] | None:
        """Return *signal* if its confidence meets the threshold, else ``None``."""
        if signal.get("direction", "NEUTRAL") == "NEUTRAL":
            return None
        if float(signal.get("confidence", 0.0)) < _CONFIDENCE_THRESHOLD:
            logger.debug(
                "Signal confidence %.2f below threshold — suppressed",
                signal.get("confidence", 0.0),
            )
            return None
        return signal


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

    screeners: list[BaseAgent] = []
    for name in config_llm.screener_agents[:2]:
        if name == "gpt4o":
            screeners.append(GPTAgent(api_key=config_llm.gpt4o_api_key, model=config_llm.gpt4o_model))
        elif name == "grok":
            screeners.append(GrokAgent(api_key=config_llm.grok_api_key, model=config_llm.grok_model))
        elif name == "claude":
            screeners.append(ClaudeAgent(api_key=config_llm.claude_api_key, model=config_llm.claude_model))
        else:
            raise ValueError(f"Unknown screener agent: {name!r}")

    decision_name = config_llm.decision_agent
    if decision_name == "claude":
        decision: BaseAgent = ClaudeAgent(api_key=config_llm.claude_api_key, model=config_llm.claude_model)
    elif decision_name == "gpt4o":
        decision = GPTAgent(api_key=config_llm.gpt4o_api_key, model=config_llm.gpt4o_model)
    elif decision_name == "grok":
        decision = GrokAgent(api_key=config_llm.grok_api_key, model=config_llm.grok_model)
    else:
        raise ValueError(f"Unknown decision agent: {decision_name!r}")

    return AgentPipeline(
        screener_agents=screeners,
        decision_agent=decision,
        screener_confidence_gate=config_llm.screener_confidence_gate,
    )
