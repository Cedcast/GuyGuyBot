"""
main.py
-------
GuyGuyBot entry point.

Startup sequence
----------------
1. Load config from config.yaml
2. Initialise SQLite database
3. Create SignalLogger and RiskGate
4. Build the LLM AgentPipeline
5. Instantiate ExchangeClient and SentimentFetcher
6. Instantiate the trade engines (wired to ExchangeClient)
7. Start the Telegram bot
8. Start APScheduler
9. Run all scan loops and the bot concurrently in a single asyncio event loop
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agents.pipeline import build_pipeline
from core.config import load_config
from core.database import Database
from core.signal_logger import SignalLogger
from data.exchange_client import ExchangeClient
from engines.scalping_engine import ScalpingEngine
from engines.swing_engine import SwingEngine
from news.sentiment import SentimentFetcher
from risk.gating import RiskGate
from telegram.bot import GuyGuyBot

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("guygubot.main")


# ---------------------------------------------------------------------------
# Engine scan loop
# ---------------------------------------------------------------------------

async def _engine_loop(
    engine: ScalpingEngine | SwingEngine,
    pipeline: Any,
    signal_logger: SignalLogger,
    risk_gate: RiskGate,
    bot: GuyGuyBot,
    scan_interval: int,
    sentiment_fetcher: SentimentFetcher | None = None,
) -> None:
    """Continuously run *engine*.scan() every *scan_interval* seconds.

    For each candidate signal that passes risk gating, the agent pipeline
    is invoked.  A confirmed signal is logged and sent to Telegram.
    News context is fetched once per scan cycle and injected into each
    pipeline call.
    """
    name = engine.engine_name
    logger.info("Engine loop started: %s (interval=%ds)", name, scan_interval)

    while True:
        try:
            # Fetch news context once per scan cycle
            news_context: dict[str, Any] = {}
            if sentiment_fetcher is not None:
                try:
                    pairs = engine.get_pairs()
                    news_context = await sentiment_fetcher.get_market_context(pairs)
                    logger.debug("News context fetched: F&G=%s, sentiment=%.2f",
                                 news_context.get("fear_greed_index"),
                                 news_context.get("market_sentiment_score", 0))
                except Exception as exc:
                    logger.warning("Failed to fetch news context: %s", exc)

            candidates = await engine.scan()

            for candidate in candidates:
                pair = candidate["pair"]
                engine_name = candidate["engine"]

                if not risk_gate.can_trade(pair, engine_name):
                    logger.debug("Skipping %s/%s — risk gate blocked", pair, engine_name)
                    continue

                market_data = candidate.get("market_data", candidate)
                context = {"news_context": news_context}
                final_signal = await pipeline.run_pipeline(market_data, context=context)

                if final_signal is None:
                    logger.debug("No actionable signal for %s/%s", pair, engine_name)
                    continue

                # Merge engine metadata into the pipeline output
                full_signal: dict[str, Any] = {
                    **candidate,
                    **final_signal,
                    "pair": pair,
                    "engine": engine_name,
                }
                # Attach news context to signal for Telegram message formatting
                if news_context:
                    full_signal["news_context"] = news_context

                signal_id = signal_logger.log_signal(full_signal)
                full_signal["signal_id"] = signal_id
                full_signal["id"] = signal_id

                await bot.send_signal(full_signal)

        except asyncio.CancelledError:
            logger.info("Engine loop cancelled: %s", name)
            break
        except Exception as exc:
            logger.exception("Unhandled error in %s engine loop: %s", name, exc)

        await asyncio.sleep(scan_interval)


# ---------------------------------------------------------------------------
# Trade monitor loop
# ---------------------------------------------------------------------------

async def _trade_monitor_loop(
    signal_logger: SignalLogger,
    risk_gate: RiskGate,
    bot: GuyGuyBot,
    exchange_client: ExchangeClient,
    poll_interval: int = 30,
) -> None:
    """Periodically check open trades for TP/SL hits using live prices."""
    logger.info("Trade monitor loop started (poll_interval=%ds)", poll_interval)

    while True:
        try:
            open_trades = signal_logger.get_open_trades()
            for trade in open_trades:
                close_price = await exchange_client.fetch_current_price(trade["pair"])
                if close_price is None:
                    continue

                outcome = _check_trade_outcome(trade, close_price)
                if outcome:
                    closed = signal_logger.close_trade(trade["id"], close_price, outcome)
                    risk_gate.close_position(trade["pair"], trade["engine"])
                    await bot.send_trade_outcome(closed)

        except asyncio.CancelledError:
            logger.info("Trade monitor loop cancelled")
            break
        except Exception as exc:
            logger.exception("Trade monitor error: %s", exc)

        await asyncio.sleep(poll_interval)


def _check_trade_outcome(trade: dict[str, Any], current_price: float) -> str | None:
    """Return ``'TP'``, ``'SL'``, or ``None`` based on *current_price*."""
    entry = float(trade["entry"])
    stop_loss = float(trade["stop_loss"])
    take_profit = float(trade["take_profit"])
    direction = trade["direction"]

    if direction == "LONG":
        if current_price >= take_profit:
            return "TP"
        if current_price <= stop_loss:
            return "SL"
    else:  # SHORT
        if current_price <= take_profit:
            return "TP"
        if current_price >= stop_loss:
            return "SL"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Bootstrap all components and run the event loop."""
    # 1. Config
    config = load_config()
    logger.info("Config loaded — pairs: %s", config.trading.pairs)

    # Paper trading warning
    if config.paper_trading.enabled:
        logger.warning("=" * 60)
        logger.warning("PAPER TRADING MODE ENABLED — no real orders will be placed")
        logger.warning(config.paper_trading.note)
        logger.warning("=" * 60)

    # 2. Database
    db = Database(config.database.path)
    db.initialize()

    # 3. Signal logger + risk gate
    signal_logger = SignalLogger(
        db=db,
        account_size=config.trading.account_size,
        risk_per_trade=config.trading.risk_per_trade,
    )
    risk_gate = RiskGate(db)

    # 4. Agent pipeline
    pipeline = build_pipeline(config.llm)

    # 5. ExchangeClient + SentimentFetcher
    exchange_client = ExchangeClient()
    sentiment_fetcher: SentimentFetcher | None = None
    if config.news.enabled:
        sentiment_fetcher = SentimentFetcher(
            cryptopanic_token=config.news.cryptopanic_token,
            cache_ttl=config.news.cache_ttl_seconds,
        )
        logger.info("News/sentiment fetcher enabled")

    # 6. Trade engines (wired to ExchangeClient)
    engines_tasks: list[asyncio.Task] = []
    active_engines = []

    if config.engines.scalping.enabled:
        scalping = ScalpingEngine(
            pairs=config.trading.pairs,
            timeframes=config.engines.scalping.timeframes,
            exchange_client=exchange_client,
        )
        active_engines.append((scalping, config.engines.scalping.scan_interval))

    if config.engines.swing.enabled:
        swing = SwingEngine(
            pairs=config.trading.pairs,
            timeframes=config.engines.swing.timeframes,
            exchange_client=exchange_client,
        )
        active_engines.append((swing, config.engines.swing.scan_interval))

    # 7. Telegram bot
    bot = GuyGuyBot(config=config, signal_logger=signal_logger, risk_gate=risk_gate)
    await bot.start_polling()

    # 8. APScheduler
    scheduler = AsyncIOScheduler(timezone="UTC")
    bot.setup_scheduler(scheduler)
    scheduler.start()

    # 9. Launch all tasks
    for eng, interval in active_engines:
        task = asyncio.create_task(
            _engine_loop(eng, pipeline, signal_logger, risk_gate, bot, interval, sentiment_fetcher),
            name=f"engine_{eng.engine_name}",
        )
        engines_tasks.append(task)

    monitor_task = asyncio.create_task(
        _trade_monitor_loop(signal_logger, risk_gate, bot, exchange_client),
        name="trade_monitor",
    )

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown_handler)

    logger.info("GuyGuyBot running — press Ctrl+C to stop")
    await stop_event.wait()

    logger.info("Shutting down…")
    for task in engines_tasks + [monitor_task]:
        task.cancel()
    await asyncio.gather(*engines_tasks, monitor_task, return_exceptions=True)

    scheduler.shutdown(wait=False)
    await bot.stop()
    if sentiment_fetcher is not None:
        await sentiment_fetcher.close()
    await exchange_client.close()
    logger.info("GuyGuyBot stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
