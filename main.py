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
5. Instantiate the trade engines
6. Start the Telegram bot
7. Start APScheduler
8. Run all scan loops and the bot concurrently in a single asyncio event loop
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
from engines.scalping_engine import ScalpingEngine
from engines.swing_engine import SwingEngine
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
) -> None:
    """Continuously run *engine*.scan() every *scan_interval* seconds.

    For each candidate signal that passes risk gating, the agent pipeline
    is invoked.  A confirmed signal is logged and sent to Telegram.
    """
    name = engine.engine_name
    logger.info("Engine loop started: %s (interval=%ds)", name, scan_interval)

    while True:
        try:
            candidates = await engine.scan()

            for candidate in candidates:
                pair = candidate["pair"]
                engine_name = candidate["engine"]

                if not risk_gate.can_trade(pair, engine_name):
                    logger.debug("Skipping %s/%s — risk gate blocked", pair, engine_name)
                    continue

                market_data = candidate.get("market_data", candidate)
                final_signal = await pipeline.run_pipeline(market_data)

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
    poll_interval: int = 30,
) -> None:
    """Periodically check open trades for TP/SL hits.

    **Stub**: the close price is fetched from a placeholder; replace with
    a real Binance price call to make this functional.
    """
    logger.info("Trade monitor loop started (poll_interval=%ds)", poll_interval)

    while True:
        try:
            open_trades = signal_logger.get_open_trades()
            for trade in open_trades:
                close_price = await _fetch_current_price(trade["pair"])
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


async def _fetch_current_price(pair: str) -> float | None:
    """Fetch the current mark price for *pair* from Binance.

    **Stub** — returns ``None`` (no real price fetching yet).

    To implement, call the Binance REST API::

        async with aiohttp.ClientSession() as session:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={pair}"
            async with session.get(url) as resp:
                data = await resp.json()
                return float(data["price"])
    """
    # TODO: Replace with real Binance price fetch.
    return None


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

    # 5. Trade engines
    engines_tasks: list[asyncio.Task] = []
    active_engines = []

    if config.engines.scalping.enabled:
        scalping = ScalpingEngine(
            pairs=config.trading.pairs,
            timeframes=config.engines.scalping.timeframes,
        )
        active_engines.append((scalping, config.engines.scalping.scan_interval))

    if config.engines.swing.enabled:
        swing = SwingEngine(
            pairs=config.trading.pairs,
            timeframes=config.engines.swing.timeframes,
        )
        active_engines.append((swing, config.engines.swing.scan_interval))

    # 6. Telegram bot
    bot = GuyGuyBot(config=config, signal_logger=signal_logger, risk_gate=risk_gate)
    await bot.start_polling()

    # 7. APScheduler
    scheduler = AsyncIOScheduler(timezone="UTC")
    bot.setup_scheduler(scheduler)
    scheduler.start()

    # 8. Launch all tasks
    for eng, interval in active_engines:
        task = asyncio.create_task(
            _engine_loop(eng, pipeline, signal_logger, risk_gate, bot, interval),
            name=f"engine_{eng.engine_name}",
        )
        engines_tasks.append(task)

    monitor_task = asyncio.create_task(
        _trade_monitor_loop(signal_logger, risk_gate, bot),
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
    logger.info("GuyGuyBot stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
