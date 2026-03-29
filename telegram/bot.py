"""
telegram/bot.py
---------------
Telegram bot integration using python-telegram-bot (v20+, async).

Features
--------
* /start  — welcome message
* /status — show open positions
* /stats  — show latest weekly stats
* Signal messages with [✅ Opened Trade] / [❌ Skipped Trade] inline buttons
* Trade outcome notifications (TP / SL)
* Periodic stats via APScheduler (weekly / monthly / quarterly)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from telegram.notifications import (
    format_signal_message,
    format_stats_message,
    format_status_message,
    format_trade_outcome_message,
)

if TYPE_CHECKING:
    from core.config import Config
    from core.signal_logger import SignalLogger
    from risk.gating import RiskGate

logger = logging.getLogger(__name__)

# Callback data prefixes for inline buttons
_CB_OPEN = "open_trade:"
_CB_SKIP = "skip_trade:"


class GuyGuyBot:
    """Telegram bot for GuyGuyBot.

    Parameters
    ----------
    config:
        Loaded application config.
    signal_logger:
        :class:`~core.signal_logger.SignalLogger` instance.
    risk_gate:
        :class:`~risk.gating.RiskGate` instance.
    """

    def __init__(
        self,
        config: "Config",
        signal_logger: "SignalLogger",
        risk_gate: "RiskGate",
    ) -> None:
        self._config = config
        self._signal_logger = signal_logger
        self._risk_gate = risk_gate
        self._chat_id = config.telegram.chat_id

        self._app: Application = (
            Application.builder()
            .token(config.telegram.bot_token)
            .build()
        )
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("stats", self._cmd_stats))
        self._app.add_handler(
            CallbackQueryHandler(self._cb_trade_button, pattern=f"^({_CB_OPEN}|{_CB_SKIP})")
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        await update.message.reply_text(
            "👋 <b>GuyGuyBot</b> is online!\n\n"
            "Use /status to see open positions.\n"
            "Use /stats to view the latest weekly performance.",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /status command — show open positions."""
        open_trades = self._signal_logger.get_open_trades()
        text = format_status_message(open_trades)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /stats command — show the latest weekly stats."""
        stats = self._signal_logger.get_stats("weekly")
        if stats:
            text = format_stats_message(stats)
        else:
            text = "📊 No stats available yet. Stats are computed weekly."
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # ------------------------------------------------------------------
    # Callback query handler (inline buttons)
    # ------------------------------------------------------------------

    async def _cb_trade_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle [✅ Opened Trade] and [❌ Skipped Trade] button taps."""
        query = update.callback_query
        await query.answer()

        data: str = query.data or ""
        try:
            if data.startswith(_CB_OPEN):
                signal_id = int(data[len(_CB_OPEN):])
                await self._handle_open_trade(query, signal_id)
            elif data.startswith(_CB_SKIP):
                signal_id = int(data[len(_CB_SKIP):])
                await self._handle_skip_trade(query, signal_id)
        except (ValueError, TypeError) as exc:
            logger.error("Button callback parse error: %s", exc)
            await query.edit_message_reply_markup(reply_markup=None)

    async def _handle_open_trade(self, query: Any, signal_id: int) -> None:
        """Mark a signal as OPENED and create a trade record."""
        signal = self._signal_logger.get_signal(signal_id)
        if signal is None:
            logger.warning("Open button: signal #%d not found", signal_id)
            return
        if signal["status"] != "PENDING":
            await query.edit_message_reply_markup(reply_markup=None)
            return

        pair = signal["pair"]
        engine = signal["engine"]

        if not self._risk_gate.can_trade(pair, engine):
            await query.edit_message_text(
                query.message.text + "\n\n⚠️ <i>Trade blocked — position already open for this pair/engine.</i>",
                parse_mode=ParseMode.HTML,
            )
            self._signal_logger.update_signal_status(signal_id, "SKIPPED")
            return

        try:
            trade_id = self._signal_logger.open_trade(signal_id)
            self._risk_gate.open_position(pair, engine, signal_id)
        except Exception as exc:
            logger.error("Failed to open trade for signal #%d: %s", signal_id, exc)
            return

        await query.edit_message_text(
            query.message.text + f"\n\n✅ <b>Trade Opened</b>  <code>Trade #{trade_id}</code>",
            parse_mode=ParseMode.HTML,
        )
        logger.info("Signal #%d opened as trade #%d via Telegram button", signal_id, trade_id)

    async def _handle_skip_trade(self, query: Any, signal_id: int) -> None:
        """Mark a signal as SKIPPED."""
        signal = self._signal_logger.get_signal(signal_id)
        if signal is None:
            return
        if signal["status"] != "PENDING":
            await query.edit_message_reply_markup(reply_markup=None)
            return

        self._signal_logger.update_signal_status(signal_id, "SKIPPED")
        await query.edit_message_text(
            query.message.text + "\n\n❌ <b>Trade Skipped</b>",
            parse_mode=ParseMode.HTML,
        )
        logger.info("Signal #%d skipped via Telegram button", signal_id)

    # ------------------------------------------------------------------
    # Outbound notification helpers
    # ------------------------------------------------------------------

    async def send_signal(self, signal: dict[str, Any]) -> None:
        """Send a signal message with Opened/Skipped inline buttons.

        Parameters
        ----------
        signal:
            Signal dict including ``signal_id`` from the database.
        """
        signal_id = signal.get("signal_id") or signal.get("id")
        text = format_signal_message(signal)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Opened Trade", callback_data=f"{_CB_OPEN}{signal_id}"),
                InlineKeyboardButton("❌ Skipped Trade", callback_data=f"{_CB_SKIP}{signal_id}"),
            ]
        ])
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        except Exception as exc:
            logger.error("Failed to send signal message: %s", exc)

    async def send_trade_outcome(self, trade: dict[str, Any]) -> None:
        """Send a TP or SL notification message."""
        text = format_trade_outcome_message(trade)
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.error("Failed to send trade outcome message: %s", exc)

    async def send_periodic_stats(self, period_type: str) -> None:
        """Compute and send a stats message for *period_type*.

        Called automatically by the APScheduler jobs.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)

        if period_type == "weekly":
            start = now - timedelta(weeks=1)
        elif period_type == "monthly":
            start = now.replace(day=1)
        elif period_type == "quarterly":
            quarter_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter_month, day=1)
        else:
            start = now - timedelta(days=7)

        period_start = start.isoformat()
        period_end = now.isoformat()

        try:
            stats = self._signal_logger.compute_stats(period_type, period_start, period_end)
            text = format_stats_message(stats)
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
            logger.info("Sent %s stats to Telegram", period_type)
        except Exception as exc:
            logger.error("Failed to send %s stats: %s", period_type, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_scheduler(self, scheduler: Any) -> None:
        """Register APScheduler jobs for periodic stats messages.

        Parameters
        ----------
        scheduler:
            A running :class:`apscheduler.schedulers.asyncio.AsyncIOScheduler`.
        """
        schedule_cfg = self._config.telegram.stats_schedule

        if schedule_cfg.weekly:
            scheduler.add_job(
                self.send_periodic_stats,
                trigger="cron",
                day_of_week="mon",
                hour=9,
                minute=0,
                args=["weekly"],
                id="stats_weekly",
                replace_existing=True,
            )
            logger.info("Scheduled weekly stats (Mon 09:00 UTC)")

        if schedule_cfg.monthly:
            scheduler.add_job(
                self.send_periodic_stats,
                trigger="cron",
                day=1,
                hour=9,
                minute=0,
                args=["monthly"],
                id="stats_monthly",
                replace_existing=True,
            )
            logger.info("Scheduled monthly stats (1st of month 09:00 UTC)")

        if schedule_cfg.quarterly:
            scheduler.add_job(
                self.send_periodic_stats,
                trigger="cron",
                month="1,4,7,10",
                day=1,
                hour=9,
                minute=0,
                args=["quarterly"],
                id="stats_quarterly",
                replace_existing=True,
            )
            logger.info("Scheduled quarterly stats (1st of quarter 09:00 UTC)")

    async def start_polling(self) -> None:
        """Initialise the bot and begin polling for updates."""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started polling")

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot stopped")
