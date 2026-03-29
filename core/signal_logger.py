"""
core/signal_logger.py
---------------------
High-level API over the database for logging signals, managing trades,
and computing performance statistics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from core.database import Database

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SignalLogger:
    """Provides all read/write operations for signals, trades, and stats.

    Parameters
    ----------
    db:
        An initialised :class:`~core.database.Database` instance.
    account_size:
        Paper account size in USD (used for PnL calculations).
    risk_per_trade:
        Fraction of account to risk per trade (e.g. 0.01 = 1 %).
    """

    def __init__(self, db: Database, account_size: float = 1000.0, risk_per_trade: float = 0.01) -> None:
        self._db = db
        self._account_size = account_size
        self._risk_per_trade = risk_per_trade

    # ------------------------------------------------------------------
    # Signal operations
    # ------------------------------------------------------------------

    def log_signal(self, signal_data: dict[str, Any]) -> int:
        """Insert a new signal row and return its ``signal_id``.

        Expected keys in *signal_data*: ``pair``, ``engine``, ``timeframe``,
        ``direction``, ``entry``, ``stop_loss``, ``take_profit``,
        ``confidence`` (optional), ``raw_response`` (optional).
        """
        raw = json.dumps(signal_data.get("raw_response")) if signal_data.get("raw_response") else None
        signal_id = self._db.execute_write(
            """
            INSERT INTO signals
                (timestamp, pair, engine, timeframe, direction, entry,
                 stop_loss, take_profit, confidence, raw_response, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                _utcnow(),
                signal_data["pair"],
                signal_data["engine"],
                signal_data["timeframe"],
                signal_data["direction"],
                float(signal_data["entry"]),
                float(signal_data["stop_loss"]),
                float(signal_data["take_profit"]),
                float(signal_data.get("confidence", 0.0)),
                raw,
            ),
        )
        logger.info(
            "Logged signal #%d  %s %s %s @ %.4f",
            signal_id,
            signal_data["pair"],
            signal_data["direction"],
            signal_data["engine"],
            float(signal_data["entry"]),
        )
        return signal_id

    def update_signal_status(self, signal_id: int, status: str) -> None:
        """Update the *status* column of signal *signal_id*.

        Valid values: ``PENDING``, ``OPENED``, ``SKIPPED``, ``TP_HIT``, ``SL_HIT``.
        """
        self._db.execute_write(
            "UPDATE signals SET status = ? WHERE id = ?",
            (status, signal_id),
        )
        logger.info("Signal #%d status → %s", signal_id, status)

    def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        """Return a signal row as a plain dict, or ``None`` if not found."""
        row = self._db.execute_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Trade operations
    # ------------------------------------------------------------------

    def open_trade(self, signal_id: int) -> int:
        """Create a trade record for *signal_id* and return the trade id.

        Also updates the signal status to ``OPENED``.
        """
        signal = self.get_signal(signal_id)
        if signal is None:
            raise ValueError(f"Signal {signal_id} not found")

        trade_id = self._db.execute_write(
            """
            INSERT INTO trades
                (signal_id, pair, engine, direction, entry, stop_loss,
                 take_profit, open_time, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (
                signal_id,
                signal["pair"],
                signal["engine"],
                signal["direction"],
                signal["entry"],
                signal["stop_loss"],
                signal["take_profit"],
                _utcnow(),
            ),
        )
        self.update_signal_status(signal_id, "OPENED")
        logger.info("Opened trade #%d for signal #%d (%s %s)", trade_id, signal_id, signal["pair"], signal["direction"])
        return trade_id

    def close_trade(self, trade_id: int, close_price: float, outcome: str) -> dict[str, Any]:
        """Close *trade_id* at *close_price* with *outcome* (``TP`` or ``SL``).

        Computes PnL in USD and R-multiple, updates the trade row, and
        updates the parent signal status accordingly.

        Returns the updated trade as a dict.
        """
        row = self._db.execute_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        if row is None:
            raise ValueError(f"Trade {trade_id} not found")

        trade = dict(row)
        entry = float(trade["entry"])
        stop_loss = float(trade["stop_loss"])
        direction = trade["direction"]

        risk_amount = self._account_size * self._risk_per_trade
        risk_in_price = abs(entry - stop_loss)

        if risk_in_price == 0:
            logger.warning("Trade #%d has zero risk distance; skipping PnL calculation", trade_id)
            r_multiple = 0.0
            pnl = 0.0
        else:
            position_size = risk_amount / risk_in_price  # units of asset
            if direction == "LONG":
                pnl = (close_price - entry) * position_size
            else:  # SHORT
                pnl = (entry - close_price) * position_size
            r_multiple = pnl / risk_amount

        self._db.execute_write(
            """
            UPDATE trades
               SET close_time = ?, close_price = ?, outcome = ?,
                   pnl = ?, r_multiple = ?
             WHERE id = ?
            """,
            (_utcnow(), close_price, outcome, round(pnl, 4), round(r_multiple, 4), trade_id),
        )

        signal_status = "TP_HIT" if outcome == "TP" else "SL_HIT"
        self.update_signal_status(int(trade["signal_id"]), signal_status)

        logger.info(
            "Closed trade #%d  outcome=%s  pnl=%.2f  R=%.2f",
            trade_id, outcome, pnl, r_multiple,
        )

        updated = self._db.execute_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        return dict(updated) if updated else {}

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return all trades with outcome ``OPEN``."""
        rows = self._db.execute_all("SELECT * FROM trades WHERE outcome = 'OPEN'")
        return [dict(r) for r in rows]

    def get_trade(self, trade_id: int) -> dict[str, Any] | None:
        """Return a trade row as a plain dict, or ``None``."""
        row = self._db.execute_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def compute_stats(self, period_type: str, period_start: str, period_end: str) -> dict[str, Any]:
        """Compute and persist statistics for the given period.

        Parameters
        ----------
        period_type:
            One of ``weekly``, ``monthly``, ``quarterly``.
        period_start / period_end:
            ISO-8601 UTC strings bounding the period.

        Returns the newly inserted stats row as a dict.
        """
        rows = self._db.execute_all(
            """
            SELECT s.status, t.pnl, t.r_multiple
              FROM signals s
         LEFT JOIN trades t ON t.signal_id = s.id
             WHERE s.timestamp >= ? AND s.timestamp <= ?
            """,
            (period_start, period_end),
        )

        total_signals = len(rows)
        opened = sum(1 for r in rows if r["status"] in ("OPENED", "TP_HIT", "SL_HIT"))
        skipped = sum(1 for r in rows if r["status"] == "SKIPPED")
        wins = sum(1 for r in rows if r["status"] == "TP_HIT")
        losses = sum(1 for r in rows if r["status"] == "SL_HIT")
        closed = wins + losses

        total_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None)
        winrate = (wins / closed) if closed > 0 else 0.0

        r_mults = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
        avg_r = (sum(r_mults) / len(r_mults)) if r_mults else 0.0

        # Simple drawdown: running cumulative PnL trough from peak
        max_drawdown = self._calc_max_drawdown(rows)

        stats_id = self._db.execute_write(
            """
            INSERT INTO stats
                (period_type, period_start, period_end, total_signals,
                 opened_trades, skipped_trades, wins, losses, total_pnl,
                 winrate, avg_r_multiple, max_drawdown, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                period_type, period_start, period_end,
                total_signals, opened, skipped, wins, losses,
                round(total_pnl, 4), round(winrate, 4),
                round(avg_r, 4), round(max_drawdown, 4),
                _utcnow(),
            ),
        )
        logger.info("Computed %s stats (id=%d): %d signals, winrate=%.1f%%", period_type, stats_id, total_signals, winrate * 100)
        row = self._db.execute_one("SELECT * FROM stats WHERE id = ?", (stats_id,))
        return dict(row) if row else {}

    def get_stats(self, period_type: str) -> dict[str, Any] | None:
        """Return the most recently computed stats for *period_type*."""
        row = self._db.execute_one(
            "SELECT * FROM stats WHERE period_type = ? ORDER BY computed_at DESC LIMIT 1",
            (period_type,),
        )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_max_drawdown(rows: list) -> float:
        """Return the maximum peak-to-trough drawdown from a list of trade rows."""
        pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
        if not pnls:
            return 0.0
        peak = 0.0
        cumulative = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd
