"""
risk/gating.py
--------------
RiskGate enforces the one-open-trade-per-pair/engine rule.

State is persisted in the ``positions`` table so the gate survives restarts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.database import Database

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RiskGate:
    """Manages position locks per (pair, engine) combination.

    Parameters
    ----------
    db:
        An initialised :class:`~core.database.Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def can_trade(self, pair: str, engine: str) -> bool:
        """Return ``True`` if no open position exists for *pair* / *engine*.

        This is the primary guard called before generating or acting on a
        signal for a given pair.
        """
        row = self._db.execute_one(
            "SELECT id FROM positions WHERE pair = ? AND engine = ?",
            (pair, engine),
        )
        allowed = row is None
        if not allowed:
            logger.debug("RiskGate blocked %s/%s — position already open", pair, engine)
        return allowed

    def open_position(self, pair: str, engine: str, signal_id: int) -> None:
        """Record an open position for *pair* / *engine*, linked to *signal_id*.

        Raises ``RuntimeError`` if a position is already open for this
        pair/engine (prevents double-entry).
        """
        if not self.can_trade(pair, engine):
            raise RuntimeError(f"Position already open for {pair}/{engine}")

        self._db.execute_write(
            "INSERT INTO positions (pair, engine, signal_id, opened_at) VALUES (?, ?, ?, ?)",
            (pair, engine, signal_id, _utcnow()),
        )
        logger.info("RiskGate opened position: %s/%s signal_id=%d", pair, engine, signal_id)

    def close_position(self, pair: str, engine: str) -> None:
        """Release the position lock for *pair* / *engine*.

        Safe to call even if no position is currently tracked (idempotent).
        """
        self._db.execute_write(
            "DELETE FROM positions WHERE pair = ? AND engine = ?",
            (pair, engine),
        )
        logger.info("RiskGate closed position: %s/%s", pair, engine)

    def get_open_positions(self) -> list[dict]:
        """Return all currently tracked open positions."""
        rows = self._db.execute_all("SELECT * FROM positions")
        return [dict(r) for r in rows]
