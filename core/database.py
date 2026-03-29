"""
core/database.py
----------------
SQLite persistence layer for GuyGuyBot.

Tables
------
- signals  : every generated signal and its lifecycle status
- trades   : open/closed paper trades with PnL and R-multiple
- stats    : computed weekly/monthly/quarterly performance snapshots
- positions: active pair/engine locks used by the risk gate
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_DDL_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,              -- ISO-8601 UTC
    pair          TEXT    NOT NULL,
    engine        TEXT    NOT NULL,              -- 'scalping' | 'swing'
    timeframe     TEXT    NOT NULL,
    direction     TEXT    NOT NULL,              -- 'LONG' | 'SHORT'
    entry         REAL    NOT NULL,
    stop_loss     REAL    NOT NULL,
    take_profit   REAL    NOT NULL,
    confidence    REAL    NOT NULL DEFAULT 0.0,  -- 0.0–1.0
    raw_response  TEXT,                          -- JSON blob from agents
    status        TEXT    NOT NULL DEFAULT 'PENDING'
    -- status values: PENDING | OPENED | SKIPPED | TP_HIT | SL_HIT
);
"""

_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     INTEGER NOT NULL REFERENCES signals(id),
    pair          TEXT    NOT NULL,
    engine        TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    entry         REAL    NOT NULL,
    stop_loss     REAL    NOT NULL,
    take_profit   REAL    NOT NULL,
    open_time     TEXT    NOT NULL,              -- ISO-8601 UTC
    close_time    TEXT,
    close_price   REAL,
    outcome       TEXT    NOT NULL DEFAULT 'OPEN',
    -- outcome values: OPEN | TP | SL
    pnl           REAL,
    r_multiple    REAL
);
"""

_DDL_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period_type     TEXT    NOT NULL,  -- 'weekly' | 'monthly' | 'quarterly'
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    total_signals   INTEGER NOT NULL DEFAULT 0,
    opened_trades   INTEGER NOT NULL DEFAULT 0,
    skipped_trades  INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    total_pnl       REAL    NOT NULL DEFAULT 0.0,
    winrate         REAL    NOT NULL DEFAULT 0.0,
    avg_r_multiple  REAL    NOT NULL DEFAULT 0.0,
    max_drawdown    REAL    NOT NULL DEFAULT 0.0,
    computed_at     TEXT    NOT NULL              -- ISO-8601 UTC
);
"""

_DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pair       TEXT    NOT NULL,
    engine     TEXT    NOT NULL,
    signal_id  INTEGER NOT NULL REFERENCES signals(id),
    opened_at  TEXT    NOT NULL,
    UNIQUE(pair, engine)
);
"""


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class Database:
    """Thin wrapper around a SQLite connection.

    Use :meth:`connection` as a context manager for transactional writes::

        with db.connection() as conn:
            conn.execute("INSERT INTO signals ...")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables if they do not already exist."""
        with self.connection() as conn:
            conn.execute(_DDL_SIGNALS)
            conn.execute(_DDL_TRADES)
            conn.execute(_DDL_STATS)
            conn.execute(_DDL_POSITIONS)
        logger.info("Database initialised at %s", self._path)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an open :class:`sqlite3.Connection` with WAL mode enabled.

        Commits on clean exit, rolls back on exception.
        """
        conn = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute *sql* and return the first result row, or ``None``."""
        with self.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchone()

    def execute_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute *sql* and return all result rows."""
        with self.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE and return ``lastrowid`` or ``rowcount``."""
        with self.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.lastrowid or cur.rowcount
