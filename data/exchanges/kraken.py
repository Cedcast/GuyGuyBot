"""
data/exchanges/kraken.py
------------------------
KrakenExchange — Kraken spot REST endpoints (no auth required).
Spot only — no funding rate, OI, or L/S ratio.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from data.exchanges.base import BaseExchange

logger = logging.getLogger(__name__)

_BASE = "https://api.kraken.com/0/public"

_PRICE_CACHE_TTL = 10

_PAIR_MAP: dict[str, str] = {
    "BTCUSDT": "XBTUSD",
    "ETHUSDT": "ETHUSD",
    "SOLUSDT": "SOLUSD",
    "BNBUSDT": "BNBUSD",
    "XRPUSDT": "XRPUSD",
    "ADAUSDT": "ADAUSD",
    "DOGEUSDT": "XDGUSD",
}

_TF_MAP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


class KrakenExchange(BaseExchange):
    """Kraken spot REST adapter.

    Spot only — returns ``None`` for all futures-specific methods
    (funding rate, open interest, long/short ratio).
    All methods return ``None`` (or safe defaults) on failure.
    """

    name = "kraken"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._price_cache: dict[str, tuple[float, float]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _to_kraken_pair(self, pair: str) -> str:
        return _PAIR_MAP.get(pair, pair)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV candles from Kraken OHLC endpoint."""
        kraken_pair = self._to_kraken_pair(pair)
        interval = _TF_MAP.get(timeframe, 30)
        url = f"{_BASE}/OHLC"
        params = {"pair": kraken_pair, "interval": interval}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            errors = data.get("error", [])
            if errors:
                logger.warning("KrakenExchange.fetch_ohlcv %s errors: %s", pair, errors)
                return []
            result: dict = data.get("result", {})
            # Result has the pair name as key and a "last" key
            candle_key = next((k for k in result if k != "last"), None)
            if candle_key is None:
                return []
            raw: list[list] = result[candle_key]
            # Kraken: [time, open, high, low, close, vwap, volume, count]
            candles = [
                {
                    "timestamp": int(k[0]) * 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[6]),
                }
                for k in raw
            ]
            return candles[-limit:]
        except Exception as exc:
            logger.warning("KrakenExchange.fetch_ohlcv %s/%s failed: %s", pair, timeframe, exc)
            return []

    # ------------------------------------------------------------------
    # Current price
    # ------------------------------------------------------------------

    async def fetch_current_price(self, pair: str) -> float | None:
        now = time.monotonic()
        cached = self._price_cache.get(pair)
        if cached and (now - cached[1]) < _PRICE_CACHE_TTL:
            return cached[0]

        kraken_pair = self._to_kraken_pair(pair)
        url = f"{_BASE}/Ticker"
        params = {"pair": kraken_pair}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            errors = data.get("error", [])
            if errors:
                logger.warning("KrakenExchange.fetch_current_price %s errors: %s", pair, errors)
                return None
            result: dict = data.get("result", {})
            if not result:
                return None
            ticker_key = next(iter(result))
            price = float(result[ticker_key]["c"][0])
            self._price_cache[pair] = (price, now)
            return price
        except Exception as exc:
            logger.warning("KrakenExchange.fetch_current_price %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Order book depth
    # ------------------------------------------------------------------

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        kraken_pair = self._to_kraken_pair(pair)
        url = f"{_BASE}/Depth"
        params = {"pair": kraken_pair, "count": 20}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            errors = data.get("error", [])
            if errors:
                logger.warning("KrakenExchange.fetch_order_book_depth %s errors: %s", pair, errors)
                return None
            result: dict = data.get("result", {})
            if not result:
                return None
            book_key = next(iter(result))
            book = result[book_key]
            bids: list[list] = book.get("bids", [])
            asks: list[list] = book.get("asks", [])
            bid_wall = sum(float(b[1]) for b in bids)
            ask_wall = sum(float(a[1]) for a in asks)
            ratio = bid_wall / ask_wall if ask_wall > 0 else 1.0
            return {"bid_wall": bid_wall, "ask_wall": ask_wall, "bid_ask_ratio": ratio}
        except Exception as exc:
            logger.warning("KrakenExchange.fetch_order_book_depth %s failed: %s", pair, exc)
            return None
