"""
data/exchanges/bybit.py
-----------------------
BybitExchange — Bybit V5 public REST endpoints (no auth required).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from data.exchanges.base import BaseExchange

logger = logging.getLogger(__name__)

_BASE = "https://api.bybit.com/v5"

_PRICE_CACHE_TTL = 10

_TF_MAP: dict[str, str] = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


class BybitExchange(BaseExchange):
    """Bybit V5 public REST adapter (perpetual linear contracts).

    All methods return ``None`` (or safe defaults) on failure.
    """

    name = "bybit"

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

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV candles from Bybit V5 kline endpoint."""
        interval = _TF_MAP.get(timeframe, timeframe)
        url = f"{_BASE}/market/kline"
        params = {
            "category": "linear",
            "symbol": pair,
            "interval": interval,
            "limit": limit,
        }
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            raw: list[list] = data.get("result", {}).get("list", [])
            # Bybit returns newest first — reverse to oldest-first
            raw = list(reversed(raw))
            return [
                {
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
            ]
        except Exception as exc:
            logger.warning("BybitExchange.fetch_ohlcv %s/%s failed: %s", pair, timeframe, exc)
            return []

    # ------------------------------------------------------------------
    # Current price + funding rate (same endpoint)
    # ------------------------------------------------------------------

    async def _fetch_ticker(self, pair: str) -> dict[str, Any] | None:
        """Fetch linear ticker data (price + funding rate)."""
        url = f"{_BASE}/market/tickers"
        params = {"category": "linear", "symbol": pair}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("result", {}).get("list", [])
            if not items:
                return None
            return items[0]
        except Exception as exc:
            logger.warning("BybitExchange._fetch_ticker %s failed: %s", pair, exc)
            return None

    async def fetch_current_price(self, pair: str) -> float | None:
        now = time.monotonic()
        cached = self._price_cache.get(pair)
        if cached and (now - cached[1]) < _PRICE_CACHE_TTL:
            return cached[0]

        ticker = await self._fetch_ticker(pair)
        if ticker is None:
            return None
        try:
            price = float(ticker["lastPrice"])
            self._price_cache[pair] = (price, now)
            return price
        except Exception as exc:
            logger.warning("BybitExchange.fetch_current_price %s failed: %s", pair, exc)
            return None

    async def fetch_funding_rate(self, pair: str) -> float | None:
        ticker = await self._fetch_ticker(pair)
        if ticker is None:
            return None
        try:
            return float(ticker["fundingRate"])
        except Exception as exc:
            logger.warning("BybitExchange.fetch_funding_rate %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def fetch_open_interest(self, pair: str) -> dict[str, float] | None:
        url = f"{_BASE}/market/open-interest"
        params = {
            "category": "linear",
            "symbol": pair,
            "intervalTime": "5min",
            "limit": 1,
        }
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("result", {}).get("list", [])
            if not items:
                return None
            return {"oi": float(items[0]["openInterest"]), "change_pct": 0.0}
        except Exception as exc:
            logger.warning("BybitExchange.fetch_open_interest %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Long/Short ratio
    # ------------------------------------------------------------------

    async def fetch_long_short_ratio(self, pair: str) -> dict[str, float] | None:
        url = f"{_BASE}/market/account-ratio"
        params = {
            "category": "linear",
            "symbol": pair,
            "period": "5min",
            "limit": 1,
        }
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("result", {}).get("list", [])
            if not items:
                return None
            row = items[0]
            buy_ratio = float(row["buyRatio"])
            sell_ratio = float(row["sellRatio"])
            ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
            return {
                "long_short_ratio": ls_ratio,
                "long_account": buy_ratio,
                "short_account": sell_ratio,
            }
        except Exception as exc:
            logger.warning("BybitExchange.fetch_long_short_ratio %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Order book depth
    # ------------------------------------------------------------------

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        url = f"{_BASE}/market/orderbook"
        params = {"category": "linear", "symbol": pair, "limit": 20}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            book = data.get("result", {})
            bids: list[list[str]] = book.get("b", [])
            asks: list[list[str]] = book.get("a", [])
            bid_wall = sum(float(b[1]) for b in bids)
            ask_wall = sum(float(a[1]) for a in asks)
            ratio = bid_wall / ask_wall if ask_wall > 0 else 1.0
            return {"bid_wall": bid_wall, "ask_wall": ask_wall, "bid_ask_ratio": ratio}
        except Exception as exc:
            logger.warning("BybitExchange.fetch_order_book_depth %s failed: %s", pair, exc)
            return None
