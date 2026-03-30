"""
data/exchanges/binance.py
-------------------------
BinanceExchange — Binance Futures public REST endpoints (no auth required).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from data.exchanges.base import BaseExchange

logger = logging.getLogger(__name__)

_FAPI_BASE = "https://fapi.binance.com/fapi/v1"
_FDATA_BASE = "https://fapi.binance.com/futures/data"

_PRICE_CACHE_TTL = 10

_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}


class BinanceExchange(BaseExchange):
    """Binance Futures public REST adapter.

    All methods return ``None`` (or safe defaults) on failure — they never
    raise so callers can proceed without crashing on transient API errors.
    """

    name = "binance"

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._external_session = session
        self._session: aiohttp.ClientSession | None = session
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._prev_oi: dict[str, float] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._external_session:
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
        """Fetch OHLCV candles from Binance Futures."""
        interval = _TF_MAP.get(timeframe, timeframe)
        url = f"{_FAPI_BASE}/klines"
        params = {"symbol": pair, "interval": interval, "limit": limit}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                raw: list[list] = await resp.json()
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
            logger.warning("BinanceExchange.fetch_ohlcv %s/%s failed: %s", pair, timeframe, exc)
            return []

    # ------------------------------------------------------------------
    # Current price
    # ------------------------------------------------------------------

    async def fetch_current_price(self, pair: str) -> float | None:
        """Fetch the latest mark price from Binance Futures ticker."""
        now = time.monotonic()
        cached = self._price_cache.get(pair)
        if cached and (now - cached[1]) < _PRICE_CACHE_TTL:
            return cached[0]

        url = f"{_FAPI_BASE}/ticker/price"
        params = {"symbol": pair}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            price = float(data["price"])
            self._price_cache[pair] = (price, now)
            return price
        except Exception as exc:
            logger.warning("BinanceExchange.fetch_current_price %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, pair: str) -> float | None:
        """Fetch the current funding rate from Binance Futures premium index."""
        url = f"{_FAPI_BASE}/premiumIndex"
        params = {"symbol": pair}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return float(data["lastFundingRate"])
        except Exception as exc:
            logger.warning("BinanceExchange.fetch_funding_rate %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def fetch_open_interest(self, pair: str) -> dict[str, float] | None:
        """Fetch open interest and compute change vs. previous fetch."""
        url = f"{_FAPI_BASE}/openInterest"
        params = {"symbol": pair}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            oi = float(data["openInterest"])
            prev = self._prev_oi.get(pair)
            change_pct = ((oi - prev) / prev * 100) if prev else 0.0
            self._prev_oi[pair] = oi
            return {"oi": oi, "change_pct": change_pct}
        except Exception as exc:
            logger.warning("BinanceExchange.fetch_open_interest %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Long/Short ratio
    # ------------------------------------------------------------------

    async def fetch_long_short_ratio(self, pair: str) -> dict[str, float] | None:
        """Fetch global long/short account ratio (5-minute period)."""
        url = f"{_FDATA_BASE}/globalLongShortAccountRatio"
        params = {"symbol": pair, "period": "5m", "limit": 1}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data: list[dict] = await resp.json()
            if not data:
                return None
            row = data[0]
            return {
                "long_short_ratio": float(row["longShortRatio"]),
                "long_account": float(row["longAccount"]),
                "short_account": float(row["shortAccount"]),
            }
        except Exception as exc:
            logger.warning("BinanceExchange.fetch_long_short_ratio %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Order book depth
    # ------------------------------------------------------------------

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        """Fetch order book and return aggregated bid/ask walls."""
        url = f"{_FAPI_BASE}/depth"
        params = {"symbol": pair, "limit": 20}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            bids: list[list[str]] = data.get("bids", [])
            asks: list[list[str]] = data.get("asks", [])
            bid_wall = sum(float(b[1]) for b in bids)
            ask_wall = sum(float(a[1]) for a in asks)
            ratio = bid_wall / ask_wall if ask_wall > 0 else 1.0
            return {"bid_wall": bid_wall, "ask_wall": ask_wall, "bid_ask_ratio": ratio}
        except Exception as exc:
            logger.warning("BinanceExchange.fetch_order_book_depth %s failed: %s", pair, exc)
            return None
