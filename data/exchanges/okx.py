"""
data/exchanges/okx.py
---------------------
OKXExchange — OKX V5 public REST endpoints (no auth required).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from data.exchanges.base import BaseExchange

logger = logging.getLogger(__name__)

_BASE = "https://www.okx.com/api/v5"

_PRICE_CACHE_TTL = 10

_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "1w": "1W",
}


def _to_okx_inst(pair: str) -> str:
    """Convert ``BTCUSDT`` → ``BTC-USDT-SWAP``."""
    base = pair.replace("USDT", "").replace("BUSD", "")
    return f"{base}-USDT-SWAP"


class OKXExchange(BaseExchange):
    """OKX V5 public REST adapter (perpetual swap contracts).

    All methods return ``None`` (or safe defaults) on failure.
    """

    name = "okx"

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
        """Fetch OHLCV candles from OKX V5 candles endpoint."""
        inst_id = _to_okx_inst(pair)
        bar = _TF_MAP.get(timeframe, timeframe)
        url = f"{_BASE}/market/candles"
        params = {"instId": inst_id, "bar": bar, "limit": limit}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            raw: list[list] = data.get("data", [])
            # OKX returns newest first — reverse to oldest-first
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
            logger.warning("OKXExchange.fetch_ohlcv %s/%s failed: %s", pair, timeframe, exc)
            return []

    # ------------------------------------------------------------------
    # Current price
    # ------------------------------------------------------------------

    async def fetch_current_price(self, pair: str) -> float | None:
        now = time.monotonic()
        cached = self._price_cache.get(pair)
        if cached and (now - cached[1]) < _PRICE_CACHE_TTL:
            return cached[0]

        inst_id = _to_okx_inst(pair)
        url = f"{_BASE}/market/ticker"
        params = {"instId": inst_id}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("data", [])
            if not items:
                return None
            price = float(items[0]["last"])
            self._price_cache[pair] = (price, now)
            return price
        except Exception as exc:
            logger.warning("OKXExchange.fetch_current_price %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, pair: str) -> float | None:
        inst_id = _to_okx_inst(pair)
        url = f"{_BASE}/public/funding-rate"
        params = {"instId": inst_id}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("data", [])
            if not items:
                return None
            return float(items[0]["fundingRate"])
        except Exception as exc:
            logger.warning("OKXExchange.fetch_funding_rate %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def fetch_open_interest(self, pair: str) -> dict[str, float] | None:
        inst_id = _to_okx_inst(pair)
        url = f"{_BASE}/public/open-interest"
        params = {"instType": "SWAP", "instId": inst_id}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("data", [])
            if not items:
                return None
            return {"oi": float(items[0]["oi"]), "change_pct": 0.0}
        except Exception as exc:
            logger.warning("OKXExchange.fetch_open_interest %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Long/Short ratio
    # ------------------------------------------------------------------

    async def fetch_long_short_ratio(self, pair: str) -> dict[str, float] | None:
        base = pair.replace("USDT", "").replace("BUSD", "")
        url = f"{_BASE}/rubik/stat/contracts/long-short-account-ratio"
        params = {"ccy": base, "period": "5m"}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[list] = data.get("data", [])
            if not items:
                return None
            # data is [ts, longRatio, shortRatio] — most recent first
            row = items[0]
            long_ratio = float(row[1])
            short_ratio = float(row[2])
            ls_ratio = long_ratio / short_ratio if short_ratio > 0 else 1.0
            return {
                "long_short_ratio": ls_ratio,
                "long_account": long_ratio,
                "short_account": short_ratio,
            }
        except Exception as exc:
            logger.warning("OKXExchange.fetch_long_short_ratio %s failed: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Order book depth
    # ------------------------------------------------------------------

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        inst_id = _to_okx_inst(pair)
        url = f"{_BASE}/market/books"
        params = {"instId": inst_id, "sz": 20}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            items: list[dict] = data.get("data", [])
            if not items:
                return None
            book = items[0]
            bids: list[list[str]] = book.get("bids", [])
            asks: list[list[str]] = book.get("asks", [])
            bid_wall = sum(float(b[1]) for b in bids)
            ask_wall = sum(float(a[1]) for a in asks)
            ratio = bid_wall / ask_wall if ask_wall > 0 else 1.0
            return {"bid_wall": bid_wall, "ask_wall": ask_wall, "bid_ask_ratio": ratio}
        except Exception as exc:
            logger.warning("OKXExchange.fetch_order_book_depth %s failed: %s", pair, exc)
            return None
