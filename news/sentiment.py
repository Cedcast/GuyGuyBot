"""
news/sentiment.py
-----------------
Fetches real-time crypto news sentiment and market fear/greed index.
Sources:
  - CryptoPanic public API (no key needed for basic feed)
  - Alternative.me Fear & Greed Index API (free)
  - LunarCrush social sentiment API (free key from lunarcrush.com)
  - CoinGlass liquidation data (free public API)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # 5 minutes


class SentimentFetcher:
    """Async fetcher for crypto market sentiment data.

    Results are cached for :attr:`cache_ttl` seconds to avoid hammering
    free-tier APIs.

    Parameters
    ----------
    cryptopanic_token:
        Public CryptoPanic token.  Defaults to ``"Pub"`` (unauthenticated).
    cache_ttl:
        Cache lifetime in seconds.  Defaults to 300 (5 minutes).
    lunarcrush_api_key:
        Free LunarCrush API key from lunarcrush.com.  Optional — falls back
        to neutral defaults when not configured.
    """

    def __init__(
        self,
        cryptopanic_token: str = "Pub",
        cache_ttl: int = _CACHE_TTL,
        lunarcrush_api_key: str = "",
    ) -> None:
        self._token = cryptopanic_token
        self._cache_ttl = cache_ttl
        self._lunarcrush_key = lunarcrush_api_key
        self._session: aiohttp.ClientSession | None = None
        # Cache entries: {key: (data, timestamp)}
        self._cache: dict[str, tuple[Any, float]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _from_cache(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry[1]) < self._cache_ttl:
            return entry[0]
        return None

    def _store_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.monotonic())

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------

    async def fetch_fear_greed(self) -> dict[str, Any]:
        """Fetch the Alternative.me Fear & Greed Index.

        Returns
        -------
        ``{value: int, classification: str, timestamp: str}``
        or safe defaults on failure.
        """
        safe_default: dict[str, Any] = {
            "value": 50,
            "classification": "Neutral",
            "timestamp": "",
        }
        cached = self._from_cache("fear_greed")
        if cached is not None:
            return cached

        url = "https://api.alternative.me/fng/?limit=1"
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            entry = data["data"][0]
            result: dict[str, Any] = {
                "value": int(entry["value"]),
                "classification": entry["value_classification"],
                "timestamp": entry.get("timestamp", ""),
            }
            self._store_cache("fear_greed", result)
            return result
        except Exception as exc:
            logger.warning("fetch_fear_greed failed: %s", exc)
            return safe_default

    # ------------------------------------------------------------------
    # CryptoPanic news
    # ------------------------------------------------------------------

    async def fetch_crypto_news(
        self,
        currencies: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch recent crypto news from CryptoPanic.

        Parameters
        ----------
        currencies:
            Optional list of ticker symbols to filter by (e.g. ``["BTC", "ETH"]``).
        limit:
            Max number of posts to return.

        Returns
        -------
        List of ``{title, url, source, published_at}`` dicts.
        Returns an empty list on failure.
        """
        cache_key = f"news_{','.join(currencies or [])}"
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        url = "https://cryptopanic.com/api/free/v1/posts/"
        params: dict[str, Any] = {
            "auth_token": self._token,
            "public": "true",
            "kind": "news",
        }
        if currencies:
            params["currencies"] = ",".join(currencies)

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            posts = data.get("results", [])[:limit]
            results: list[dict[str, Any]] = []
            for post in posts:
                title = post.get("title", "")
                results.append(
                    {
                        "title": title,
                        "url": post.get("url", ""),
                        "source": post.get("source", {}).get("title", ""),
                        "published_at": post.get("published_at", ""),
                    }
                )
            self._store_cache(cache_key, results)
            return results
        except Exception as exc:
            logger.warning("fetch_crypto_news failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # LunarCrush social sentiment
    # ------------------------------------------------------------------

    async def fetch_lunarcrush_sentiment(
        self,
        coins: list[str],
    ) -> dict[str, Any]:
        """Fetch social sentiment from LunarCrush.

        Uses the free public API v4.  No paid plan needed.
        Returns galaxy_score (0-100), social_volume, and sentiment label.
        Falls back to neutral defaults if API key not configured or call fails.
        """
        if not self._lunarcrush_key:
            return {
                "galaxy_score": 50,
                "social_volume_24h": 0,
                "sentiment": "neutral",
                "source": "unavailable",
            }

        cache_key = f"lunarcrush_{'_'.join(coins[:5])}"
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        url = "https://lunarcrush.com/api4/public/coins/list/v2"
        headers = {"Authorization": f"Bearer {self._lunarcrush_key}"}
        params = {"symbols": ",".join(coins[:5])}

        try:
            session = await self._get_session()
            async with session.get(url, headers=headers, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            items = data.get("data", [])
            if not items:
                return {"galaxy_score": 50, "social_volume_24h": 0, "sentiment": "neutral"}

            avg_galaxy = sum(i.get("galaxy_score", 50) for i in items) / len(items)
            total_social_vol = sum(i.get("social_volume_24h", 0) for i in items)

            sentiment = "bullish" if avg_galaxy > 60 else "bearish" if avg_galaxy < 40 else "neutral"

            result: dict[str, Any] = {
                "galaxy_score": round(avg_galaxy, 1),
                "social_volume_24h": total_social_vol,
                "sentiment": sentiment,
                "source": "lunarcrush",
            }
            self._store_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("fetch_lunarcrush_sentiment failed: %s", exc)
            return {"galaxy_score": 50, "social_volume_24h": 0, "sentiment": "neutral", "source": "error"}

    # ------------------------------------------------------------------
    # CoinGlass liquidation data
    # ------------------------------------------------------------------

    async def fetch_coinglass_liquidations(
        self,
        pair: str,
    ) -> dict[str, Any]:
        """Fetch liquidation data from CoinGlass.

        Uses the free public API.  Returns estimated USD liquidation values.
        Falls back to empty dict on failure.
        """
        cache_key = f"coinglass_{pair}"
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        url = "https://open-api.coinglass.com/public/v2/liquidation_ex"
        params = {"symbol": pair, "interval": "h4"}

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

            liq_data = data.get("data", {})
            long_liq = float(liq_data.get("longLiquidationUsd24h", 0))
            short_liq = float(liq_data.get("shortLiquidationUsd24h", 0))

            result: dict[str, Any] = {
                "long_liquidations_24h_usd": long_liq,
                "short_liquidations_24h_usd": short_liq,
                "liquidation_bias": (
                    "longs" if long_liq > short_liq
                    else "shorts" if short_liq > long_liq
                    else "neutral"
                ),
            }
            self._store_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("fetch_coinglass_liquidations failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Combined market context
    # ------------------------------------------------------------------

    async def get_market_context(self, pairs: list[str]) -> dict[str, Any]:
        """Build a combined market context dict ready for injection into agent prompts.

        Parameters
        ----------
        pairs:
            Trading pair symbols, e.g. ``["BTCUSDT", "ETHUSDT"]``.
            The base currencies (``BTC``, ``ETH``) are extracted for news filtering.

        Returns
        -------
        ::

            {
                "fear_greed_index": 71,
                "fear_greed_label": "Greed",
                "market_sentiment_score": 0.0,
                "recent_headlines": [...],   # raw — let Grok interpret
                "sentiment_warning": False,
                "lunarcrush_galaxy_score": 65,
                "lunarcrush_social_volume_24h": 120000,
                "lunarcrush_sentiment": "bullish",
                "long_liquidations_24h_usd": 50000000.0,
                "short_liquidations_24h_usd": 30000000.0,
                "liquidation_bias": "longs",
            }
        """
        safe_default: dict[str, Any] = {
            "fear_greed_index": 50,
            "fear_greed_label": "Neutral",
            "market_sentiment_score": 0.0,
            "recent_headlines": [],
            "sentiment_warning": False,
            "lunarcrush_galaxy_score": 50,
            "lunarcrush_social_volume_24h": 0,
            "lunarcrush_sentiment": "neutral",
            "long_liquidations_24h_usd": 0,
            "short_liquidations_24h_usd": 0,
            "liquidation_bias": "neutral",
        }

        # Extract base currencies: "BTCUSDT" → "BTC"
        currencies = [p.replace("USDT", "").replace("BUSD", "") for p in pairs]
        # Primary pair base for CoinGlass (use first pair)
        primary_base = currencies[0] if currencies else "BTC"

        try:
            fg, news, lc, liq = await _gather_safe(
                self.fetch_fear_greed(),
                self.fetch_crypto_news(currencies=currencies, limit=10),
                self.fetch_lunarcrush_sentiment(currencies),
                self.fetch_coinglass_liquidations(primary_base),
            )
            fg = fg or safe_default
            news = news or []
            lc = lc or {}
            liq = liq or {}

            fg_value = int(fg.get("value", 50))
            sentiment_warning = fg_value < 25

            return {
                "fear_greed_index": fg_value,
                "fear_greed_label": fg.get("classification", "Neutral"),
                "market_sentiment_score": 0.0,  # kept for backward compat; headlines now passed raw
                "recent_headlines": [item["title"] for item in news[:5]],
                "sentiment_warning": sentiment_warning,
                # LunarCrush social
                "lunarcrush_galaxy_score": lc.get("galaxy_score", 50),
                "lunarcrush_social_volume_24h": lc.get("social_volume_24h", 0),
                "lunarcrush_sentiment": lc.get("sentiment", "neutral"),
                # CoinGlass liquidations
                "long_liquidations_24h_usd": liq.get("long_liquidations_24h_usd", 0),
                "short_liquidations_24h_usd": liq.get("short_liquidations_24h_usd", 0),
                "liquidation_bias": liq.get("liquidation_bias", "neutral"),
            }
        except Exception as exc:
            logger.warning("get_market_context failed: %s", exc)
            return safe_default


async def _gather_safe(*coros: Any) -> list[Any]:
    """Run coroutines concurrently, returning ``None`` for any that raise."""
    import asyncio

    results = await asyncio.gather(*coros, return_exceptions=True)
    return [None if isinstance(r, Exception) else r for r in results]
