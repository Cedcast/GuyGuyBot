"""
news/sentiment.py
-----------------
Fetches real-time crypto news sentiment and market fear/greed index.
Sources:
  - CryptoPanic public API (no key needed for basic feed)
  - Alternative.me Fear & Greed Index API (free)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Simple keyword-based sentiment scoring
# -----------------------------------------------------------------------
_POSITIVE_WORDS = {
    "bull", "bullish", "surge", "surges", "surging",
    "breakout", "rally", "rallies", "moon", "mooning",
    "pump", "pumping", "all-time-high", "ath", "gain",
    "gains", "rise", "rising", "green", "recovery",
    "adoption", "buy", "long", "outperform",
}
_NEGATIVE_WORDS = {
    "bear", "bearish", "crash", "crashes", "crashing",
    "dump", "dumping", "plunge", "plunges", "plunging",
    "fud", "hack", "hacked", "ban", "banned", "sec",
    "lawsuit", "scam", "fraud", "drop", "drops", "fall",
    "falling", "red", "loss", "losses", "sell", "short",
    "liquidation", "exploit",
}

_CACHE_TTL = 300  # 5 minutes


def _score_title(title: str) -> float:
    """Return a sentiment score in [-1.0, 1.0] based on keyword matching."""
    words = set(title.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


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
    """

    def __init__(
        self,
        cryptopanic_token: str = "Pub",
        cache_ttl: int = _CACHE_TTL,
    ) -> None:
        self._token = cryptopanic_token
        self._cache_ttl = cache_ttl
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
        List of ``{title, url, source, published_at, sentiment_score}`` dicts.
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
                        "sentiment_score": _score_title(title),
                    }
                )
            self._store_cache(cache_key, results)
            return results
        except Exception as exc:
            logger.warning("fetch_crypto_news failed: %s", exc)
            return []

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
                "market_sentiment_score": 0.4,
                "recent_headlines": [...],
                "sentiment_warning": False,
            }
        """
        safe_default: dict[str, Any] = {
            "fear_greed_index": 50,
            "fear_greed_label": "Neutral",
            "market_sentiment_score": 0.0,
            "recent_headlines": [],
            "sentiment_warning": False,
        }

        # Extract base currencies: "BTCUSDT" → "BTC"
        currencies = [p.replace("USDT", "").replace("BUSD", "") for p in pairs]

        try:
            fg, news = await _gather_safe(
                self.fetch_fear_greed(),
                self.fetch_crypto_news(currencies=currencies, limit=10),
            )
            fg = fg or safe_default
            news = news or []

            # Aggregate sentiment across headlines
            scores = [item["sentiment_score"] for item in news if "sentiment_score" in item]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            fg_value = int(fg.get("value", 50))
            sentiment_warning = avg_score < -0.5 or fg_value < 25

            return {
                "fear_greed_index": fg_value,
                "fear_greed_label": fg.get("classification", "Neutral"),
                "market_sentiment_score": round(avg_score, 3),
                "recent_headlines": [item["title"] for item in news[:5]],
                "sentiment_warning": sentiment_warning,
            }
        except Exception as exc:
            logger.warning("get_market_context failed: %s", exc)
            return safe_default


async def _gather_safe(*coros: Any) -> list[Any]:
    """Run coroutines concurrently, returning ``None`` for any that raise."""
    import asyncio

    results = await asyncio.gather(*coros, return_exceptions=True)
    return [None if isinstance(r, Exception) else r for r in results]
