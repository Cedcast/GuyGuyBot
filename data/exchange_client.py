"""
data/exchange_client.py
-----------------------
MultiExchangeClient — aggregates price, OHLCV, and futures data
across Binance, Bybit, OKX, and Kraken concurrently.

All exchange APIs used are public endpoints — no API keys required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from data.exchanges.binance import BinanceExchange
from data.exchanges.bybit import BybitExchange
from data.exchanges.kraken import KrakenExchange
from data.exchanges.okx import OKXExchange

logger = logging.getLogger(__name__)

_ALL_EXCHANGES = ["binance", "bybit", "okx", "kraken"]
# Futures-capable exchanges (have funding rate, OI, L/S)
_FUTURES_EXCHANGES = ["binance", "bybit", "okx"]

# Threshold for price divergence warning (0.1%)
_PRICE_DIVERGENCE_WARN = 0.001

# Backward-compat alias so existing ``from data.exchange_client import ExchangeClient``
# imports keep working.
ExchangeClient = None  # replaced at bottom of file after class definition


class MultiExchangeClient:
    """Orchestrates all 4 exchanges and aggregates their data.

    Parameters
    ----------
    enabled_exchanges:
        List of exchange names to use.  Defaults to all 4.
    ohlcv_source:
        Which exchange to use as primary OHLCV source.  Default ``"binance"``.
    """

    def __init__(
        self,
        enabled_exchanges: list[str] | None = None,
        ohlcv_source: str = "binance",
    ) -> None:
        names = enabled_exchanges or _ALL_EXCHANGES
        self._ohlcv_source = ohlcv_source

        # Instantiate requested exchanges
        _factory = {
            "binance": BinanceExchange,
            "bybit": BybitExchange,
            "okx": OKXExchange,
            "kraken": KrakenExchange,
        }
        self._exchanges: dict[str, Any] = {}
        for name in names:
            if name in _factory:
                self._exchanges[name] = _factory[name]()
            else:
                logger.warning("MultiExchangeClient: unknown exchange %r — skipped", name)

        # OHLCV fallback order (skip exchanges not in self._exchanges)
        self._ohlcv_order = [ohlcv_source] + [
            n for n in ["binance", "bybit", "okx"] if n != ohlcv_source
        ]

        # Previous OI totals for change-pct calculation: {pair: float}
        self._prev_oi_total: dict[str, float] = {}

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV from the primary source; fall back to next available.

        Returns
        -------
        List of ``{timestamp, open, high, low, close, volume}`` dicts or
        an empty list on total failure.
        """
        for name in self._ohlcv_order:
            ex = self._exchanges.get(name)
            if ex is None:
                continue
            candles = await ex.fetch_ohlcv(pair, timeframe, limit)
            if candles:
                return candles
            logger.debug("MultiExchangeClient: OHLCV fallback from %s for %s/%s", name, pair, timeframe)
        return []

    # ------------------------------------------------------------------
    # Current price — averaged across all exchanges
    # ------------------------------------------------------------------

    async def fetch_current_price(self, pair: str) -> float | None:
        """Fetch price from all exchanges concurrently; return average."""
        exchange_names = list(self._exchanges.keys())
        tasks = [self._exchanges[n].fetch_current_price(pair) for n in exchange_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices: dict[str, float] = {}
        for name, result in zip(exchange_names, results):
            if isinstance(result, Exception) or result is None:
                continue
            prices[name] = float(result)

        if not prices:
            return None

        avg = sum(prices.values()) / len(prices)

        # Log divergences > 0.1%
        for name, price in prices.items():
            if avg > 0 and abs(price - avg) / avg > _PRICE_DIVERGENCE_WARN:
                logger.warning(
                    "Price divergence for %s on %s: %.2f vs avg %.2f (%.3f%%)",
                    pair, name, price, avg,
                    abs(price - avg) / avg * 100,
                )

        return avg

    # ------------------------------------------------------------------
    # Aggregated futures data
    # ------------------------------------------------------------------

    async def fetch_aggregated_futures_data(self, pair: str) -> dict[str, Any]:
        """Fetch and aggregate futures data across all futures exchanges.

        Returns a rich dict with per-exchange and aggregated values.
        Returns safe defaults for any exchange that fails.
        """
        futures_exchanges = [
            n for n in _FUTURES_EXCHANGES if n in self._exchanges
        ]
        all_exchanges = list(self._exchanges.keys())

        # Fetch concurrently: funding rates, OI, L/S, and prices
        fr_tasks = [self._exchanges[n].fetch_funding_rate(pair) for n in futures_exchanges]
        oi_tasks = [self._exchanges[n].fetch_open_interest(pair) for n in futures_exchanges]
        ls_tasks = [self._exchanges[n].fetch_long_short_ratio(pair) for n in futures_exchanges]
        price_tasks = [self._exchanges[n].fetch_current_price(pair) for n in all_exchanges]

        fr_results, oi_results, ls_results, price_results = await asyncio.gather(
            asyncio.gather(*fr_tasks, return_exceptions=True),
            asyncio.gather(*oi_tasks, return_exceptions=True),
            asyncio.gather(*ls_tasks, return_exceptions=True),
            asyncio.gather(*price_tasks, return_exceptions=True),
        )

        result: dict[str, Any] = {}

        # ---- Funding rates ----
        funding_rates: list[float] = []
        for name, fr in zip(futures_exchanges, fr_results):
            if isinstance(fr, Exception) or fr is None:
                result[f"funding_rate_{name}"] = None
            else:
                val = float(fr)
                result[f"funding_rate_{name}"] = val
                funding_rates.append(val)

        if funding_rates:
            result["funding_rate_avg"] = sum(funding_rates) / len(funding_rates)
            result["funding_rate_max"] = max(funding_rates)
            result["funding_rate_min"] = min(funding_rates)
        else:
            result["funding_rate_avg"] = None
            result["funding_rate_max"] = None
            result["funding_rate_min"] = None

        # ---- Open interest ----
        oi_values: list[float] = []
        for name, oi in zip(futures_exchanges, oi_results):
            if isinstance(oi, Exception) or oi is None:
                result[f"oi_{name}"] = None
            else:
                val = float(oi.get("oi", 0))
                result[f"oi_{name}"] = val
                oi_values.append(val)

        oi_total = sum(oi_values) if oi_values else 0.0
        result["oi_total"] = oi_total
        prev_oi = self._prev_oi_total.get(pair)
        result["oi_change_pct"] = ((oi_total - prev_oi) / prev_oi * 100) if prev_oi else 0.0
        if oi_total > 0:
            self._prev_oi_total[pair] = oi_total

        # ---- Long/Short ratios ----
        ls_ratios: list[float] = []
        for name, ls in zip(futures_exchanges, ls_results):
            if isinstance(ls, Exception) or ls is None:
                result[f"ls_ratio_{name}"] = None
            else:
                val = float(ls.get("long_short_ratio", 1.0))
                result[f"ls_ratio_{name}"] = val
                ls_ratios.append(val)

        result["ls_ratio_avg"] = sum(ls_ratios) / len(ls_ratios) if ls_ratios else None

        # ---- Prices ----
        valid_prices: dict[str, float] = {}
        for name, price in zip(all_exchanges, price_results):
            if isinstance(price, Exception) or price is None:
                result[f"price_{name}"] = None
            else:
                val = float(price)
                result[f"price_{name}"] = val
                valid_prices[name] = val

        if valid_prices:
            avg_price = sum(valid_prices.values()) / len(valid_prices)
            result["price_avg"] = avg_price
            if avg_price > 0:
                max_price = max(valid_prices.values())
                min_price = min(valid_prices.values())
                result["price_divergence_pct"] = (max_price - min_price) / avg_price * 100
            else:
                result["price_divergence_pct"] = 0.0
        else:
            result["price_avg"] = None
            result["price_divergence_pct"] = 0.0

        # ---- Exchange consensus ----
        fr_avg = result.get("funding_rate_avg")
        ls_avg = result.get("ls_ratio_avg")
        oi_change = result.get("oi_change_pct", 0.0)

        if (
            fr_avg is not None
            and ls_avg is not None
            and fr_avg > 0.0005
            and oi_change > 0
            and ls_avg > 1.3
        ):
            consensus = "BULLISH"
        elif (
            fr_avg is not None
            and fr_avg < -0.0002
        ) or (
            ls_avg is not None
            and ls_avg < 0.8
            and oi_change > 0
        ):
            consensus = "BEARISH"
        else:
            consensus = "NEUTRAL"
        result["exchange_consensus"] = consensus

        return result

    # ------------------------------------------------------------------
    # Order book depth — Binance only (most liquid)
    # ------------------------------------------------------------------

    async def fetch_order_book_depth(self, pair: str) -> dict[str, float] | None:
        """Fetch order book depth from Binance (most liquid, lowest latency)."""
        ex = self._exchanges.get("binance")
        if ex is None:
            # Fallback to first available exchange
            for name in self._exchanges:
                ex = self._exchanges[name]
                break
        if ex is None:
            return None
        return await ex.fetch_order_book_depth(pair)

    # ------------------------------------------------------------------
    # Backward-compatible single-value methods
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, pair: str) -> float | None:
        """Return average funding rate across all futures exchanges."""
        data = await self.fetch_aggregated_futures_data(pair)
        return data.get("funding_rate_avg")

    async def fetch_open_interest(self, pair: str) -> dict[str, float] | None:
        """Return aggregated OI for backward compatibility."""
        data = await self.fetch_aggregated_futures_data(pair)
        oi_total = data.get("oi_total")
        if oi_total is None:
            return None
        return {"oi": oi_total, "change_pct": data.get("oi_change_pct", 0.0)}

    async def fetch_long_short_ratio(self, pair: str) -> dict[str, float] | None:
        """Return averaged long/short ratio for backward compatibility."""
        data = await self.fetch_aggregated_futures_data(pair)
        ls_avg = data.get("ls_ratio_avg")
        if ls_avg is None:
            return None
        return {
            "long_short_ratio": ls_avg,
            "long_account": ls_avg / (1 + ls_avg),
            "short_account": 1 / (1 + ls_avg),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all exchange sessions."""
        await asyncio.gather(
            *[ex.close() for ex in self._exchanges.values()],
            return_exceptions=True,
        )


# Backward-compat alias — existing code that does
#   ``from data.exchange_client import ExchangeClient``
# will get MultiExchangeClient.
ExchangeClient = MultiExchangeClient
