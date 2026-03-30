"""
engines/swing_engine.py
-----------------------
Swing engine — monitors 4h and 1d timeframes.

Uses real OHLCV data from Binance Futures via ExchangeClient and
calculates real technical indicators via the indicators module.
Win-rate boosters: funding rate filter, OI confirmation, long/short
ratio filter, RSI divergence, MACD trend, and minimum confidence gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from data.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    detect_ema_crossover,
    detect_rsi_divergence,
)
from engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from data.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

_MIN_CANDLES = 100

# Minimum confidence (after all adjustments) to pass signal to pipeline
_MIN_CONFIDENCE = 0.68


class SwingEngine(BaseEngine):
    """Medium-to-longer-term trade engine scanning 4h and 1d timeframes.

    Parameters
    ----------
    pairs:
        Symbols to scan.
    timeframes:
        Timeframes to iterate (default: ``["4h", "1d"]``).
    exchange_client:
        Optional :class:`~data.exchange_client.ExchangeClient` instance for
        real Binance Futures data.  When not provided the engine returns no
        candidates (fail-safe).
    """

    def __init__(
        self,
        pairs: list[str],
        timeframes: list[str] | None = None,
        exchange_client: "ExchangeClient | None" = None,
    ) -> None:
        super().__init__(pairs=pairs, timeframes=timeframes)
        self._client = exchange_client

    @property
    def engine_name(self) -> str:
        return "swing"

    async def scan(self) -> list[dict[str, Any]]:
        """Scan all pairs across all swing timeframes."""
        if self._client is None:
            logger.warning("SwingEngine: no ExchangeClient configured — skipping scan")
            return []

        candidates: list[dict[str, Any]] = []

        for pair in self._pairs:
            for timeframe in self._timeframes:
                market_data = await self._fetch_market_data(pair, timeframe)
                if market_data is None:
                    continue

                if not self._has_setup(market_data):
                    continue

                signal = await self._build_signal(pair, market_data)
                if signal is None:
                    continue

                candidates.append(signal)

        logger.info("SwingEngine scan complete — %d candidate(s) found", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Market data fetching
    # ------------------------------------------------------------------

    async def _fetch_market_data(self, pair: str, timeframe: str) -> dict[str, Any] | None:
        """Fetch OHLCV data and compute indicators for *pair*/*timeframe*."""
        assert self._client is not None
        try:
            candles = await self._client.fetch_ohlcv(pair, timeframe, limit=300)
            if len(candles) < _MIN_CANDLES:
                logger.debug("SwingEngine: insufficient candles for %s/%s (%d)", pair, timeframe, len(candles))
                return None

            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]

            rsi = calculate_rsi(closes)
            ema50 = calculate_ema(closes, 50)
            ema200 = calculate_ema(closes, 200)
            macd = calculate_macd(closes)
            ema_crossover = detect_ema_crossover(closes, 50, 200)
            atr = calculate_atr(highs, lows, closes)

            # RSI series for divergence detection — compute full RSI series once via pandas
            import pandas as pd
            s = pd.Series(closes, dtype=float)
            delta = s.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, float("nan"))
            rsi_full = (100 - (100 / (1 + rs))).fillna(50).tolist()
            lookback = 30
            rsi_series = rsi_full[-lookback:]
            price_window = closes[-lookback:]
            rsi_divergence = detect_rsi_divergence(price_window, rsi_series, lookback=20)

            return {
                "pair": pair,
                "timeframe": timeframe,
                "close": closes[-1],
                "open": candles[-1]["open"],
                "high": candles[-1]["high"],
                "low": candles[-1]["low"],
                "volume": candles[-1]["volume"],
                "candles": candles,
                "closes": closes,
                "highs": highs,
                "lows": lows,
                "indicators": {
                    "rsi": round(rsi, 2),
                    "ema50": round(ema50, 6),
                    "ema200": round(ema200, 6),
                    "ema_crossover": ema_crossover,
                    "macd": {
                        "macd": round(macd["macd"], 6),
                        "signal": round(macd["signal"], 6),
                        "histogram": round(macd["histogram"], 6),
                    },
                    "atr": round(atr, 6),
                    "rsi_divergence": rsi_divergence,
                },
            }
        except Exception as exc:
            logger.error("SwingEngine: failed to fetch %s/%s: %s", pair, timeframe, exc)
            return None

    # ------------------------------------------------------------------
    # Setup detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_setup(market_data: dict[str, Any]) -> bool:
        """Return True if at least 2 of 3 swing setup conditions are met.

        Conditions:
          1. RSI divergence detected (BULLISH_DIV or BEARISH_DIV)
          2. EMA50 vs EMA200 alignment (crossover or clear separation)
          3. MACD histogram turning (positive = bullish, negative = bearish)
        """
        indicators = market_data.get("indicators", {})
        rsi_div = indicators.get("rsi_divergence", "NONE")
        ema_crossover = indicators.get("ema_crossover", "NONE")
        ema50 = float(indicators.get("ema50", 0))
        ema200 = float(indicators.get("ema200", 0))
        macd = indicators.get("macd", {})
        macd_hist = float(macd.get("histogram", 0))

        condition_1 = rsi_div in ("BULLISH_DIV", "BEARISH_DIV")
        condition_2 = ema_crossover != "NONE" or (ema50 != 0 and ema200 != 0 and abs(ema50 - ema200) / ema200 > 0.005)
        condition_3 = abs(macd_hist) > 0  # histogram is non-zero (turning)

        met = sum([condition_1, condition_2, condition_3])
        return met >= 2

    # ------------------------------------------------------------------
    # Signal building with futures filters
    # ------------------------------------------------------------------

    async def _build_signal(self, pair: str, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Build a signal with real ATR-based SL/TP and futures data filters."""
        assert self._client is not None
        indicators = market_data.get("indicators", {})
        close = float(market_data["close"])
        atr = float(indicators.get("atr", close * 0.02))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        ema50 = float(indicators.get("ema50", 0))
        ema200 = float(indicators.get("ema200", 1))
        rsi_div = indicators.get("rsi_divergence", "NONE")
        macd_hist = float(indicators.get("macd", {}).get("histogram", 0))

        # Determine direction
        if ema_crossover == "BULLISH" or (ema50 > ema200 and rsi_div == "BULLISH_DIV") or (ema50 > ema200 and macd_hist > 0):
            direction = "LONG"
        elif ema_crossover == "BEARISH" or (ema50 < ema200 and rsi_div == "BEARISH_DIV") or (ema50 < ema200 and macd_hist < 0):
            direction = "SHORT"
        else:
            return None  # No clear direction

        # SL = 2.5x ATR; TP = 2x SL (minimum 2:1 RR)
        sl_distance = atr * 2.5
        tp_distance = sl_distance * 2.0

        if sl_distance <= 0:
            return None

        if direction == "LONG":
            stop_loss = round(close - sl_distance, 6)
            take_profit = round(close + tp_distance, 6)
        else:
            stop_loss = round(close + sl_distance, 6)
            take_profit = round(close - tp_distance, 6)

        # Validate minimum 2:1 RR
        risk = abs(close - stop_loss)
        reward = abs(take_profit - close)
        if risk <= 0 or (reward / risk) < 2.0:
            return None

        # Base confidence from EMA alignment and divergence signals
        confidence = 0.58
        if ema_crossover != "NONE":
            confidence += 0.10
        if rsi_div in ("BULLISH_DIV", "BEARISH_DIV"):
            confidence += 0.08

        # Fetch futures data — one aggregated call across all exchanges
        futures_data: dict[str, Any] = {}
        try:
            futures_data = await self._client.fetch_aggregated_futures_data(pair)
        except Exception as exc:
            logger.debug("Failed to fetch aggregated futures data for %s: %s", pair, exc)

        # Funding rate filter — use average across all exchanges
        funding_rate = futures_data.get("funding_rate_avg")
        if funding_rate is not None:
            fr = float(funding_rate)
            if direction == "LONG" and fr > 0.0005:
                confidence -= 0.20
            elif direction == "SHORT" and fr < -0.0005:
                confidence -= 0.20

        # Open interest filter — use aggregated total OI change
        oi_change = float(futures_data.get("oi_change_pct", 0))
        if oi_change != 0:
            if (direction == "LONG" and oi_change > 0) or (direction == "SHORT" and oi_change < 0):
                confidence += 0.10

        # Long/short ratio filter — use averaged ratio
        ls_avg = futures_data.get("ls_ratio_avg")
        if ls_avg is not None:
            ls_ratio = float(ls_avg)
            if direction == "LONG" and ls_ratio > 2.0:
                confidence -= 0.15

        # Minimum confidence gate
        confidence = round(min(0.95, max(0.0, confidence)), 4)
        if confidence < _MIN_CONFIDENCE:
            logger.debug("SwingEngine: confidence %.2f below gate for %s — skipped", confidence, pair)
            return None

        return {
            "pair": pair,
            "engine": "swing",
            "timeframe": market_data["timeframe"],
            "direction": direction,
            "entry": round(close, 6),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "reasoning": (
                f"EMA50/200 {'bullish' if ema50 > ema200 else 'bearish'} "
                f"({ema_crossover}), RSI div: {rsi_div}, MACD hist: {macd_hist:.4f}"
            ),
            "market_data": {
                **market_data,
                "futures": futures_data,
            },
        }
