"""
engines/scalping_engine.py
--------------------------
Scalping engine — monitors 30m and 1h timeframes.

Uses real OHLCV data from Binance Futures via ExchangeClient and
calculates real technical indicators via the indicators module.
Win-rate boosters: funding rate filter, OI confirmation, multi-timeframe
confluence, long/short ratio filter, and minimum confidence gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from data.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_volume_sma,
    detect_ema_crossover,
)
from engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from data.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

# Minimum number of candles required for indicator calculation
_MIN_CANDLES = 50

# Minimum confidence (after all adjustments) to pass signal to pipeline
_MIN_CONFIDENCE = 0.68


class ScalpingEngine(BaseEngine):
    """Short-term trade engine scanning 30m and 1h timeframes.

    Parameters
    ----------
    pairs:
        Symbols to scan.
    timeframes:
        Timeframes to iterate (default: ``["30m", "1h"]``).
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
        return "scalping"

    async def scan(self) -> list[dict[str, Any]]:
        """Scan all pairs across all scalping timeframes."""
        if self._client is None:
            logger.warning("ScalpingEngine: no ExchangeClient configured — skipping scan")
            return []

        candidates: list[dict[str, Any]] = []

        for pair in self._pairs:
            for timeframe in self._timeframes:
                market_data = await self._fetch_market_data(pair, timeframe)
                if market_data is None:
                    continue

                if not self._has_setup(market_data):
                    continue

                # Multi-timeframe confluence: check 1h trend agrees
                if not await self._check_htf_confluence(pair, market_data):
                    logger.debug("ScalpingEngine: HTF confluence failed for %s/%s", pair, timeframe)
                    continue

                signal = await self._build_signal(pair, market_data)
                if signal is None:
                    continue

                candidates.append(signal)

        logger.info("ScalpingEngine scan complete — %d candidate(s) found", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Market data fetching
    # ------------------------------------------------------------------

    async def _fetch_market_data(self, pair: str, timeframe: str) -> dict[str, Any] | None:
        """Fetch OHLCV data and compute indicators for *pair*/*timeframe*."""
        assert self._client is not None
        try:
            candles = await self._client.fetch_ohlcv(pair, timeframe, limit=200)
            if len(candles) < _MIN_CANDLES:
                logger.debug("ScalpingEngine: insufficient candles for %s/%s (%d)", pair, timeframe, len(candles))
                return None

            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]
            volumes = [c["volume"] for c in candles]

            rsi = calculate_rsi(closes)
            ema_fast = calculate_ema(closes, 9)
            ema_slow = calculate_ema(closes, 21)
            ema_crossover = detect_ema_crossover(closes, 9, 21)
            atr = calculate_atr(highs, lows, closes)
            vol_sma = calculate_volume_sma(volumes, 20)
            current_volume = volumes[-1]
            volume_ratio = current_volume / vol_sma if vol_sma > 0 else 1.0

            return {
                "pair": pair,
                "timeframe": timeframe,
                "close": closes[-1],
                "open": candles[-1]["open"],
                "high": candles[-1]["high"],
                "low": candles[-1]["low"],
                "volume": current_volume,
                "candles": candles,
                "closes": closes,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
                "indicators": {
                    "rsi": round(rsi, 2),
                    "ema_fast": round(ema_fast, 6),
                    "ema_slow": round(ema_slow, 6),
                    "ema_crossover": ema_crossover,
                    "atr": round(atr, 6),
                    "volume_ratio": round(volume_ratio, 2),
                },
            }
        except Exception as exc:
            logger.error("ScalpingEngine: failed to fetch %s/%s: %s", pair, timeframe, exc)
            return None

    # ------------------------------------------------------------------
    # Setup detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_setup(market_data: dict[str, Any]) -> bool:
        """Return True if at least 2 of 3 scalping conditions are met.

        Conditions:
          1. RSI < 35 (oversold) or RSI > 65 (overbought)
          2. EMA9 vs EMA21 crossover detected
          3. Current volume > 1.5x 20-period average
        """
        indicators = market_data.get("indicators", {})
        rsi = float(indicators.get("rsi", 50))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        volume_ratio = float(indicators.get("volume_ratio", 1.0))

        condition_1 = rsi < 35 or rsi > 65
        condition_2 = ema_crossover in ("BULLISH", "BEARISH")
        condition_3 = volume_ratio > 1.5

        met = sum([condition_1, condition_2, condition_3])
        return met >= 2

    # ------------------------------------------------------------------
    # Multi-timeframe confluence
    # ------------------------------------------------------------------

    async def _check_htf_confluence(self, pair: str, market_data: dict[str, Any]) -> bool:
        """Check that the 1h EMA trend agrees with the signal direction."""
        assert self._client is not None
        indicators = market_data.get("indicators", {})
        ema_crossover = indicators.get("ema_crossover", "NONE")

        # Only check confluence when there's a clear crossover direction
        if ema_crossover == "NONE":
            return True

        try:
            htf_candles = await self._client.fetch_ohlcv(pair, "1h", limit=50)
            if len(htf_candles) < 30:
                return True  # Not enough data → don't filter
            htf_closes = [c["close"] for c in htf_candles]
            htf_ema9 = calculate_ema(htf_closes, 9)
            htf_ema21 = calculate_ema(htf_closes, 21)
            htf_bullish = htf_ema9 > htf_ema21
            if ema_crossover == "BULLISH" and not htf_bullish:
                return False
            if ema_crossover == "BEARISH" and htf_bullish:
                return False
            return True
        except Exception as exc:
            logger.debug("HTF confluence check failed for %s: %s", pair, exc)
            return True  # Fail open

    # ------------------------------------------------------------------
    # Signal building with futures filters
    # ------------------------------------------------------------------

    async def _build_signal(self, pair: str, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Build a signal with real ATR-based SL/TP and futures data filters."""
        assert self._client is not None
        indicators = market_data.get("indicators", {})
        close = float(market_data["close"])
        atr = float(indicators.get("atr", close * 0.005))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        rsi = float(indicators.get("rsi", 50))

        # Determine direction from EMA crossover; fall back to RSI extremes
        if ema_crossover == "BULLISH":
            direction = "LONG"
        elif ema_crossover == "BEARISH":
            direction = "SHORT"
        elif rsi < 35:
            direction = "LONG"
        elif rsi > 65:
            direction = "SHORT"
        else:
            return None  # No clear direction

        # SL = 1.5x ATR; TP = 2x SL (minimum 2:1 RR)
        sl_distance = atr * 1.5
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

        # Base confidence from RSI extremity and crossover
        rsi_extremity = max(0.0, (abs(rsi - 50) - 15) / 35)  # 0→1 as RSI moves from 50 to 35/65
        confidence = 0.55 + (rsi_extremity * 0.2)

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
            logger.debug("ScalpingEngine: confidence %.2f below gate for %s — skipped", confidence, pair)
            return None

        return {
            "pair": pair,
            "engine": "scalping",
            "timeframe": market_data["timeframe"],
            "direction": direction,
            "entry": round(close, 6),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "reasoning": f"EMA crossover {ema_crossover}, RSI {rsi:.1f}, vol ratio {indicators.get('volume_ratio', 0):.1f}x",
            "market_data": {
                **market_data,
                "futures": futures_data,
            },
        }
