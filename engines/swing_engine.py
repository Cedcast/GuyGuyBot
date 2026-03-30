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

import datetime
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from data.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    calculate_volume_sma,
    calculate_vwap,
    classify_market_regime,
    detect_ema_crossover,
    detect_rsi_divergence,
    detect_support_resistance,
)
from engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from data.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

_MIN_CANDLES = 100

# Minimum confidence (after all adjustments) to pass signal to pipeline
_MIN_CONFIDENCE = 0.73

# Seconds per timeframe — used for candle-age and completion calculations
_TF_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


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

                # HTF confluence: for 4h signals, require 1d EMA50/200 to agree
                if not await self._check_htf_confluence(pair, market_data):
                    logger.debug("SwingEngine: HTF confluence failed for %s/%s", pair, timeframe)
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

            volumes = [c["volume"] for c in candles]
            vol_sma = calculate_volume_sma(volumes, 20)
            current_volume = volumes[-1]
            volume_ratio = current_volume / vol_sma if vol_sma > 0 else 1.0
            bb = calculate_bollinger_bands(closes)
            stoch = calculate_stochastic(highs, lows, closes)
            adx_data = calculate_adx(highs, lows, closes)
            sr = detect_support_resistance(highs, lows, closes, lookback=80)
            vwap = calculate_vwap(highs, lows, closes, volumes)
            regime = classify_market_regime(
                adx_data["adx"], adx_data["plus_di"], adx_data["minus_di"]
            )

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
                    "bb": {
                        "upper": round(bb["upper"], 6),
                        "middle": round(bb["middle"], 6),
                        "lower": round(bb["lower"], 6),
                        "bandwidth": round(bb["bandwidth"], 4),
                    },
                    "stoch": {
                        "k": round(stoch["k"], 2),
                        "d": round(stoch["d"], 2),
                    },
                    "adx": round(adx_data["adx"], 2),
                    "plus_di": round(adx_data["plus_di"], 2),
                    "minus_di": round(adx_data["minus_di"], 2),
                    "support": round(sr["support"], 6),
                    "resistance": round(sr["resistance"], 6),
                    "vwap": round(vwap, 6),
                    "regime": regime,
                    "volume_ratio": round(volume_ratio, 2),
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

        An ADX gate is applied first: ranging markets with ADX < 20 are skipped.

        Conditions:
          1. RSI divergence detected (BULLISH_DIV or BEARISH_DIV)
          2. EMA50 vs EMA200 alignment (crossover or clear separation)
          3. MACD histogram turning (positive = bullish, negative = bearish)
        """
        indicators = market_data.get("indicators", {})

        # ADX gate: require at least weak trend for swing trades
        regime = indicators.get("regime", "RANGING")
        adx_val = float(indicators.get("adx", 0))
        if regime == "RANGING" and adx_val < 20:
            return False

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
    # HTF confluence
    # ------------------------------------------------------------------

    async def _check_htf_confluence(self, pair: str, market_data: dict[str, Any]) -> bool:
        """For 4h signals, check 1d EMA50/200 direction agrees. 1d signals always pass."""
        assert self._client is not None
        timeframe = market_data.get("timeframe", "")
        if timeframe != "4h":
            return True  # 1d is already the highest TF we use

        indicators = market_data.get("indicators", {})
        ema50 = float(indicators.get("ema50", 0))
        ema200 = float(indicators.get("ema200", 1))
        signal_bullish = ema50 > ema200

        try:
            htf_candles = await self._client.fetch_ohlcv(pair, "1d", limit=210)
            if len(htf_candles) < 200:
                return True  # not enough data — don't block
            htf_closes = [float(c["close"]) for c in htf_candles]
            htf_ema50 = calculate_ema(htf_closes, 50)
            htf_ema200 = calculate_ema(htf_closes, 200)
            htf_bullish = htf_ema50 > htf_ema200
            return signal_bullish == htf_bullish
        except Exception:
            return True  # fail open

    # ------------------------------------------------------------------
    # Signal building with futures filters
    # ------------------------------------------------------------------

    async def _build_signal(self, pair: str, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Build a signal with S/R-aware SL/TP and weighted confidence scoring."""
        assert self._client is not None

        # Candle-age gate: only enter in the last 20% of a candle's lifespan
        tf = market_data.get("timeframe", "")
        tf_secs = _TF_SECONDS.get(tf, 0)
        if tf_secs > 0:
            now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
            candle_position = now_ts % tf_secs
            candle_completion = candle_position / tf_secs
            if candle_completion < 0.80:
                logger.debug(
                    "SwingEngine: candle only %.0f%% complete for %s/%s — skipping mid-candle entry",
                    candle_completion * 100, pair, tf,
                )
                return None

        indicators = market_data.get("indicators", {})
        close = float(market_data["close"])
        atr = float(indicators.get("atr", close * 0.02))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        ema50 = float(indicators.get("ema50", 0))
        ema200 = float(indicators.get("ema200", 1))
        rsi_div = indicators.get("rsi_divergence", "NONE")
        macd_hist = float(indicators.get("macd", {}).get("histogram", 0))
        vwap = float(indicators.get("vwap", close))
        adx = float(indicators.get("adx", 0))
        plus_di = float(indicators.get("plus_di", 0))
        minus_di = float(indicators.get("minus_di", 0))
        support = float(indicators.get("support", 0))
        resistance = float(indicators.get("resistance", 0))
        volume_ratio = float(indicators.get("volume_ratio", 1.0))
        stoch = indicators.get("stoch", {})
        regime = indicators.get("regime", "RANGING")

        # Determine direction
        if ema_crossover == "BULLISH" or (ema50 > ema200 and rsi_div == "BULLISH_DIV") or (ema50 > ema200 and macd_hist > 0):
            direction = "LONG"
        elif ema_crossover == "BEARISH" or (ema50 < ema200 and rsi_div == "BEARISH_DIV") or (ema50 < ema200 and macd_hist < 0):
            direction = "SHORT"
        else:
            return None  # No clear direction

        # Volume gate: require ≥1.2× average volume to confirm directional move
        if volume_ratio < 1.2:
            logger.debug("SwingEngine: insufficient volume (%.2fx) for %s — skipped", volume_ratio, pair)
            return None

        # S/R-aware SL: place behind nearest S/R level; fall back to 3.0× ATR
        sl_distance = atr * 3.0  # was 2.5
        if direction == "LONG" and support > 0 and support < close:
            sr_distance = close - support
            if atr * 0.5 < sr_distance < atr * 5:
                sl_distance = sr_distance + atr * 0.3
        elif direction == "SHORT" and resistance > 0 and resistance > close:
            sr_distance = resistance - close
            if atr * 0.5 < sr_distance < atr * 5:
                sl_distance = sr_distance + atr * 0.3
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

        # Weighted confidence scoring
        confidence = 0.50

        # ADX contribution (max +0.18)
        adx_score = min(1.0, max(0.0, (adx - 15) / 35))
        confidence += adx_score * 0.18

        # EMA crossover bonus
        if ema_crossover != "NONE":
            confidence += 0.08

        # RSI divergence bonus
        if rsi_div in ("BULLISH_DIV", "BEARISH_DIV"):
            confidence += 0.07

        # S/R proximity (max +0.12)
        if direction == "LONG" and support > 0 and close > 0:
            proximity = 1.0 - min(1.0, (close - support) / (close * 0.03))
            confidence += proximity * 0.12
        elif direction == "SHORT" and resistance > 0 and close > 0:
            proximity = 1.0 - min(1.0, (resistance - close) / (close * 0.03))
            confidence += proximity * 0.12

        # Volume contribution (max +0.08)
        vol_score = min(1.0, max(0.0, (volume_ratio - 1.0) / 2.0))
        confidence += vol_score * 0.08

        # DI alignment bonus
        if direction == "LONG" and plus_di > minus_di:
            confidence += 0.05
        elif direction == "SHORT" and minus_di > plus_di:
            confidence += 0.05

        # VWAP alignment bonus
        if direction == "LONG" and close > vwap:
            confidence += 0.04
        elif direction == "SHORT" and close < vwap:
            confidence += 0.04

        # Fetch futures data — one aggregated call across all exchanges
        futures_data: dict[str, Any] = {}
        try:
            futures_data = await self._client.fetch_aggregated_futures_data(pair)
        except Exception as exc:
            logger.debug("Failed to fetch aggregated futures data for %s: %s", pair, exc)

        # Multi-exchange consensus gate: require funding sign to agree across ≥2 exchanges
        exchanges_futures = futures_data.get("by_exchange", {})
        if exchanges_futures:
            funding_signs = [
                1 if float(v.get("funding_rate", 0)) > 0 else (-1 if float(v.get("funding_rate", 0)) < 0 else 0)
                for v in exchanges_futures.values()
                if v.get("funding_rate") is not None
            ]
            if len(funding_signs) >= 2:
                positive_count = funding_signs.count(1)
                negative_count = funding_signs.count(-1)
                if direction == "LONG" and positive_count > negative_count:
                    logger.debug("SwingEngine: cross-exchange funding consensus against LONG for %s — skipped", pair)
                    return None
                if direction == "SHORT" and negative_count > positive_count:
                    logger.debug("SwingEngine: cross-exchange funding consensus against SHORT for %s — skipped", pair)
                    return None

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

        # Candle completion percentage (for Claude context)
        candle_completion_pct: float | None = None
        if tf_secs > 0:
            completion_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
            candle_completion_pct = round(((completion_ts % tf_secs) / tf_secs) * 100, 1)

        return {
            "pair": pair,
            "engine": "swing",
            "timeframe": market_data["timeframe"],
            "direction": direction,
            "entry": round(close, 6),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profit_1": round(close + sl_distance * 1.0, 6) if direction == "LONG" else round(close - sl_distance * 1.0, 6),
            "take_profit_2": round(close + sl_distance * 3.0, 6) if direction == "LONG" else round(close - sl_distance * 3.0, 6),
            "candle_completion_pct": candle_completion_pct,
            "confidence": confidence,
            "adx": round(adx, 2),
            "regime": regime,
            "support": round(support, 6),
            "resistance": round(resistance, 6),
            "vwap": round(vwap, 6),
            "reasoning": (
                f"Regime: {regime}, ADX: {adx:.1f}, "
                f"EMA50/200 {'bullish' if ema50 > ema200 else 'bearish'} ({ema_crossover}), "
                f"RSI div: {rsi_div}, MACD hist: {macd_hist:.4f}, "
                f"VWAP: {'above' if close > vwap else 'below'}"
            ),
            "market_data": {
                **market_data,
                "futures": futures_data,
            },
        }
