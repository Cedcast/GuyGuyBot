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

import datetime
import logging
from typing import TYPE_CHECKING, Any

from data.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_rsi,
    calculate_stochastic,
    calculate_volume_sma,
    calculate_vwap,
    classify_market_regime,
    detect_ema_crossover,
    detect_support_resistance,
)
from engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from data.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

# Minimum number of candles required for indicator calculation
_MIN_CANDLES = 50

# Minimum confidence (after all adjustments) to pass signal to pipeline
_MIN_CONFIDENCE = 0.73

# Seconds per timeframe — used for candle-age and completion calculations
_TF_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


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
            bb = calculate_bollinger_bands(closes)
            stoch = calculate_stochastic(highs, lows, closes)
            adx_data = calculate_adx(highs, lows, closes)
            sr = detect_support_resistance(highs, lows, closes)
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
                "volumes": volumes,
                "indicators": {
                    "rsi": round(rsi, 2),
                    "ema_fast": round(ema_fast, 6),
                    "ema_slow": round(ema_slow, 6),
                    "ema_crossover": ema_crossover,
                    "atr": round(atr, 6),
                    "volume_ratio": round(volume_ratio, 2),
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
        """Return True when at least 3 of 5 directional signals agree.

        Signals:
          1. Price vs VWAP
          2. RSI extreme (< 40 bullish, > 60 bearish)
          3. Bollinger Band touch
          4. EMA9/21 crossover
          5. Stochastic K extreme (< 25 bullish, > 75 bearish)

        An ADX gate is applied first: ranging markets are skipped entirely.
        """
        indicators = market_data.get("indicators", {})
        rsi = float(indicators.get("rsi", 50))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        volume_ratio = float(indicators.get("volume_ratio", 1.0))
        bb = indicators.get("bb", {})
        close = float(market_data.get("close", 0))
        vwap = float(indicators.get("vwap", close))
        regime = indicators.get("regime", "RANGING")
        stoch = indicators.get("stoch", {})

        # ADX gate: skip ranging markets entirely
        if regime == "RANGING":
            return False

        # Build 5-signal vote
        bullish_votes = 0
        bearish_votes = 0

        # Signal 1: VWAP
        if close > vwap:
            bullish_votes += 1
        elif close < vwap:
            bearish_votes += 1

        # Signal 2: RSI
        if rsi < 40:
            bullish_votes += 1
        elif rsi > 60:
            bearish_votes += 1

        # Signal 3: Bollinger Bands
        bb_lower = float(bb.get("lower", 0))
        bb_upper = float(bb.get("upper", float("inf")))
        if bb_lower > 0 and close <= bb_lower * 1.005:
            bullish_votes += 1
        elif bb_upper > 0 and close >= bb_upper * 0.995:
            bearish_votes += 1

        # Signal 4: EMA crossover
        if ema_crossover == "BULLISH":
            bullish_votes += 1
        elif ema_crossover == "BEARISH":
            bearish_votes += 1

        # Signal 5: Stochastic
        stoch_k = float(stoch.get("k", 50))
        if stoch_k < 25:
            bullish_votes += 1
        elif stoch_k > 75:
            bearish_votes += 1

        if bullish_votes > bearish_votes and bullish_votes >= 4:
            return True
        if bearish_votes > bullish_votes and bearish_votes >= 4:
            return True
        return False

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
        """Build a signal with S/R-aware SL/TP and weighted confidence scoring."""
        assert self._client is not None

        # Time-of-day blackout: UTC 00:00–04:00 (low-liquidity, fakeout-prone)
        _now_utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 0 <= _now_utc_hour < 4:
            logger.debug("ScalpingEngine: low-liquidity window (UTC %02d:xx) — skipping %s", _now_utc_hour, pair)
            return None

        # Candle-age gate: only enter in the last 20% of a candle's lifespan
        tf = market_data.get("timeframe", "")
        tf_secs = _TF_SECONDS.get(tf, 0)
        if tf_secs > 0:
            now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
            candle_position = now_ts % tf_secs
            candle_completion = candle_position / tf_secs
            if candle_completion < 0.80:
                logger.debug(
                    "ScalpingEngine: candle only %.0f%% complete for %s/%s — skipping mid-candle entry",
                    candle_completion * 100, pair, tf,
                )
                return None

        indicators = market_data.get("indicators", {})
        close = float(market_data["close"])
        atr = float(indicators.get("atr", close * 0.005))
        ema_crossover = indicators.get("ema_crossover", "NONE")
        rsi = float(indicators.get("rsi", 50))
        vwap = float(indicators.get("vwap", close))
        bb = indicators.get("bb", {})
        stoch = indicators.get("stoch", {})
        adx = float(indicators.get("adx", 0))
        plus_di = float(indicators.get("plus_di", 0))
        minus_di = float(indicators.get("minus_di", 0))
        support = float(indicators.get("support", 0))
        resistance = float(indicators.get("resistance", 0))
        volume_ratio = float(indicators.get("volume_ratio", 1.0))
        regime = indicators.get("regime", "RANGING")

        # BB bandwidth volatility gate: skip already-broken squeezes
        bb_bw = float(bb.get("bandwidth", 0))
        if bb_bw > 0.08:
            logger.debug("ScalpingEngine: BB bandwidth %.4f too wide for %s — skipped", bb_bw, pair)
            return None

        # Recompute direction vote (mirrors _has_setup)
        bullish_votes = 0
        bearish_votes = 0
        if close > vwap:
            bullish_votes += 1
        elif close < vwap:
            bearish_votes += 1
        if rsi < 40:
            bullish_votes += 1
        elif rsi > 60:
            bearish_votes += 1
        bb_lower = float(bb.get("lower", 0))
        bb_upper = float(bb.get("upper", float("inf")))
        if bb_lower > 0 and close <= bb_lower * 1.005:
            bullish_votes += 1
        elif bb_upper > 0 and close >= bb_upper * 0.995:
            bearish_votes += 1
        if ema_crossover == "BULLISH":
            bullish_votes += 1
        elif ema_crossover == "BEARISH":
            bearish_votes += 1
        stoch_k = float(stoch.get("k", 50))
        if stoch_k < 25:
            bullish_votes += 1
        elif stoch_k > 75:
            bearish_votes += 1

        if bullish_votes > bearish_votes and bullish_votes >= 4:
            direction = "LONG"
        elif bearish_votes > bullish_votes and bearish_votes >= 4:
            direction = "SHORT"
        else:
            return None

        # Volume gate: require ≥1.2× average volume to confirm directional move
        if volume_ratio < 1.2:
            logger.debug("ScalpingEngine: insufficient volume (%.2fx) for %s — skipped", volume_ratio, pair)
            return None

        # S/R-aware SL: place behind nearest S/R level; fall back to 2.0× ATR
        sl_distance = atr * 2.0  # was 1.5 — wider fallback keeps SL outside noise
        if direction == "LONG" and support > 0 and support < close:
            sr_distance = close - support
            if atr * 0.5 < sr_distance < atr * 3:
                sl_distance = sr_distance + atr * 0.2  # small buffer beyond S/R
        elif direction == "SHORT" and resistance > 0 and resistance > close:
            sr_distance = resistance - close
            if atr * 0.5 < sr_distance < atr * 3:
                sl_distance = sr_distance + atr * 0.2

        # BB SL check: skip if SL would land inside the Bollinger Band
        bb_lower_val = float(bb.get("lower", 0))
        bb_upper_val = float(bb.get("upper", float("inf")))
        if direction == "LONG" and bb_lower_val > 0 and (close - sl_distance) > bb_lower_val:
            logger.debug("ScalpingEngine: SL inside BB lower for %s — skipped", pair)
            return None
        if direction == "SHORT" and bb_upper_val < float("inf") and (close + sl_distance) < bb_upper_val:
            logger.debug("ScalpingEngine: SL inside BB upper for %s — skipped", pair)
            return None

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

        # ADX contribution (25% weight, max +0.20)
        adx_score = min(1.0, max(0.0, (adx - 20) / 30))
        confidence += adx_score * 0.20

        # S/R proximity contribution (20% weight, max +0.15)
        if direction == "LONG" and support > 0 and close > 0:
            proximity = 1.0 - min(1.0, (close - support) / (close * 0.02))
            confidence += proximity * 0.15
        elif direction == "SHORT" and resistance > 0 and close > 0:
            proximity = 1.0 - min(1.0, (resistance - close) / (close * 0.02))
            confidence += proximity * 0.15

        # Volume contribution (15% weight, max +0.10)
        vol_score = min(1.0, max(0.0, (volume_ratio - 1.0) / 2.0))
        confidence += vol_score * 0.10

        # Vote strength contribution (max +0.10)
        vote_strength = max(bullish_votes, bearish_votes) / 5.0
        confidence += vote_strength * 0.10

        # DI alignment bonus (+0.05 if DI agrees with direction)
        if direction == "LONG" and plus_di > minus_di:
            confidence += 0.05
        elif direction == "SHORT" and minus_di > plus_di:
            confidence += 0.05

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
                    logger.debug("ScalpingEngine: cross-exchange funding consensus against LONG for %s — skipped", pair)
                    return None
                if direction == "SHORT" and negative_count > positive_count:
                    logger.debug("ScalpingEngine: cross-exchange funding consensus against SHORT for %s — skipped", pair)
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
            logger.debug("ScalpingEngine: confidence %.2f below gate for %s — skipped", confidence, pair)
            return None

        # Candle completion percentage (for Claude context)
        candle_completion_pct: float | None = None
        if tf_secs > 0:
            completion_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
            candle_completion_pct = round(((completion_ts % tf_secs) / tf_secs) * 100, 1)

        return {
            "pair": pair,
            "engine": "scalping",
            "timeframe": market_data["timeframe"],
            "direction": direction,
            "entry": round(close, 6),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profit_1": round(close + (sl_distance * 1.0), 6) if direction == "LONG" else round(close - (sl_distance * 1.0), 6),
            "take_profit_2": round(close + (sl_distance * 3.0), 6) if direction == "LONG" else round(close - (sl_distance * 3.0), 6),
            "candle_completion_pct": candle_completion_pct,
            "confidence": confidence,
            "adx": round(adx, 2),
            "regime": regime,
            "support": round(support, 6),
            "resistance": round(resistance, 6),
            "vwap": round(vwap, 6),
            "reasoning": (
                f"Regime: {regime}, ADX: {adx:.1f}, "
                f"votes: {max(bullish_votes, bearish_votes)}/5, "
                f"RSI: {rsi:.1f}, VWAP: {'above' if close > vwap else 'below'}"
            ),
            "market_data": {
                **market_data,
                "futures": futures_data,
            },
        }
