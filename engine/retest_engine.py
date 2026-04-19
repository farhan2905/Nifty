"""
RetestEngine: Runs every 15m, validates predictions across timeframes, updates signal status.
Core idea: We set targets on H1, D1, W1 timeframes. Every 15m candle is a retest opportunity.
If price behaves consistently → signal CONFIRMED, confidence UP
If price diverges → signal INVALIDATED, exit signal fired
If news changes → signal REASSESSED with updated sentiment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import logging
import uuid
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & Data Classes
# ---------------------------------------------------------------------------

class SignalStatus(Enum):
    PENDING = "pending"          # Signal issued, not yet confirmed
    CONFIRMED = "confirmed"      # Multiple retests passed
    TRACKING = "tracking"        # In-progress, some retests passed
    INVALIDATED = "invalidated"  # Retest failed — exit
    COMPLETED = "completed"      # Target reached


class RetestResult(Enum):
    PASS = "pass"                # Price behavior consistent with thesis
    FAIL = "fail"                # Price diverged from thesis
    NEUTRAL = "neutral"          # Ambiguous


@dataclass
class ActiveSignal:
    signal_id: str
    direction: int               # 1=long, -1=short
    entry_price: float
    timestamp: datetime

    # Multi-timeframe targets
    target_1h: float             # H1 target price
    target_1d: float             # Daily target
    target_1w: float             # Weekly target
    stop_loss: float             # ATR-based stop

    # Retest tracking
    retest_count: int = 0
    retest_passes: int = 0
    retest_fails: int = 0
    confidence: float = 0.5
    status: SignalStatus = SignalStatus.PENDING

    # Sentiment tracker
    last_sentiment: float = 0.0
    sentiment_shift_alert: bool = False

    # History
    retest_history: List[Dict] = field(default_factory=list)

    # Internal state
    last_candle_close: float = 0.0
    last_candle_high: float = 0.0
    last_candle_low: float = 0.0
    time_since_entry_hours: float = 0.0
    soft_invalidation_warned: bool = False


# ---------------------------------------------------------------------------
# Regime Constants
# ---------------------------------------------------------------------------

REGIME_NAMES = {0: "ranging", 1: "weak_bull", 2: "strong_bull", 3: "weak_bear", 4: "strong_bear"}

REGIME_TARGET_MULTIPLIERS = {
    "strong_bull": 1.20,
    "weak_bull":   1.00,
    "ranging":     0.70,
    "weak_bear":   1.00,
    "strong_bear": 1.20,
}


# ---------------------------------------------------------------------------
# RetestEngine
# ---------------------------------------------------------------------------

class RetestEngine:
    """
    Core retest logic running every 15 minutes.

    Usage
    -----
    engine = RetestEngine(config, model, sentiment_analyzer)
    result = engine.on_new_15m_candle(candle, market_data, news_articles)
    """

    # Confidence thresholds for status transitions
    CONFIRM_THRESHOLD:     float = 0.70
    TRACKING_THRESHOLD:    float = 0.55
    INVALIDATE_THRESHOLD:  float = 0.30

    # Volume z-score needed for confirmation
    VOLUME_CONFIRM_ZSCORE: float = 0.5

    # Minimum directional progress per candle (fraction of ATR)
    MIN_PROGRESS_PER_CANDLE: float = 0.001   # 0.1% of price

    # VIX spike threshold (absolute daily change)
    VIX_SPIKE_THRESHOLD: float = 3.0

    # Time-decay: soft invalidation warning after N hours without target progress
    SOFT_INVALIDATION_HOURS: float = 4.0

    # Bayesian update factors
    PASS_ALPHA:    float = 0.10   # fraction of remaining confidence gap added on PASS
    FAIL_BETA:     float = 0.20   # fraction of current confidence removed on FAIL

    # News urgency threshold that triggers a sentiment alert
    NEWS_URGENCY_THRESHOLD: float = 0.70

    def __init__(self, config: dict, model, sentiment_analyzer):
        """
        Parameters
        ----------
        config           : dict with keys: N_ENSEMBLE, CONFIDENCE_GATE, ATR_PERIOD, etc.
        model            : trained Hi-LSTM ensemble (exposes .predict(market_data) → dict)
        sentiment_analyzer: exposes .score(articles) → list[{headline, score, urgency}]
        """
        self.config = config
        self.model = model
        self.sentiment = sentiment_analyzer
        self.active_signals: Dict[str, ActiveSignal] = {}
        self.signal_log: List[Dict] = []
        self._candle_buffer: List[dict] = []   # rolling window for volume baseline
        logger.info("RetestEngine initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_new_15m_candle(
        self,
        candle: dict,
        market_data: dict,
        news_articles: list,
    ) -> dict:
        """
        Main entry point. Called every 15 minutes with new candle data.

        Parameters
        ----------
        candle       : {open, high, low, close, volume, timestamp}
        market_data  : {x_15m, x_1h, x_1d, x_1w, vix, fii_net, pcr, max_pain}
        news_articles: list of recent news dicts {headline, body, timestamp}

        Returns
        -------
        {
          "new_signals":          [...],   # Newly fired signals
          "updated_signals":      [...],   # Updated active signals
          "invalidated_signals":  [...],   # Signals that failed retest
          "completed_signals":    [...],   # Target reached
          "sentiment_alerts":     [...],   # News-driven warnings
          "model_prediction":     {...},   # Fresh prediction from model
        }
        """
        ts = candle.get("timestamp", datetime.utcnow())
        logger.debug("on_new_15m_candle | %s | close=%.2f", ts, candle["close"])

        # — 0. Buffer candle for volume baseline —
        self._candle_buffer.append(candle)
        if len(self._candle_buffer) > 96:          # ~1 day of 15m bars
            self._candle_buffer.pop(0)

        # — 1. Fresh model inference —
        prediction = self._run_model_prediction(market_data)
        logger.debug("Model prediction: %s", prediction)

        # — 2. Sentiment scoring —
        sentiment_scores = self.sentiment.score(news_articles) if news_articles else []
        agg_sentiment = self._aggregate_sentiment(sentiment_scores)

        # — 3. Possibly fire a new signal —
        new_signals: List[dict] = []
        if self._should_fire_signal(prediction, market_data):
            sig = self._fire_new_signal(prediction, candle, market_data)
            new_signals.append(self.generate_signal_report(sig))
            logger.info("New signal fired: %s direction=%d conf=%.3f",
                        sig.signal_id, sig.direction, sig.confidence)

        # — 4. Evaluate all active signals —
        updated_signals: List[dict] = []
        invalidated_signals: List[dict] = []
        completed_signals: List[dict] = []
        sentiment_alerts: List[dict] = []

        for sid, signal in list(self.active_signals.items()):
            # 4a. Check target reached
            if self._is_target_reached(signal, candle):
                signal.status = SignalStatus.COMPLETED
                self._log_signal(signal, "target_reached")
                completed_signals.append(self.generate_signal_report(signal))
                del self.active_signals[sid]
                continue

            # 4b. Check stop-loss hit
            if self._is_stop_hit(signal, candle):
                signal.status = SignalStatus.INVALIDATED
                self._log_signal(signal, "stop_loss")
                invalidated_signals.append(self.generate_signal_report(signal))
                del self.active_signals[sid]
                continue

            # 4c. Sentiment shift check
            shift = self._check_sentiment_shift(signal, sentiment_scores)
            if shift:
                signal.sentiment_shift_alert = True
                sentiment_alerts.append({
                    "signal_id": sid,
                    "alert": "Sentiment regime shift detected",
                    "agg_sentiment": agg_sentiment,
                    "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                })

            # 4d. Core retest evaluation
            result = self._evaluate_retest(signal, candle, agg_sentiment)
            signal.retest_count += 1
            if result == RetestResult.PASS:
                signal.retest_passes += 1
            elif result == RetestResult.FAIL:
                signal.retest_fails += 1

            # 4e. Confidence update
            new_conf = self._update_signal_confidence(signal, result, prediction["confidence"])
            signal.confidence = new_conf

            # 4f. Time-decay soft invalidation
            hours_elapsed = (ts - signal.timestamp).total_seconds() / 3600.0 \
                if isinstance(ts, datetime) and isinstance(signal.timestamp, datetime) \
                else 0.0
            signal.time_since_entry_hours = hours_elapsed
            if (hours_elapsed > self.SOFT_INVALIDATION_HOURS
                    and not signal.soft_invalidation_warned
                    and signal.status not in (SignalStatus.CONFIRMED,)):
                signal.soft_invalidation_warned = True
                sentiment_alerts.append({
                    "signal_id": sid,
                    "alert": f"Soft-invalidation: {hours_elapsed:.1f}h without target progress",
                    "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                })

            # 4g. Status transition
            signal.status = self._resolve_status(signal)

            # 4h. Record retest history entry
            signal.retest_history.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "result": result.value,
                "confidence": round(new_conf, 4),
                "status": signal.status.value,
                "close": candle["close"],
            })
            signal.last_candle_close = candle["close"]
            signal.last_candle_high  = candle["high"]
            signal.last_candle_low   = candle["low"]

            # 4i. Hard invalidation check
            if signal.confidence < self.INVALIDATE_THRESHOLD:
                signal.status = SignalStatus.INVALIDATED
                self._log_signal(signal, "low_confidence")
                invalidated_signals.append(self.generate_signal_report(signal))
                del self.active_signals[sid]
            else:
                updated_signals.append(self.generate_signal_report(signal))

        return {
            "new_signals":         new_signals,
            "updated_signals":     updated_signals,
            "invalidated_signals": invalidated_signals,
            "completed_signals":   completed_signals,
            "sentiment_alerts":    sentiment_alerts,
            "model_prediction":    prediction,
        }

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _run_model_prediction(self, market_data: dict) -> dict:
        """
        Run fresh model inference.

        Returns
        -------
        {direction, magnitude, confidence, regime, uncertainty, attention_weights}
        """
        try:
            raw = self.model.predict(market_data)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            # Graceful degradation: neutral prediction
            return {
                "direction":        0,
                "magnitude":        0.0,
                "confidence":       0.0,
                "regime":           "ranging",
                "uncertainty":      1.0,
                "attention_weights": {},
            }

        # Normalise output shape — model may return tensors or numpy arrays
        direction  = int(np.sign(float(raw.get("direction_logit", 0))))
        magnitude  = float(raw.get("magnitude", 0.0))
        confidence = float(np.clip(raw.get("confidence", 0.5), 0.0, 1.0))
        regime_idx = int(raw.get("regime", 0))
        regime     = REGIME_NAMES.get(regime_idx, "ranging")
        uncertainty = float(raw.get("uncertainty", 1.0 - confidence))
        attn       = raw.get("attention_weights", {})
        if hasattr(attn, "tolist"):
            attn = attn.tolist()

        return {
            "direction":        direction,
            "magnitude":        magnitude,
            "confidence":       confidence,
            "regime":           regime,
            "uncertainty":      uncertainty,
            "attention_weights": attn,
        }

    # ------------------------------------------------------------------
    # Retest evaluation
    # ------------------------------------------------------------------

    def _evaluate_retest(
        self,
        signal: ActiveSignal,
        candle: dict,
        current_sentiment: float,
    ) -> RetestResult:
        """
        Evaluate whether the current 15m candle is consistent with the signal thesis.

        Checks
        ------
        1. Price structure  — Higher lows (long) / lower highs (short)
        2. Volume confirmation — Above 20-bar average?
        3. Distance progress — Are we moving toward H1 target?
        4. Sentiment consistency — Has sentiment flipped?
        5. VIX spike — Sudden spike invalidates trend signals
        6. Time decay — >4h without progress → soft-invalidation warning

        Returns PASS / FAIL / NEUTRAL
        """
        checks: List[Optional[bool]] = []   # True=bullish, False=bearish, None=neutral

        close  = candle["close"]
        high   = candle["high"]
        low    = candle["low"]
        volume = candle.get("volume", 0)
        vix    = candle.get("vix", None)   # some callers inject VIX into candle dict

        prev_close = signal.last_candle_close if signal.last_candle_close else signal.entry_price
        prev_high  = signal.last_candle_high  if signal.last_candle_high  else signal.entry_price
        prev_low   = signal.last_candle_low   if signal.last_candle_low   else signal.entry_price

        # ── Check 1: Price structure ──────────────────────────────────────
        if signal.direction == 1:   # LONG
            # Higher low is bullish structure
            structure_ok = low >= prev_low * 0.9995   # small tolerance
            checks.append(structure_ok)
        else:                       # SHORT
            # Lower high is bearish structure
            structure_ok = high <= prev_high * 1.0005
            checks.append(structure_ok)

        # ── Check 2: Volume confirmation ──────────────────────────────────
        if len(self._candle_buffer) >= 5:
            avg_vol = np.mean([c.get("volume", 0) for c in self._candle_buffer[-20:]])
            if avg_vol > 0:
                vol_ratio = volume / avg_vol
                # Volume > 1x average is mildly confirming, < 0.5x is mildly rejecting
                if vol_ratio >= 1.0:
                    checks.append(True)
                elif vol_ratio < 0.5:
                    checks.append(False)
                else:
                    checks.append(None)
            else:
                checks.append(None)
        else:
            checks.append(None)

        # ── Check 3: Progress toward H1 target ───────────────────────────
        dist_to_target = (signal.target_1h - close) * signal.direction
        entry_to_target = (signal.target_1h - signal.entry_price) * signal.direction
        if entry_to_target > 0:
            progress_pct = 1.0 - dist_to_target / entry_to_target
            # Expect at least MIN_PROGRESS per retest
            expected_progress = signal.retest_count * self.MIN_PROGRESS_PER_CANDLE
            checks.append(progress_pct >= expected_progress)
        else:
            checks.append(None)

        # ── Check 4: Sentiment consistency ────────────────────────────────
        if abs(current_sentiment) > 0.4:
            sentiment_aligned = (signal.direction == 1 and current_sentiment > 0) or \
                                (signal.direction == -1 and current_sentiment < 0)
            checks.append(sentiment_aligned)
        else:
            checks.append(None)   # Neutral sentiment → no strong opinion

        # ── Check 5: VIX spike ───────────────────────────────────────────
        if vix is not None and signal.retest_count > 0:
            prior_vix = getattr(signal, "_last_vix", vix)
            vix_delta = vix - prior_vix
            if vix_delta > self.VIX_SPIKE_THRESHOLD:
                logger.warning("VIX spike detected: Δ%.2f  → FAIL override", vix_delta)
                signal._last_vix = vix
                return RetestResult.FAIL
            signal._last_vix = vix

        # ── Aggregate checks ─────────────────────────────────────────────
        valid  = [c for c in checks if c is not None]
        passes = sum(1 for c in valid if c)
        fails  = sum(1 for c in valid if not c)
        total  = len(valid)

        if total == 0:
            return RetestResult.NEUTRAL

        pass_ratio = passes / total
        if pass_ratio >= 0.60:
            return RetestResult.PASS
        elif pass_ratio <= 0.35:
            return RetestResult.FAIL
        else:
            return RetestResult.NEUTRAL

    # ------------------------------------------------------------------
    # Confidence update
    # ------------------------------------------------------------------

    def _update_signal_confidence(
        self,
        signal: ActiveSignal,
        result: RetestResult,
        model_confidence: float,
    ) -> float:
        """
        Bayesian-style confidence update:
          PASS    → new_conf = conf + (1 - conf) * PASS_ALPHA * model_conf
          FAIL    → new_conf = conf - conf * FAIL_BETA
          NEUTRAL → no change

        Clipped to [0.0, 1.0].
        """
        conf = signal.confidence

        if result == RetestResult.PASS:
            delta = (1.0 - conf) * self.PASS_ALPHA * model_confidence
            new_conf = conf + delta
        elif result == RetestResult.FAIL:
            new_conf = conf - conf * self.FAIL_BETA
        else:
            new_conf = conf

        new_conf = float(np.clip(new_conf, 0.0, 1.0))
        logger.debug(
            "Confidence update | signal=%s result=%s %.4f → %.4f",
            signal.signal_id, result.value, conf, new_conf,
        )
        return new_conf

    # ------------------------------------------------------------------
    # Sentiment shift detection
    # ------------------------------------------------------------------

    def _check_sentiment_shift(
        self,
        signal: ActiveSignal,
        scored_articles: list,
    ) -> bool:
        """
        Detect regime-changing news.

        - LONG signal + negative articles with urgency > NEWS_URGENCY_THRESHOLD → alert
        - SHORT signal + strong positive news (RBI rate cut, strong DII buying) → alert

        Returns True if a significant shift is detected.
        """
        if not scored_articles:
            return False

        high_urgency = [
            a for a in scored_articles
            if float(a.get("urgency", 0)) > self.NEWS_URGENCY_THRESHOLD
        ]

        if not high_urgency:
            return False

        for article in high_urgency:
            score    = float(article.get("score", 0))
            urgency  = float(article.get("urgency", 0))
            headline = article.get("headline", "")

            # Long signal hit by strongly negative news
            if signal.direction == 1 and score < -0.5:
                logger.warning(
                    "Sentiment shift alert (LONG) | signal=%s | headline='%s' score=%.2f urgency=%.2f",
                    signal.signal_id, headline[:80], score, urgency,
                )
                signal.last_sentiment = score
                return True

            # Short signal hit by strongly positive catalyst
            if signal.direction == -1 and score > 0.5:
                positive_keywords = ["rbi cut", "rate cut", "dii buying", "strong gdp",
                                     "buyback", "dividend", "record high", "upgrade"]
                if any(kw in headline.lower() for kw in positive_keywords) or score > 0.7:
                    logger.warning(
                        "Sentiment shift alert (SHORT) | signal=%s | headline='%s' score=%.2f urgency=%.2f",
                        signal.signal_id, headline[:80], score, urgency,
                    )
                    signal.last_sentiment = score
                    return True

        return False

    # ------------------------------------------------------------------
    # Target calculation
    # ------------------------------------------------------------------

    def _calculate_targets(
        self,
        prediction: dict,
        entry_price: float,
        atr: float,
        regime: str,
    ) -> dict:
        """
        Calculate multi-timeframe targets from model prediction.

        Base multipliers
        ----------------
        H1 target  : entry ± atr * 1.5
        D1 target  : entry ± atr * 3.0
        W1 target  : entry ± atr * 6.0
        Stop-loss  : entry ∓ atr * 0.8

        Regime adjustments
        ------------------
        strong_bull / strong_bear : multipliers × 1.20
        ranging                   : multipliers × 0.70
        weak_bull / weak_bear     : multipliers × 1.00
        """
        direction = int(np.sign(prediction.get("direction", 0))) or 1
        mult = REGIME_TARGET_MULTIPLIERS.get(regime, 1.0)

        h1_mult   = 1.5 * mult
        d1_mult   = 3.0 * mult
        w1_mult   = 6.0 * mult
        stop_mult = 0.8             # Stop never widened by regime

        target_1h = entry_price + direction * atr * h1_mult
        target_1d = entry_price + direction * atr * d1_mult
        target_1w = entry_price + direction * atr * w1_mult
        stop_loss = entry_price - direction * atr * stop_mult

        logger.debug(
            "Targets | entry=%.2f atr=%.2f regime=%s mult=%.2f | H1=%.2f D1=%.2f W1=%.2f SL=%.2f",
            entry_price, atr, regime, mult, target_1h, target_1d, target_1w, stop_loss,
        )
        return {
            "target_1h": round(target_1h, 2),
            "target_1d": round(target_1d, 2),
            "target_1w": round(target_1w, 2),
            "stop_loss": round(stop_loss, 2),
        }

    # ------------------------------------------------------------------
    # Signal report & dashboard
    # ------------------------------------------------------------------

    def generate_signal_report(self, signal: ActiveSignal) -> dict:
        """
        Full signal report with recommendation.

        Recommendation logic
        --------------------
        CONFIRMED + passes > 3       → ADD
        TRACKING                     → HOLD
        INVALIDATED / low-confidence → EXIT
        sentiment_shift_alert        → REDUCE
        """
        current_price = signal.last_candle_close or signal.entry_price
        pnl_pct = (current_price - signal.entry_price) / signal.entry_price * 100.0 * signal.direction

        # Recommendation
        if signal.status == SignalStatus.INVALIDATED:
            recommendation = "EXIT"
            reasoning = "Retest failed or confidence below threshold — exit immediately."
        elif signal.status == SignalStatus.COMPLETED:
            recommendation = "EXIT"
            reasoning = "Target reached — close position and book profit."
        elif signal.sentiment_shift_alert:
            recommendation = "REDUCE"
            reasoning = "Adverse sentiment shift detected — reduce position size."
        elif signal.status == SignalStatus.CONFIRMED and signal.retest_passes >= 3:
            recommendation = "ADD"
            reasoning = (
                f"Signal CONFIRMED after {signal.retest_passes} retests with "
                f"confidence={signal.confidence:.2%}. Consider adding to position."
            )
        elif signal.status in (SignalStatus.TRACKING, SignalStatus.CONFIRMED):
            recommendation = "HOLD"
            reasoning = (
                f"Signal is {signal.status.value}. "
                f"{signal.retest_passes}/{signal.retest_count} retests passed. "
                f"Confidence={signal.confidence:.2%}."
            )
        else:
            recommendation = "HOLD"
            reasoning = "Signal is PENDING — awaiting first confirmed retest."

        retest_summary = {
            "total":   signal.retest_count,
            "passes":  signal.retest_passes,
            "fails":   signal.retest_fails,
            "pass_rate": (
                round(signal.retest_passes / signal.retest_count, 3)
                if signal.retest_count > 0 else 0.0
            ),
        }

        return {
            "signal_id":      signal.signal_id,
            "direction":      signal.direction,
            "entry":          round(signal.entry_price, 2),
            "current_price":  round(current_price, 2),
            "pnl_pct":        round(pnl_pct, 3),
            "targets": {
                "h1": round(signal.target_1h, 2),
                "d1": round(signal.target_1d, 2),
                "w1": round(signal.target_1w, 2),
            },
            "stop_loss":       round(signal.stop_loss, 2),
            "confidence":      round(signal.confidence, 4),
            "status":          signal.status.value,
            "retest_summary":  retest_summary,
            "sentiment_alert": signal.sentiment_shift_alert,
            "recommendation":  recommendation,
            "reasoning":       reasoning,
            "timestamp":       signal.timestamp.isoformat()
                               if isinstance(signal.timestamp, datetime)
                               else str(signal.timestamp),
        }

    def get_dashboard_state(self) -> dict:
        """Returns full system state for dashboard display."""
        return {
            "active_signal_count": len(self.active_signals),
            "active_signals": [
                self.generate_signal_report(s)
                for s in self.active_signals.values()
            ],
            "signal_log_tail": self.signal_log[-20:],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_fire_signal(self, prediction: dict, market_data: dict) -> bool:
        """
        Gate logic for generating a new signal:
          - Model confidence must exceed CONFIDENCE_GATE
          - Direction must be non-zero (not flat)
          - No existing signal in the same direction
        """
        gate = self.config.get("CONFIDENCE_GATE", 0.60)
        direction = prediction.get("direction", 0)
        confidence = prediction.get("confidence", 0.0)

        if confidence < gate:
            return False
        if direction == 0:
            return False

        # Avoid duplicate directional signals
        for sig in self.active_signals.values():
            if sig.direction == direction and sig.status not in (
                SignalStatus.INVALIDATED, SignalStatus.COMPLETED
            ):
                return False

        return True

    def _fire_new_signal(
        self,
        prediction: dict,
        candle: dict,
        market_data: dict,
    ) -> ActiveSignal:
        """Create and register a new ActiveSignal."""
        direction  = prediction["direction"]
        entry      = candle["close"]
        atr        = self._estimate_atr(market_data)
        regime     = prediction.get("regime", "ranging")
        targets    = self._calculate_targets(prediction, entry, atr, regime)

        sig = ActiveSignal(
            signal_id   = str(uuid.uuid4())[:8],
            direction   = direction,
            entry_price = entry,
            timestamp   = candle.get("timestamp", datetime.utcnow()),
            target_1h   = targets["target_1h"],
            target_1d   = targets["target_1d"],
            target_1w   = targets["target_1w"],
            stop_loss   = targets["stop_loss"],
            confidence  = prediction["confidence"],
            last_candle_close = entry,
            last_candle_high  = candle["high"],
            last_candle_low   = candle["low"],
        )
        self.active_signals[sig.signal_id] = sig
        return sig

    def _is_target_reached(self, signal: ActiveSignal, candle: dict) -> bool:
        """Check if price has hit H1 target (first milestone)."""
        if signal.direction == 1:
            return candle["high"] >= signal.target_1h
        else:
            return candle["low"] <= signal.target_1h

    def _is_stop_hit(self, signal: ActiveSignal, candle: dict) -> bool:
        """Check if stop-loss has been breached."""
        if signal.direction == 1:
            return candle["low"] <= signal.stop_loss
        else:
            return candle["high"] >= signal.stop_loss

    def _resolve_status(self, signal: ActiveSignal) -> SignalStatus:
        """State machine: resolve current SignalStatus from confidence + history."""
        if signal.status in (SignalStatus.INVALIDATED, SignalStatus.COMPLETED):
            return signal.status
        if signal.confidence >= self.CONFIRM_THRESHOLD and signal.retest_passes >= 2:
            return SignalStatus.CONFIRMED
        if signal.confidence >= self.TRACKING_THRESHOLD:
            return SignalStatus.TRACKING
        return SignalStatus.PENDING

    def _estimate_atr(self, market_data: dict, period: int = 14) -> float:
        """
        Estimate ATR from 15m price data in market_data.
        Falls back to a default fraction of the last close if data is unavailable.
        """
        x_15m = market_data.get("x_15m")
        try:
            if x_15m is not None:
                arr = np.array(x_15m)
                if arr.ndim == 3:
                    arr = arr[0]   # (seq_len, features)
                # Assume features: [open, high, low, close, volume, ...]
                highs  = arr[-period:, 1]
                lows   = arr[-period:, 2]
                closes = arr[-period:, 3]
                prev_closes = np.concatenate([[closes[0]], closes[:-1]])
                tr = np.maximum(
                    highs - lows,
                    np.maximum(
                        np.abs(highs - prev_closes),
                        np.abs(lows - prev_closes),
                    ),
                )
                atr = float(np.mean(tr))
                return max(atr, closes[-1] * 0.002)   # floor at 0.2%
        except Exception as exc:
            logger.debug("ATR estimation fallback: %s", exc)

        return 100.0   # Hard fallback (Nifty ~20,000 → 100pts ~ 0.5%)

    def _aggregate_sentiment(self, scored_articles: list) -> float:
        """Weighted average sentiment score across articles, weighted by urgency."""
        if not scored_articles:
            return 0.0
        scores   = np.array([float(a.get("score",   0)) for a in scored_articles])
        urgency  = np.array([float(a.get("urgency", 0.5)) for a in scored_articles])
        total_u  = urgency.sum()
        if total_u == 0:
            return float(scores.mean())
        return float((scores * urgency).sum() / total_u)

    def _log_signal(self, signal: ActiveSignal, reason: str):
        """Append finalised signal to the signal log."""
        self.signal_log.append({
            "signal_id":   signal.signal_id,
            "direction":   signal.direction,
            "entry":       signal.entry_price,
            "exit":        signal.last_candle_close,
            "status":      signal.status.value,
            "reason":      reason,
            "retest_count":  signal.retest_count,
            "retest_passes": signal.retest_passes,
            "final_conf":  round(signal.confidence, 4),
            "timestamp":   datetime.utcnow().isoformat(),
        })
        logger.info(
            "Signal finalised | id=%s status=%s reason=%s conf=%.3f",
            signal.signal_id, signal.status.value, reason, signal.confidence,
        )
