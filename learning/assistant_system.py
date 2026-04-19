from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from learning.daily_feedback import DailyFeedbackLearner, PredictionRecord, RewardEngine


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class ForecastSnapshot:
    """Normalized forecast package that keeps the assistant output consistent."""

    timestamp: str
    direction: str
    confidence: float
    magnitude_pct: float
    regime: str
    horizon_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_probs: Dict[str, float] = field(default_factory=dict)
    uncertainty_pct: float = 0.0
    sentiment: Dict[str, Any] = field(default_factory=dict)
    macro: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)
    adaptive_policy: Dict[str, Any] = field(default_factory=dict)
    spot_price: float = 0.0
    model_prediction: Dict[str, Any] = field(default_factory=dict)


class ForecastEngine:
    """Converts raw model/ensemble outputs into an assistant-ready forecast."""

    def __init__(self, confidence_floor: float = 0.0):
        self.confidence_floor = confidence_floor

    @staticmethod
    def _direction_label(idx: int) -> str:
        return {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}.get(int(idx), "FLAT")

    def build(self, prediction: Dict[str, Any], *, timestamp: Optional[datetime] = None) -> ForecastSnapshot:
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or datetime.utcnow())
        direction_probs = np.asarray(prediction.get("direction_probs", [0.0, 1.0, 0.0]), dtype=float)
        regime_probs = np.asarray(prediction.get("regime_probs", []), dtype=float)

        direction_idx = int(np.argmax(direction_probs)) if direction_probs.size else 1
        confidence = float(prediction.get("confidence_mean", prediction.get("confidence", 0.0)))
        confidence = _clip(confidence, 0.0, 1.0)
        if confidence < self.confidence_floor:
            direction_idx = 1
        direction = self._direction_label(direction_idx)

        horizon_predictions = prediction.get("horizon_predictions_mean") or prediction.get("horizon_predictions") or {}
        normalized_horizon = {}
        for horizon, payload in horizon_predictions.items():
            if not isinstance(payload, dict):
                continue
            d_probs = np.asarray(payload.get("direction_probs", [0.0, 1.0, 0.0]), dtype=float)
            d_idx = int(np.argmax(d_probs)) if d_probs.size else 1
            normalized_horizon[str(horizon)] = {
                "direction_probs": d_probs.tolist(),
                "direction_idx": d_idx,
                "direction_label": self._direction_label(d_idx),
                "confidence_mean": _clip(float(payload.get("confidence_mean", confidence)), 0.0, 1.0),
                "magnitude_mean": float(payload.get("magnitude_mean", prediction.get("magnitude_mean", 0.0))),
                "magnitude_pct": float(payload.get("magnitude_pct", payload.get("magnitude_mean", 0.0) * 100.0)),
            }

        regime_name = prediction.get("regime", "Unknown")
        if regime_probs.size:
            regime_idx = int(np.argmax(regime_probs))
            regime_name = prediction.get("regime_names", {}).get(regime_idx, regime_name) if isinstance(prediction.get("regime_names"), dict) else regime_name

        return ForecastSnapshot(
            timestamp=ts,
            direction=direction,
            confidence=confidence,
            magnitude_pct=float(prediction.get("magnitude_mean", prediction.get("magnitude_pct", 0.0)) * 100.0 if abs(prediction.get("magnitude_mean", 0.0)) <= 1.0 else prediction.get("magnitude_mean", 0.0)),
            regime=str(regime_name),
            horizon_predictions=normalized_horizon,
            regime_probs={str(i): float(v) for i, v in enumerate(regime_probs.tolist())} if regime_probs.size else dict(prediction.get("regime_probs", {})),
            uncertainty_pct=float(prediction.get("uncertainty_pct", prediction.get("magnitude_std", 0.0) * 100.0)),
            sentiment=dict(prediction.get("sentiment") or {}),
            macro=dict(prediction.get("macro") or {}),
            targets=dict(prediction.get("targets") or {}),
            adaptive_policy=dict(prediction.get("adaptive_policy") or {}),
            spot_price=float(prediction.get("spot_price", 0.0)),
            model_prediction=dict(prediction.get("model_prediction") or prediction),
        )


class Verifier:
    """Compares the forecast to the realized candle outcome."""

    def __init__(self, direction_threshold: float = 0.0):
        self.direction_threshold = float(direction_threshold)

    def verify(self, forecast: ForecastSnapshot, actual_return: float, realized_pnl: Optional[float] = None) -> Dict[str, Any]:
        pred_dir = forecast.direction
        actual_dir = "FLAT"
        if actual_return > self.direction_threshold:
            actual_dir = "BULLISH"
        elif actual_return < -self.direction_threshold:
            actual_dir = "BEARISH"

        direction_hit = pred_dir == actual_dir
        direction_score = 1.0 if direction_hit else -1.0
        if pred_dir == "FLAT" and actual_dir == "FLAT":
            direction_score = 0.5

        mag_error_pct = abs((forecast.magnitude_pct / 100.0) - actual_return) * 100.0
        conf_alignment = 1.0 - min(1.0, mag_error_pct / 2.0)
        pnl = float(realized_pnl if realized_pnl is not None else 0.0)
        return {
            "forecast_direction": pred_dir,
            "actual_direction": actual_dir,
            "direction_hit": direction_hit,
            "direction_score": direction_score,
            "magnitude_error_pct": mag_error_pct,
            "confidence_alignment": conf_alignment,
            "realized_pnl": pnl,
        }


class MemoryBuffer:
    """Recent-case memory with recency weighting and regime buckets."""

    def __init__(self, maxlen: int = 500):
        self.maxlen = maxlen
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(dict(item))
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen:]

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.items[-n:]

    def by_regime(self, regime: str, n: int = 50) -> List[Dict[str, Any]]:
        regime_l = (regime or "").lower()
        matches = [x for x in self.items if str(x.get("regime", "")).lower() == regime_l]
        return matches[-n:]

    def similarity_context(self, forecast: ForecastSnapshot, n: int = 5) -> List[Dict[str, Any]]:
        candidates = self.by_regime(forecast.regime, n=50) or self.recent(50)
        if not candidates:
            return []
        scored = []
        for row in candidates:
            score = 0.0
            score += 1.0 if row.get("direction") == forecast.direction else 0.0
            score += 1.0 - min(1.0, abs(float(row.get("confidence", 0.0)) - forecast.confidence))
            score += 1.0 - min(1.0, abs(float(row.get("magnitude_pct", 0.0)) - forecast.magnitude_pct) / max(1.0, abs(forecast.magnitude_pct) + 1e-9))
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n]]


class OnlineCalibrator:
    """Adapts confidence gate, temperature and review sensitivity from recent errors."""

    def __init__(self, confidence_gate: float = 0.65, temperature: float = 1.0):
        self.confidence_gate = confidence_gate
        self.temperature = temperature
        self.error_ema = 0.0
        self.reward_ema = 0.0
        self.overconf_ema = 0.0

    def update(self, verification: Dict[str, Any], reward: float) -> Dict[str, float]:
        mag_err = float(verification.get("magnitude_error_pct", 0.0))
        direction_hit = bool(verification.get("direction_hit", False))
        overconf = max(0.0, 1.0 - float(verification.get("confidence_alignment", 0.0))) if not direction_hit else 0.0

        self.error_ema = 0.9 * self.error_ema + 0.1 * mag_err
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * float(reward)
        self.overconf_ema = 0.9 * self.overconf_ema + 0.1 * overconf

        target_gate = 0.50 + 0.18 * min(1.0, self.error_ema / 2.0) + 0.18 * min(1.0, self.overconf_ema)
        target_gate = _clip(target_gate, 0.45, 0.85)
        self.confidence_gate = 0.85 * self.confidence_gate + 0.15 * target_gate

        target_temp = 1.0 + 0.25 * min(1.0, self.overconf_ema * 1.5) + 0.15 * min(1.0, self.error_ema / 2.0)
        self.temperature = 0.9 * self.temperature + 0.1 * _clip(target_temp, 0.75, 2.0)

        return self.recommend()

    def recommend(self) -> Dict[str, float]:
        return {
            "confidence_gate": round(float(self.confidence_gate), 4),
            "temperature": round(float(self.temperature), 4),
            "error_ema": round(float(self.error_ema), 4),
            "reward_ema": round(float(self.reward_ema), 4),
            "overconf_ema": round(float(self.overconf_ema), 4),
        }


class DriftDetector:
    """Lightweight detector for performance or regime drift."""

    def __init__(self, reward_window: int = 20, error_window: int = 20):
        self.reward_window = reward_window
        self.error_window = error_window
        self.recent_rewards: List[float] = []
        self.recent_errors: List[float] = []
        self.recent_regimes: List[str] = []
        self.baseline_reward = 0.0
        self.baseline_error = 0.0

    def update(self, reward: float, error: float, regime: str) -> Dict[str, Any]:
        self.recent_rewards.append(float(reward))
        self.recent_errors.append(float(error))
        self.recent_regimes.append(str(regime))
        self.recent_rewards = self.recent_rewards[-self.reward_window:]
        self.recent_errors = self.recent_errors[-self.error_window:]
        self.recent_regimes = self.recent_regimes[-max(self.reward_window, self.error_window):]

        if len(self.recent_rewards) >= max(8, self.reward_window // 2):
            self.baseline_reward = 0.95 * self.baseline_reward + 0.05 * float(np.mean(self.recent_rewards))
        if len(self.recent_errors) >= max(8, self.error_window // 2):
            self.baseline_error = 0.95 * self.baseline_error + 0.05 * float(np.mean(self.recent_errors))

        reward_drop = self.baseline_reward - float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        error_rise = float(np.mean(self.recent_errors)) - self.baseline_error if self.recent_errors else 0.0
        regime_changes = sum(1 for a, b in zip(self.recent_regimes, self.recent_regimes[1:]) if a != b)
        drift_score = max(0.0, reward_drop) + max(0.0, error_rise) + 0.05 * regime_changes
        drift_flag = drift_score > 0.35
        return {
            "drift_flag": drift_flag,
            "drift_score": round(float(drift_score), 4),
            "reward_drop": round(float(reward_drop), 4),
            "error_rise": round(float(error_rise), 4),
            "regime_changes": int(regime_changes),
        }


class ExplanationLayer:
    """Generates compact human-readable reasons for the forecast."""

    def summarize(self, forecast: ForecastSnapshot, verification: Optional[Dict[str, Any]] = None, memory: Optional[MemoryBuffer] = None, drift: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        reasons: List[str] = []
        reasons.append(f"Base direction: {forecast.direction} with {forecast.confidence:.1%} confidence")
        reasons.append(f"Primary regime: {forecast.regime}")

        if forecast.horizon_predictions:
            top_h = []
            for horizon in ("15m", "1h", "1d", "1w"):
                hp = forecast.horizon_predictions.get(horizon)
                if hp:
                    top_h.append(f"{horizon}:{hp.get('direction_label')}({float(hp.get('confidence_mean', 0.0)):.0%})")
            if top_h:
                reasons.append("Horizon agreement: " + ", ".join(top_h[:4]))

        if forecast.sentiment:
            label = forecast.sentiment.get("label")
            score = forecast.sentiment.get("score")
            if label is not None:
                reasons.append(f"Sentiment: {label} ({float(score or 0.0):+.2f})")

        if forecast.macro:
            vix = forecast.macro.get("india_vix")
            vix_regime = forecast.macro.get("vix_regime")
            if vix is not None:
                reasons.append(f"Macro: India VIX {float(vix):.1f} ({vix_regime})")

        if verification:
            reasons.append(
                f"Last check: hit={verification.get('direction_hit')} error={float(verification.get('magnitude_error_pct', 0.0)):.2f}%"
            )

        if memory is not None:
            similar = memory.similarity_context(forecast, n=3)
            if similar:
                hits = sum(1 for x in similar if x.get("direction_hit"))
                reasons.append(f"Similar recent cases: {hits}/{len(similar)} correct")

        if drift and drift.get("drift_flag"):
            reasons.append(f"Drift detected: score={drift.get('drift_score')} → confidence will be tightened")

        return {
            "summary": reasons,
            "top_reasons": reasons[:5],
        }


class AssistantOrchestrator:
    """Unifies forecast, verification, reward, memory, calibration, drift, and explanation."""

    def __init__(self, feedback: DailyFeedbackLearner, direction_threshold: float = 0.0):
        self.feedback = feedback
        self.forecast_engine = ForecastEngine()
        self.verifier = Verifier(direction_threshold=direction_threshold)
        self.memory = MemoryBuffer(maxlen=getattr(feedback, "history_limit", 500))
        self.calibrator = OnlineCalibrator()
        self.drift_detector = DriftDetector()
        self.explainer = ExplanationLayer()
        self.reward_engine = RewardEngine()
        self.last_drift: Dict[str, Any] = {}

    def package_forecast(self, raw_prediction: Dict[str, Any], timestamp: Optional[datetime] = None) -> ForecastSnapshot:
        forecast = self.forecast_engine.build(raw_prediction, timestamp=timestamp)
        forecast.adaptive_policy = {
            **(forecast.adaptive_policy or {}),
            **self.calibrator.recommend(),
        }
        return forecast

    def log_prediction(self, forecast: ForecastSnapshot) -> PredictionRecord:
        payload = {
            "timestamp": forecast.timestamp,
            "direction": forecast.direction,
            "confidence": forecast.confidence,
            "magnitude_pct": forecast.magnitude_pct,
            "regime": forecast.regime,
            "sentiment": forecast.sentiment,
            "macro": forecast.macro,
            "adaptive_policy": forecast.adaptive_policy,
        }
        rec = self.feedback.log_prediction(payload)
        self.memory.add({**payload, "status": rec.status})
        return rec

    def verify_and_learn(self, timestamp: str | datetime, actual_return: float, realized_pnl: Optional[float] = None, regime: Optional[str] = None) -> Dict[str, Any]:
        result = self.feedback.update_outcome(timestamp, actual_return=actual_return, realized_pnl=realized_pnl, regime=regime)
        verification = {
            "direction_hit": True,
            "magnitude_error_pct": abs(actual_return) * 100.0,
            "confidence_alignment": 1.0,
        }
        if self.feedback.records:
            last = self.feedback.records[-1]
            forecast = ForecastSnapshot(
                timestamp=last.timestamp,
                direction=last.direction,
                confidence=last.confidence,
                magnitude_pct=last.magnitude_pct,
                regime=last.regime,
                sentiment={"score": last.sentiment_score, "n_articles": last.news_count},
                macro={"india_vix": last.macro_vix},
                adaptive_policy=last.adaptive_policy,
            )
            verification = self.verifier.verify(forecast, actual_return, realized_pnl=realized_pnl)
            reward = float(result.get("reward", 0.0))
            self.memory.add({
                "timestamp": last.timestamp,
                "regime": last.regime,
                "direction": last.direction,
                "confidence": last.confidence,
                "magnitude_pct": last.magnitude_pct,
                "direction_hit": verification["direction_hit"],
                "reward": reward,
                "actual_return": actual_return,
                "realized_pnl": verification["realized_pnl"],
            })
            self.calibrator.update(verification, reward)
            drift = self.drift_detector.update(reward=reward, error=verification["magnitude_error_pct"], regime=regime or last.regime)
            self.last_drift = drift
            result["verification"] = verification
            result["drift"] = drift
            result["explanation"] = self.explainer.summarize(forecast, verification=verification, memory=self.memory, drift=drift)
        return result

    def explain(self, forecast: ForecastSnapshot, verification: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        drift = self.last_drift if self.last_drift else None
        return self.explainer.summarize(forecast, verification=verification, memory=self.memory, drift=drift)

    def recommendations(self) -> Dict[str, Any]:
        rec = self.calibrator.recommend()
        rec.update(self.feedback.recommendations())
        return rec
