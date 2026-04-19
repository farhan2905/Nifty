from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_sign(x: float, threshold: float = 0.0) -> int:
    if x > threshold:
        return 1
    if x < -threshold:
        return -1
    return 0


def _direction_index(direction: str) -> int:
    d = (direction or "").upper()
    if d in {"BULL", "BULLISH", "BUY", "LONG"}:
        return 1
    if d in {"BEAR", "BEARISH", "SELL", "SHORT"}:
        return -1
    return 0


def _default_confidence_gate() -> float:
    return 0.65


@dataclass
class PredictionRecord:
    timestamp: str
    direction: str
    confidence: float
    magnitude_pct: float
    regime: str = "Unknown"
    sentiment_score: float = 0.0
    news_count: int = 0
    macro_vix: Optional[float] = None
    adaptive_policy: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    actual_return: Optional[float] = None
    realized_pnl: Optional[float] = None
    reward: Optional[float] = None


@dataclass
class MarketMemoryBank:
    history_limit: int = 500
    recent_rewards: List[float] = field(default_factory=list)
    recent_accuracy: List[float] = field(default_factory=list)
    recent_overconfidence: List[float] = field(default_factory=list)
    recent_pnl: List[float] = field(default_factory=list)
    regime_scores: Dict[str, List[float]] = field(default_factory=dict)
    confidence_bins: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pending: List[PredictionRecord] = field(default_factory=list)

    ema_reward: float = 0.0
    ema_accuracy: float = 0.5
    ema_overconfidence: float = 0.0
    ema_pnl: float = 0.0
    ema_error: float = 0.0

    def update(self, record: PredictionRecord, reward: float, actual_return: float, pnl: float) -> None:
        correct = self._is_correct(record.direction, actual_return)
        overconf = max(0.0, record.confidence - 0.5) * (0.0 if correct else 1.0)
        err = abs(actual_return)

        self.ema_reward = 0.92 * self.ema_reward + 0.08 * reward
        self.ema_accuracy = 0.92 * self.ema_accuracy + 0.08 * (1.0 if correct else 0.0)
        self.ema_overconfidence = 0.9 * self.ema_overconfidence + 0.1 * overconf
        self.ema_pnl = 0.92 * self.ema_pnl + 0.08 * pnl
        self.ema_error = 0.92 * self.ema_error + 0.08 * err

        self.recent_rewards.append(float(reward))
        self.recent_accuracy.append(1.0 if correct else 0.0)
        self.recent_overconfidence.append(float(overconf))
        self.recent_pnl.append(float(pnl))

        self.regime_scores.setdefault(record.regime or "Unknown", []).append(float(reward))
        cb = self.confidence_bins.setdefault(self._bin_conf(record.confidence), {"count": 0.0, "reward": 0.0, "accuracy": 0.0})
        cb["count"] += 1.0
        cb["reward"] += float(reward)
        cb["accuracy"] += 1.0 if correct else 0.0

        self._trim()

    def _bin_conf(self, conf: float) -> str:
        if conf < 0.4:
            return "low"
        if conf < 0.7:
            return "mid"
        return "high"

    def _trim(self) -> None:
        for lst_name in ("recent_rewards", "recent_accuracy", "recent_overconfidence", "recent_pnl"):
            lst = getattr(self, lst_name)
            if len(lst) > self.history_limit:
                setattr(self, lst_name, lst[-self.history_limit:])
        for k, values in list(self.regime_scores.items()):
            if len(values) > self.history_limit:
                self.regime_scores[k] = values[-self.history_limit:]

    def _is_correct(self, direction: str, actual_return: float, threshold: float = 0.0) -> bool:
        pred = _direction_index(direction)
        actual = _safe_sign(actual_return, threshold=threshold)
        if pred == 0:
            return abs(actual_return) <= threshold * 2
        return pred == actual

    def snapshot(self) -> Dict[str, Any]:
        regime_means = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in self.regime_scores.items()}
        confidence_bins = {}
        for k, v in self.confidence_bins.items():
            cnt = max(v.get("count", 0.0), 1.0)
            confidence_bins[k] = {
                "count": int(v.get("count", 0.0)),
                "avg_reward": float(v.get("reward", 0.0) / cnt),
                "avg_accuracy": float(v.get("accuracy", 0.0) / cnt),
            }
        return {
            "ema_reward": self.ema_reward,
            "ema_accuracy": self.ema_accuracy,
            "ema_overconfidence": self.ema_overconfidence,
            "ema_pnl": self.ema_pnl,
            "ema_error": self.ema_error,
            "recent_rewards": self.recent_rewards[-80:],
            "recent_accuracy": self.recent_accuracy[-80:],
            "recent_overconfidence": self.recent_overconfidence[-80:],
            "recent_pnl": self.recent_pnl[-80:],
            "regime_scores": regime_means,
            "confidence_bins": confidence_bins,
            "pending_count": len(self.pending),
        }


@dataclass
class RewardEngine:
    direction_weight: float = 1.0
    pnl_weight: float = 0.75
    confidence_penalty: float = 0.35
    regime_bonus: float = 0.10
    overconfidence_penalty: float = 0.75
    threshold: float = 0.001

    def compute_reward(
        self,
        direction: str,
        actual_return: float,
        confidence: float,
        realized_pnl: Optional[float] = None,
        regime: str = "Unknown",
        sentiment_score: float = 0.0,
    ) -> float:
        pred = _direction_index(direction)
        actual = _safe_sign(actual_return, threshold=self.threshold)
        if pred == 0:
            direction_score = 0.3 if abs(actual_return) <= self.threshold * 1.5 else -0.25
        elif pred == actual:
            direction_score = 1.0
        else:
            direction_score = -1.0

        pnl = float(realized_pnl if realized_pnl is not None else pred * actual_return)
        pnl_score = math.tanh(pnl * 20.0)
        conf_gap = max(0.0, confidence - 0.5)
        overconf = conf_gap if direction_score < 0 else 0.0
        sent_bonus = 0.05 * float(np.tanh(sentiment_score * 2.0))

        regime_bias = 0.0
        regime_l = (regime or "").lower()
        if "bull" in regime_l and pred >= 0:
            regime_bias += 0.1
        if "bear" in regime_l and pred <= 0:
            regime_bias += 0.1
        if "rang" in regime_l and pred == 0:
            regime_bias += 0.08

        reward = (
            self.direction_weight * direction_score
            + self.pnl_weight * pnl_score
            - self.confidence_penalty * conf_gap
            - self.overconfidence_penalty * overconf
            + self.regime_bonus * regime_bias
            + sent_bonus
        )
        return float(_clip(reward, -2.0, 2.0))


@dataclass
class DailyPolicyTuner:
    confidence_gate: float = field(default_factory=_default_confidence_gate)
    magnitude_gate: float = 0.0
    position_size: float = 1.0
    temperature: float = 1.0
    risk_bias: float = 0.0

    def update(self, memory: MarketMemoryBank, reward: float, actual_return: float) -> None:
        acc = memory.ema_accuracy
        over = memory.ema_overconfidence
        rew = memory.ema_reward

        target_gate = 0.48 + 0.22 * (1.0 - acc) + 0.18 * over
        target_gate = _clip(target_gate, 0.45, 0.85)
        self.confidence_gate = float(0.85 * self.confidence_gate + 0.15 * target_gate)

        target_size = 0.95 + 0.20 * np.tanh(rew) - 0.10 * over
        self.position_size = float(_clip(0.85 * self.position_size + 0.15 * target_size, 0.25, 1.5))

        target_temp = 1.0 + 0.35 * over + 0.20 * max(0.0, 0.52 - acc)
        self.temperature = float(_clip(0.9 * self.temperature + 0.1 * target_temp, 0.75, 2.0))

        self.risk_bias = float(_clip(0.8 * self.risk_bias + 0.2 * np.tanh(-reward), -1.0, 1.0))

        self.magnitude_gate = float(_clip(0.9 * self.magnitude_gate + 0.1 * abs(actual_return), 0.0, 0.20))

    def recommend(self) -> Dict[str, float]:
        return {
            "confidence_gate": round(self.confidence_gate, 4),
            "magnitude_gate": round(self.magnitude_gate, 4),
            "position_size": round(self.position_size, 4),
            "temperature": round(self.temperature, 4),
            "risk_bias": round(self.risk_bias, 4),
        }


class DailyFeedbackLearner:
    def __init__(self, state_path: str | Path, history_limit: int = 500):
        self.state_path = Path(state_path)
        self.history_limit = history_limit
        self.memory = MarketMemoryBank(history_limit=history_limit)
        self.policy = DailyPolicyTuner()
        self.reward_engine = RewardEngine()
        self.records: List[PredictionRecord] = []
        self.load()

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            mem = payload.get("memory", {})
            self.memory.ema_reward = float(mem.get("ema_reward", 0.0))
            self.memory.ema_accuracy = float(mem.get("ema_accuracy", 0.5))
            self.memory.ema_overconfidence = float(mem.get("ema_overconfidence", 0.0))
            self.memory.ema_pnl = float(mem.get("ema_pnl", 0.0))
            self.memory.ema_error = float(mem.get("ema_error", 0.0))
            self.memory.recent_rewards = list(mem.get("recent_rewards", []))[-self.history_limit:]
            self.memory.recent_accuracy = list(mem.get("recent_accuracy", []))[-self.history_limit:]
            self.memory.recent_overconfidence = list(mem.get("recent_overconfidence", []))[-self.history_limit:]
            self.memory.recent_pnl = list(mem.get("recent_pnl", []))[-self.history_limit:]
            self.memory.regime_scores = {k: list(v) for k, v in mem.get("regime_scores", {}).items()}
            self.memory.confidence_bins = {k: dict(v) for k, v in mem.get("confidence_bins", {}).items()}
            self.policy.confidence_gate = float(payload.get("policy", {}).get("confidence_gate", self.policy.confidence_gate))
            self.policy.magnitude_gate = float(payload.get("policy", {}).get("magnitude_gate", self.policy.magnitude_gate))
            self.policy.position_size = float(payload.get("policy", {}).get("position_size", self.policy.position_size))
            self.policy.temperature = float(payload.get("policy", {}).get("temperature", self.policy.temperature))
            self.policy.risk_bias = float(payload.get("policy", {}).get("risk_bias", self.policy.risk_bias))
            recs = payload.get("records", [])
            self.records = [PredictionRecord(**r) for r in recs[-self.history_limit:]]
        except Exception:
            return

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "memory": self.memory.snapshot(),
            "policy": self.policy.recommend(),
            "records": [asdict(r) for r in self.records[-self.history_limit:]],
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log_prediction(self, prediction: Dict[str, Any]) -> PredictionRecord:
        rec = PredictionRecord(
            timestamp=str(prediction.get("timestamp", datetime.utcnow().isoformat())),
            direction=str(prediction.get("direction", "FLAT")),
            confidence=float(prediction.get("confidence", 0.0)),
            magnitude_pct=float(prediction.get("magnitude_pct", 0.0)),
            regime=str(prediction.get("regime", "Unknown")),
            sentiment_score=float((prediction.get("sentiment") or {}).get("score", 0.0)),
            news_count=int((prediction.get("sentiment") or {}).get("n_articles", 0)),
            macro_vix=(prediction.get("macro") or {}).get("india_vix"),
            adaptive_policy=dict(prediction.get("adaptive_policy") or self.policy.recommend()),
        )
        self.records.append(rec)
        self.memory.pending.append(rec)
        self.records = self.records[-self.history_limit:]
        self.memory.pending = self.memory.pending[-self.history_limit:]
        self.save()
        return rec

    def update_outcome(
        self,
        timestamp: str | datetime,
        actual_return: float,
        realized_pnl: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        match = None
        for rec in reversed(self.records):
            if rec.timestamp == ts:
                match = rec
                break
        if match is None:
            for rec in reversed(self.records):
                if rec.timestamp.startswith(ts[:10]):
                    match = rec
                    break
        if match is None:
            raise ValueError(f"No prediction record found for {ts}")

        reward = self.reward_engine.compute_reward(
            direction=match.direction,
            actual_return=actual_return,
            confidence=match.confidence,
            realized_pnl=realized_pnl,
            regime=regime or match.regime,
            sentiment_score=match.sentiment_score,
        )
        pnl = float(realized_pnl if realized_pnl is not None else actual_return * _direction_index(match.direction))
        self.memory.update(match, reward=reward, actual_return=actual_return, pnl=pnl)
        self.policy.update(self.memory, reward=reward, actual_return=actual_return)

        match.status = "realized"
        match.actual_return = float(actual_return)
        match.realized_pnl = pnl
        match.reward = reward
        self.save()
        return {
            "timestamp": ts,
            "reward": reward,
            "actual_return": actual_return,
            "realized_pnl": pnl,
            "policy": self.policy.recommend(),
            "memory": self.memory.snapshot(),
        }

    def recommendations(self) -> Dict[str, Any]:
        rec = self.policy.recommend()
        rec.update({
            "ema_reward": round(self.memory.ema_reward, 4),
            "ema_accuracy": round(self.memory.ema_accuracy, 4),
            "ema_overconfidence": round(self.memory.ema_overconfidence, 4),
            "ema_pnl": round(self.memory.ema_pnl, 4),
            "ema_error": round(self.memory.ema_error, 4),
        })
        return rec

    def snapshot(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.snapshot(),
            "policy": self.policy.recommend(),
            "records_count": len(self.records),
            "pending_count": len(self.memory.pending),
        }
