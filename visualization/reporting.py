from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


class PredictionVisualizer:
    """Lightweight visual dashboard generator for backtests and live predictions."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)

    def _save_fig(self, fig, filename: str) -> Path:
        path = self.output_dir / self._safe_name(filename)
        fig.savefig(path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return path

    def _write_html(self, title: str, summary: Dict[str, Any], images: Iterable[Path], filename: str) -> Path:
        html_path = self.output_dir / self._safe_name(filename)
        parts = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            "<style>body{font-family:Arial,sans-serif;background:#0b1020;color:#e8eefc;padding:24px}"
            ".card{background:#121a33;padding:18px;border-radius:16px;margin:16px 0;box-shadow:0 6px 20px rgba(0,0,0,.18)}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}"
            "img{max-width:100%;border-radius:12px;border:1px solid #2d3553}"
            "pre{white-space:pre-wrap;word-break:break-word;background:#0f1630;padding:14px;border-radius:12px;overflow:auto}"
            "h1,h2,h3{margin-top:0}"
            "</style></head><body>",
            f"<h1>{title}</h1>",
            "<div class='card'><h2>Summary</h2><pre>",
            json.dumps(summary, indent=2, default=str),
            "</pre></div>",
            "<div class='grid'>",
        ]
        for img in images:
            parts.append(f"<div class='card'><img src='{img.name}' alt='{img.name}'></div>")
        parts.extend(["</div>", "</body></html>"])
        html_path.write_text("\n".join(parts), encoding="utf-8")
        return html_path


    def _plot_feedback_panel(self, report: Dict[str, Any]) -> list[Path]:
        images: list[Path] = []
        fb = report.get("rl_feedback") or {}
        memory = fb.get("memory") or {}
        policy = fb.get("policy") or {}
        recent_rewards = memory.get("recent_rewards") or []
        recent_accuracy = memory.get("recent_accuracy") or []
        if recent_rewards:
            fig = plt.figure(figsize=(10, 4))
            plt.plot(recent_rewards)
            plt.title("Daily Reward Memory")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.tight_layout()
            images.append(self._save_fig(fig, "rl_reward_memory.png"))
        if recent_accuracy:
            fig = plt.figure(figsize=(10, 4))
            plt.plot(np.array(recent_accuracy) * 100.0)
            plt.title("Daily Accuracy Memory")
            plt.xlabel("Step")
            plt.ylabel("Accuracy %")
            plt.tight_layout()
            images.append(self._save_fig(fig, "rl_accuracy_memory.png"))
        if policy:
            fig = plt.figure(figsize=(9, 4))
            labels = ["conf_gate", "mag_gate", "pos_size", "temp", "risk_bias"]
            values = [
                float(policy.get("confidence_gate", 0.0)),
                float(policy.get("magnitude_gate", 0.0)),
                float(policy.get("position_size", 0.0)),
                float(policy.get("temperature", 0.0)),
                float(policy.get("risk_bias", 0.0)),
            ]
            plt.bar(labels, values)
            plt.title("Adaptive Policy State")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            images.append(self._save_fig(fig, "rl_policy_state.png"))
        return images

    def save_backtest_dashboard(self, report: Dict[str, Any], results: Any | None = None) -> Dict[str, str]:
        backtest = report.get("backtest", {})
        folds = backtest.get("folds", [])
        images = []

        fig = plt.figure(figsize=(9, 5))
        labels = ["OOS Acc", "Sharpe", "Drawdown"]
        values = [
            float(backtest.get("oos_acc_mean", 0.0)) * 100.0,
            float(backtest.get("sharpe_ratio", 0.0)),
            float(backtest.get("max_drawdown", 0.0)) * 100.0,
        ]
        plt.bar(labels, values)
        plt.title("Backtest KPIs")
        plt.ylabel("Percent / Ratio")
        plt.tight_layout()
        images.append(self._save_fig(fig, "backtest_kpis.png"))

        equity = None
        if results is not None and getattr(results, "combined_equity", None):
            equity = np.asarray(results.combined_equity, dtype=float)
        elif backtest.get("combined_equity"):
            equity = np.asarray(backtest["combined_equity"], dtype=float)
        if equity is not None and len(equity) > 1:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(equity)
            plt.title("Combined Out-of-Sample Equity Curve")
            plt.xlabel("Step")
            plt.ylabel("Equity")
            plt.tight_layout()
            images.append(self._save_fig(fig, "backtest_equity.png"))

        if folds:
            fig = plt.figure(figsize=(10, 5))
            fold_ids = [f.get("fold_idx", i) for i, f in enumerate(folds)]
            accs = [float(f.get("test_acc", 0.0)) * 100.0 for f in folds]
            plt.bar([str(i) for i in fold_ids], accs)
            plt.title("Fold Test Accuracy")
            plt.xlabel("Fold")
            plt.ylabel("Accuracy %")
            plt.tight_layout()
            images.append(self._save_fig(fig, "backtest_fold_accuracy.png"))

            fig = plt.figure(figsize=(10, 5))
            maes = [float(f.get("test_mag_mae", 0.0)) for f in folds]
            plt.bar([str(i) for i in fold_ids], maes)
            plt.title("Fold Magnitude MAE")
            plt.xlabel("Fold")
            plt.ylabel("MAE")
            plt.tight_layout()
            images.append(self._save_fig(fig, "backtest_fold_mae.png"))

        images.extend(self._plot_feedback_panel(report))
        summary = {
            "timestamp": report.get("timestamp"),
            "device": report.get("device"),
            "features": report.get("features", {}),
            "model": report.get("model", {}),
            "rl_feedback": report.get("rl_feedback", {}),
            "backtest": backtest,
        }
        html = self._write_html(
            title="Nifty Hi-LSTM v2 — Backtest Dashboard",
            summary=summary,
            images=images,
            filename=f"backtest_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        )
        return {"html": str(html), "images": [str(p) for p in images]}

    def save_live_snapshot(
        self,
        engine_output: Dict[str, Any],
        history: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        images = []
        model_pred = engine_output.get("model_prediction", {}) or {}
        horizon_preds = engine_output.get("horizon_predictions", {}) or {}

        dir_probs = np.asarray(model_pred.get("direction_probs", []), dtype=float)
        if dir_probs.size == 3:
            fig = plt.figure(figsize=(8, 4))
            plt.bar(["BEAR", "FLAT", "BULL"], dir_probs * 100.0)
            plt.title("Direction Probabilities")
            plt.ylabel("Probability %")
            plt.tight_layout()
            images.append(self._save_fig(fig, "live_direction_probs.png"))

        regime_probs = engine_output.get("regime_probs", {}) or {}
        if regime_probs:
            labels = list(regime_probs.keys())
            values = [float(v) for v in regime_probs.values()]
            fig = plt.figure(figsize=(9, 4))
            plt.bar(labels, values)
            plt.title("Regime Probabilities")
            plt.ylabel("Probability %")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            images.append(self._save_fig(fig, "live_regime_probs.png"))

        if horizon_preds:
            labels, confs, mags = [], [], []
            for horizon in ("15m", "1h", "1d"):
                hp = horizon_preds.get(horizon)
                if not hp:
                    continue
                labels.append(horizon)
                confs.append(float(hp.get("confidence_mean", 0.0)) * 100.0)
                mags.append(float(hp.get("magnitude_pct", 0.0)))
            if labels:
                x = np.arange(len(labels))
                fig = plt.figure(figsize=(9, 4))
                plt.bar(x, confs)
                plt.xticks(x, labels)
                plt.title("Horizon Confidence")
                plt.ylabel("Confidence %")
                plt.tight_layout()
                images.append(self._save_fig(fig, "live_horizon_confidence.png"))

                fig = plt.figure(figsize=(9, 4))
                plt.bar(x, mags)
                plt.xticks(x, labels)
                plt.title("Horizon Magnitude")
                plt.ylabel("Expected Move %")
                plt.tight_layout()
                images.append(self._save_fig(fig, "live_horizon_magnitude.png"))

        if history:
            conf_hist = [float(h.get("confidence", 0.0)) * 100.0 for h in history][-80:]
            if conf_hist:
                fig = plt.figure(figsize=(10, 4))
                plt.plot(conf_hist)
                plt.title("Recent Confidence History")
                plt.ylabel("Confidence %")
                plt.tight_layout()
                images.append(self._save_fig(fig, "live_confidence_history.png"))

        summary = {
            "timestamp": engine_output.get("timestamp"),
            "direction": engine_output.get("direction"),
            "confidence": engine_output.get("confidence"),
            "magnitude_pct": engine_output.get("magnitude_pct"),
            "uncertainty_pct": engine_output.get("uncertainty_pct"),
            "regime": engine_output.get("regime"),
            "targets": engine_output.get("targets", {}),
            "model_prediction": model_pred,
            "horizon_predictions": horizon_preds,
        }
        html = self._write_html(
            title="Nifty Hi-LSTM v2 — Live Signal Snapshot",
            summary=summary,
            images=images,
            filename=f"live_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        )
        return {"html": str(html), "images": [str(p) for p in images]}
