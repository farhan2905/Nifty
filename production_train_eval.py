
"""
production_train_eval.py
========================
End-to-end production runner for Nifty Hi-LSTM v2.

This script:
1. downloads / loads the full historical timeframes,
2. engineers production features,
3. runs walk-forward training and evaluation,
4. executes the retest/backtest engine,
5. saves a compact report with the results.

It is intentionally conservative with assumptions so it can run inside the
existing package without needing extra glue code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import torch

from config import Config
from data.data_collector import NiftyDataCollector
from data.feature_engineer import FeatureEngineer
from models.architecture import AdaptiveHiLSTMv2
from training.walk_forward import WalkForwardBacktester
from visualization.reporting import PredictionVisualizer
from learning.daily_feedback import DailyFeedbackLearner


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    )


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Open", "High", "Low", "Close", "Volume", "target_return", "Date"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def run_production_pipeline(device: str = "auto") -> Dict[str, Any]:
    setup_logging()
    cfg = Config()
    dev = resolve_device(device)

    logger.info("Starting production pipeline on %s", dev)
    collector = NiftyDataCollector(cfg.DATA_DIR)
    frames = collector.merge_all_timeframes()

    fe = FeatureEngineer(add_target_return=True, clip_sigma=5.0, include_calendar_features=True)
    engineered = {}
    for tf, frame in frames.items():
        logger.info("Engineering features for %s (%d rows)", tf, len(frame))
        engineered[tf] = fe.compute_technical_features(frame)
        logger.info("%s engineered rows=%d features=%d", tf, len(engineered[tf]), fe.feature_count())

    feature_cols = _feature_columns(engineered["1d"])
    n_features = min(len(feature_cols), getattr(cfg, "N_FEATURES", 92))
    logger.info("Using %d features per timeframe", n_features)

    def model_factory() -> AdaptiveHiLSTMv2:
        return AdaptiveHiLSTMv2(
            features_15m=n_features,
            features_1h=n_features,
            features_1d=n_features,
            features_1w=n_features,
            hidden_15m=cfg.LSTM_HIDDEN_15M,
            hidden_1h=cfg.LSTM_HIDDEN_1H,
            hidden_1d=cfg.LSTM_HIDDEN_1D,
            hidden_1w=cfg.LSTM_HIDDEN_1W,
            layers_15m=cfg.LSTM_LAYERS_15M,
            layers_1h=cfg.LSTM_LAYERS_1H,
            layers_1d=cfg.LSTM_LAYERS_1D,
            layers_1w=cfg.LSTM_LAYERS_1W,
            dropout=cfg.DROPOUT,
        )

    model = model_factory().to(dev)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{params:,}")

    backtester = WalkForwardBacktester(cfg, device=str(dev))
    results = backtester.run(
        model_factory=model_factory,
        df_15m=engineered["15m"],
        df_1h=engineered["1h"],
        df_1d=engineered["1d"],
        df_1w=engineered["1w"],
        n_features=n_features,
        verbose=True,
    )

    feedback = DailyFeedbackLearner(Path(cfg.RL_STATE_FILE), history_limit=getattr(cfg, "FEEDBACK_HISTORY_LIMIT", 500))

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev),
        "features": {
            "count": n_features,
            "engineered_count": fe.feature_count(),
            "names": feature_cols[:n_features],
        },
        "model": {
            "trainable_parameters": params,
        },
        "rl_feedback": feedback.snapshot(),
        "backtest": {
            "oos_acc_mean": results.oos_acc_mean,
            "oos_acc_std": results.oos_acc_std,
            "oos_mag_mae": results.oos_mag_mae,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "folds": [f.__dict__ for f in results.folds],
        },
    }

    out_dir = Path(cfg.LOG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    visualizer = PredictionVisualizer(Path(cfg.REPORT_DIR))
    viz_paths = visualizer.save_backtest_dashboard(report, results)
    report["visuals"] = viz_paths

    logger.info("Saved report to %s", report_path)
    logger.info("Saved visual dashboard to %s", viz_paths.get("html"))
    logger.info("%s", results.summary())
    return report


if __name__ == "__main__":
    run_production_pipeline()
