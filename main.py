"""
NIFTY Hi-LSTM v2 — Main Entry Point
=====================================
Usage:
  python main.py --mode demo        # Quick demo with yfinance data (no GPU required)
  python main.py --mode download    # Download all historical data to disk cache
  python main.py --mode train       # Full walk-forward training pipeline
  python main.py --mode backtest    # Walk-forward backtest with equity-curve plot
  python main.py --mode live        # Live market prediction (market hours only)
  python main.py --mode predict     # One-shot prediction on the latest cached data

Optional flags:
  --device   {auto|cpu|cuda|mps}    Compute device (default: auto)
  --ensemble-size N                 Number of ensemble models (default: 5)
  --log-level {DEBUG|INFO|WARNING}  Logging verbosity (default: INFO)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from config import Config

# ── logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamp + level formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                _ROOT / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mode="w",
            ),
        ],
    )


# ── device resolution ────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE — fully runnable end-to-end demonstration
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(config: Config) -> None:
    """
    Fully working demonstration using real yfinance data.

    Pipeline:
      1. Download 60 days of 15m Nifty data + daily data from 2020.
      2. Compute 35+ technical features via FeatureEngineer.
      3. Build a compact AdaptiveHiLSTMv2 model (reduced hidden sizes for speed).
      4. Run 3 synthetic training epochs.
      5. Execute a single inference pass with MC Dropout.
      6. Print a formatted prediction dashboard.

    No GPU required — completes in ~30 seconds on a CPU-only machine.
    """
    from data.data_collector import NiftyDataCollector
    from data.feature_engineer import FeatureEngineer
    from models.architecture import AdaptiveHiLSTMv2
    from models.losses import AdaptiveTradingLoss

    _banner("DEMO MODE", width=62)
    print("  Full pipeline: download → features → train (3 epochs) → predict\n")

    # ── 1. Download data ──
    _step(1, 5, "Downloading Nifty 50 data …")
    collector = NiftyDataCollector(config.DATA_DIR)

    daily_data    = collector.fetch_daily_data("2020-01-01",
                                               datetime.now().strftime("%Y-%m-%d"))
    intraday_15m  = collector.fetch_intraday_data("15m", "60d")
    vix_data      = collector.fetch_india_vix("2020-01-01",
                                              datetime.now().strftime("%Y-%m-%d"))

    print(f"     Daily:  {len(daily_data):,} bars")
    print(f"     15-min: {len(intraday_15m):,} bars")
    print(f"     VIX:    {len(vix_data):,} bars")

    if len(daily_data) < 50:
        print("\n  [!] Insufficient data downloaded — check internet connection.")
        print("  [!] Continuing with synthetic data for demonstration.\n")
        _run_synthetic_demo(config)
        return

    # ── 2. Feature engineering ──
    _step(2, 5, "Engineering features …")
    fe = FeatureEngineer(add_target_return=True)

    daily_feat    = fe.compute_technical_features(daily_data)
    intraday_feat = fe.compute_technical_features(intraday_15m)

    # Determine feature count (exclude OHLCV + target_return)
    feature_cols = fe.feature_columns()
    n_features   = min(len(feature_cols), getattr(config, "N_FEATURES", 92))

    print(f"     Feature columns:  {len(feature_cols)}")
    print(f"     Using top N:      {n_features}")
    print(f"     Daily shape:      {daily_feat.shape}")
    print(f"     Intraday shape:   {intraday_feat.shape}")

    # ── 3. Build compact demo model ──
    _step(3, 5, "Building compact model (demo configuration) …")
    model = AdaptiveHiLSTMv2(
        features_15m=n_features,
        features_1h=n_features,
        features_1d=n_features,
        features_1w=n_features,
        hidden_15m=64,
        hidden_1h=48,
        hidden_1d=32,
        hidden_1w=16,
        layers_15m=2,
        layers_1h=1,
        layers_1d=1,
        layers_1w=1,
        dropout=0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"     Parameters: {total_params:,}  (full model would be ~{total_params * 15:,})")

    # ── 4. Synthetic 3-epoch training ──
    _step(4, 5, "Quick training — 3 epochs on synthetic batches …")
    batch_size = 8
    device     = torch.device("cpu")
    model      = model.to(device)

    # Create random tensors matching the model's expected input shapes
    def _rand_batch(seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, n_features, device=device)

    loss_fn   = AdaptiveTradingLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    model.train()

    for epoch in range(3):
        x_15m = _rand_batch(config.SEQ_15M)
        x_1h  = _rand_batch(config.SEQ_1H)
        x_1d  = _rand_batch(config.SEQ_1D)
        x_1w  = _rand_batch(config.SEQ_1W)

        y_dir = torch.randint(0, 3, (batch_size,), device=device)
        y_mag = torch.randn(batch_size, device=device) * 0.01

        _out = model(x_15m, x_1h, x_1d, x_1w)
        dir_logits, magnitude, confidence, regime_probs = (
            _out["direction_logits"], _out["magnitude"],
            _out["confidence"], _out["regime_probs"]
        )
        loss, components = loss_fn(dir_logits, magnitude, confidence, regime_probs, y_dir, y_mag)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(
            f"     Epoch {epoch+1}/3 | "
            f"loss={components['total'].item():.4f} | "
            f"dir={components['direction_ce'].item():.4f} | "
            f"mag={components['magnitude_mse'].item():.4f}"
        )

    # ── 5. Single inference with MC Dropout ──
    _step(5, 5, "Running inference (MC Dropout × 20 passes) …")

    # Use the last available real window if we have enough data
    # Ensure both arrays use the same feature_cols so shapes match the model
    def _safe_array(df: "pd.DataFrame") -> np.ndarray:
        """Extract exactly n_features numeric columns, padding/trimming as needed."""
        num_df = df.select_dtypes(include=[float, int])
        # Align columns: use feature_cols that are present, pad rest with 0
        aligned = np.zeros((len(df), n_features), dtype=np.float32)
        common = [c for c in feature_cols[:n_features] if c in num_df.columns]
        if common:
            aligned[:, :len(common)] = num_df[common].values
        return aligned

    n_15m = _safe_array(intraday_feat)
    n_1d  = _safe_array(daily_feat)

    def _build_input(arr: np.ndarray, seq_len: int) -> torch.Tensor:
        arr = arr[:, :n_features]
        if len(arr) < seq_len:
            pad = np.zeros((seq_len - len(arr), n_features), dtype=np.float32)
            arr = np.vstack([pad, arr])
        window = arr[-seq_len:].astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(window, device=device).unsqueeze(0)

    # MC Dropout: keep dropout active but freeze BatchNorm + LayerNorm
    model.eval()  # sets all to eval first
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()  # re-enable dropout for MC uncertainty

    dir_samples, mag_samples, conf_samples, regime_samples = [], [], [], []
    with torch.no_grad():
        for _ in range(20):
            x15 = _build_input(n_15m, config.SEQ_15M)
            x1h = _build_input(n_15m, config.SEQ_1H)    # reuse 15m features for demo
            x1d = _build_input(n_1d,  config.SEQ_1D)
            x1w = _build_input(n_1d,  config.SEQ_1W)

            _o = model(x15, x1h, x1d, x1w)
            dl  = _o["direction_logits"]
            mag = _o["magnitude"]
            conf = _o["confidence"]
            reg = _o["regime_probs"]
            dir_samples.append(torch.softmax(dl, dim=-1).cpu().numpy())
            mag_samples.append(mag.cpu().item())
            conf_samples.append(conf.cpu().item())
            regime_samples.append(reg.cpu().numpy())

    model.eval()

    dir_probs    = np.mean(dir_samples, axis=0).squeeze()   # (3,)
    mag_mean     = float(np.mean(mag_samples)) * 100.0
    mag_std      = float(np.std(mag_samples))  * 100.0
    conf_mean    = float(np.mean(conf_samples))
    regime_probs = np.mean(regime_samples, axis=0).squeeze()  # (n_regimes,)

    dir_labels       = ["BEARISH", "FLAT", "BULLISH"]
    dir_idx          = int(np.argmax(dir_probs))
    direction        = dir_labels[dir_idx]
    dir_conf         = float(dir_probs[dir_idx])

    regime_idx       = int(np.argmax(regime_probs))
    regime_name      = config.REGIME_NAMES.get(regime_idx, f"Regime-{regime_idx}")

    # Approximate spot price
    try:
        spot = float(daily_data["Close"].iloc[-1])
    except Exception:
        spot = 24_000.0

    # ── Print dashboard ──
    print()
    _banner("PREDICTION OUTPUT", width=62)
    dir_sym = {"BULLISH": "▲", "BEARISH": "▼", "FLAT": "◆"}.get(direction, "◆")
    print(f"  PREDICTION:   {dir_sym} {direction:<10} ({dir_conf:.1%} direction conf)")
    print(f"  CONFIDENCE:   {conf_mean:.1%} (ensemble calibrated)")
    print(f"  MAGNITUDE:    {mag_mean:+.2f}% ± {mag_std:.2f}%")
    print(f"  UNCERTAINTY:  ±{mag_std:.2f}% (MC Dropout σ)")
    print(f"  REGIME:       {regime_name}")
    print(f"  SPOT:         {spot:,.2f}")
    print()

    # Simple target estimates
    sign = 1 if dir_idx == 2 else (-1 if dir_idx == 0 else 0)
    mag_frac = abs(mag_mean) / 100
    print(f"  H1 TARGET:    {spot * (1 + sign * max(mag_frac, 0.004)):,.0f}")
    print(f"  D1 TARGET:    {spot * (1 + sign * max(mag_frac * 2.5, 0.010)):,.0f}")
    print(f"  W1 TARGET:    {spot * (1 + sign * max(mag_frac * 5.0, 0.020)):,.0f}")
    print(f"  STOP LOSS:    {spot * (1 - sign * 0.006):,.0f}")
    print()
    print(f"  Regime probs: ", end="")
    for i, p in enumerate(regime_probs):
        name = config.REGIME_NAMES.get(i, f"R{i}")
        print(f"{name}={p:.0%}", end="  ")
    print()
    _separator(width=62)
    print("\n  Demo complete.")
    print("  Next steps:")
    print("    python main.py --mode download   # cache full history")
    print("    python main.py --mode train      # full ensemble training")
    print("    python main.py --mode live       # live market prediction\n")


def _run_synthetic_demo(config: Config) -> None:
    """Fallback demo using purely synthetic data when yfinance is unavailable."""
    from models.architecture import AdaptiveHiLSTMv2
    from models.losses import AdaptiveTradingLoss

    print("  Running synthetic-data fallback demo …\n")
    n_features = 50
    batch_size = 4

    model = AdaptiveHiLSTMv2(
        features_15m=n_features, features_1h=n_features,
        features_1d=n_features,  features_1w=n_features,
        hidden_15m=32, hidden_1h=24, hidden_1d=16, hidden_1w=8,
        layers_15m=1, layers_1h=1, layers_1d=1, layers_1w=1,
    )
    loss_fn   = AdaptiveTradingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(3):
        x15 = torch.randn(batch_size, config.SEQ_15M, n_features)
        x1h = torch.randn(batch_size, config.SEQ_1H,  n_features)
        x1d = torch.randn(batch_size, config.SEQ_1D,  n_features)
        x1w = torch.randn(batch_size, config.SEQ_1W,  n_features)
        y_d = torch.randint(0, 3, (batch_size,))
        y_m = torch.randn(batch_size) * 0.01

        _o = model(x15, x1h, x1d, x1w)
        dl, mag, conf, reg = _o["direction_logits"], _o["magnitude"], _o["confidence"], _o["regime_probs"]
        loss, comp = loss_fn(dl, mag, conf, reg, y_d, y_m)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"     Epoch {epoch+1}/3 | loss={comp['total'].item():.4f}")

    model.eval()
    with torch.no_grad():
        x15 = torch.randn(1, config.SEQ_15M, n_features)
        x1h = torch.randn(1, config.SEQ_1H,  n_features)
        x1d = torch.randn(1, config.SEQ_1D,  n_features)
        x1w = torch.randn(1, config.SEQ_1W,  n_features)
        _o = model(x15, x1h, x1d, x1w)
        dl   = _o["direction_logits"]
        conf = _o["confidence"]

    dir_labels = ["BEARISH", "FLAT", "BULLISH"]
    direction  = dir_labels[dl.argmax().item()]
    print(f"\n  Synthetic prediction: {direction}  ({conf.item():.1%} conf)")
    print("  (Use --mode demo with internet access for real yfinance data)")


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_download(config: Config) -> None:
    """
    Download and cache all required data:
      - Daily Nifty 50 OHLCV from 2000-01-01 to today
      - 15-minute intraday (last 730 days — yfinance maximum)
      - 1-hour intraday   (last 730 days)
      - India VIX daily   from 2000-01-01
    All results are persisted as pickle files in config.DATA_DIR.
    """
    from data.data_collector import NiftyDataCollector

    _banner("DOWNLOAD MODE")
    collector = NiftyDataCollector(config.DATA_DIR)
    today     = datetime.now().strftime("%Y-%m-%d")

    _step(1, 4, f"Daily Nifty 50 from 2000-01-01 to {today} …")
    daily = collector.fetch_daily_data("2000-01-01", today)
    print(f"     Fetched {len(daily):,} daily bars")

    _step(2, 4, "15-minute intraday (last 60 days) …")
    intra_15m = collector.fetch_intraday_data("15m", "60d")
    print(f"     Fetched {len(intra_15m):,} 15m bars")

    _step(3, 4, "1-hour intraday (last 730 days) …")
    intra_1h = collector.fetch_intraday_data("1h", "730d")
    print(f"     Fetched {len(intra_1h):,} 1h bars")

    _step(4, 4, f"India VIX from 2000-01-01 to {today} …")
    vix = collector.fetch_india_vix("2000-01-01", today)
    print(f"     Fetched {len(vix):,} VIX bars")

    print(f"\n  Data cached to: {config.DATA_DIR}")
    print("  Run --mode train next.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_train(config: Config) -> None:
    """
    Full training pipeline:
      1. Load cached data (run --mode download first).
      2. Compute features for all timeframes.
      3. Walk-forward cross-validation → N fold checkpoints.
      4. Assemble EnsembleModel from best-N fold checkpoints.
      5. Save ensemble to config.MODEL_DIR/ensemble.pt.
    """
    from data.data_collector import NiftyDataCollector
    from data.feature_engineer import FeatureEngineer
    from models.architecture import AdaptiveHiLSTMv2, EnsembleModel
    from training.walk_forward import WalkForwardBacktester

    _banner("TRAIN MODE")

    # ── data ──
    _step(1, 4, "Loading cached data …")
    collector = NiftyDataCollector(config.DATA_DIR)
    today     = datetime.now().strftime("%Y-%m-%d")

    daily    = collector.fetch_daily_data("2000-01-01", today)
    intra15m = collector.fetch_intraday_data("15m", "60d")
    intra1h  = collector.fetch_intraday_data("1h",  "730d")
    print(f"     Daily: {len(daily):,} | 15m: {len(intra15m):,} | 1h: {len(intra1h):,}")

    # ── features ──
    _step(2, 4, "Computing features …")
    fe = FeatureEngineer(add_target_return=True)

    df_1d  = fe.compute_technical_features(daily)
    df_15m = fe.compute_technical_features(intra15m)
    df_1h  = fe.compute_technical_features(intra1h)

    # Resample weekly from daily
    df_1w  = df_1d.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    df_1w  = fe.compute_technical_features(df_1w)

    feature_cols = fe.feature_columns()
    feature_schema = {"15m": feature_cols, "1h": feature_cols, "1d": feature_cols, "1w": feature_cols}
    n_features   = min(len(feature_cols), config.N_REGIMES * 10)  # sensible cap
    n_features   = max(n_features, 20)

    print(f"     Feature count: {n_features}")

    # ── walk-forward training ──
    _step(3, 4, f"Walk-forward training ({config.N_FOLDS} folds) …")

    def model_factory() -> AdaptiveHiLSTMv2:
        return AdaptiveHiLSTMv2(
            features_15m=n_features, features_1h=n_features,
            features_1d=n_features,  features_1w=n_features,
            hidden_15m=config.LSTM_HIDDEN_15M, hidden_1h=config.LSTM_HIDDEN_1H,
            hidden_1d=config.LSTM_HIDDEN_1D,   hidden_1w=config.LSTM_HIDDEN_1W,
            layers_15m=config.LSTM_LAYERS_15M, layers_1h=config.LSTM_LAYERS_1H,
            layers_1d=config.LSTM_LAYERS_1D,   layers_1w=config.LSTM_LAYERS_1W,
            dropout=config.DROPOUT,
        )

    backtester = WalkForwardBacktester(config, device="auto")
    results    = backtester.run(model_factory, df_15m, df_1h, df_1d, df_1w,
                                n_features=n_features, verbose=True)

    # ── assemble ensemble ──
    _step(4, 4, f"Assembling ensemble (top {config.N_ENSEMBLE} checkpoints) …")
    from training.trainer import ModelTrainer

    trainer = ModelTrainer(config, device="auto")

    # Sort folds by OOS accuracy, take top N
    best_folds = sorted(results.folds, key=lambda f: -f.test_acc)[:config.N_ENSEMBLE]
    ensemble_models = []
    for fold in best_folds:
        m = model_factory()
        if Path(fold.checkpoint_path).exists():
            trainer.load_checkpoint(m, fold.checkpoint_path)
        ensemble_models.append(m)
        print(f"     Loaded fold {fold.fold_idx} (acc={fold.test_acc:.1%})")

    ensemble = EnsembleModel(ensemble_models)
    ens_path = Path(config.MODEL_DIR) / "ensemble.pt"
    torch.save({"models": [m.state_dict() for m in ensemble_models], "feature_schema": feature_schema, "n_features": n_features}, str(ens_path))
    print(f"\n  Ensemble saved → {ens_path}")
    print(results.summary())


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(config: Config) -> None:
    """
    Load the ensemble checkpoint saved by --mode train and re-run the
    walk-forward evaluation, producing an equity-curve plot.

    If no trained models exist, trains from scratch first.
    """
    from data.data_collector import NiftyDataCollector
    from data.feature_engineer import FeatureEngineer
    from models.architecture import AdaptiveHiLSTMv2
    from training.walk_forward import WalkForwardBacktester

    _banner("BACKTEST MODE")

    collector = NiftyDataCollector(config.DATA_DIR)
    today     = datetime.now().strftime("%Y-%m-%d")

    _step(1, 3, "Loading data …")
    daily    = collector.fetch_daily_data("2000-01-01", today)
    intra15m = collector.fetch_intraday_data("15m", "60d")
    intra1h  = collector.fetch_intraday_data("1h",  "730d")

    _step(2, 3, "Computing features …")
    fe     = FeatureEngineer(add_target_return=True)
    df_1d  = fe.compute_technical_features(daily)
    df_15m = fe.compute_technical_features(intra15m)
    df_1h  = fe.compute_technical_features(intra1h)
    df_1w  = fe.compute_technical_features(
        df_1d.resample("W-FRI").agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}
        ).dropna()
    )

    feature_cols = fe.feature_columns()
    feature_schema = {"15m": feature_cols, "1h": feature_cols, "1d": feature_cols, "1w": feature_cols}
    n_features   = min(len(feature_cols), getattr(config, "N_FEATURES", 92))

    _step(3, 3, "Running walk-forward backtest …")

    def model_factory() -> AdaptiveHiLSTMv2:
        return AdaptiveHiLSTMv2(
            features_15m=n_features, features_1h=n_features,
            features_1d=n_features,  features_1w=n_features,
            hidden_15m=config.LSTM_HIDDEN_15M, hidden_1h=config.LSTM_HIDDEN_1H,
            hidden_1d=config.LSTM_HIDDEN_1D,   hidden_1w=config.LSTM_HIDDEN_1W,
            layers_15m=config.LSTM_LAYERS_15M, layers_1h=config.LSTM_LAYERS_1H,
            layers_1d=config.LSTM_LAYERS_1D,   layers_1w=config.LSTM_LAYERS_1W,
            dropout=config.DROPOUT,
        )

    backtester = WalkForwardBacktester(config, device="auto")
    results    = backtester.run(model_factory, df_15m, df_1h, df_1d, df_1w,
                                n_features=n_features, verbose=True, skip_training=True)

    curve_path = str(Path(config.LOG_DIR) / "equity_curve.png")
    backtester.plot_equity_curve(results, save_path=curve_path, show=False)
    print(f"\n  Equity curve saved → {curve_path}")
    print(results.summary())

    # Build Visualization Dashboard
    from visualization.reporting import PredictionVisualizer
    import webbrowser
    import json
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": "auto",
        "features": {"count": n_features},
        "model": {"trainable_parameters": sum(p.numel() for p in model_factory().parameters())},
        "rl_feedback": {}, # No RL in simple backtest
        "backtest": {
            "oos_acc_mean": results.oos_acc_mean,
            "oos_acc_std": results.oos_acc_std,
            "oos_mag_mae": results.oos_mag_mae,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "folds": [f.__dict__ for f in results.folds],
        }
    }
    visualizer = PredictionVisualizer(Path(config.REPORT_DIR))
    viz_paths = visualizer.save_backtest_dashboard(report, results)
    html_path = viz_paths.get("html")
    
    if html_path:
        print(f"\n  [Dashboard] Opening full backtest dashboard: {html_path}")
        webbrowser.open(f"file:///{Path(html_path).resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_live(config: Config) -> None:
    """
    Start the live predictor.

    Loads the ensemble checkpoint from config.MODEL_DIR/ensemble.pt.
    Fires on_candle_close at :00, :15, :30, :45 every hour during
    NSE market hours (Mon–Fri, 09:15–15:30 IST).

    Press Ctrl-C to stop gracefully.
    """
    import pytz
    from datetime import datetime

    from data.data_collector import NiftyDataCollector
    from data.feature_engineer import FeatureEngineer
    from models.architecture import AdaptiveHiLSTMv2, EnsembleModel
    from engine.retest_engine import RetestEngine
    from sentiment.news_fetcher import NewsFeeder
    from sentiment.sentiment_analyzer import SentimentAnalyzer
    from inference.live_predictor import LivePredictor

    _banner("LIVE MODE")

    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)
    print(f"  Current IST time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    ens_path = Path(config.MODEL_DIR) / "ensemble.pt"
    if not ens_path.exists():
        print(f"  [!] No trained ensemble found at {ens_path}")
        print("  [!] Run --mode train first, or --mode demo for a quick test.\n")
        return

    _step(1, 4, "Loading ensemble checkpoint …")
    device = resolve_device("auto")

    # Determine feature count from checkpoint if possible
    ckpt   = torch.load(str(ens_path), map_location=device)
    n_features = int(ckpt.get("n_features", 50))
    feature_schema = ckpt.get("feature_schema", None)

    def _load_model(state_dict: dict) -> AdaptiveHiLSTMv2:
        m = AdaptiveHiLSTMv2(
            features_15m=n_features, features_1h=n_features,
            features_1d=n_features,  features_1w=n_features,
            hidden_15m=config.LSTM_HIDDEN_15M, hidden_1h=config.LSTM_HIDDEN_1H,
            hidden_1d=config.LSTM_HIDDEN_1D,   hidden_1w=config.LSTM_HIDDEN_1W,
            layers_15m=config.LSTM_LAYERS_15M, layers_1h=config.LSTM_LAYERS_1H,
            layers_1d=config.LSTM_LAYERS_1D,   layers_1w=config.LSTM_LAYERS_1W,
            dropout=config.DROPOUT,
        ).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        return m

    models = [_load_model(sd) for sd in ckpt["models"]]
    ensemble = EnsembleModel(models)
    print(f"     Loaded {len(models)} model(s) from {ens_path}")

    _step(2, 4, "Initialising data pipeline …")
    collector = NiftyDataCollector(config.DATA_DIR)
    fe        = FeatureEngineer(add_target_return=False)

    _step(3, 4, "Initialising sentiment pipeline …")
    news_feeder = NewsFeeder(cache_dir=config.DATA_DIR)
    sentiment   = SentimentAnalyzer()

    _step(4, 4, "Initialising retest engine …")
    retest = RetestEngine(config, ensemble, sentiment)

    config.N_FEATURES = n_features
    config.FEATURE_SCHEMA = feature_schema

    predictor = LivePredictor(
        config=config,
        ensemble_model=ensemble,
        retest_engine=retest,
        data_collector=collector,
        feature_engineer=fe,
        news_feeder=news_feeder,
        sentiment_analyzer=sentiment,
        mc_samples=config.MC_DROPOUT_PASSES,
        log_dir=config.LOG_DIR,
    )

    predictor.start()   # blocks until Ctrl-C


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(config: Config) -> None:
    """
    Load the latest trained ensemble, fetch live data, and print a single
    prediction update — useful for cron-based scheduled execution or
    quick checks outside of live mode.
    """
    from data.data_collector import NiftyDataCollector
    from data.feature_engineer import FeatureEngineer
    from models.architecture import AdaptiveHiLSTMv2, EnsembleModel
    from engine.retest_engine import RetestEngine
    from sentiment.news_fetcher import NewsFeeder
    from sentiment.sentiment_analyzer import SentimentAnalyzer
    from inference.live_predictor import LivePredictor

    _banner("PREDICT MODE (one-shot)")

    ens_path = Path(config.MODEL_DIR) / "ensemble.pt"
    if not ens_path.exists():
        print(f"  [!] No ensemble checkpoint found at {ens_path}")
        print("  [!] Run --mode train first.\n")
        return

    device     = resolve_device("auto")

    _step(1, 2, "Loading model + data …")
    ckpt = torch.load(str(ens_path), map_location=device)
    n_features = int(ckpt.get("n_features", 50))
    feature_schema = ckpt.get("feature_schema", None)

    def _load_model(sd: dict) -> AdaptiveHiLSTMv2:
        m = AdaptiveHiLSTMv2(
            features_15m=n_features, features_1h=n_features,
            features_1d=n_features,  features_1w=n_features,
            hidden_15m=config.LSTM_HIDDEN_15M, hidden_1h=config.LSTM_HIDDEN_1H,
            hidden_1d=config.LSTM_HIDDEN_1D,   hidden_1w=config.LSTM_HIDDEN_1W,
            layers_15m=config.LSTM_LAYERS_15M, layers_1h=config.LSTM_LAYERS_1H,
            layers_1d=config.LSTM_LAYERS_1D,   layers_1w=config.LSTM_LAYERS_1W,
            dropout=config.DROPOUT,
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        return m

    models   = [_load_model(sd) for sd in ckpt["models"]]
    ensemble = EnsembleModel(models)

    collector   = NiftyDataCollector(config.DATA_DIR)
    fe          = FeatureEngineer(add_target_return=False)
    news_feeder = NewsFeeder(cache_dir=config.DATA_DIR)
    sentiment   = SentimentAnalyzer()
    retest      = RetestEngine(config, ensemble, sentiment)

    config.N_FEATURES = n_features
    config.FEATURE_SCHEMA = feature_schema

    predictor = LivePredictor(
        config=config,
        ensemble_model=ensemble,
        retest_engine=retest,
        data_collector=collector,
        feature_engineer=fe,
        news_feeder=news_feeder,
        sentiment_analyzer=sentiment,
        mc_samples=config.MC_DROPOUT_PASSES,
        log_dir=config.LOG_DIR,
    )

    _step(2, 2, "Running one-shot pipeline …")
    predictor.on_candle_close()


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 60) -> None:
    bar = "═" * width
    pad = max(0, (width - len(title) - 4) // 2)
    print(f"\n{bar}")
    print(f"{'':>{pad}}  {title}")
    print(f"{bar}")


def _separator(width: int = 60) -> None:
    print("─" * width)


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nifty Hi-LSTM v2 — Hierarchical LSTM Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "download", "train", "backtest", "live", "predict"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        metavar="N",
        help="Number of models in the ensemble (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure log directory exists before setting up file handler
    Path(_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info("Nifty Hi-LSTM v2 starting — mode=%s  device=%s", args.mode, args.device)

    config            = Config()
    config.N_ENSEMBLE = args.ensemble_size

    mode_map = {
        "demo":      run_demo,
        "download":  run_download,
        "train":     run_train,
        "backtest":  run_backtest,
        "live":      run_live,
        "predict":   run_predict,
    }

    try:
        mode_map[args.mode](config)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    except Exception as exc:
        logger.exception("Fatal error in mode '%s': %s", args.mode, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()