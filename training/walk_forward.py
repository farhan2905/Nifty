"""
training/walk_forward.py
========================
WalkForwardBacktester — Implements expanding-window walk-forward validation
for the Nifty Hi-LSTM v2 ensemble.

Walk-Forward Logic
------------------
The daily dataset is partitioned into N chronological folds:

    ┌──────────────── Total history ─────────────────────────────┐
    │  TRAIN (5y)  │ TEST (6m) │                                │  fold 0
    │  TRAIN (5y + 6m)  │ TEST │                               │  fold 1
    │  TRAIN (5y + 12m) │ TEST │                              │  fold 2
    │  ...                                                     │
    └────────────────────────────────────────────────────────────┘

Each fold:
    1. Train a fresh model on the training window.
    2. Evaluate on the test window — collect directional accuracy,
       magnitude MAE, and a simple equity-curve simulation.
    3. Save the per-fold model checkpoint.

After all folds:
    4. Aggregate metrics across folds.
    5. Build and display a combined out-of-sample equity curve.

The ensemble for live inference is assembled from the N best-checkpoint
models saved during walk-forward.

Usage
-----
    backtester = WalkForwardBacktester(config, device="cuda")
    results = backtester.run(model_factory, df_15m, df_1h, df_1d, df_1w,
                             n_features=50)
    backtester.plot_equity_curve(results)
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx:       int
    train_start:    str
    train_end:      str
    test_start:     str
    test_end:       str
    val_acc:        float      # directional accuracy on validation set during training
    test_acc:       float      # directional accuracy on hold-out test window
    test_mag_mae:   float      # magnitude MAE on test window
    equity_curve:   List[float] = field(default_factory=list)
    checkpoint_path: str = ""


@dataclass
class BacktestResults:
    folds:          List[FoldResult]
    oos_acc_mean:   float
    oos_acc_std:    float
    oos_mag_mae:    float
    combined_equity: List[float]
    sharpe_ratio:   float
    max_drawdown:   float

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Walk-Forward Backtest Summary",
            "=" * 55,
            f"  Folds:             {len(self.folds)}",
            f"  OOS Dir. Accuracy: {self.oos_acc_mean:.1%} ± {self.oos_acc_std:.1%}",
            f"  OOS Magnitude MAE: {self.oos_mag_mae:.4f}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}",
            f"  Max Drawdown:      {self.max_drawdown:.1%}",
            "=" * 55,
        ]
        for f in self.folds:
            lines.append(
                f"  Fold {f.fold_idx}: test {f.test_start}–{f.test_end} | "
                f"acc={f.test_acc:.1%} | mag_mae={f.test_mag_mae:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Equity simulation
# ---------------------------------------------------------------------------

def _simulate_equity(
    directions_pred: np.ndarray,     # (T,) predicted class: 0=Bear, 1=Flat, 2=Bull
    returns_actual:  np.ndarray,     # (T,) actual log-returns
    confidence:      np.ndarray,     # (T,) predicted confidence ∈ (0, 1)
    direction_threshold: float = 0.003,
    confidence_gate:     float = 0.55,
    position_size:       float = 1.0,
) -> np.ndarray:
    """
    Simple long/flat/short equity simulation.

    Rules:
    * Bull prediction + conf > gate → LONG  (+1 position)
    * Bear prediction + conf > gate → SHORT (-1 position)
    * Flat or low-confidence        → FLAT  (0 position)
    * Position size scales linearly with confidence above the gate.

    Returns an equity curve array (T,) starting at 1.0.
    """
    equity = np.ones(len(directions_pred) + 1)
    for i, (d, r, c) in enumerate(zip(directions_pred, returns_actual, confidence)):
        if c < confidence_gate:
            pos = 0.0
        elif d == 2:   # Bull
            pos =  position_size * min(1.0, (c - confidence_gate) / (1 - confidence_gate) + 0.5)
        elif d == 0:   # Bear
            pos = -position_size * min(1.0, (c - confidence_gate) / (1 - confidence_gate) + 0.5)
        else:
            pos = 0.0
        equity[i + 1] = equity[i] * np.exp(pos * r)
    return equity[1:]


def _sharpe(equity: np.ndarray, rf: float = 0.065, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a daily-bar equity curve."""
    log_ret = np.log(equity[1:] / equity[:-1])
    excess  = log_ret - rf / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve."""
    cummax   = np.maximum.accumulate(equity)
    drawdown = (cummax - equity) / cummax
    return float(drawdown.max())


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Expanding-window walk-forward validation.

    Parameters
    ----------
    config : Config
        Uses TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS, N_FOLDS,
        SEQ_*, DIRECTION_THRESHOLD, CONFIDENCE_GATE, MODEL_DIR.
    device : str
        Compute device (auto / cpu / cuda / mps).
    """

    def __init__(self, config, device: str = "auto"):
        self.config  = config
        self.device  = device
        self.model_dir = Path(getattr(config, "MODEL_DIR", "models/saved"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ── fold generation ──────────────────────────────────────────────────────

    def _generate_folds(
        self, df_1d: pd.DataFrame
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Return a list of (train_idx, test_idx) pairs.

        The first fold trains on TRAIN_WINDOW_YEARS years of data.
        Each subsequent fold extends the training window by TEST_WINDOW_MONTHS months.
        """
        cfg = self.config
        if not isinstance(df_1d.index, pd.DatetimeIndex):
            df_1d = df_1d.copy()
            df_1d.index = pd.to_datetime(df_1d.index)

        dates  = df_1d.index.sort_values()
        n      = len(dates)
        folds  = []

        train_bars = int(cfg.TRAIN_WINDOW_YEARS * 252)
        test_bars  = int(cfg.TEST_WINDOW_MONTHS * 21)   # ~21 trading days/month

        if train_bars + test_bars > n:
            raise ValueError(
                f"Not enough data for walk-forward: need {train_bars + test_bars} bars, "
                f"have {n}."
            )

        start = 0
        for fold in range(cfg.N_FOLDS):
            train_end   = start + train_bars
            test_end    = train_end + test_bars
            if test_end > n:
                break
            train_idx = dates[start:train_end]
            test_idx  = dates[train_end:test_end]
            folds.append((train_idx, test_idx))
            start += test_bars     # expand window by one test period

        logger.info("Generated %d walk-forward folds.", len(folds))
        return folds

    # ── evaluation helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_fold(
        self,
        model: torch.nn.Module,
        df_15m: pd.DataFrame,
        df_1h:  pd.DataFrame,
        df_1d:  pd.DataFrame,
        df_1w:  pd.DataFrame,
        test_idx: pd.DatetimeIndex,
        n_features: int,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the model on the test window.

        Returns (test_acc, test_mag_mae, pred_directions, actual_returns, confidences)
        """
        from training.trainer import MultiTimeframeDataset

        cfg = self.config

        # Filter DataFrames to test window
        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.index, pd.DatetimeIndex):
                return df[df.index <= test_idx[-1]].tail(
                    len(df) // 2 + len(test_idx) * 2
                )
            return df

        ds = MultiTimeframeDataset(
            _filter(df_15m), _filter(df_1h), _filter(df_1d), _filter(df_1w),
            seq_15m=cfg.SEQ_15M, seq_1h=cfg.SEQ_1H,
            seq_1d=cfg.SEQ_1D,   seq_1w=cfg.SEQ_1W,
            n_features=n_features,
            direction_threshold=cfg.DIRECTION_THRESHOLD,
        )

        full_ds = ds.to_tensor_dataset()

        # Use last len(test_idx) samples as the test set
        n_test = min(len(test_idx), len(full_ds))
        if n_test == 0:
            return 0.0, 0.0, np.array([]), np.array([]), np.array([])

        test_subset = torch.utils.data.Subset(full_ds, list(range(len(full_ds) - n_test, len(full_ds))))
        loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

        model.eval()
        dev = next(model.parameters()).device

        all_logits:  List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_mags:    List[torch.Tensor] = []
        all_y_mags:  List[torch.Tensor] = []
        all_confs:   List[torch.Tensor] = []

        for batch in loader:
            x15, x1h, x1d, x1w, y_dir, y_mag = [b.to(dev) for b in batch]
            _o = model(x15, x1h, x1d, x1w)
            dir_logits = _o["direction_logits"]
            magnitude  = _o["magnitude"]
            confidence = _o["confidence"]
            all_logits.append(dir_logits.cpu())
            all_targets.append(y_dir.cpu())
            all_mags.append(magnitude.cpu().squeeze())
            all_y_mags.append(y_mag.cpu())
            all_confs.append(confidence.cpu().squeeze())

        logits  = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        mags    = torch.cat(all_mags).numpy()
        y_mags  = torch.cat(all_y_mags).numpy()
        confs   = torch.cat(all_confs).numpy()

        acc     = (logits.argmax(dim=-1) == targets).float().mean().item()
        mag_mae = float(np.abs(mags - y_mags).mean())

        return acc, mag_mae, logits.argmax(dim=-1).numpy(), y_mags, confs

    # ── main API ──────────────────────────────────────────────────────────────

    def run(
        self,
        model_factory: Callable[[], torch.nn.Module],
        df_15m: pd.DataFrame,
        df_1h:  pd.DataFrame,
        df_1d:  pd.DataFrame,
        df_1w:  pd.DataFrame,
        n_features: int = 50,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        model_factory : callable
            Zero-argument callable that returns a freshly initialised model.
        df_15m, df_1h, df_1d, df_1w : pd.DataFrame
            Full-history feature DataFrames (output of FeatureEngineer).
        n_features : int
            Number of features to use per timeframe.
        verbose : bool
            Print fold-by-fold progress.

        Returns
        -------
        BacktestResults
        """
        from training.trainer import ModelTrainer

        cfg         = self.config
        trainer     = ModelTrainer(cfg, device=self.device)
        folds_idx   = self._generate_folds(df_1d)

        fold_results:   List[FoldResult] = []
        combined_equity = [1.0]

        for fold_idx, (train_idx, test_idx) in enumerate(folds_idx):
            if verbose:
                print(
                    f"\n─── Fold {fold_idx + 1}/{len(folds_idx)} "
                    f"| train: {train_idx[0].date()} → {train_idx[-1].date()} "
                    f"| test: {test_idx[0].date()} → {test_idx[-1].date()} ───"
                )

            # Filter training data
            def _slice(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
                if isinstance(df.index, pd.DatetimeIndex):
                    return df[df.index <= idx[-1]]
                frac = len(idx) / max(len(df_1d), 1)
                return df.iloc[:max(1, int(len(df) * frac))]

            train_df_1d  = _slice(df_1d,  train_idx)
            train_df_15m = _slice(df_15m, train_idx)
            train_df_1h  = _slice(df_1h,  train_idx)
            train_df_1w  = _slice(df_1w,  train_idx)

            # Fresh model
            model = model_factory()

            # Train
            checkpoint_path = str(self.model_dir / f"fold_{fold_idx:02d}.pt")
            feature_cols = [c for c in df_1d.columns if c not in {"Open", "High", "Low", "Close", "Volume", "target_return", "Date"}]
            history = trainer.fit(
                model,
                train_df_15m, train_df_1h, train_df_1d, train_df_1w,
                n_features=n_features,
                checkpoint_path=checkpoint_path,
                feature_schema={"15m": feature_cols, "1h": feature_cols, "1d": feature_cols, "1w": feature_cols},
            )
            best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0

            # Evaluate on test window
            test_acc, test_mae, pred_dirs, actual_rets, confs = self._evaluate_fold(
                model, df_15m, df_1h, df_1d, df_1w, test_idx, n_features
            )

            # Equity simulation
            if len(pred_dirs) > 0:
                eq = _simulate_equity(
                    pred_dirs, actual_rets, confs,
                    direction_threshold=cfg.DIRECTION_THRESHOLD,
                    confidence_gate=cfg.CONFIDENCE_GATE,
                )
                fold_eq = (eq * combined_equity[-1]).tolist()
                combined_equity.extend(fold_eq)
            else:
                fold_eq = []

            fr = FoldResult(
                fold_idx=fold_idx,
                train_start=str(train_idx[0].date()),
                train_end=str(train_idx[-1].date()),
                test_start=str(test_idx[0].date()),
                test_end=str(test_idx[-1].date()),
                val_acc=best_val_acc,
                test_acc=test_acc,
                test_mag_mae=test_mae,
                equity_curve=fold_eq,
                checkpoint_path=checkpoint_path,
            )
            fold_results.append(fr)

            if verbose:
                print(
                    f"  Fold {fold_idx + 1} complete | "
                    f"val_acc={best_val_acc:.1%} | "
                    f"test_acc={test_acc:.1%} | "
                    f"mag_mae={test_mae:.4f}"
                )

        # ── aggregate ──
        test_accs = [f.test_acc    for f in fold_results]
        test_maes = [f.test_mag_mae for f in fold_results]

        equity_arr = np.array(combined_equity[1:]) if len(combined_equity) > 1 else np.array([1.0])
        sharpe     = _sharpe(equity_arr)
        mdd        = _max_drawdown(equity_arr)

        results = BacktestResults(
            folds=fold_results,
            oos_acc_mean=float(np.mean(test_accs))  if test_accs else 0.0,
            oos_acc_std=float(np.std(test_accs))    if test_accs else 0.0,
            oos_mag_mae=float(np.mean(test_maes))   if test_maes else 0.0,
            combined_equity=combined_equity,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
        )

        if verbose:
            print("\n" + results.summary())

        return results

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot_equity_curve(
        self,
        results: BacktestResults,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot the combined OOS equity curve and per-fold boundaries.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not installed — skipping equity curve plot.")
            return

        equity = np.array(results.combined_equity)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Top: equity curve
        ax = axes[0]
        ax.plot(equity, color="#2196F3", linewidth=1.5, label="Strategy OOS")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Nifty Hi-LSTM v2 — Walk-Forward OOS Equity Curve", fontsize=13)
        ax.set_ylabel("Portfolio Value (normalised)")
        ax.set_xlabel("Trading Days (OOS)")

        # Shade fold boundaries
        colours = plt.cm.tab10.colors
        pos = 0
        for i, fold in enumerate(results.folds):
            width = len(fold.equity_curve)
            ax.axvspan(pos, pos + width, alpha=0.07, color=colours[i % 10])
            ax.text(
                pos + width / 2, equity[pos : pos + width].min() * 0.995,
                f"F{i+1}\n{fold.test_acc:.0%}",
                ha="center", va="top", fontsize=7, color=colours[i % 10],
            )
            pos += width

        ax.legend()

        # Bottom: drawdown
        ax2 = axes[1]
        cummax   = np.maximum.accumulate(equity)
        drawdown = (cummax - equity) / cummax
        ax2.fill_between(range(len(drawdown)), drawdown * 100, alpha=0.6, color="salmon")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Trading Days (OOS)")
        ax2.set_title(
            f"Sharpe={results.sharpe_ratio:.2f}  |  Max DD={results.max_drawdown:.1%}  |  "
            f"OOS Acc={results.oos_acc_mean:.1%}",
            fontsize=10,
        )

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Equity curve saved → %s", save_path)
        if show:
            plt.show()
        plt.close()
