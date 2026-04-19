"""
training/trainer.py
===================
ModelTrainer — Orchestrates the full training loop for a single
AdaptiveHiLSTMv2 model instance.

Features
--------
* Multi-timeframe DataLoader construction from aligned OHLCV DataFrames.
* AdaptiveTradingLoss with Kendall-Gal uncertainty weighting.
* AdamW optimiser + OneCycleLR scheduler.
* Early stopping on validation directional accuracy.
* Gradient clipping (norm = 1.0).
* Checkpoint save/restore (best validation model).
* Mixed-precision (torch.autocast) when device = CUDA.
* Training / validation loss + metrics logged every epoch.

Usage
-----
    trainer = ModelTrainer(config, device="cuda")
    history = trainer.fit(model, daily_df, intraday_15m_df,
                          intraday_1h_df, weekly_df)
    trainer.save_checkpoint(model, path="models/saved/fold_0.pt")
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class MultiTimeframeDataset:
    """
    Converts aligned OHLCV feature DataFrames into sliding-window tensors
    suitable for the four-encoder Hi-LSTM architecture.

    Parameters
    ----------
    df_15m, df_1h, df_1d, df_1w : pd.DataFrame
        Feature DataFrames (already processed by FeatureEngineer).
    seq_15m, seq_1h, seq_1d, seq_1w : int
        Look-back window lengths for each timeframe.
    n_features : int
        Number of feature columns to use from each DataFrame.
        If a DataFrame has more columns they are silently truncated;
        if fewer, zero-padded.
    direction_threshold : float
        Minimum absolute next-bar log-return to classify as up/down (else flat).
    target_col : str
        Name of the target column in the daily DataFrame.
    """

    def __init__(
        self,
        df_15m: pd.DataFrame,
        df_1h:  pd.DataFrame,
        df_1d:  pd.DataFrame,
        df_1w:  pd.DataFrame,
        seq_15m: int = 96,
        seq_1h:  int = 48,
        seq_1d:  int = 252,
        seq_1w:  int = 52,
        n_features: int = 50,
        direction_threshold: float = 0.003,
        target_col: str = "target_return",
        feature_schema: Optional[Dict[str, List[str]]] = None,
    ):
        self.seq    = {"15m": seq_15m, "1h": seq_1h, "1d": seq_1d, "1w": seq_1w}
        self.n_feat = n_features
        self.thresh = direction_threshold
        self.target_col = target_col
        self.feature_schema = feature_schema or {}

        self.dfs = {"15m": df_15m, "1h": df_1h, "1d": df_1d, "1w": df_1w}
        self._timeline = None

        # Use daily bars as the "master clock" for labels
        self._build()

    def _get_feature_array(self, df: pd.DataFrame, tf: str) -> np.ndarray:
        """Return a (T, n_features) float32 array from a DataFrame."""
        schema = self.feature_schema.get(tf)
        if schema:
            work = df.reindex(columns=schema, fill_value=0.0)
        else:
            work = df.select_dtypes(include=[float, int]).copy()
        arr = work.to_numpy(dtype=np.float32)
        T, F = arr.shape
        if F < self.n_feat:
            pad = np.zeros((T, self.n_feat - F), dtype=np.float32)
            arr = np.hstack([arr, pad])
        else:
            arr = arr[:, :self.n_feat]
        return arr

    def _build(self):
        """Build aligned (X_15m, X_1h, X_1d, X_1w, y_dir, y_mag) arrays."""
        def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if not isinstance(out.index, pd.DatetimeIndex):
                if "Date" in out.columns:
                    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                    out = out.set_index("Date", drop=False)
                else:
                    out.index = pd.to_datetime(out.index, errors="coerce")
            return out.sort_index()

        self.dfs = {tf: _to_dt_index(df) for tf, df in self.dfs.items()}
        arrs = {tf: self._get_feature_array(df, tf) for tf, df in self.dfs.items()}

        # Extract labels from daily DataFrame
        if self.target_col in self.dfs["1d"].columns:
            raw_returns = self.dfs["1d"][self.target_col].values.astype(np.float32)
        else:
            close = self.dfs["1d"]["Close"].values.astype(np.float32)
            raw_returns = np.log(close[1:] / close[:-1])
            raw_returns = np.append(raw_returns, 0.0)

        def to_direction(r: float) -> int:
            if r > self.thresh:
                return 2
            if r < -self.thresh:
                return 0
            return 1

        daily_df = self.dfs["1d"]
        n_daily = len(daily_df)
        start_idx = max(self.seq["1d"] - 1, 0)
        end_idx = max(start_idx + 1, n_daily - 1)
        if end_idx <= start_idx:
            raise ValueError(
                f"Not enough daily bars ({n_daily}) for sequences of length {self.seq['1d']}."
            )

        self.X_15m, self.X_1h, self.X_1d, self.X_1w = [], [], [], []
        self.y_dir, self.y_mag = [], []

        for d_end in range(start_idx, n_daily - 1):
            d_start = d_end - self.seq["1d"] + 1
            if d_start < 0:
                continue

            day_ts = pd.Timestamp(daily_df.index[d_end]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            day_cut = daily_df.index[d_start:d_end + 1]

            def _slice_tf(tf: str, seq_len: int) -> np.ndarray:
                df = self.dfs[tf]
                eligible = df.index[df.index <= day_ts]
                if len(eligible) == 0:
                    return np.zeros((seq_len, self.n_feat), dtype=np.float32)
                end_label = eligible[-1]
                block = df.loc[:end_label]
                if self.feature_schema.get(tf):
                    block = block.reindex(columns=self.feature_schema[tf], fill_value=0.0)
                else:
                    block = block.select_dtypes(include=[float, int])
                arr = block.to_numpy(dtype=np.float32)
                return self._pad(arr, seq_len)

            x15 = _slice_tf("15m", self.seq["15m"])
            x1h = _slice_tf("1h", self.seq["1h"])
            x1d = arrs["1d"][max(0, d_end - self.seq["1d"] + 1):d_end + 1]
            x1d = self._pad(x1d, self.seq["1d"])
            x1w = _slice_tf("1w", self.seq["1w"])

            label_ret = float(raw_returns[d_end]) if d_end < len(raw_returns) else 0.0
            self.X_15m.append(x15)
            self.X_1h.append(x1h)
            self.X_1d.append(x1d)
            self.X_1w.append(x1w)
            self.y_dir.append(to_direction(label_ret))
            self.y_mag.append(label_ret)

        self.X_15m = np.array(self.X_15m, dtype=np.float32)
        self.X_1h  = np.array(self.X_1h,  dtype=np.float32)
        self.X_1d  = np.array(self.X_1d,  dtype=np.float32)
        self.X_1w  = np.array(self.X_1w,  dtype=np.float32)
        self.y_dir = np.array(self.y_dir, dtype=np.int64)
        self.y_mag = np.array(self.y_mag, dtype=np.float32)

    def _pad(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Zero-pad arr along axis 0 to target_len rows at the front."""
        arr = np.asarray(arr, dtype=np.float32)
        n = arr.shape[0]
        nf = self.n_feat
        if n == 0:
            return np.zeros((target_len, nf), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] < nf:
            pad_cols = np.zeros((arr.shape[0], nf - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad_cols])
        elif arr.shape[1] > nf:
            arr = arr[:, :nf]
        if n >= target_len:
            return arr[-target_len:]
        pad = np.zeros((target_len - n, nf), dtype=np.float32)
        return np.vstack([pad, arr])

    def to_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(self.X_15m),
            torch.from_numpy(self.X_1h),
            torch.from_numpy(self.X_1d),
            torch.from_numpy(self.X_1w),
            torch.from_numpy(self.y_dir),
            torch.from_numpy(self.y_mag),
        )

    def __len__(self) -> int:
        return len(self.y_dir)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _directional_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds   = logits.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    return correct


def _per_class_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=-1).cpu()
    tgt   = targets.cpu()
    result = {}
    for cls, name in enumerate(["Bear", "Flat", "Bull"]):
        mask    = (tgt == cls)
        if mask.sum() == 0:
            result[name] = float("nan")
        else:
            result[name] = (preds[mask] == cls).float().mean().item()
    return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Full training orchestrator for a single AdaptiveHiLSTMv2 model.

    Parameters
    ----------
    config : Config
        Global configuration (batch size, epochs, LR, patience, etc.).
    device : str | torch.device
        Target compute device. "auto" auto-selects CUDA → MPS → CPU.
    """

    def __init__(self, config, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        logger.info("ModelTrainer initialised — device=%s", self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    # ── data preparation ─────────────────────────────────────────────────────

    def build_dataloaders(
        self,
        df_15m: pd.DataFrame,
        df_1h:  pd.DataFrame,
        df_1d:  pd.DataFrame,
        df_1w:  pd.DataFrame,
        val_fraction: float = 0.15,
        n_features: int = 50,
        feature_schema: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build training and validation DataLoaders from four aligned DataFrames.
        The split is chronological (last val_fraction of the data → validation).
        """
        cfg = self.config

        dataset = MultiTimeframeDataset(
            df_15m, df_1h, df_1d, df_1w,
            seq_15m=cfg.SEQ_15M, seq_1h=cfg.SEQ_1H,
            seq_1d=cfg.SEQ_1D,   seq_1w=cfg.SEQ_1W,
            n_features=n_features,
            direction_threshold=cfg.DIRECTION_THRESHOLD,
        )

        full_ds = dataset.to_tensor_dataset()
        n_total = len(full_ds)
        n_val   = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val

        # Chronological split — no shuffling for time series
        indices_train = list(range(n_train))
        indices_val   = list(range(n_train, n_total))

        train_ds = torch.utils.data.Subset(full_ds, indices_train)
        val_ds   = torch.utils.data.Subset(full_ds, indices_val)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=False, drop_last=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=False, drop_last=False, num_workers=0,
        )
        logger.info(
            "Dataset: %d train samples, %d val samples — features=%d",
            n_train, n_val, n_features,
        )
        return train_loader, val_loader

    # ── training loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        model: nn.Module,
        df_15m: pd.DataFrame,
        df_1h:  pd.DataFrame,
        df_1d:  pd.DataFrame,
        df_1w:  pd.DataFrame,
        n_features: int = 50,
        val_fraction: float = 0.15,
        checkpoint_path: Optional[str] = None,
        feature_schema: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'val_acc'
                  (list of values, one per epoch).
        """
        from models.losses import AdaptiveTradingLoss

        cfg = self.config
        model = model.to(self.device)

        train_loader, val_loader = self.build_dataloaders(
            df_15m, df_1h, df_1d, df_1w, val_fraction, n_features, feature_schema
        )
        self._feature_schema = feature_schema or {}

        loss_fn  = AdaptiveTradingLoss().to(self.device)
        from models.losses import ConfidenceCalibrationLoss
        conf_loss_fn = ConfidenceCalibrationLoss().to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.LEARNING_RATE * 10,
            steps_per_epoch=len(train_loader),
            epochs=cfg.EPOCHS,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        use_amp = (self.device.type == "cuda")
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [],
        }
        best_val_acc = -1.0
        patience_counter = 0
        best_state: Optional[Dict] = None

        for epoch in range(cfg.EPOCHS):
            # ── train ──
            model.train()
            loss_fn.train()
            train_losses = []
            t0 = time.perf_counter()

            for batch in train_loader:
                x15, x1h, x1d, x1w, y_dir, y_mag = [b.to(self.device) for b in batch]

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        _o = model(x15, x1h, x1d, x1w)
                        dir_logits, magnitude, confidence, regime_probs = (
                            _o["direction_logits"], _o["magnitude"],
                            _o["confidence"], _o["regime_probs"]
                        )
                        loss, _ = loss_fn(dir_logits, magnitude, confidence, regime_probs, y_dir, y_mag)
                        loss = loss + 0.2 * conf_loss_fn(confidence, dir_logits, y_dir)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _o = model(x15, x1h, x1d, x1w)
                    dir_logits, magnitude, confidence, regime_probs = (
                        _o["direction_logits"], _o["magnitude"],
                        _o["confidence"], _o["regime_probs"]
                    )
                    loss, _ = loss_fn(dir_logits, magnitude, confidence, regime_probs, y_dir, y_mag)
                    loss = loss + 0.2 * conf_loss_fn(confidence, dir_logits, y_dir)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                train_losses.append(loss.item())

            # ── validate ──
            model.eval()
            val_losses = []
            all_logits: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []

            with torch.no_grad():
                for batch in val_loader:
                    x15, x1h, x1d, x1w, y_dir, y_mag = [b.to(self.device) for b in batch]
                    _o = model(x15, x1h, x1d, x1w)
                    dir_logits, magnitude, confidence, regime_probs = (
                        _o["direction_logits"], _o["magnitude"],
                        _o["confidence"], _o["regime_probs"]
                    )
                    loss, _ = loss_fn(dir_logits, magnitude, confidence, regime_probs, y_dir, y_mag)
                    val_losses.append(loss.item())
                    all_logits.append(dir_logits.cpu())
                    all_targets.append(y_dir.cpu())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            val_acc    = _directional_accuracy(
                torch.cat(all_logits), torch.cat(all_targets)
            )
            elapsed = time.perf_counter() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            lr_cur = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.LEARNING_RATE
            logger.info(
                "Epoch %3d/%d | train=%.4f | val=%.4f | acc=%.1f%% | lr=%.2e | %.1fs",
                epoch + 1, cfg.EPOCHS, train_loss, val_loss, val_acc * 100, lr_cur, elapsed
            )

            # ── early stopping ──
            if val_acc > best_val_acc + 1e-4:
                best_val_acc     = val_acc
                patience_counter = 0
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    self.save_checkpoint(model, checkpoint_path, epoch, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d — best val acc: %.1f%%",
                        epoch + 1, best_val_acc * 100,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Restored best model weights (val acc: %.1f%%)", best_val_acc * 100)

        return history

    # ── checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        epoch: int = 0,
        val_acc: float = 0.0,
    ) -> None:
        """Save model state dict + metadata to *path* (creates parent dirs)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_acc":          val_acc,
                "config_snapshot":  {
                    "SEQ_15M": self.config.SEQ_15M,
                    "SEQ_1H":  self.config.SEQ_1H,
                    "SEQ_1D":  self.config.SEQ_1D,
                    "SEQ_1W":  self.config.SEQ_1W,
                },
                "feature_schema": getattr(self, "_feature_schema", {}) or {},
            },
            path,
        )
        logger.info("Checkpoint saved → %s  (epoch=%d, val_acc=%.1f%%)",
                    path, epoch, val_acc * 100)

    def load_checkpoint(self, model: nn.Module, path: str) -> Dict[str, Any]:
        """Load a checkpoint into *model* in-place and return the metadata dict."""
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        self._feature_schema = ckpt.get("feature_schema", {}) or {}
        logger.info(
            "Checkpoint loaded ← %s  (epoch=%d, val_acc=%.1f%%)",
            path, ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0) * 100,
        )
        return ckpt
