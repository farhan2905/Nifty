"""
LivePredictor: Runs in real-time during market hours.
Every 15 minutes (after 15m candle close):
  1. Fetch latest 15m, 1h, 1d, 1w data
  2. Fetch latest news
  3. Run model (ensemble + MC Dropout)
  4. Run retest engine
  5. Print/log signal update

Schedule: fires at HH:00, HH:15, HH:30, HH:45 every hour,
          but only executes the full pipeline during NSE market hours
          (Mon–Fri 09:15–15:30 IST).

Usage:
    predictor = LivePredictor(config, ensemble_model, retest_engine,
                              data_collector, feature_engineer,
                              news_feeder, sentiment_analyzer)
    predictor.start()   # blocks; Ctrl-C to stop
"""

import os
import csv
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from visualization.reporting import PredictionVisualizer
from learning.daily_feedback import DailyFeedbackLearner
from learning.assistant_system import AssistantOrchestrator

import numpy as np
import torch
import schedule
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _now_ist() -> datetime:
    """Return the current wall-clock time as a timezone-aware IST datetime."""
    return datetime.now(IST)


def _candle_align_15m(dt: datetime) -> datetime:
    """
    Snap *dt* back to the most recently closed 15-minute candle boundary.

    E.g.  09:23 IST  →  09:15 IST
          10:30 IST  →  10:30 IST
          15:47 IST  →  15:45 IST
    """
    minute_slot = (dt.minute // 15) * 15
    return dt.replace(minute=minute_slot, second=0, microsecond=0)


def _colour(text: str, code: str) -> str:
    """
    Wrap *text* in an ANSI colour escape if the terminal supports it,
    otherwise return *text* unchanged.
    """
    if not os.isatty(1):           # stdout is not a TTY (e.g. file redirect)
        return text
    return f"\033[{code}m{text}\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictor:
    """
    Orchestrates real-time inference for the Nifty Hi-LSTM v2 system.

    Parameters
    ----------
    config : Config
        Global configuration object (SEQ_15M, SEQ_1H, SEQ_1D, SEQ_1W,
        REGIME_NAMES, LOG_DIR, MC_SAMPLES, etc.).
    ensemble_model : EnsembleModel
        Pre-loaded, pre-trained ensemble of AdaptiveHiLSTMv2 instances.
    retest_engine : RetestEngine
        Stateful retest/signal-tracking engine.
    data_collector : NiftyDataCollector
        Handles yfinance / NSEpy data fetching and caching.
    feature_engineer : FeatureEngineer
        Computes all technical + cross-timeframe features.
    news_feeder : NewsFeeder
        Fetches and caches recent RSS / web news articles.
    sentiment_analyzer : SentimentAnalyzer
        Scores articles with FinBERT / VADER.
    mc_samples : int
        Number of forward passes for MC Dropout uncertainty estimation (default 30).
    log_dir : str | Path | None
        Directory for daily CSV prediction logs. Defaults to config.LOG_DIR.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        config,
        ensemble_model,
        retest_engine,
        data_collector,
        feature_engineer,
        news_feeder,
        sentiment_analyzer,
        mc_samples: int = 30,
        log_dir: Optional[str] = None,
        feedback_learner: Optional[DailyFeedbackLearner] = None,
    ):
        self.config = config
        self.ensemble = ensemble_model
        self.retest = retest_engine
        self.collector = data_collector
        self.fe = feature_engineer
        self.news = news_feeder
        self.sentiment = sentiment_analyzer
        self.mc_samples = mc_samples

        self.log_dir = Path(log_dir or getattr(config, "LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(getattr(config, "REPORT_DIR", self.log_dir / "reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = PredictionVisualizer(self.report_dir)

        self._running = False
        self._stop_event = threading.Event()
        self._last_candle: Optional[datetime] = None
        self._prediction_history: List[Dict[str, Any]] = []
        self.feature_schema = getattr(config, "FEATURE_SCHEMA", None) or {}
        self.feedback = feedback_learner or DailyFeedbackLearner(
            Path(getattr(config, "RL_STATE_FILE", self.log_dir / "rl_feedback_state.json")),
            history_limit=getattr(config, "FEEDBACK_HISTORY_LIMIT", 500),
        )
        self.assistant = AssistantOrchestrator(
            self.feedback,
            direction_threshold=float(getattr(config, "DIRECTION_THRESHOLD", 0.0)),
        )
        self._last_forecast = None

        logger.info("LivePredictor initialised — log_dir=%s", self.log_dir)

    # ── market-hours helpers ──────────────────────────────────────────────────

    def is_market_hours(self) -> bool:
        """
        Return True if the current IST time falls within NSE trading hours:
        Monday–Friday, 09:15–15:30 IST (inclusive of open, exclusive of close).
        """
        now = _now_ist()
        if now.weekday() >= 5:          # Saturday = 5, Sunday = 6
            return False
        open_time  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
        close_time = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
        return open_time <= now < close_time

    def get_next_candle_time(self) -> datetime:
        """
        Return the IST datetime of the next 15-minute candle close after *now*.

        E.g. if now is 10:22 IST, the next candle closes at 10:30 IST.
        """
        now = _now_ist()
        minutes_past = now.minute % 15
        minutes_to_next = 15 - minutes_past if minutes_past != 0 else 15
        return (now + timedelta(minutes=minutes_to_next)).replace(
            second=0, microsecond=0
        )

    # ── data pipeline ─────────────────────────────────────────────────────────

    def _fetch_all_timeframes(self) -> Dict[str, Any]:
        """
        Pull the freshest bars for each timeframe and return a dict:
            { '15m': DataFrame, '1h': DataFrame, '1d': DataFrame, '1w': DataFrame }
        Raises on any fetch failure so the caller can skip this cycle.
        """
        logger.debug("Fetching multi-timeframe data …")
        data = {
            "15m": self.collector.fetch_intraday_data("15m", "5d"),
            "1h":  self.collector.fetch_intraday_data("1h",  "30d"),
            "1d":  self.collector.fetch_daily_data(
                       (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
                       datetime.now().strftime("%Y-%m-%d")
                   ),
        }
        # Weekly: resample from daily to save an API call
        d_df = data["1d"].copy()
        d_df.index = d_df.index if hasattr(d_df.index, "freq") else d_df.index
        try:
            data["1w"] = d_df.resample("W-FRI").agg(
                {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
            ).dropna()
        except Exception:
            data["1w"] = data["1d"].iloc[::5].copy()   # fallback: every 5th daily bar
        return data

    def _build_model_inputs(
        self,
        raw_data: Dict[str, Any],
        n_features: int,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Run FeatureEngineer on each timeframe, slice the last
        SEQ_* rows, and return (x_15m, x_1h, x_1d, x_1w) tensors
        of shape (1, seq_len, n_features) ready for the model.

        Returns None if any timeframe lacks enough history.
        """
        seq_map = {
            "15m": self.config.SEQ_15M,
            "1h":  self.config.SEQ_1H,
            "1d":  self.config.SEQ_1D,
            "1w":  self.config.SEQ_1W,
        }
        tensors = []
        for tf in ("15m", "1h", "1d", "1w"):
            seq_len = seq_map[tf]
            try:
                feat_df = self.fe.compute_technical_features(raw_data[tf])
                schema = self.feature_schema.get(tf)
                if schema:
                    feat_df = feat_df.reindex(columns=schema, fill_value=0.0)
                numeric = feat_df.select_dtypes(include=[float, int])
                cols = numeric.columns.tolist()
                if len(cols) < n_features:
                    logger.warning(
                        "Timeframe %s has only %d features (need %d); padding with zeros.",
                        tf, len(cols), n_features,
                    )
                    arr = numeric.values
                    pad = np.zeros((arr.shape[0], n_features - arr.shape[1]), dtype=np.float32)
                    arr = np.hstack([arr, pad])
                else:
                    arr = numeric.iloc[:, :n_features].values

                if len(arr) < seq_len:
                    logger.warning(
                        "Timeframe %s has only %d bars (need %d); skipping cycle.",
                        tf, len(arr), seq_len,
                    )
                    return None

                window = arr[-seq_len:].astype(np.float32)
                # Replace NaN / Inf with 0
                window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
                tensors.append(
                    torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                )
            except Exception as exc:
                logger.error("Feature engineering failed for %s: %s", tf, exc)
                return None

        return tuple(tensors)

    # ── horizon utilities ──────────────────────────────────────────────────────

    def _summarise_horizon_predictions(self, horizon_predictions: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, Any]]:
        """Convert raw tensor horizon outputs to a JSON / display friendly dict."""
        summary: Dict[str, Dict[str, Any]] = {}
        for horizon, vals in (horizon_predictions or {}).items():
            if not vals:
                continue
            dir_probs = torch.softmax(vals["direction_logits"], dim=-1).detach().cpu().numpy()
            magnitude = vals["magnitude"].detach().cpu().numpy().reshape(-1)
            confidence = vals["confidence"].detach().cpu().numpy().reshape(-1)
            summary[horizon] = {
                "direction_probs": dir_probs.tolist(),
                "direction_idx": int(np.argmax(dir_probs)),
                "direction_label": ["BEARISH", "FLAT", "BULLISH"][int(np.argmax(dir_probs))],
                "confidence_mean": float(np.mean(confidence)),
                "magnitude_mean": float(np.mean(magnitude)),
                "magnitude_pct": float(np.mean(magnitude) * 100.0),
            }
        return summary

    # ── MC Dropout uncertainty ────────────────────────────────────────────────

    @torch.no_grad()
    def _mc_dropout_predict(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        n_samples: int,
    ) -> Dict[str, Any]:
        """
        Estimate predictive uncertainty via MC Dropout:
          - Enable dropout at inference time
          - Forward-pass *n_samples* times
          - Aggregate: mean direction probability, magnitude, confidence, regime

        Returns a dict with keys:
            direction_probs (3,), magnitude_mean, magnitude_std,
            confidence_mean, regime_probs (4,), uncertainty_pct
        """
        # Activate dropout layers
        model.train()           # keeps dropout active
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                m.eval()        # keep normalisation stable

        dir_samples    = []
        mag_samples    = []
        conf_samples   = []
        regime_samples = []
        horizon_samples: List[Dict[str, Dict[str, Any]]] = []

        for _ in range(n_samples):
            out = model(*inputs)
            dir_logits   = out["direction_logits"]
            magnitude    = out["magnitude"]
            confidence   = out["confidence"]
            regime_probs = out["regime_probs"]
            dir_samples.append(torch.softmax(dir_logits, dim=-1).cpu().numpy())
            mag_samples.append(magnitude.cpu().numpy())
            conf_samples.append(confidence.cpu().numpy())
            regime_samples.append(regime_probs.cpu().numpy())
            horizon_samples.append(self._summarise_horizon_predictions(out.get("horizon_predictions", {})))

        model.eval()

        dir_arr    = np.array(dir_samples).squeeze()   # (n_samples, 3)
        mag_arr    = np.array(mag_samples).squeeze()   # (n_samples,)
        conf_arr   = np.array(conf_samples).squeeze()  # (n_samples,)
        regime_arr = np.array(regime_samples).squeeze() # (n_samples, 4)

        # Handle edge-case: single sample or 1-d collapse
        if dir_arr.ndim == 1:
            dir_arr = dir_arr[np.newaxis, :]
        if regime_arr.ndim == 1:
            regime_arr = regime_arr[np.newaxis, :]

        horizon_means: Dict[str, Dict[str, Any]] = {}
        if horizon_samples:
            for horizon in {h for sample in horizon_samples for h in sample.keys()}:
                hdir = np.array([s[horizon]["direction_probs"] for s in horizon_samples if horizon in s])
                hconf = np.array([s[horizon]["confidence_mean"] for s in horizon_samples if horizon in s])
                hmag = np.array([s[horizon]["magnitude_mean"] for s in horizon_samples if horizon in s])
                if len(hdir):
                    avg_hdir = hdir.mean(axis=0)
                    horizon_means[horizon] = {
                        "direction_probs": avg_hdir.tolist(),
                        "direction_idx": int(np.argmax(avg_hdir)),
                        "direction_label": ["BEARISH", "FLAT", "BULLISH"][int(np.argmax(avg_hdir))],
                        "confidence_mean": float(hconf.mean()),
                        "magnitude_mean": float(hmag.mean()),
                        "magnitude_pct": float(hmag.mean() * 100.0),
                    }

        magnitude_mean  = float(mag_arr.mean())
        magnitude_std   = float(mag_arr.std())
        uncertainty_pct = magnitude_std * 100.0  # express in percent

        return {
            "direction_probs":  dir_arr.mean(axis=0),        # (3,)
            "magnitude_mean":   magnitude_mean,
            "magnitude_std":    magnitude_std,
            "uncertainty_pct":  uncertainty_pct,
            "confidence_mean":  float(conf_arr.mean()),
            "regime_probs":     regime_arr.mean(axis=0),     # (4,)
            "horizon_predictions_mean": horizon_means,
        }

    def _ensemble_predict(
        self,
        inputs: Tuple[torch.Tensor, ...],
    ) -> Dict[str, Any]:
        """
        Average MC-Dropout predictions across all models in the ensemble.
        Returns the same dict schema as _mc_dropout_predict but averaged.
        """
        results = []
        for model in self.ensemble.models:
            results.append(self._mc_dropout_predict(model, inputs, self.mc_samples))

        avg_dir    = np.mean([r["direction_probs"] for r in results], axis=0)
        avg_mag    = float(np.mean([r["magnitude_mean"]  for r in results]))
        avg_std    = float(np.mean([r["magnitude_std"]   for r in results]))
        avg_conf   = float(np.mean([r["confidence_mean"] for r in results]))
        avg_regime = np.mean([r["regime_probs"] for r in results], axis=0)

        horizon_agg: Dict[str, Dict[str, Any]] = {}
        if results and results[0].get("horizon_predictions_mean"):
            horizon_names = {h for r in results for h in r.get("horizon_predictions_mean", {}).keys()}
            for horizon in horizon_names:
                h_dirs, h_confs, h_mags = [], [], []
                for r in results:
                    h = r.get("horizon_predictions_mean", {}).get(horizon)
                    if not h:
                        continue
                    h_dirs.append(h["direction_probs"])
                    h_confs.append(h["confidence_mean"])
                    h_mags.append(h["magnitude_mean"])
                if h_dirs:
                    avg_hdir = np.mean(h_dirs, axis=0)
                    horizon_agg[horizon] = {
                        "direction_probs": avg_hdir.tolist(),
                        "direction_idx": int(np.argmax(avg_hdir)),
                        "direction_label": ["BEARISH", "FLAT", "BULLISH"][int(np.argmax(avg_hdir))],
                        "confidence_mean": float(np.mean(h_confs)),
                        "magnitude_mean": float(np.mean(h_mags)),
                        "magnitude_pct": float(np.mean(h_mags) * 100.0),
                    }

        return {
            "direction_probs":  avg_dir,
            "magnitude_mean":   avg_mag,
            "magnitude_std":    avg_std,
            "uncertainty_pct":  avg_std * 100.0,
            "confidence_mean":  avg_conf,
            "regime_probs":     avg_regime,
            "horizon_predictions_mean": horizon_agg,
        }

    # ── full pipeline ─────────────────────────────────────────────────────────

    def on_candle_close(self):
        """
        Full 15-minute pipeline:

        1.  Guard: skip if outside market hours.
        2.  Fetch fresh OHLCV bars for 15m / 1h / 1d / 1w.
        3.  Compute technical + cross-timeframe features.
        4.  Fetch news from the last 2 hours.
        5.  Run sentiment analysis.
        6.  Build model input tensors.
        7.  Run ensemble with MC Dropout → predictions + uncertainty.
        8.  Feed into RetestEngine for signal tracking.
        9.  Log and pretty-print the full dashboard.
        """
        candle_time = _candle_align_15m(_now_ist())

        # ── guard ──
        if not self.is_market_hours():
            logger.debug("Outside market hours — skipping candle at %s", candle_time)
            return

        # Deduplicate: ensure we don't process the same candle twice
        if self._last_candle == candle_time:
            logger.debug("Candle %s already processed — skipping.", candle_time)
            return
        self._last_candle = candle_time

        logger.info("── Candle close: %s ──", candle_time.strftime("%H:%M IST"))
        cycle_start = time.perf_counter()

        # ── Step 1: fetch data ──
        try:
            raw_data = self._fetch_all_timeframes()
        except Exception as exc:
            logger.error("Data fetch failed: %s", exc)
            return

        # ── Step 2: feature engineering ──
        n_features = getattr(self.config, "N_FEATURES", 50)
        inputs = self._build_model_inputs(raw_data, n_features)
        if inputs is None:
            logger.warning("Insufficient data — skipping inference this cycle.")
            return

        # ── Step 3: news + sentiment ──
        sentiment_output = {"score": 0.0, "label": "Neutral",
                            "n_articles": 0, "urgency": False}
        try:
            articles = self.news.fetch_recent(hours=2)
            if articles:
                sentiment_output = self.sentiment.analyze_batch(articles)
                sentiment_output["n_articles"] = len(articles)
        except Exception as exc:
            logger.warning("Sentiment pipeline failed: %s — continuing without it.", exc)

        # ── Step 4: macro signals ──
        macro = self._fetch_macro_signals()

        # ── Step 5: ensemble inference ──
        try:
            pred = self._ensemble_predict(inputs)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            return

        adaptive_policy = self.assistant.recommendations()

        # ── Step 6: decode outputs ──
        regime_names = getattr(
            self.config, "REGIME_NAMES",
            ["Strong Bull", "Weak Bull", "Ranging", "Bearish"]
        )
        dir_labels   = ["BEARISH", "FLAT", "BULLISH"]
        dir_idx      = int(np.argmax(pred["direction_probs"]))
        direction    = dir_labels[dir_idx]
        conf         = pred["confidence_mean"]
        mag_pct      = pred["magnitude_mean"] * 100.0
        uncertainty  = pred["uncertainty_pct"]

        regime_idx   = int(np.argmax(pred["regime_probs"]))
        regime_name  = regime_names[regime_idx]
        regime_pct   = float(pred["regime_probs"][regime_idx]) * 100.0

        # ── Step 7: price targets ──
        spot = self._get_spot_price(raw_data)
        targets = self._compute_targets(spot, pred, raw_data)

        assistant_raw = {
            "timestamp": candle_time,
            "direction_probs": pred["direction_probs"].tolist(),
            "magnitude_mean": pred["magnitude_mean"],
            "magnitude_std": pred["magnitude_std"],
            "confidence_mean": pred["confidence_mean"],
            "regime_probs": pred["regime_probs"].tolist(),
            "regime_names": {i: regime_names[i] for i in range(len(regime_names))},
            "horizon_predictions_mean": pred.get("horizon_predictions_mean", {}),
            "targets": targets,
            "adaptive_policy": adaptive_policy,
            "spot_price": spot,
            "sentiment": sentiment_output,
            "macro": macro,
            "regime": regime_name,
            "uncertainty_pct": uncertainty,
            "model_prediction": {
                "direction_probs": pred["direction_probs"].tolist(),
                "magnitude_mean": pred["magnitude_mean"],
                "magnitude_std": pred["magnitude_std"],
                "confidence_mean": pred["confidence_mean"],
                "regime_probs": pred["regime_probs"].tolist(),
                "horizon_predictions": pred.get("horizon_predictions_mean", {}),
            },
        }
        forecast = self.assistant.package_forecast(assistant_raw, timestamp=candle_time)
        self._last_forecast = forecast

        # ── Step 8: retest engine ──
        engine_input = {
            "timestamp":      candle_time,
            "direction":      forecast.direction,
            "confidence":     forecast.confidence,
            "magnitude_pct":  forecast.magnitude_pct,
            "uncertainty_pct": forecast.uncertainty_pct,
            "regime":         forecast.regime,
            "regime_probs":   {regime_names[i]: float(pred["regime_probs"][i]) * 100
                               for i in range(len(regime_names))},
            "adaptive_policy": forecast.adaptive_policy,
            "model_prediction": forecast.model_prediction,
            "horizon_predictions": forecast.horizon_predictions,
            "targets":        targets,
            "spot_price":     spot,
            "sentiment":      sentiment_output,
            "macro":          macro,
            "assistant_explanation": self.assistant.explain(forecast),
        }
        try:
            engine_output = self.retest.process(engine_input)
        except Exception as exc:
            logger.error("Retest engine failed: %s — using raw prediction.", exc)
            engine_output = engine_input.copy()
            engine_output.setdefault("active_signals", [])

        elapsed = time.perf_counter() - cycle_start
        logger.info("Pipeline completed in %.2fs", elapsed)

        # ── Step 9: display + log ──
        self.print_signal_update(engine_output)
        self.save_prediction_log(engine_output, candle_time)
        try:
            self.assistant.log_prediction(forecast)
        except Exception as exc:
            logger.debug("Assistant log failed: %s", exc)
        try:
            self.visualizer.save_live_snapshot(engine_output, self._prediction_history[-120:])
        except Exception as exc:
            logger.debug("Live dashboard render failed: %s", exc)
        self._prediction_history.append(engine_output)

    # ── macro helper ─────────────────────────────────────────────────────────

    def _fetch_macro_signals(self) -> Dict[str, Any]:
        """
        Best-effort macro data fetch (VIX, FII, PCR).
        Returns sensible defaults on failure.
        """
        defaults = {
            "india_vix":     None,
            "fii_flow_cr":   None,
            "pcr":           None,
            "vix_regime":    "Unknown",
        }
        try:
            vix_df = self.collector.fetch_india_vix(
                (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            )
            if len(vix_df) > 0:
                vix_val = float(vix_df["Close"].iloc[-1])
                defaults["india_vix"] = vix_val
                if vix_val < 12:
                    defaults["vix_regime"] = "Very Low"
                elif vix_val < 16:
                    defaults["vix_regime"] = "Normal"
                elif vix_val < 20:
                    defaults["vix_regime"] = "Elevated"
                else:
                    defaults["vix_regime"] = "High Fear"
        except Exception as exc:
            logger.debug("VIX fetch skipped: %s", exc)

        try:
            fii = getattr(self.collector, "fetch_fii_data", None)
            if callable(fii):
                fii_df = fii()
                if len(fii_df) > 0:
                    defaults["fii_flow_cr"] = float(fii_df["net"].iloc[-1])
        except Exception:
            pass

        return defaults

    # ── price targets ─────────────────────────────────────────────────────────

    def _get_spot_price(self, raw_data: Dict[str, Any]) -> float:
        """Extract the latest close price from 15m bars."""
        try:
            return float(raw_data["15m"]["Close"].iloc[-1])
        except Exception:
            return 0.0

    def _compute_targets(
        self,
        spot: float,
        pred: Dict[str, Any],
        raw_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute H1 / D1 / W1 price targets and a dynamic stop-loss.

        The model outputs a *normalised* magnitude (fraction of price).
        We scale by timeframe multipliers drawn from empirical Nifty volatility.
        Stop-loss is set at 1.5× the 15m ATR below/above spot (direction-aware).
        """
        mag = abs(pred["magnitude_mean"])      # normalised fraction
        sign = 1 if pred["direction_probs"][2] > pred["direction_probs"][0] else -1

        # Approximate annualised vol → per-tf fraction
        tf_multipliers = {"h1": 1.0, "d1": 2.5, "w1": 5.5}

        # Attempt real ATR from 15m bars
        atr_pct = 0.004   # 0.40% default for Nifty intraday
        try:
            df15 = raw_data["15m"]
            tr = np.maximum(
                (df15["High"] - df15["Low"]).values,
                np.maximum(
                    abs(df15["High"] - df15["Close"].shift(1)).values,
                    abs(df15["Low"]  - df15["Close"].shift(1)).values,
                )
            )
            atr_abs  = float(np.nanmean(tr[-14:]))
            atr_pct  = atr_abs / spot if spot > 0 else atr_pct
        except Exception:
            pass

        # Use model magnitude or fallback to ATR-based estimate
        h1_move  = max(mag * tf_multipliers["h1"],  atr_pct * 1.0)
        d1_move  = max(mag * tf_multipliers["d1"],  atr_pct * 2.5)
        w1_move  = max(mag * tf_multipliers["w1"],  atr_pct * 5.0)
        sl_move  = atr_pct * 1.5

        return {
            "h1_target": round(spot * (1 + sign * h1_move),  2),
            "d1_target": round(spot * (1 + sign * d1_move),  2),
            "w1_target": round(spot * (1 + sign * w1_move),  2),
            "stop_loss": round(spot * (1 - sign * sl_move),  2),
            "h1_pct":    round(sign * h1_move * 100, 2),
            "d1_pct":    round(sign * d1_move * 100, 2),
            "w1_pct":    round(sign * w1_move * 100, 2),
            "sl_pct":    round(-sign * sl_move * 100, 2),
        }

    # ── display ───────────────────────────────────────────────────────────────

    def print_signal_update(self, engine_output: Dict[str, Any]):
        """
        Pretty-print the current prediction dashboard to stdout.

        ════════════════════════════════════════════════════════════
        NIFTY Hi-LSTM v2 | 15:15 IST
        ════════════════════════════════════════════════════════════
        PREDICTION:  ▲ BULLISH  (73.4% conf)
        MAGNITUDE:   +0.42% expected
        REGIME:      Weak Bull (42%) | Ranging (33%)
        UNCERTAINTY: ±0.18% (MC Dropout)

        H1 TARGET:  24,180  (+0.38%)
        D1 TARGET:  24,320  (+0.96%)
        W1 TARGET:  24,580  (+2.04%)
        STOP LOSS:  23,920  (-0.60%)

        ACTIVE SIGNALS: 2
          [SIG-001] LONG  @ 23,985 | Retests: 4/5 PASS | Conf: 78% | H1T: +0.8%
          ...
        ════════════════════════════════════════════════════════════
        """
        ts:  datetime      = engine_output.get("timestamp", _now_ist())
        ts_str = ts.strftime("%H:%M IST") if hasattr(ts, "strftime") else str(ts)

        direction    = engine_output.get("direction", "FLAT")
        conf         = engine_output.get("confidence",     0.0)
        mag_pct      = engine_output.get("magnitude_pct",  0.0)
        uncertainty  = engine_output.get("uncertainty_pct", 0.0)
        regime       = engine_output.get("regime", "Unknown")
        spot         = engine_output.get("spot_price", 0.0)
        targets      = engine_output.get("targets", {})
        sentiment    = engine_output.get("sentiment", {})
        macro        = engine_output.get("macro", {})
        active_sigs  = engine_output.get("active_signals", [])
        regime_probs = engine_output.get("regime_probs", {})

        # Direction symbol + colour
        dir_symbol = {"BULLISH": "▲", "BEARISH": "▼", "FLAT": "◆"}.get(direction, "◆")
        dir_colour = {"BULLISH": "32", "BEARISH": "31", "FLAT": "33"}.get(direction, "0")

        # ── header ──
        BAR = "═" * 60
        print("\n" + _colour(BAR, "36"))
        print(_colour(f"  NIFTY Hi-LSTM v2  |  {ts_str}", "1"))
        print(_colour(BAR, "36"))

        # ── prediction block ──
        dir_str = _colour(f"{dir_symbol} {direction}", dir_colour)
        print(f"  PREDICTION:   {dir_str}  ({conf:.1%} conf)")
        print(f"  MAGNITUDE:    {mag_pct:+.2f}% expected")

        # Regime line: show top-2 regimes
        regime_line = regime
        if regime_probs:
            sorted_r = sorted(regime_probs.items(), key=lambda x: -x[1])
            regime_line = " | ".join(f"{n} ({v:.0f}%)" for n, v in sorted_r[:2])
        print(f"  REGIME:       {regime_line}")
        print(f"  UNCERTAINTY:  ±{uncertainty:.2f}% (MC Dropout)")

        adaptive = engine_output.get("adaptive_policy", {})
        if adaptive:
            print(f"  ADAPTIVE:     conf_gate={adaptive.get('confidence_gate', 0.0):.2f} | pos={adaptive.get('position_size', 1.0):.2f} | temp={adaptive.get('temperature', 1.0):.2f}")

        # ── horizon view ──
        horizon_preds = engine_output.get("horizon_predictions", {})
        if horizon_preds:
            print()
            print("  HORIZON VIEW:")
            for horizon in ("15m", "1h", "1d"):
                hp = horizon_preds.get(horizon)
                if not hp:
                    continue
                arrow = {"BULLISH": "▲", "BEARISH": "▼", "FLAT": "◆"}.get(hp.get("direction_label", "FLAT"), "◆")
                print(
                    f"    {horizon:>3}: {arrow} {hp.get('direction_label', 'FLAT'):8s} "
                    f"| conf {float(hp.get('confidence_mean', 0.0)):.1%} "
                    f"| mag {float(hp.get('magnitude_pct', 0.0)):+.2f}%"
                )

        # ── targets ──
        print()
        if spot and targets:
            h1t = targets.get("h1_target", 0)
            d1t = targets.get("d1_target", 0)
            w1t = targets.get("w1_target", 0)
            sl  = targets.get("stop_loss", 0)
            h1p = targets.get("h1_pct", 0)
            d1p = targets.get("d1_pct", 0)
            w1p = targets.get("w1_pct", 0)
            slp = targets.get("sl_pct", 0)
            print(f"  H1 TARGET:    {h1t:,.0f}  ({h1p:+.2f}%)")
            print(f"  D1 TARGET:    {d1t:,.0f}  ({d1p:+.2f}%)")
            print(f"  W1 TARGET:    {w1t:,.0f}  ({w1p:+.2f}%)")
            sl_str = _colour(f"  STOP LOSS:    {sl:,.0f}  ({slp:+.2f}%)", "31")
            print(sl_str)
        else:
            print("  (No spot price available — targets not computed)")

        # ── active signals ──
        print()
        print(f"  ACTIVE SIGNALS: {len(active_sigs)}")
        for sig in active_sigs[:10]:          # cap at 10 for readability
            sig_id   = sig.get("id",        "???")
            side     = sig.get("side",      "LONG")
            entry    = sig.get("entry",     0.0)
            retests  = sig.get("retests",   "0/0 PASS")
            s_conf   = sig.get("confidence", 0.0)
            status   = sig.get("status",    "")
            h1t_sig  = sig.get("h1t_pct",   0.0)
            side_col = "32" if side == "LONG" else "31"
            side_str = _colour(side, side_col)
            print(
                f"    [{sig_id}] {side_str} @ {entry:,.0f} | "
                f"Retests: {retests} | Conf: {s_conf:.0%} | "
                f"H1T: {h1t_sig:+.1f}% | {status}"
            )
        if not active_sigs:
            print("    — No active signals —")

        explanation = engine_output.get("assistant_explanation", {})
        reasons = explanation.get("summary", []) if isinstance(explanation, dict) else []
        if reasons:
            print()
            print("  WHY THIS FORECAST:")
            for reason in reasons[:4]:
                print(f"    - {reason}")

        # ── macro / sentiment footer ──
        print()
        sent_score = sentiment.get("score",      0.0)
        sent_label = sentiment.get("label",      "Neutral")
        n_articles = sentiment.get("n_articles", 0)
        urgency    = "URGENT" if sentiment.get("urgency") else "No urgency"

        vix_val    = macro.get("india_vix",   None)
        vix_regime = macro.get("vix_regime",  "—")
        fii_flow   = macro.get("fii_flow_cr", None)
        fii_str    = f"₹{fii_flow:,.0f} Cr" if fii_flow is not None else "N/A"
        fii_label  = "(bearish)" if fii_flow and fii_flow < 0 else "(bullish)"
        vix_str    = f"{vix_val:.1f} ({vix_regime})" if vix_val is not None else "N/A"

        print(f"  SENTIMENT:    {sent_label} ({sent_score:+.2f}) | "
              f"{n_articles} articles | {urgency}")
        print(f"  FII TODAY:    {fii_str} {fii_label}")
        print(f"  INDIA VIX:    {vix_str}")
        print(_colour(BAR, "36") + "\n")

    # ── logging ───────────────────────────────────────────────────────────────

    def save_prediction_log(self, prediction: Dict[str, Any], timestamp: datetime):
        """
        Append one row to the daily CSV log at:
            <log_dir>/predictions_YYYY-MM-DD.csv

        Columns written:
            timestamp, direction, confidence, magnitude_pct, uncertainty_pct,
            regime, spot_price, h1_target, d1_target, w1_target, stop_loss,
            sentiment_score, sentiment_label, india_vix, fii_flow_cr
        """
        date_str  = timestamp.strftime("%Y-%m-%d")
        log_file  = self.log_dir / f"predictions_{date_str}.csv"
        is_new    = not log_file.exists()

        fieldnames = [
            "timestamp", "direction", "confidence", "magnitude_pct",
            "uncertainty_pct", "regime", "spot_price",
            "h1_target", "d1_target", "w1_target", "stop_loss",
            "sentiment_score", "sentiment_label",
            "india_vix", "fii_flow_cr",
            "horizon_predictions_json",
            "adaptive_confidence_gate",
            "adaptive_position_size",
            "adaptive_temperature",
            "memory_ema_reward",
            "memory_ema_accuracy",
        ]

        targets  = prediction.get("targets", {})
        sentiment = prediction.get("sentiment", {})
        macro    = prediction.get("macro", {})

        row = {
            "timestamp":       timestamp.isoformat(),
            "direction":       prediction.get("direction",      "FLAT"),
            "confidence":      round(prediction.get("confidence",    0.0), 4),
            "magnitude_pct":   round(prediction.get("magnitude_pct", 0.0), 4),
            "uncertainty_pct": round(prediction.get("uncertainty_pct", 0.0), 4),
            "regime":          prediction.get("regime",         "Unknown"),
            "spot_price":      prediction.get("spot_price",     0.0),
            "h1_target":       targets.get("h1_target",         ""),
            "d1_target":       targets.get("d1_target",         ""),
            "w1_target":       targets.get("w1_target",         ""),
            "stop_loss":       targets.get("stop_loss",         ""),
            "sentiment_score": sentiment.get("score",           ""),
            "sentiment_label": sentiment.get("label",           ""),
            "india_vix":       macro.get("india_vix",           ""),
            "fii_flow_cr":     macro.get("fii_flow_cr",         ""),
            "horizon_predictions_json": json.dumps(prediction.get("horizon_predictions", {}), default=str),
            "adaptive_confidence_gate": (prediction.get("adaptive_policy", {}) or {}).get("confidence_gate", ""),
            "adaptive_position_size": (prediction.get("adaptive_policy", {}) or {}).get("position_size", ""),
            "adaptive_temperature": (prediction.get("adaptive_policy", {}) or {}).get("temperature", ""),
            "memory_ema_reward": (self.feedback.snapshot().get("memory", {}) or {}).get("ema_reward", ""),
            "memory_ema_accuracy": (self.feedback.snapshot().get("memory", {}) or {}).get("ema_accuracy", ""),
        }

        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

        logger.debug("Logged prediction to %s", log_file)


    def apply_daily_outcome(self, timestamp: datetime | str, actual_return: float, realized_pnl: Optional[float] = None, regime: Optional[str] = None) -> Dict[str, Any]:
        """Verify the latest forecast and update reward, memory, calibration, and drift state."""
        result = self.assistant.verify_and_learn(timestamp, actual_return=actual_return, realized_pnl=realized_pnl, regime=regime)
        logger.info(
            "Updated assistant feedback: reward=%.4f gate=%.3f temp=%.3f",
            result.get("reward", 0.0),
            result.get("policy", {}).get("confidence_gate", 0.0),
            result.get("policy", {}).get("temperature", 0.0),
        )
        return result

    def get_adaptive_policy(self) -> Dict[str, Any]:
        """Return the current assistant policy recommendations."""
        return self.assistant.recommendations()

    # ── scheduler ────────────────────────────────────────────────────────────

    def start(self):
        """
        Start the live predictor scheduler.

        Registers on_candle_close at the four quarter-hour marks of each
        hour (:00, :15, :30, :45).  The loop runs in the calling thread
        and blocks until stop() is called or a KeyboardInterrupt is raised.
        """
        if self._running:
            logger.warning("LivePredictor.start() called while already running.")
            return

        self._running    = True
        self._stop_event.clear()

        # Schedule at every quarter-hour
        schedule.every().hour.at(":00").do(self.on_candle_close)
        schedule.every().hour.at(":15").do(self.on_candle_close)
        schedule.every().hour.at(":30").do(self.on_candle_close)
        schedule.every().hour.at(":45").do(self.on_candle_close)

        logger.info("LivePredictor started — firing at :00 :15 :30 :45 each hour.")
        print(
            "\n"
            + _colour("  Nifty Hi-LSTM v2  — Live Mode Started", "1;36")
            + "\n"
            + f"  Scheduling candle-close jobs at :00 :15 :30 :45 each hour.\n"
            + f"  Market hours: Mon–Fri  09:15 – 15:30 IST\n"
            + f"  Press Ctrl-C to stop.\n"
        )

        try:
            while not self._stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — stopping.")
        finally:
            self.stop()

    def stop(self):
        """
        Gracefully stop the scheduler.
        Clears all scheduled jobs and signals the run-loop to exit.
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        schedule.clear()
        logger.info("LivePredictor stopped. %d predictions logged this session.",
                    len(self._prediction_history))
        print(
            "\n"
            + _colour("  LivePredictor stopped.", "33")
            + f"  Predictions this session: {len(self._prediction_history)}\n"
        )

    # ── context-manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        state = "running" if self._running else "idle"
        return (
            f"<LivePredictor state={state} "
            f"mc_samples={self.mc_samples} "
            f"predictions={len(self._prediction_history)}>"
        )
