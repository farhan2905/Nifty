# Nifty Hi-LSTM v2 — Hierarchical Multi-Timeframe LSTM Prediction System

A production-grade deep-learning system for Nifty 50 directional forecasting.
Combines hierarchical Bi-LSTM encoders, adaptive market-regime detection,
FinBERT-powered sentiment analysis, and a stateful signal-retest engine.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              INPUT LAYER (4 Timeframes)          │
                    ├──────────┬──────────┬──────────┬────────────────┤
                    │  15-min  │   1-hr   │   1-day  │   1-week       │
                    │  96 bars │  48 bars │ 252 bars │  52 bars       │
                    │  ~50 feat│  ~50 feat│  ~50 feat│  ~50 feat      │
                    └────┬─────┴────┬─────┴────┬─────┴────┬───────────┘
                         │          │           │          │
              ┌──────────▼──┐  ┌───▼────┐  ┌──▼─────┐  ┌▼──────────┐
              │ Bi-LSTM     │  │Bi-LSTM │  │Bi-LSTM │  │ Bi-LSTM   │
              │ + MH-Attn   │  │+MH-Attn│  │+MH-Attn│  │ +MH-Attn  │
              │ (15m enc)   │  │(1h enc)│  │(1d enc)│  │ (1w enc)  │
              └──────┬──────┘  └───┬────┘  └──┬─────┘  └─────┬─────┘
                     │             │           │              │
                     └─────────────┴─────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Hierarchical Gating │
                              │  1w → 1d → 1h → 15m │
                              │  (coarse controls    │
                              │   fine-grained enc.) │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Regime Classifier   │
                              │  5 states (HMM-like) │
                              │  Strong Bull / Weak  │
                              │  Bull / Ranging /    │
                              │  Weak Bear / Strong  │
                              │  Bear                │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Regime-Conditioned  │
                              │  Decoder             │
                              └──────┬──────┬────────┘
                                     │      │
                          ┌──────────▼┐   ┌▼──────────────────┐
                          │ Direction │   │ Magnitude (Huber)  │
                          │ Head      │   │ Confidence Head    │
                          │ 3-class   │   │ Scalar calibration │
                          │ CE loss   │   │ BCE calibration    │
                          └───────────┘   └────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Ensemble (N=5)      │
                     │  MC Dropout × 30     │
                     │  Uncertainty: ±σ%    │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Retest Engine      │
                     │   Signal lifecycle   │
                     │   H1/D1/W1 targets  │
                     │   Multi-bar confirm  │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Live Predictor      │
                     │  :00/:15/:30/:45     │
                     │  09:15 – 15:30 IST  │
                     └─────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd nifty_hilstm_v2
pip install -r requirements.txt
```

For GPU acceleration (CUDA 11.8):
```bash
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (MPS):
```bash
pip install torch>=2.1.0
# MPS backend is included in standard macOS wheels since PyTorch 2.0
```

### 2. Run the Demo (no GPU needed, ~30 seconds)

```bash
python main.py --mode demo
```

The demo downloads 60 days of real Nifty 15-minute data via yfinance,
engineers features, trains a compact model for 3 epochs, then prints a
formatted prediction dashboard.

---

## Execution Modes

| Mode | Command | Description |
|---|---|---|
| `demo` | `python main.py --mode demo` | End-to-end demo with real yfinance data. No GPU required. |
| `download` | `python main.py --mode download` | Download and cache all data (daily from 2000, 15m/1h last 730d). |
| `train` | `python main.py --mode train` | Full walk-forward training → ensemble checkpoint. |
| `backtest` | `python main.py --mode backtest` | Walk-forward evaluation + equity curve plot. |
| `live` | `python main.py --mode live` | Live predictor (runs 09:15–15:30 IST, Mon–Fri). |
| `predict` | `python main.py --mode predict` | One-shot inference on latest cached data. |

### Optional Flags

```bash
python main.py --mode train \
    --device cuda \
    --ensemble-size 7 \
    --log-level DEBUG
```

---

## Architecture In Detail

### 1. Four Bi-LSTM Encoders

Each timeframe is encoded independently by a bidirectional LSTM stack
followed by multi-head self-attention:

```
Input → Bi-LSTM (n_layers) → Multi-Head Self-Attention → Context vector
```

Look-back windows are calibrated to capture behaviorally meaningful periods:
- **15m**: 96 bars = 24 hours of intraday microstructure
- **1h**: 48 bars = 2 days of session-level momentum
- **1d**: 252 bars = 1 full trading year of trend + seasonality
- **1w**: 52 bars = 1 calendar year of macro cycle

### 2. Hierarchical Cross-Timeframe Gating

Weekly context gates daily encodings; daily gates hourly; hourly gates 15-minute.
This enforces the economic intuition that short-term signals are only meaningful
in the context of the prevailing longer-term regime.

```
G_15m = σ(W_h · H_1h) ⊙ H_15m     # 1h gates 15m
G_1h  = σ(W_d · H_1d) ⊙ H_1h      # 1d gates 1h
G_1d  = σ(W_w · H_1w) ⊙ H_1d      # 1w gates 1d
```

### 3. Market Regime Classifier

A 5-state softmax head over the fused cross-timeframe representation:

| Regime | Typical conditions |
|---|---|
| Strong Bull | High positive momentum, expanding range, low VIX |
| Weak Bull | Positive drift, narrowing range, moderate VIX |
| Ranging | Mean-reverting, low directional bias, moderate volume |
| Weak Bear | Negative drift, elevated volatility, FII outflows |
| Strong Bear | Fast downside, VIX spike, broad sell-off |

The decoder is conditioned on regime probabilities so the direction, magnitude,
and confidence outputs adapt to the prevailing market character.

### 4. Three Output Heads

| Head | Output | Loss |
|---|---|---|
| Direction | 3-class softmax (Bear / Flat / Bull) | Cross-entropy with class weights |
| Magnitude | Scalar (normalised return %) | Huber regression |
| Confidence | Scalar ∈ (0, 1) | Binary CE vs empirical correctness |

Loss weights are learned via Kendall-Gal multi-task uncertainty weighting,
so no manual balancing is required.

### 5. Ensemble + MC Dropout Uncertainty

- **N=5** independently initialised models, each trained on the same data.
- At inference time, each model performs **30 MC Dropout forward passes**
  with dropout enabled to sample from the approximate posterior.
- Predictions are averaged across all `5 × 30 = 150` stochastic passes.
- The standard deviation of predicted magnitudes gives ±σ% uncertainty.

### 6. Retest Engine

The RetestEngine maintains a stateful registry of active signals:

```
Signal lifecycle:

   PENDING → TRACKING → CONFIRMED → COMPLETED
                  ↓
            INVALIDATED (stop-loss hit or retest fail)
```

Every 15-minute candle is scored as a retest. Signals accumulate
pass/fail counts that drive the displayed confidence and the automated
exit recommendation.

### 7. Live Predictor

Fires `on_candle_close()` at `:00`, `:15`, `:30`, `:45` each hour
using the `schedule` library. Execution is guarded by `is_market_hours()`
which enforces NSE trading hours (Mon–Fri, 09:15–15:30 IST).

Each cycle runs the full pipeline:
1. Fetch fresh OHLCV data for all four timeframes
2. Recompute features
3. Fetch news published in the last 2 hours
4. Run FinBERT / VADER sentiment scoring
5. Build model input tensors
6. Run ensemble inference with MC Dropout
7. Update the retest engine
8. Print the prediction dashboard
9. Append one row to the daily CSV log

---

## Performance Expectations

> These are realistic estimates based on walk-forward backtesting.
> Past performance does not guarantee future results.
> This system is a research tool — not financial advice.

| Metric | Expected range |
|---|---|
| Directional accuracy (OOS) | 54% – 63% |
| Magnitude MAE | 0.15% – 0.35% |
| Sharpe ratio (backtest) | 0.6 – 1.4 |
| Max drawdown (backtest) | 8% – 22% |
| Win rate (signal-to-exit) | 52% – 60% |

**Why not higher?** Nifty 50 is a semi-strong efficient market with highly
informed institutional participants. A 58% directional accuracy sustained
over 3+ years of walk-forward OOS periods is a meaningful edge. Claims of
90%+ accuracy should be treated with extreme scepticism — they virtually
always reflect look-ahead bias or overfitting.

---

## Data Sources

| Source | Data | Access method |
|---|---|---|
| Yahoo Finance (`yfinance`) | Nifty 50 OHLCV (daily + intraday), India VIX | `yfinance.download()` |
| NSE India (web) | FII / DII net flows, circuit breakers | HTTP + BeautifulSoup |
| Economic Times RSS | Market news for sentiment | `feedparser` |
| Moneycontrol RSS | Market news for sentiment | `feedparser` |
| Google News RSS | Broad Nifty / FII query results | `feedparser` |
| RBI press releases | Monetary policy announcements | HTTP + BeautifulSoup |

All data is cached to disk with a 24-hour TTL so repeated runs do not
re-fetch unchanged data.

---

## Configuration Reference

All tuneable parameters live in `config.py`:

```python
class Config:
    # Data
    TICKER              = "^NSEI"          # Yahoo Finance symbol for Nifty 50
    VIX_TICKER          = "^INDIAVIX"
    START_DATE_DAILY    = "2000-01-01"
    INTRADAY_PERIOD     = "730d"           # yfinance limit for sub-daily data

    # Sequence lengths (look-back windows)
    SEQ_15M             = 96              # 24 h of 15-minute candles
    SEQ_1H              = 48             # 2 days of hourly candles
    SEQ_1D              = 252            # 1 trading year of daily candles
    SEQ_1W              = 52             # 1 calendar year of weekly candles

    # Architecture
    LSTM_HIDDEN_15M     = 256
    LSTM_HIDDEN_1H      = 192
    LSTM_HIDDEN_1D      = 128
    LSTM_HIDDEN_1W      = 64
    LSTM_LAYERS_*       = 3 / 2 / 2 / 2
    DROPOUT             = 0.3
    ATTENTION_HEADS     = 8

    # Training
    BATCH_SIZE          = 64
    EPOCHS              = 150
    LEARNING_RATE       = 5e-4
    WEIGHT_DECAY        = 1e-5
    PATIENCE            = 20             # Early stopping epochs
    N_ENSEMBLE          = 5
    MC_DROPOUT_PASSES   = 50

    # Signal generation
    DIRECTION_THRESHOLD = 0.003          # Min return to classify as up/down
    CONFIDENCE_GATE     = 0.65           # Min confidence to emit a signal

    # Regime
    N_REGIMES           = 5
    REGIME_NAMES        = {0: "strong_bull", 1: "weak_bull",
                            2: "ranging", 3: "weak_bear", 4: "strong_bear"}

    # Walk-forward
    TRAIN_WINDOW_YEARS  = 5
    TEST_WINDOW_MONTHS  = 6
    N_FOLDS             = 8
```

---

## Live Trading Integration

> The system generates directional signals — it does not place orders.
> Integration with a broker API is the user's responsibility.

### Recommended Integration Pattern

```python
# In your broker integration script:
from inference.live_predictor import LivePredictor

class MyBrokerPredictor(LivePredictor):
    def on_candle_close(self):
        super().on_candle_close()    # runs full pipeline

    def print_signal_update(self, engine_output):
        super().print_signal_update(engine_output)   # prints dashboard

        # Extract signal
        direction   = engine_output.get("direction")
        confidence  = engine_output.get("confidence", 0)
        targets     = engine_output.get("targets", {})
        active_sigs = engine_output.get("active_signals", [])

        # Only act on high-confidence CONFIRMED signals
        for sig in active_sigs:
            if (sig.get("status") == "CONFIRMED"
                    and confidence > 0.70
                    and len(active_sigs) < 3):
                self._place_order(sig, targets)

    def _place_order(self, signal, targets):
        # Your broker API call here
        pass
```

### CSV Log Integration

Each prediction cycle appends one row to:
```
logs/predictions_YYYY-MM-DD.csv
```

Columns: `timestamp, direction, confidence, magnitude_pct, uncertainty_pct,
regime, spot_price, h1_target, d1_target, w1_target, stop_loss,
sentiment_score, sentiment_label, india_vix, fii_flow_cr`

This log can be consumed by any downstream system (Excel, Grafana, etc.)
for monitoring and post-trade analysis.

---

## Project Structure

```
nifty_hilstm_v2/
├── main.py                      ← Entry point (all modes)
├── config.py                    ← Central configuration
├── requirements.txt
├── README.md
│
├── data/
│   ├── data_collector.py        ← yfinance + NSE data fetching + caching
│   ├── feature_engineer.py      ← Technical analysis features (35+)
│   └── cache/                   ← Disk-cached pickles (auto-created)
│
├── models/
│   ├── architecture.py          ← AdaptiveHiLSTMv2, EnsembleModel
│   ├── losses.py                ← AdaptiveTradingLoss (4-component)
│   └── saved/                   ← Model checkpoints (auto-created)
│
├── training/
│   ├── trainer.py               ← ModelTrainer (single-model loop)
│   └── walk_forward.py          ← WalkForwardBacktester
│
├── engine/
│   └── retest_engine.py         ← RetestEngine (signal lifecycle)
│
├── sentiment/
│   ├── news_fetcher.py          ← RSS + web news collection
│   ├── sentiment_analyzer.py    ← FinBERT / VADER scoring
│   └── macro_signals.py         ← VIX regime, FII flows
│
├── inference/
│   └── live_predictor.py        ← LivePredictor (15m scheduler)
│
└── logs/                        ← Daily CSV prediction logs
```

---

## Development Notes

### Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=.
```

### Code Style

```bash
pip install black isort
black .
isort .
```

### Extending the Model

To add a new timeframe (e.g., monthly bars):
1. Add a `TemporalEncoder` in `architecture.py`
2. Add the corresponding `SEQ_1M` in `config.py`
3. Update `_fetch_all_timeframes()` in `live_predictor.py`
4. Update `MultiTimeframeDataset` in `training/trainer.py`

To add a new feature group:
1. Add a `_add_<group>_features()` method to `FeatureEngineer`
2. Call it from `compute_technical_features()`

---

## Disclaimer

This software is provided for educational and research purposes only.
It does not constitute financial advice. Trading financial instruments
involves substantial risk of loss. The authors accept no liability for
any losses incurred by use of this system.
