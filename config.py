"""
config.py
=========
System-wide configuration for the Nifty 50 Hi-LSTM v2 prediction system.

All hyper-parameters, paths, and tunable constants are centralised here so
every downstream module can ``from config import Config`` without duplicating
magic numbers.
"""

import os


class Config:
    """Central configuration container.

    Attributes are grouped into logical sections:

    * **Paths** – filesystem locations for cache, models, logs.
    * **Data** – tickers, date ranges, timeframe periods.
    * **Sequences** – look-back window lengths per timeframe.
    * **Architecture** – LSTM layer/hidden-unit settings, dropout, attention.
    * **Training** – batch size, epochs, learning rate, ensemble size.
    * **Targets** – direction threshold, confidence gate.
    * **Retest engine** – polling interval and alignment tolerance.
    * **News / Sentiment** – RSS feed URLs and FinBERT model name.
    * **Regime** – HMM / K-Means regime labels.
    * **Walk-forward** – rolling-window back-test settings.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data", "cache")
    MODEL_DIR: str = os.path.join(BASE_DIR, "models", "saved")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    REPORT_DIR: str = os.path.join(BASE_DIR, "reports")
    RL_STATE_FILE: str = os.path.join(LOG_DIR, "rl_feedback_state.json")
    FEEDBACK_HISTORY_LIMIT: int = 500

    # ------------------------------------------------------------------
    # Data settings
    # ------------------------------------------------------------------
    TICKER: str = "^NSEI"
    VIX_TICKER: str = "^INDIAVIX"
    START_DATE_DAILY: str = "2000-01-01"
    INTRADAY_PERIOD: str = "730d"          # yfinance max for sub-daily

    # Supported intraday intervals
    INTRADAY_INTERVALS: tuple = ("15m", "1h")
    N_FEATURES: int = 92
    ENABLE_MULTI_HORIZON: bool = True

    # ------------------------------------------------------------------
    # Sequence lengths  (look-back windows per timeframe)
    # ------------------------------------------------------------------
    SEQ_15M: int = 96    # 24 h of 15-minute candles  (96 × 15 min = 24 h)
    SEQ_1H: int = 48     # 48 h of hourly candles
    SEQ_1D: int = 252    # ≈ 1 trading year of daily candles
    SEQ_1W: int = 52     # 1 calendar year of weekly candles

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    LSTM_HIDDEN_15M: int = 256
    LSTM_HIDDEN_1H: int = 192
    LSTM_HIDDEN_1D: int = 128
    LSTM_HIDDEN_1W: int = 64

    LSTM_LAYERS_15M: int = 3
    LSTM_LAYERS_1H: int = 2
    LSTM_LAYERS_1D: int = 2
    LSTM_LAYERS_1W: int = 2

    DROPOUT: float = 0.3
    ATTENTION_HEADS: int = 8

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 64
    EPOCHS: int = 150
    LEARNING_RATE: float = 0.0005
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 20          # Early-stopping patience (epochs)
    N_ENSEMBLE: int = 5         # Number of models in ensemble
    MC_DROPOUT_PASSES: int = 50  # Monte-Carlo dropout passes for uncertainty

    # ------------------------------------------------------------------
    # Targets / signal generation
    # ------------------------------------------------------------------
    # Minimum absolute return to classify as up/down (else "flat")
    DIRECTION_THRESHOLD: float = 0.003   # 0.3 %
    # Minimum ensemble confidence to emit a trading signal
    CONFIDENCE_GATE: float = 0.65

    # ------------------------------------------------------------------
    # Retest engine
    # ------------------------------------------------------------------
    RETEST_INTERVAL_MINUTES: int = 15
    # Maximum price distance for a retest to "confirm" a H1/D1/W1 level
    ALIGNMENT_TOLERANCE: float = 0.002   # 0.2 %

    # ------------------------------------------------------------------
    # News / Sentiment
    # ------------------------------------------------------------------
    NEWS_SOURCES: list = [
        "https://economictimes.indiatimes.com/markets/stocks/rss",
        "https://www.moneycontrol.com/rss/marketstats.xml",
    ]
    # HuggingFace model ID for financial sentiment classification
    SENTIMENT_MODEL: str = "ProsusAI/finbert"

    # ------------------------------------------------------------------
    # Market regime
    # ------------------------------------------------------------------
    REGIME_NAMES: dict = {
        0: "strong_bull",
        1: "weak_bull",
        2: "ranging",
        3: "weak_bear",
        4: "strong_bear",
    }
    N_REGIMES: int = 5

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------
    TRAIN_WINDOW_YEARS: int = 5   # Training window length
    TEST_WINDOW_MONTHS: int = 6   # Out-of-sample test window
    N_FOLDS: int = 8              # Number of walk-forward folds


# ---------------------------------------------------------------------------
# Bootstrap – create required directories on import
# ---------------------------------------------------------------------------
for _d in (Config.DATA_DIR, Config.MODEL_DIR, Config.LOG_DIR, Config.REPORT_DIR):
    os.makedirs(_d, exist_ok=True)
