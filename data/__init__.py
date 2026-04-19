
"""
data/__init__.py
================
Data sub-package for the Nifty 50 Hi-LSTM v2 prediction system.

Exposes the primary classes for convenience, but degrades gracefully when
optional download dependencies such as yfinance are not available.
"""

try:  # Optional, because some environments do not have yfinance installed.
    from data.data_collector import NiftyDataCollector  # noqa: F401
except Exception:  # pragma: no cover - import-time fallback
    NiftyDataCollector = None  # type: ignore[assignment]

from data.feature_engineer import FeatureEngineer  # noqa: F401

__all__ = ["NiftyDataCollector", "FeatureEngineer"]
