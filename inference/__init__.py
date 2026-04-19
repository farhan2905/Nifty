"""
Nifty 50 Hi-LSTM v2 — Inference Package
========================================
Provides:
  - LivePredictor : Real-time 15m candle-close prediction engine
"""

from .live_predictor import LivePredictor

__all__ = ["LivePredictor"]
