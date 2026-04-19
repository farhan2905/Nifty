"""
Nifty 50 Hi-LSTM v2 — Sentiment & Macro Intelligence Module
============================================================
Subpackage providing:
  - NewsFeeder        : RSS + RBI scraping, caching
  - SentimentAnalyzer : FinBERT / VADER scorer + aggregation
  - MacroSignalExtractor : VIX regime, FII flow, options signals, global corr
"""

from .news_fetcher import NewsArticle, NewsFeeder
from .sentiment_analyzer import SentimentScore, SentimentAnalyzer
from .macro_signals import MacroSignalExtractor

__all__ = [
    "NewsArticle",
    "NewsFeeder",
    "SentimentScore",
    "SentimentAnalyzer",
    "MacroSignalExtractor",
]
