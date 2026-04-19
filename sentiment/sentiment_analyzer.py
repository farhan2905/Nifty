"""
SentimentAnalyzer: Scores financial news using FinBERT (ProsusAI/finbert).

Falls back to VADER if the ``transformers`` library is not available or if the
model cannot be downloaded.

Features produced per article
------------------------------
  positive / negative / neutral : raw FinBERT/VADER probabilities (0–1)
  compound                       : weighted score in [-1, 1]
  event_type                     : dominant category string
  urgency                        : float in [0, 1] driven by keyword matches

Aggregated window features (from aggregate_for_window)
-------------------------------------------------------
  sentiment_compound       – urgency-weighted average compound score
  sentiment_std            – standard deviation (disagreement proxy)
  positive_ratio           – fraction of positive articles
  negative_ratio           – fraction of negative articles
  urgency_score            – max urgency in window
  event_flags              – one-hot dict keyed by event_type
  n_articles               – article count
  sentiment_momentum       – current compound minus previous-window compound
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps  (graceful degradation)
# ---------------------------------------------------------------------------

_FINBERT_AVAILABLE = False
_finbert_pipeline = None

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _FINBERT_AVAILABLE = True
except ImportError:
    logger.info("transformers not available – will use VADER for sentiment.")

_VADER_AVAILABLE = False
_vader_analyzer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSIA
    _VADER_AVAILABLE = True
except ImportError:
    logger.info("vaderSentiment not available – basic fallback scoring active.")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SentimentScore:
    """Sentiment scores and metadata for a single piece of text.

    Attributes
    ----------
    positive    : Probability of positive sentiment (0–1).
    negative    : Probability of negative sentiment (0–1).
    neutral     : Probability of neutral sentiment (0–1).
    compound    : Signed aggregate in [-1, 1]; positive = bullish.
    event_type  : Dominant event category detected by keyword matching.
                  One of: rbi_policy / fii_flow / earnings / global_macro /
                          options_expiry / domestic / other
    urgency     : Urgency score in [0, 1] derived from keyword severity buckets.
    model_used  : "finbert" | "vader" | "keyword_fallback"
    """

    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 1.0
    compound: float = 0.0
    event_type: str = "other"
    urgency: float = 0.0
    model_used: str = "keyword_fallback"

    def to_dict(self) -> dict:
        return {
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral, 4),
            "compound": round(self.compound, 4),
            "event_type": self.event_type,
            "urgency": round(self.urgency, 4),
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Scores financial text and aggregates scores into window-level features.

    Parameters
    ----------
    use_finbert : bool
        Attempt to load ProsusAI/finbert.  Set False to force VADER/fallback.
    finbert_model : str
        HuggingFace model ID for FinBERT (default: "ProsusAI/finbert").
    batch_size : int
        Batch size for FinBERT inference.
    max_length : int
        Token truncation length for FinBERT.
    """

    # ------------------------------------------------------------------
    # Class-level keyword dictionaries
    # ------------------------------------------------------------------

    URGENCY_KEYWORDS: Dict[str, List[str]] = {
        "high": [
            "crash", "circuit", "ban", "halt", "crisis", "war", "surge",
            "rally", "rate cut", "rate hike", "default", "bankruptcy",
            "suspension", "collapse", "panic", "meltdown", "record high",
            "record low", "historic",
        ],
        "medium": [
            "decline", "gain", "fall", "rise", "quarterly", "earnings",
            "revenue", "profit", "loss", "miss", "beat", "upgrade", "downgrade",
            "buyback", "dividend", "merger", "acquisition",
        ],
        "low": [
            "outlook", "guidance", "target", "estimate", "forecast",
            "expectation", "projection", "plan",
        ],
    }

    # Urgency weight per bucket
    _URGENCY_WEIGHTS: Dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.2}

    EVENT_PATTERNS: Dict[str, List[str]] = {
        "rbi_policy": [
            "RBI", "repo rate", "monetary policy", "MPC", "inflation target",
            "reverse repo", "CRR", "SLR", "liquidity", "policy rate",
        ],
        "fii_flow": [
            "FII", "FPI", "foreign institutional", "DII", "outflow", "inflow",
            "net buying", "net selling", "foreign portfolio",
        ],
        "earnings": [
            "Q1", "Q2", "Q3", "Q4", "quarterly", "earnings", "PAT", "revenue",
            "results", "net profit", "EBITDA", "margin",
        ],
        "global_macro": [
            "Fed", "Federal Reserve", "China", "US GDP", "oil", "crude",
            "dollar", "rupee", "FOMC", "ECB", "Bank of England", "inflation",
            "recession", "yield curve",
        ],
        "options_expiry": [
            "expiry", "rollover", "settlement", "F&O", "futures", "options",
            "open interest", "max pain", "PCR",
        ],
        "domestic": [
            "budget", "GST", "GDP", "IIP", "PMI", "trade deficit",
            "fiscal", "current account", "RBI data",
        ],
    }

    def __init__(
        self,
        use_finbert: bool = True,
        finbert_model: str = "ProsusAI/finbert",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self._model_name = "keyword_fallback"

        if use_finbert and _FINBERT_AVAILABLE:
            self._load_finbert(finbert_model)
        elif _VADER_AVAILABLE:
            self._load_vader()
        else:
            logger.warning(
                "Neither transformers nor vaderSentiment available. "
                "Using keyword-only scoring."
            )

        # Pre-compile urgency regex per bucket for speed
        self._urgency_patterns: Dict[str, re.Pattern] = {
            bucket: re.compile(
                "|".join(re.escape(kw) for kw in kws), re.IGNORECASE
            )
            for bucket, kws in self.URGENCY_KEYWORDS.items()
        }

        # Pre-compile event regex per type
        self._event_patterns: Dict[str, re.Pattern] = {
            evt: re.compile(
                "|".join(re.escape(p) for p in patterns), re.IGNORECASE
            )
            for evt, patterns in self.EVENT_PATTERNS.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentScore:
        """Score a single piece of text.

        Parameters
        ----------
        text : str
            Headline + description concatenated is ideal.

        Returns
        -------
        SentimentScore
        """
        if not text or not text.strip():
            return SentimentScore()

        scores = self._score_text(text)
        scores.event_type = self._detect_event_type(text)
        scores.urgency = self._compute_urgency(text)
        return scores

    def analyze_batch(self, articles: list) -> list:
        """Score a list of NewsArticle objects (or plain dicts with 'title'/'description').

        Mutates each article in-place by setting ``article.sentiment_score``
        and also returns the modified list.

        Parameters
        ----------
        articles : list
            List of NewsArticle instances or dicts.

        Returns
        -------
        list
            Same list with sentiment_score populated.
        """
        if not articles:
            return articles

        texts = []
        for art in articles:
            if hasattr(art, "title"):
                texts.append(f"{art.title}. {art.description or ''}")
            else:
                texts.append(f"{art.get('title', '')}. {art.get('description', '')}")

        if self._model_name == "finbert" and _finbert_pipeline is not None:
            raw_scores = self._finbert_batch(texts)
        elif self._model_name == "vader" and _vader_analyzer is not None:
            raw_scores = [self._vader_score(t) for t in texts]
        else:
            raw_scores = [self._keyword_score(t) for t in texts]

        for art, text, raw in zip(articles, texts, raw_scores):
            score = SentimentScore(
                positive=raw["positive"],
                negative=raw["negative"],
                neutral=raw["neutral"],
                compound=raw["compound"],
                model_used=self._model_name,
            )
            score.event_type = self._detect_event_type(text)
            score.urgency = self._compute_urgency(text)

            if hasattr(art, "sentiment_score"):
                art.sentiment_score = score.to_dict()
            else:
                art["sentiment_score"] = score.to_dict()

        return articles

    def aggregate_for_window(self, scored_articles: list) -> dict:
        """Aggregate a window of scored articles into a single feature vector.

        Parameters
        ----------
        scored_articles : list
            NewsArticle objects (or dicts) that already have ``sentiment_score``
            populated by :meth:`analyze_batch`.

        Returns
        -------
        dict
            Keys:
              sentiment_compound, sentiment_std,
              positive_ratio, negative_ratio,
              urgency_score, event_flags (dict),
              n_articles, sentiment_momentum (0.0 – needs prior window)
        """
        # Build a list of known event types for one-hot
        all_event_types = list(self.EVENT_PATTERNS.keys()) + ["other"]

        empty = {
            "sentiment_compound": 0.0,
            "sentiment_std": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "urgency_score": 0.0,
            "event_flags": {f"evt_{k}": 0 for k in all_event_types},
            "n_articles": 0,
            "sentiment_momentum": 0.0,
        }

        if not scored_articles:
            return empty

        compounds: List[float] = []
        urgencies: List[float] = []
        positives: List[float] = []
        negatives: List[float] = []
        event_counts: Dict[str, int] = {k: 0 for k in all_event_types}

        for art in scored_articles:
            sc = art.sentiment_score if hasattr(art, "sentiment_score") else art.get("sentiment_score")
            if sc is None:
                continue
            if isinstance(sc, SentimentScore):
                sc = sc.to_dict()

            compounds.append(sc.get("compound", 0.0))
            urgencies.append(sc.get("urgency", 0.0))
            positives.append(1.0 if sc.get("compound", 0.0) > 0.05 else 0.0)
            negatives.append(1.0 if sc.get("compound", 0.0) < -0.05 else 0.0)
            evt = sc.get("event_type", "other")
            if evt in event_counts:
                event_counts[evt] += 1
            else:
                event_counts["other"] += 1

        n = len(compounds)
        if n == 0:
            return empty

        # Urgency-weighted compound
        weights = np.array(urgencies) if urgencies else np.ones(n)
        weights = np.where(weights == 0, 0.1, weights)  # avoid zero weights
        weight_sum = weights.sum()
        weighted_compound = float(np.dot(weights, compounds) / weight_sum) if weight_sum else 0.0

        event_flags = {f"evt_{k}": int(v > 0) for k, v in event_counts.items()}

        return {
            "sentiment_compound": round(weighted_compound, 4),
            "sentiment_std": round(float(np.std(compounds)), 4),
            "positive_ratio": round(float(np.mean(positives)), 4),
            "negative_ratio": round(float(np.mean(negatives)), 4),
            "urgency_score": round(float(max(urgencies)) if urgencies else 0.0, 4),
            "event_flags": event_flags,
            "n_articles": n,
            "sentiment_momentum": 0.0,  # caller must diff consecutive windows
        }

    def create_sentiment_timeseries(
        self,
        news_cache_dir: str,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a sentiment feature DataFrame aligned to price_df's index.

        The function infers data frequency from the price DataFrame's index
        and applies an appropriate lookback window:

        =====================  ================
        Inferred frequency      Lookback window
        =====================  ================
        ≤ 15 minutes            2 hours
        ≤ 1 hour                4 hours
        Daily (or coarser)      24 hours
        =====================  ================

        Parameters
        ----------
        news_cache_dir : str
            Directory containing JSONL cache files (see :class:`NewsFeeder`).
        price_df : pd.DataFrame
            Must have a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Same index as ``price_df``.  Columns include all keys returned by
            :meth:`aggregate_for_window` (event_flags dict is flattened).
        """
        from .news_fetcher import NewsFeeder

        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df must have a DatetimeIndex.")

        # Infer lookback window
        freq_minutes = self._infer_freq_minutes(price_df)
        if freq_minutes <= 15:
            window_hours = 2
        elif freq_minutes <= 60:
            window_hours = 4
        else:
            window_hours = 24

        logger.info(
            "Sentiment timeseries: freq=%dmin  window=%dh  timestamps=%d",
            freq_minutes, window_hours, len(price_df),
        )

        feeder = NewsFeeder(cache_dir=news_cache_dir)
        rows: List[dict] = []
        prev_compound: Optional[float] = None

        for ts in price_df.index:
            ts_utc = ts if ts.tzinfo else ts.tz_localize("UTC")
            articles = feeder.get_news_at_timestamp(ts_utc, window_hours=window_hours)
            scored = self.analyze_batch(articles) if articles else []
            agg = self.aggregate_for_window(scored)

            # sentiment_momentum: compare to previous row
            if prev_compound is not None:
                agg["sentiment_momentum"] = round(
                    agg["sentiment_compound"] - prev_compound, 4
                )
            prev_compound = agg["sentiment_compound"]

            # Flatten event_flags
            flat = {k: v for k, v in agg.items() if k != "event_flags"}
            flat.update(agg.get("event_flags", {}))
            flat["timestamp"] = ts
            rows.append(flat)

        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index)
        return df

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_finbert(self, model_id: str) -> None:
        global _finbert_pipeline
        try:
            logger.info("Loading FinBERT from '%s' …", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=self.max_length,
                device=-1,  # CPU; change to 0 for GPU
            )
            self._model_name = "finbert"
            logger.info("FinBERT loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FinBERT load failed (%s). Trying VADER …", exc
            )
            self._load_vader()

    def _load_vader(self) -> None:
        global _vader_analyzer
        if _VADER_AVAILABLE:
            _vader_analyzer = _VaderSIA()
            self._model_name = "vader"
            logger.info("VADER loaded successfully.")
        else:
            logger.warning("VADER unavailable. Using keyword-only fallback.")
            self._model_name = "keyword_fallback"

    # ------------------------------------------------------------------
    # Private: scoring
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> SentimentScore:
        """Route to the appropriate backend scorer."""
        if self._model_name == "finbert" and _finbert_pipeline is not None:
            return self._finbert_score_single(text)
        elif self._model_name == "vader" and _vader_analyzer is not None:
            raw = self._vader_score(text)
            return SentimentScore(
                positive=raw["positive"],
                negative=raw["negative"],
                neutral=raw["neutral"],
                compound=raw["compound"],
                model_used="vader",
            )
        else:
            raw = self._keyword_score(text)
            return SentimentScore(
                positive=raw["positive"],
                negative=raw["negative"],
                neutral=raw["neutral"],
                compound=raw["compound"],
                model_used="keyword_fallback",
            )

    def _finbert_score_single(self, text: str) -> SentimentScore:
        """Score a single text with FinBERT."""
        try:
            results = _finbert_pipeline(text[:512])[0]
            label_map = {r["label"].lower(): r["score"] for r in results}
            pos = label_map.get("positive", 0.0)
            neg = label_map.get("negative", 0.0)
            neu = label_map.get("neutral", 0.0)
            compound = pos - neg  # maps to [-1, 1]
            return SentimentScore(
                positive=pos, negative=neg, neutral=neu,
                compound=compound, model_used="finbert"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("FinBERT single score failed: %s", exc)
            return SentimentScore()

    def _finbert_batch(self, texts: List[str]) -> List[dict]:
        """Batch-score texts with FinBERT."""
        results: List[dict] = []
        try:
            truncated = [t[:512] for t in texts]
            for i in range(0, len(truncated), self.batch_size):
                batch = truncated[i : i + self.batch_size]
                preds = _finbert_pipeline(batch)
                for pred in preds:
                    label_map = {r["label"].lower(): r["score"] for r in pred}
                    pos = label_map.get("positive", 0.0)
                    neg = label_map.get("negative", 0.0)
                    neu = label_map.get("neutral", 0.0)
                    results.append({
                        "positive": pos,
                        "negative": neg,
                        "neutral": neu,
                        "compound": pos - neg,
                    })
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT batch failed: %s. Falling back to VADER.", exc)
            results = [self._vader_score(t) if _VADER_AVAILABLE else self._keyword_score(t)
                       for t in texts]
        return results

    def _vader_score(self, text: str) -> dict:
        """Score text with VADER."""
        try:
            vs = _vader_analyzer.polarity_scores(text)
            return {
                "positive": vs["pos"],
                "negative": vs["neg"],
                "neutral": vs["neu"],
                "compound": vs["compound"],
            }
        except Exception:  # noqa: BLE001
            return self._keyword_score(text)

    def _keyword_score(self, text: str) -> dict:
        """Minimal keyword-based fallback scorer.

        Counts positive vs negative financial keywords and derives a compound.
        """
        POSITIVE_WORDS = frozenset([
            "gain", "rise", "surge", "rally", "up", "high", "buy", "bullish",
            "profit", "beat", "upgrade", "positive", "growth", "recovery",
        ])
        NEGATIVE_WORDS = frozenset([
            "fall", "drop", "crash", "decline", "down", "low", "sell",
            "bearish", "loss", "miss", "downgrade", "negative", "slowdown",
            "crisis",
        ])
        tokens = re.findall(r"\b\w+\b", text.lower())
        pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        pos = pos_count / total
        neg = neg_count / total
        compound = pos - neg
        neutral = max(0.0, 1.0 - pos - neg)
        return {
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neutral, 4),
            "compound": round(compound, 4),
        }

    # ------------------------------------------------------------------
    # Private: event type + urgency detection
    # ------------------------------------------------------------------

    def _detect_event_type(self, text: str) -> str:
        """Return the dominant event category for *text*.

        Pattern counts are compared; ties are broken by the order in
        EVENT_PATTERNS (earlier = higher priority).
        """
        best_type = "other"
        best_count = 0
        for evt, pattern in self._event_patterns.items():
            count = len(pattern.findall(text))
            if count > best_count:
                best_count = count
                best_type = evt
        return best_type

    def _compute_urgency(self, text: str) -> float:
        """Compute urgency score in [0, 1].

        High-urgency keywords push score toward 1; cumulative matches within a
        bucket are capped at 1 per bucket so a single keyword flood can't game
        the score.
        """
        total = 0.0
        max_possible = sum(self._URGENCY_WEIGHTS.values())  # 1.7

        for bucket, pattern in self._urgency_patterns.items():
            if pattern.search(text):
                total += self._URGENCY_WEIGHTS[bucket]

        return round(min(total / max_possible, 1.0), 4)

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_freq_minutes(df: pd.DataFrame) -> int:
        """Infer median interval between timestamps in minutes."""
        if len(df) < 2:
            return 1440  # assume daily
        diffs = pd.Series(df.index).diff().dropna()
        median_delta = diffs.median()
        return max(1, int(median_delta.total_seconds() / 60))
