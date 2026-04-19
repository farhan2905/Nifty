"""
NewsFeeder: Fetches and processes market news from multiple sources.

Sources:
  1. RSS feeds : Economic Times Markets, Moneycontrol Market Stats
  2. NSE announcements RSS
  3. RBI press-release page (rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx)
  4. Google News RSS for "Nifty 50" and "India stock market" / FII flow queries
  5. RBI official RSS (rbi.org.in/scripts/rss.aspx)

Caching strategy:
  - Articles are persisted to JSONL files, one per calendar day, inside cache_dir.
  - Deduplication is done by URL-based SHA-256 hash so re-fetching is safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """Represents a single fetched news article.

    Fields
    ------
    title           : Headline text
    description     : Short body / summary
    url             : Canonical article URL
    source          : Feed / scraper name (e.g. "economic_times")
    published_at    : Publication datetime (UTC-aware)
    url_hash        : SHA-256 hex digest of the URL – used for dedup
    sentiment_score : Filled by SentimentAnalyzer; None until scored
    raw_tags        : Any tag/category strings present in the feed entry
    """

    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    url_hash: str = field(init=False)
    sentiment_score: Optional[dict] = field(default=None, repr=False)
    raw_tags: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.url_hash = hashlib.sha256(self.url.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NewsArticle":
        d = dict(d)
        d["published_at"] = datetime.fromisoformat(d["published_at"])
        # url_hash is set by __post_init__; drop stale value if present
        d.pop("url_hash", None)
        return cls(**d)


# ---------------------------------------------------------------------------
# NewsFeeder
# ---------------------------------------------------------------------------

class NewsFeeder:
    """Fetches, deduplicates, caches, and retrieves market news articles.

    Parameters
    ----------
    cache_dir : str
        Root directory for JSONL cache files (one file per calendar day).
    request_timeout : int
        HTTP request timeout in seconds.
    user_agent : str
        User-agent header sent to external servers.
    """

    RSS_FEEDS: Dict[str, str] = {
        "economic_times": (
            "https://economictimes.indiatimes.com/markets/stocks/rss"
        ),
        "moneycontrol": (
            "https://www.moneycontrol.com/rss/marketstats.xml"
        ),
        "nse_announcements": (
            "https://www.nseindia.com/corporates/rss/circulars.xml"
        ),
        "google_nifty": (
            "https://news.google.com/rss/search"
            "?q=nifty+50+india+market&hl=en-IN&gl=IN&ceid=IN:en"
        ),
        "google_fii": (
            "https://news.google.com/rss/search"
            "?q=FII+India+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        ),
        "rbi": "https://rbi.org.in/scripts/rss.aspx",
    }

    RBI_PRESS_RELEASE_URL = (
        "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    )

    # Keywords that flag an article as RBI-policy-related when scraping
    _RBI_POLICY_KEYWORDS = frozenset(
        ["repo rate", "monetary policy", "mpc", "inflation", "reverse repo",
         "liquidity", "rate cut", "rate hike", "policy rate", "crr", "slr"]
    )

    def __init__(
        self,
        cache_dir: str = "data/cache/news",
        request_timeout: int = 15,
        user_agent: str = (
            "Mozilla/5.0 (compatible; NiftyHiLSTM/2.0; +https://github.com/)"
        ),
    ) -> None:
        self.cache_dir = cache_dir
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self, since_hours: int = 24) -> List[NewsArticle]:
        """Fetch from all RSS sources + RBI page, deduplicate, filter by age.

        Parameters
        ----------
        since_hours : int
            Only return articles published within the last N hours.

        Returns
        -------
        List[NewsArticle]
            Deduplicated list, newest first.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=since_hours)
        all_articles: List[NewsArticle] = []

        # Standard RSS feeds
        for feed_name, url in self.RSS_FEEDS.items():
            try:
                articles = self._parse_rss(feed_name, url)
                all_articles.extend(articles)
                logger.debug("  [%s] fetched %d articles", feed_name, len(articles))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch feed '%s': %s", feed_name, exc)

        # RBI press-release scraper
        try:
            rbi_articles = self.fetch_rbi_policy()
            all_articles.extend(rbi_articles)
            logger.debug("  [rbi_policy] fetched %d articles", len(rbi_articles))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch RBI policy page: %s", exc)

        # Deduplicate
        all_articles = self._deduplicate(all_articles)

        # Filter by age (timezone-aware comparison)
        filtered: List[NewsArticle] = []
        for art in all_articles:
            pub = art.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                filtered.append(art)

        # Sort newest first
        filtered.sort(key=lambda a: a.published_at, reverse=True)
        return filtered

    def fetch_rbi_policy(self) -> List[NewsArticle]:
        """Scrape the RBI press-release listing page.

        Returns
        -------
        List[NewsArticle]
            Articles whose title / description contains a policy keyword.
        """
        articles: List[NewsArticle] = []
        try:
            resp = self.session.get(
                self.RBI_PRESS_RELEASE_URL,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("RBI policy page request failed: %s", exc)
            return articles

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tablebg"}) or soup.find("table")

        if table is None:
            logger.warning("RBI policy page: no table found, falling back to link scan")
            rows_data = self._rbi_fallback_parse(soup)
        else:
            rows_data = self._rbi_table_parse(table)

        for item in rows_data:
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            date_str = item.get("date", "")

            if not title or not url:
                continue

            # Only keep policy-relevant items
            lower = title.lower()
            if not any(kw in lower for kw in self._RBI_POLICY_KEYWORDS):
                continue

            published_at = self._parse_date_flexible(date_str)

            articles.append(
                NewsArticle(
                    title=title,
                    description=title,  # RBI listing has no separate summary
                    url=url if url.startswith("http") else f"https://rbi.org.in{url}",
                    source="rbi_policy",
                    published_at=published_at,
                )
            )

        return articles

    def get_news_at_timestamp(
        self,
        timestamp: datetime,
        window_hours: int = 4,
    ) -> List[NewsArticle]:
        """Retrieve cached articles published within *window_hours* before *timestamp*.

        Useful for backtesting: loads only cached JSONL files that can
        overlap with the requested window.

        Parameters
        ----------
        timestamp : datetime
            Reference point (inclusive upper bound).
        window_hours : int
            How far back to look (default 4 h).

        Returns
        -------
        List[NewsArticle]
            Articles whose published_at falls in [timestamp-window, timestamp].
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        lower_bound = timestamp - timedelta(hours=window_hours)
        results: List[NewsArticle] = []

        # Determine which calendar dates we need to load
        dates_to_check = set()
        current = lower_bound.date()
        while current <= timestamp.date():
            dates_to_check.add(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        for date_str in sorted(dates_to_check):
            for art in self.load_cached_news(date_str):
                pub = art.published_at
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if lower_bound <= pub <= timestamp:
                    results.append(art)

        results.sort(key=lambda a: a.published_at, reverse=True)
        return results

    def cache_news(self, articles: List[NewsArticle]) -> None:
        """Append articles to per-day JSONL files.

        Each line is a JSON object.  Existing hashes for the same day are
        loaded first so we never write duplicates to disk.

        Parameters
        ----------
        articles : List[NewsArticle]
            Articles to persist.
        """
        # Group by date
        by_date: Dict[str, List[NewsArticle]] = {}
        for art in articles:
            pub = art.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            key = pub.strftime("%Y-%m-%d")
            by_date.setdefault(key, []).append(art)

        for date_str, day_articles in by_date.items():
            existing_hashes = {
                a.url_hash for a in self.load_cached_news(date_str)
            }
            path = self._cache_path(date_str)
            with open(path, "a", encoding="utf-8") as fh:
                for art in day_articles:
                    if art.url_hash not in existing_hashes:
                        fh.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
                        existing_hashes.add(art.url_hash)

    def load_cached_news(self, date: str) -> List[NewsArticle]:
        """Load all cached articles for a given calendar date.

        Parameters
        ----------
        date : str
            Date string in YYYY-MM-DD format.

        Returns
        -------
        List[NewsArticle]
            Empty list if no cache file exists for that date.
        """
        path = self._cache_path(date)
        if not os.path.exists(path):
            return []

        articles: List[NewsArticle] = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    articles.append(NewsArticle.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning(
                        "Cache %s line %d parse error: %s", path, lineno, exc
                    )
        return articles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_rss(self, feed_name: str, url: str) -> List[NewsArticle]:
        """Parse a single RSS/Atom feed.

        feedparser is used for parsing; requests is used for fetching so we
        can send a proper User-Agent (many feeds reject the default agent).

        Parameters
        ----------
        feed_name : str
            Logical name of the feed (stored in NewsArticle.source).
        url : str
            Full RSS URL.

        Returns
        -------
        List[NewsArticle]
        """
        articles: List[NewsArticle] = []
        try:
            resp = self.session.get(url, timeout=self.request_timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except requests.RequestException:
            # Fall back to letting feedparser do the HTTP call
            logger.debug("requests failed for %s, falling back to feedparser", feed_name)
            feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.warning(
                "Feed '%s' bozo error: %s", feed_name, feed.get("bozo_exception", "")
            )
            return articles

        for entry in feed.entries:
            title = self._clean_text(getattr(entry, "title", ""))
            description = self._clean_text(
                getattr(entry, "summary", "") or getattr(entry, "description", "")
            )
            link = getattr(entry, "link", "") or getattr(entry, "id", "")

            if not title or not link:
                continue

            published_at = self._entry_published(entry)
            tags = [t.get("term", "") for t in getattr(entry, "tags", []) if t.get("term")]

            articles.append(
                NewsArticle(
                    title=title,
                    description=description,
                    url=link,
                    source=feed_name,
                    published_at=published_at,
                    raw_tags=tags,
                )
            )

        return articles

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles by URL hash, keeping the earliest occurrence.

        Parameters
        ----------
        articles : List[NewsArticle]

        Returns
        -------
        List[NewsArticle]
            Unique articles preserving original order.
        """
        seen: set = set()
        unique: List[NewsArticle] = []
        for art in articles:
            if art.url_hash not in seen:
                seen.add(art.url_hash)
                unique.append(art)
        return unique

    def _cache_path(self, date: str) -> str:
        return os.path.join(self.cache_dir, f"news_{date}.jsonl")

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_published(entry) -> datetime:
        """Extract publication datetime from a feedparser entry.

        Attempts ``published_parsed`` then ``updated_parsed``; falls back to
        *now* (UTC) if neither is present.
        """
        for attr in ("published_parsed", "updated_parsed"):
            val = getattr(entry, attr, None)
            if val:
                try:
                    return datetime(*val[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _parse_date_flexible(date_str: str) -> datetime:
        """Best-effort parse of date strings found on the RBI website.

        Tries common Indian financial-site date formats before falling back to now.
        """
        formats = [
            "%B %d, %Y",   # January 01, 2024
            "%b %d, %Y",   # Jan 01, 2024
            "%d %B %Y",    # 01 January 2024
            "%d-%b-%Y",    # 01-Jan-2024
            "%d/%m/%Y",    # 01/01/2024
            "%Y-%m-%d",    # ISO
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip HTML tags and normalise whitespace."""
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.get_text(separator=" ").split())

    @staticmethod
    def _rbi_table_parse(table) -> List[dict]:
        """Extract rows from the RBI press-release HTML table."""
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            link_tag = cells[-1].find("a") if cells else None
            if link_tag is None:
                # Try any cell
                for cell in cells:
                    link_tag = cell.find("a")
                    if link_tag:
                        break
            if link_tag is None:
                continue
            title = link_tag.get_text(strip=True)
            href = link_tag.get("href", "")
            date_text = cells[0].get_text(strip=True) if cells else ""
            rows.append({"title": title, "url": href, "date": date_text})
        return rows

    @staticmethod
    def _rbi_fallback_parse(soup: BeautifulSoup) -> List[dict]:
        """Fallback: scan all <a> tags on the RBI page for press-release links."""
        rows = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # RBI press-release links contain 'PressRelease' or 'Notification'
            if "PressRelease" in href or "Notification" in href:
                rows.append(
                    {"title": a.get_text(strip=True), "url": href, "date": ""}
                )
        return rows
