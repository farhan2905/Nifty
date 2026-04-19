"""
data/data_collector.py
======================
Comprehensive data collection module for the Nifty 50 Hi-LSTM v2 system.

Classes
-------
NiftyDataCollector
    Fetches OHLCV data (daily + intraday), India VIX, FII/DII activity,
    and options snapshots via yfinance and public web sources.  All results
    are cached on disk with a 24-hour TTL.

Dependencies (install via pip)
------------------------------
    pip install yfinance pandas requests lxml beautifulsoup4
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required: pip install yfinance") from exc

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NSEI_TICKER = "^NSEI"
_VIX_TICKER = "^INDIAVIX"
_NSE_FII_DII_URL = "https://www.nseindia.com/reports/fii-dii"
_CACHE_MAX_AGE_SECONDS = 86_400  # 24 hours


class NiftyDataCollector:
    """Collect, cache, and merge all data feeds required by Hi-LSTM v2.

    Parameters
    ----------
    cache_dir : str
        Directory used for on-disk pickle cache.  Created automatically if it
        does not exist.

    Examples
    --------
    >>> collector = NiftyDataCollector()
    >>> daily_df = collector.fetch_daily_data("2015-01-01", "2024-12-31")
    >>> tf_dict  = collector.merge_all_timeframes()
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("NiftyDataCollector initialised. Cache dir: %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_daily_data(
        self,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download daily OHLCV for ^NSEI from *start_date* to *end_date*.

        Parameters
        ----------
        start_date : str
            ISO-8601 start date, default ``"2000-01-01"``.
        end_date : str or None
            ISO-8601 end date; defaults to today.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, Open, High, Low, Close, Volume``.
            Index is a plain RangeIndex; ``Date`` column is ``datetime64[ns]``.
        """
        cache_key = f"daily_{start_date}_{end_date or 'now'}"
        cached = self.load_from_cache(cache_key)
        if cached is not None:
            return cached

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        logger.info("Fetching daily data for %s from %s to %s", _NSEI_TICKER, start_date, end_date)
        raw = yf.download(
            _NSEI_TICKER,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {_NSEI_TICKER}")

        df = self._standardise_ohlcv(raw)
        self.save_to_cache(df, cache_key)
        return df

    def fetch_intraday_data(
        self,
        interval: str = "15m",
        period: str = "730d",
    ) -> pd.DataFrame:
        """Download intraday OHLCV for ^NSEI.

        Parameters
        ----------
        interval : str
            Candle resolution – ``"15m"`` or ``"1h"``.
        period : str
            Look-back period string accepted by yfinance, e.g. ``"60d"``,
            ``"730d"``.  Note that yfinance caps sub-daily data at 730 days.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, Open, High, Low, Close, Volume``.
        """
        if interval not in ("15m", "1h"):
            raise ValueError(f"interval must be '15m' or '1h', got '{interval}'")

        cache_key = f"intraday_{interval}_{period}"
        cached = self.load_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("Fetching %s intraday data (period=%s)", interval, period)
        raw = yf.download(
            _NSEI_TICKER,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"yfinance returned empty DataFrame for interval={interval}")

        df = self._standardise_ohlcv(raw)
        self.save_to_cache(df, cache_key)
        return df

    def fetch_fii_dii_data(self) -> pd.DataFrame:
        """Fetch NSE FII/DII net activity data.

        Attempts to retrieve the official NSE FII/DII report page.  On
        failure (network error, changed HTML structure, etc.) the method
        falls back to generating *synthetic* FII/DII proxy signals derived
        from price-volume divergence on the daily OHLCV series.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, FII_Net, DII_Net, FII_Long_Pct, FII_Short_Pct``.
        """
        cache_key = "fii_dii"
        cached = self.load_from_cache(cache_key)
        if cached is not None:
            return cached

        df = self._scrape_fii_dii_nse()
        if df is None or df.empty:
            logger.warning("NSE FII/DII scrape failed – using synthetic fallback")
            df = self._synthetic_fii_dii()

        self.save_to_cache(df, cache_key)
        return df

    def fetch_india_vix(
        self,
        start_date: str = "2007-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download India VIX (^INDIAVIX) daily data via yfinance.

        Parameters
        ----------
        start_date : str
            ISO-8601 start date.
        end_date : str or None
            ISO-8601 end date; defaults to today.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, VIX_Open, VIX_High, VIX_Low, VIX_Close``.
        """
        cache_key = f"vix_{start_date}_{end_date or 'now'}"
        cached = self.load_from_cache(cache_key)
        if cached is not None:
            return cached

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        logger.info("Fetching India VIX from %s to %s", start_date, end_date)
        raw = yf.download(
            _VIX_TICKER,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("India VIX data unavailable – returning empty DataFrame")
            return pd.DataFrame(columns=["Date", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"])

        # Flatten MultiIndex columns produced by yfinance ≥ 0.2
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()
        df.columns = [str(c) for c in df.columns]

        # Normalise datetime column name
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)

        rename_map = {
            c: f"VIX_{c}"
            for c in ("Open", "High", "Low", "Close", "Adj Close")
            if c in df.columns
        }
        df.rename(columns=rename_map, inplace=True)
        df.drop(columns=["Volume", "VIX_Adj Close"], errors="ignore", inplace=True)

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.save_to_cache(df, cache_key)
        return df

    def fetch_options_data(self) -> Dict[int, Dict[str, float]]:
        """Return a placeholder options chain structure.

        In production this would connect to a live NSE options feed.  Here
        it returns a mock structure suitable for pipeline testing.

        Returns
        -------
        dict
            ``{strike_price: {"call_oi": …, "put_oi": …,
                               "call_vol": …, "put_vol": …}}``
        """
        logger.info("fetch_options_data: returning mock options structure")

        # Derive ATM strike from latest daily close
        try:
            daily = self.fetch_daily_data(start_date="2024-01-01")
            atm = int(round(daily["Close"].iloc[-1] / 50) * 50)
        except Exception:
            atm = 22_000  # Fallback sentinel

        strikes = range(atm - 500, atm + 550, 50)
        rng = np.random.default_rng(seed=42)

        options_chain: Dict[int, Dict[str, float]] = {}
        for strike in strikes:
            moneyness = (strike - atm) / atm
            put_skew = max(0.1, 1.0 - 3 * moneyness)    # puts heavier on downside
            call_skew = max(0.1, 1.0 + 3 * moneyness)   # calls heavier on upside

            options_chain[strike] = {
                "call_oi": float(rng.integers(1_000, 200_000) * call_skew),
                "put_oi": float(rng.integers(1_000, 200_000) * put_skew),
                "call_vol": float(rng.integers(100, 50_000) * call_skew),
                "put_vol": float(rng.integers(100, 50_000) * put_skew),
                "call_iv": float(rng.uniform(0.10, 0.40)),
                "put_iv": float(rng.uniform(0.12, 0.45)),
            }

        return options_chain

    def merge_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """Fetch and align all timeframes into a single dictionary.

        Returns
        -------
        dict
            Keys: ``"15m"``, ``"1h"``, ``"1d"``, ``"1w"``.
            Each value is a ``pd.DataFrame`` with columns
            ``Date, Open, High, Low, Close, Volume``.

        Notes
        -----
        * Daily and weekly data extend back to 2000-01-01.
        * 15-minute and 1-hour data are limited to the last 730 days by the
          yfinance API.
        * Weekly data is resampled from the daily series (OHLCV-correct logic).
        """
        logger.info("Merging all timeframes …")

        tf: Dict[str, pd.DataFrame] = {}

        # --- Daily ---
        tf["1d"] = self.fetch_daily_data(start_date="2000-01-01")

        # --- Weekly (resampled from daily) ---
        tf["1w"] = self._resample_to_weekly(tf["1d"])

        # --- Intraday ---
        for interval in ("15m", "1h"):
            try:
                per = "60d" if interval == "15m" else "730d"
                tf[interval] = self.fetch_intraday_data(
                    interval=interval, period=per
                )
            except Exception as exc:
                logger.error("Could not fetch %s data: %s", interval, exc)
                tf[interval] = pd.DataFrame(
                    columns=["Date", "Open", "High", "Low", "Close", "Volume"]
                )

        logger.info(
            "Timeframe row counts – 15m: %d | 1h: %d | 1d: %d | 1w: %d",
            len(tf["15m"]),
            len(tf["1h"]),
            len(tf["1d"]),
            len(tf["1w"]),
        )
        return tf

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def save_to_cache(self, data: object, name: str) -> None:
        """Pickle *data* to ``<cache_dir>/<name>.pkl`` with a timestamp.

        Parameters
        ----------
        data : any pickle-able object
        name : str
            Cache key (used as filename stem).
        """
        payload = {"timestamp": time.time(), "data": data}
        path = self.cache_dir / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Cache written: %s", path)

    def load_from_cache(self, name: str) -> Optional[object]:
        """Load a previously cached object if it is younger than 24 hours.

        Parameters
        ----------
        name : str
            Cache key (filename stem).

        Returns
        -------
        object or None
            The cached object, or ``None`` if the cache is missing or stale.
        """
        path = self.cache_dir / f"{name}.pkl"
        if not path.exists():
            return None

        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            logger.warning("Failed to load cache %s: %s", path, exc)
            return None

        age = time.time() - payload.get("timestamp", 0)
        if age > _CACHE_MAX_AGE_SECONDS:
            logger.info("Cache stale (%.1f h) – will refresh: %s", age / 3600, name)
            return None

        logger.debug("Cache hit (%.1f h old): %s", age / 3600, name)
        return payload["data"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardise_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise a raw yfinance DataFrame to a clean OHLCV frame.

        Parameters
        ----------
        raw : pd.DataFrame
            Direct output of ``yf.download()``.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, Open, High, Low, Close, Volume``.
            All numeric columns are ``float64``; Volume is ``int64``-compatible
            but stored as ``float64`` to avoid downstream type issues.
        """
        # yfinance ≥ 0.2 may return MultiIndex columns when a single ticker
        # is downloaded – flatten them.
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()
        df.columns = [str(c) for c in df.columns]

        # Normalise the date/index column name
        for alias in ("Datetime", "index", "Date"):
            if alias in df.columns:
                df.rename(columns={alias: "Date"}, inplace=True)
                break

        # Drop adjusted-close if present (auto_adjust=True already bakes it in)
        df.drop(columns=["Adj Close"], errors="ignore", inplace=True)

        # Keep only the six canonical columns
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"yfinance data missing columns: {missing}")
        df = df[required].copy()

        # Strip timezone info for consistent arithmetic
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Cast OHLCV to float (Volume may be NaN for index data)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Drop rows where Close is entirely missing
        df.dropna(subset=["Close"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def _resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
        """Resample a daily OHLCV DataFrame to weekly (week-ending Friday).

        Parameters
        ----------
        daily_df : pd.DataFrame
            Must have a ``Date`` column and OHLCV columns.

        Returns
        -------
        pd.DataFrame
            Weekly OHLCV with same column schema as *daily_df*.
        """
        df = daily_df.copy()
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

        weekly = df.resample("W-FRI").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        weekly.dropna(subset=["Close"], inplace=True)
        weekly.reset_index(inplace=True)
        weekly.rename(columns={"index": "Date"}, errors="ignore", inplace=True)
        weekly.sort_values("Date", inplace=True)
        weekly.reset_index(drop=True, inplace=True)
        return weekly

    # ------------------------------------------------------------------
    # FII/DII helpers
    # ------------------------------------------------------------------

    def _scrape_fii_dii_nse(self) -> Optional[pd.DataFrame]:
        """Attempt to fetch FII/DII data from the NSE website.

        NSE rate-limits programmatic access heavily and periodically changes
        its page structure.  This method makes a best-effort attempt; callers
        should be prepared for a ``None`` return.

        Returns
        -------
        pd.DataFrame or None
        """
        if BeautifulSoup is None:
            logger.warning("beautifulsoup4 not installed – skipping NSE scrape")
            return None

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }

        session = requests.Session()
        try:
            # Step 1 – prime cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            # Step 2 – hit the FII/DII API endpoint
            api_url = "https://www.nseindia.com/api/fiidiiTradeReact"
            resp = session.get(api_url, headers=headers, timeout=15)
            resp.raise_for_status()
            records = resp.json()
        except Exception as exc:
            logger.warning("NSE FII/DII API request failed: %s", exc)
            return None

        rows = []
        for rec in records:
            try:
                rows.append(
                    {
                        "Date": pd.to_datetime(rec.get("date", rec.get("Date", ""))),
                        "FII_Net": float(str(rec.get("fiiNet", 0)).replace(",", "")),
                        "DII_Net": float(str(rec.get("diiNet", 0)).replace(",", "")),
                        "FII_Long_Pct": float(
                            str(rec.get("fiiLongPct", rec.get("fiiBuyPct", 50))).replace(",", "")
                        ),
                        "FII_Short_Pct": float(
                            str(rec.get("fiiShortPct", rec.get("fiiSellPct", 50))).replace(",", "")
                        ),
                    }
                )
            except (ValueError, KeyError):
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _synthetic_fii_dii(self) -> pd.DataFrame:
        """Generate synthetic FII/DII proxy signals from price-volume divergence.

        Algorithm
        ---------
        * Use daily OHLCV to compute:
          - Up-volume days  → proxy FII buying
          - Down-volume days → proxy FII selling
          - Price vs 20-day SMA direction as regime filter
        * DII is modelled as the inverse stabiliser (buys on FII outflows).

        Returns
        -------
        pd.DataFrame
            Columns: ``Date, FII_Net, DII_Net, FII_Long_Pct, FII_Short_Pct``.
        """
        logger.info("Generating synthetic FII/DII from price-volume divergence …")
        try:
            daily = self.fetch_daily_data(start_date="2000-01-01")
        except Exception as exc:
            logger.error("Cannot generate synthetic FII/DII – daily data unavailable: %s", exc)
            return pd.DataFrame(
                columns=["Date", "FII_Net", "DII_Net", "FII_Long_Pct", "FII_Short_Pct"]
            )

        df = daily.copy()
        df["ret"] = df["Close"].pct_change()
        df["vol_sma20"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_sma20"].replace(0, np.nan)
        df["sma20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["above_sma"] = (df["Close"] > df["sma20"]).astype(float)

        # FII net  ≈  vol_ratio × return × 10_000 (scaled to crore-like units)
        df["FII_Net"] = df["vol_ratio"] * df["ret"] * 10_000
        # DII net  ≈  counter-cyclical dampening
        df["DII_Net"] = -0.4 * df["FII_Net"] + np.random.default_rng(0).normal(
            0, 200, len(df)
        )

        # FII long/short pct  (sigmoid-like mapping from cumulative FII_Net)
        cum_fii = df["FII_Net"].rolling(20, min_periods=1).mean()
        df["FII_Long_Pct"] = 50 + 30 * np.tanh(cum_fii / 5_000)
        df["FII_Short_Pct"] = 100 - df["FII_Long_Pct"]

        result = df[["Date", "FII_Net", "DII_Net", "FII_Long_Pct", "FII_Short_Pct"]].copy()
        result.dropna(inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
