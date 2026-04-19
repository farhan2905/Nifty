
"""
data/feature_engineer.py
========================
Production-grade feature engineering for Nifty Hi-LSTM v2.

This module builds a broad, leakage-aware technical feature set from raw
OHLCV data. It is designed to work on daily, hourly, and 15-minute bars.

Key properties:
- 78+ engineered features (this implementation produces 110+)
- No external TA library required
- Safe handling of NaN / inf / zero-division
- Compatible with the existing training and walk-forward pipeline
- Next-bar target_return column for supervised training

Expected input columns:
    Open, High, Low, Close, Volume

Optional:
    A DatetimeIndex or a Date column for calendar features.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------

def _safe_div(numer, denom):
    numer_is_series = isinstance(numer, pd.Series)
    denom_is_series = isinstance(denom, pd.Series)
    idx = None
    if numer_is_series:
        idx = numer.index
        numer = numer.to_numpy(dtype=float)
    else:
        numer = np.asarray(numer, dtype=float)
    if denom_is_series:
        idx = denom.index if idx is None else idx
        denom = denom.to_numpy(dtype=float)
    else:
        denom = np.asarray(denom, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=np.abs(denom) > 1e-12)
    if idx is not None:
        return pd.Series(out, index=idx)
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (series - mu) / sd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    apo = ema12 - ema26
    ppo = _safe_div(apo, ema26.replace(0, np.nan)) * 100.0
    return macd_line, macd_signal, macd_hist, apo, ppo


def _bollinger(close: pd.Series, window: int = 20, mult: float = 2.0):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower, std


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100 * _safe_div((hh - close), (hh - ll))


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    k_values = 100 * _safe_div((close - ll), (hh - ll))
    k = pd.Series(k_values, index=close.index)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=period).mean()
    return _safe_div(tp - sma, 0.015 * mad)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * _safe_div(pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), atr)
    minus_di = 100 * _safe_div(pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def _aroon(high: pd.Series, low: pd.Series, period: int = 25):
    def _calc_up(s):
        idx = s.rolling(period, min_periods=period).apply(lambda x: float(np.argmax(x[::-1])) if len(x) else np.nan, raw=True)
        return 100 * _safe_div((period - idx), period)

    def _calc_down(s):
        idx = s.rolling(period, min_periods=period).apply(lambda x: float(np.argmin(x[::-1])) if len(x) else np.nan, raw=True)
        return 100 * _safe_div((period - idx), period)

    return _calc_up(high), _calc_down(low)


def _ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series):
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    avg7 = _safe_div(bp.rolling(7, min_periods=7).sum(), tr.rolling(7, min_periods=7).sum())
    avg14 = _safe_div(bp.rolling(14, min_periods=14).sum(), tr.rolling(14, min_periods=14).sum())
    avg28 = _safe_div(bp.rolling(28, min_periods=28).sum(), tr.rolling(28, min_periods=28).sum())
    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7.0


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def _vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    return (volume * ret).cumsum()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    cum_tp_v = (typical * volume).cumsum()
    cum_v = volume.cumsum().replace(0, np.nan)
    return cum_tp_v / cum_v


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    pos = raw_mf.where(tp > tp.shift(1), 0.0)
    neg = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period, min_periods=period).sum()
    neg_sum = neg.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + _safe_div(pos_sum, neg_sum)))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    mf_mult = _safe_div(((close - low) - (high - close)), (high - low))
    mf_vol = mf_mult * volume
    return _safe_div(mf_vol.rolling(period, min_periods=period).sum(), volume.rolling(period, min_periods=period).sum())


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(_safe_div(high, low))
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.pow(2).rolling(window, min_periods=window).mean())


def _garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(_safe_div(high, low))
    log_co = np.log(_safe_div(close, open_))
    term = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    return np.sqrt(term.rolling(window, min_periods=window).mean().clip(lower=0))


def _rogers_satchell_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    ho = np.log(_safe_div(high, open_))
    lo = np.log(_safe_div(low, open_))
    hc = np.log(_safe_div(high, close))
    lc = np.log(_safe_div(low, close))
    term = (ho * hc + lo * lc)
    return np.sqrt(term.rolling(window, min_periods=window).mean().clip(lower=0))


def _downside_vol(log_ret: pd.Series, window: int = 20) -> pd.Series:
    neg = log_ret.where(log_ret < 0, 0.0)
    return np.sqrt((neg.pow(2).rolling(window, min_periods=window).mean()))


def _upside_vol(log_ret: pd.Series, window: int = 20) -> pd.Series:
    pos = log_ret.where(log_ret > 0, 0.0)
    return np.sqrt((pos.pow(2).rolling(window, min_periods=window).mean()))


def _adx_strength(adx: pd.Series) -> pd.Series:
    return adx / 100.0


@dataclass
class FeatureEngineer:
    """Compute a production-grade feature matrix from raw OHLCV bars."""

    target_col: str = "Close"
    add_target_return: bool = True
    clip_sigma: float = 5.0
    clip_window: int = 252
    include_calendar_features: bool = True

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def __post_init__(self):
        self.feature_columns_: List[str] = []
        self.feature_count_: int = 0
        self.feature_schema_: List[str] = []

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_copy(df)
        df = self._add_price_features(df)
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        if self.include_calendar_features:
            df = self._add_calendar_features(df)
        if self.add_target_return:
            df = self._add_target(df)
        df = self._clean(df)
        self._finalise_feature_metadata(df)
        return df

    def _validate_and_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            col_map = {c.lower(): c for c in df.columns}
            rename = {}
            still_missing = set()
            for col in missing:
                if col.lower() in col_map:
                    rename[col_map[col.lower()]] = col
                else:
                    still_missing.add(col)
            if rename:
                df = df.rename(columns=rename)
            if still_missing:
                raise ValueError(f"FeatureEngineer: missing required columns: {still_missing}")
        df = df.copy()
        for col in self.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index("Date", drop=False)
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
        prev_close = c.shift(1)

        df["log_return"] = np.log(c / prev_close)
        df["returns_1"] = c.pct_change(1)
        df["returns_2"] = c.pct_change(2)
        df["returns_3"] = c.pct_change(3)
        df["returns_5"] = c.pct_change(5)
        df["returns_10"] = c.pct_change(10)
        df["returns_20"] = c.pct_change(20)
        df["returns_60"] = c.pct_change(60)
        df["hl_range_pct"] = _safe_div(h - l, c)
        df["body_pct"] = _safe_div(c - o, c)
        df["upper_wick_pct"] = _safe_div(h - np.maximum(o, c), c)
        df["lower_wick_pct"] = _safe_div(np.minimum(o, c) - l, c)
        df["gap_pct"] = _safe_div(o - prev_close, prev_close)
        df["close_to_open_pct"] = _safe_div(c - o, o)
        df["close_vs_prev_close"] = _safe_div(c - prev_close, prev_close)
        df["overnight_return"] = _safe_div(o - prev_close, prev_close)
        df["intraday_return"] = _safe_div(c - o, o)
        df["range_expansion_5"] = _safe_div((h - l), (h - l).rolling(5, min_periods=5).mean())
        df["range_expansion_20"] = _safe_div((h - l), (h - l).rolling(20, min_periods=20).mean())
        df["close_to_high_pct"] = _safe_div(h - c, c)
        df["close_to_low_pct"] = _safe_div(c - l, c)
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        for w in (5, 10, 20, 50, 100, 200):
            df[f"sma_{w}"] = _sma(c, w)
            df[f"ema_{w}"] = _ema(c, w)
            df[f"price_vs_sma_{w}"] = _safe_div(c - df[f"sma_{w}"], df[f"sma_{w}"])
            df[f"price_vs_ema_{w}"] = _safe_div(c - df[f"ema_{w}"], df[f"ema_{w}"])

        for w in (20, 50, 200):
            df[f"sma_{w}_slope"] = _safe_div(df[f"sma_{w}"].diff(), c)
            df[f"ema_{w}_slope"] = _safe_div(df[f"ema_{w}"].diff(), c)

        df["ema_5_20_cross"] = _safe_div(df["ema_5"] - df["ema_20"], c)
        df["ema_20_50_cross"] = _safe_div(df["ema_20"] - df["ema_50"], c)
        df["sma_20_50_cross"] = _safe_div(df["sma_20"] - df["sma_50"], c)
        df["sma_50_200_cross"] = _safe_div(df["sma_50"] - df["sma_200"], c)
        df["trend_alignment"] = np.sign(df["ema_20_slope"].fillna(0)) + np.sign(df["ema_50_slope"].fillna(0)) + np.sign(df["sma_20_slope"].fillna(0))
        df["trend_strength"] = df["price_vs_sma_50"].abs()
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]

        df["rsi_7"] = _rsi(c, 7)
        df["rsi_14"] = _rsi(c, 14)
        df["rsi_21"] = _rsi(c, 21)

        macd_line, macd_signal, macd_hist, apo, ppo = _macd(c)
        df["macd_line"] = macd_line / c
        df["macd_signal"] = macd_signal / c
        df["macd_hist"] = macd_hist / c
        df["apo"] = apo / c
        df["ppo"] = ppo / 100.0

        for w in (5, 10, 20, 60):
            df[f"roc_{w}"] = c.pct_change(w)
            df[f"mom_{w}"] = c - c.shift(w)

        df["williams_r_14"] = _williams_r(h, l, c, 14) / 100.0
        df["stoch_k"], df["stoch_d"] = _stochastic(h, l, c, 14, 3)
        df["cci_20"] = _cci(h, l, c, 20) / 100.0
        df["adx_14"], df["plus_di_14"], df["minus_di_14"] = _adx(h, l, c, 14)
        df["adx_strength"] = _adx_strength(df["adx_14"])
        df["di_diff_14"] = _safe_div(df["plus_di_14"] - df["minus_di_14"], df["plus_di_14"] + df["minus_di_14"])
        df["aroon_up_25"], df["aroon_down_25"] = _aroon(h, l, 25)
        df["aroon_osc_25"] = (df["aroon_up_25"] - df["aroon_down_25"]) / 100.0
        df["ult_osc"] = _ultimate_oscillator(h, l, c) / 100.0
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        log_ret = df["log_return"]
        atr14 = _atr(h, l, c, 14)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)

        df["atr_14_pct"] = _safe_div(atr14, c)
        df["true_range_pct"] = _safe_div(tr, c)
        df["hist_vol_5"] = _rolling_std(log_ret, 5)
        df["hist_vol_20"] = _rolling_std(log_ret, 20)
        df["hist_vol_60"] = _rolling_std(log_ret, 60)
        df["vol_zscore_20"] = _rolling_zscore(tr, 20)

        bb_up, bb_mid, bb_lo, bb_std = _bollinger(c, 20, 2.0)
        bb_width = (bb_up - bb_lo)
        df["bb_width_pct"] = _safe_div(bb_width, c)
        df["bb_position"] = _safe_div(c - bb_lo, bb_width)
        df["bb_squeeze"] = _rolling_zscore(df["bb_width_pct"].rolling(10, min_periods=10).mean(), 20)

        df["realized_vol_20"] = np.sqrt((log_ret.pow(2)).rolling(20, min_periods=20).mean())
        df["realized_vol_60"] = np.sqrt((log_ret.pow(2)).rolling(60, min_periods=60).mean())
        df["parkinson_vol_20"] = _parkinson_vol(h, l, 20)
        df["garman_klass_vol_20"] = _garman_klass_vol(o, h, l, c, 20)
        df["rogers_satchell_vol_20"] = _rogers_satchell_vol(o, h, l, c, 20)
        df["downside_vol_20"] = _downside_vol(log_ret, 20)
        df["upside_vol_20"] = _upside_vol(log_ret, 20)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        h, l, c, v, o = df["High"], df["Low"], df["Close"], df["Volume"], df["Open"]
        dollar_volume = c * v
        vwap = _vwap(h, l, c, v)
        obv = _obv(c, v)
        vpt = _vpt(c, v)

        for w in (5, 20, 60):
            vol_ma = v.rolling(w, min_periods=w).mean().replace(0, np.nan)
            df[f"volume_ratio_{w}"] = _safe_div(v, vol_ma)

        df["volume_zscore_20"] = _rolling_zscore(v, 20)
        df["dollar_volume"] = dollar_volume / 1e9
        df["dollar_volume_zscore_20"] = _rolling_zscore(dollar_volume, 20)
        obv_ma = obv.rolling(50, min_periods=50).mean().replace(0, np.nan)
        df["obv_norm"] = _safe_div(obv, obv_ma)
        df["obv_slope_5"] = _safe_div(obv.diff(5), obv.abs().rolling(20, min_periods=20).mean().replace(0, np.nan))
        df["vwap_diff_pct"] = _safe_div(c - vwap, vwap)
        df["vwap_ratio"] = _safe_div(c, vwap)
        df["mfi_14"] = _mfi(h, l, c, v, 14) / 100.0
        df["cmf_20"] = _cmf(h, l, c, v, 20)
        df["vpt"] = vpt / 1e9
        df["pvt_slope_10"] = _safe_div(vpt.diff(10), vpt.abs().rolling(20, min_periods=20).mean().replace(0, np.nan))
        df["accumulation_distribution"] = _safe_div(((c - l) - (h - c)), (h - l)) * v
        df["price_volume_trend_norm"] = vpt / np.maximum(1.0, v.cumsum().astype(float))
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "Date" in df.columns:
            idx = pd.to_datetime(df["Date"], errors="coerce")
        else:
            idx = pd.date_range("2000-01-01", periods=len(df), freq="D")

        idx = pd.DatetimeIndex(idx)
        dow = idx.dayofweek.values
        month = idx.month.values
        quarter = idx.quarter.values
        dom = idx.day.values

        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
        df["quarter_sin"] = np.sin(2 * np.pi * (quarter - 1) / 4.0)
        df["quarter_cos"] = np.cos(2 * np.pi * (quarter - 1) / 4.0)
        df["day_of_month_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
        df["day_of_month_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
        df["is_month_end"] = idx.is_month_end.astype(float)
        df["is_month_start"] = idx.is_month_start.astype(float)
        df["is_quarter_end"] = idx.is_quarter_end.astype(float)
        df["is_quarter_start"] = idx.is_quarter_start.astype(float)
        df["is_week_end"] = (idx.dayofweek == 4).astype(float)
        df["is_week_start"] = (idx.dayofweek == 0).astype(float)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["target_return"] = np.log(df["Close"].shift(-1) / df["Close"])
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        preserve_cols = list(self.REQUIRED_COLUMNS)
        if self.add_target_return and "target_return" in df.columns:
            preserve_cols.append("target_return")

        feature_cols = [c for c in df.columns if c not in preserve_cols and pd.api.types.is_numeric_dtype(df[c])]
        feature_block = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()

        # Leak-safe imputation: only forward-fill from the past.
        feature_block = feature_block.ffill()

        # Leak-safe clipping: use a rolling window of prior observations only.
        if self.clip_sigma and self.clip_sigma > 0 and self.clip_window and self.clip_window > 5:
            for col in feature_cols:
                s = feature_block[col]
                mu = s.rolling(self.clip_window, min_periods=max(20, self.clip_window // 5)).mean().shift(1)
                sig = s.rolling(self.clip_window, min_periods=max(20, self.clip_window // 5)).std(ddof=0).shift(1).replace(0, np.nan)
                lo = mu - self.clip_sigma * sig
                hi = mu + self.clip_sigma * sig
                feature_block[col] = s.where((s >= lo) & (s <= hi), other=np.nan).ffill()

        df[feature_cols] = feature_block.fillna(0.0)
        df.fillna(0.0, inplace=True)
        return df

    def _finalise_feature_metadata(self, df: pd.DataFrame) -> None:
        numeric_cols = [c for c in df.columns if c not in self.REQUIRED_COLUMNS and c != "target_return" and pd.api.types.is_numeric_dtype(df[c])]
        self.feature_columns_ = numeric_cols
        self.feature_schema_ = list(numeric_cols)
        self.feature_count_ = len(numeric_cols)

    def feature_columns(self) -> List[str]:
        return list(self.feature_columns_)

    def feature_count(self) -> int:
        return int(self.feature_count_)

    def feature_schema(self) -> List[str]:
        return list(self.feature_schema_ or self.feature_columns_)

    def align_to_schema(self, df: pd.DataFrame, schema: Sequence[str]) -> pd.DataFrame:
        out = df.copy()
        schema = list(schema)
        for col in schema:
            if col not in out.columns:
                out[col] = 0.0
        return out.reindex(columns=schema, fill_value=0.0)
