"""
MacroSignalExtractor: Derives macro-level features that capture regime changes.

Signals provided
----------------
  VIX Regime          – 5-bucket categorical + one-hot encoding
  FII Flow Momentum   – rolling net flow, acceleration, price divergence,
                        DII defence signal
  Options Signals     – max-pain pull, PCR regime, IV skew, VIX percentile,
                        expected move (±1σ)
  Global Correlation  – rolling 21-day Pearson correlation between Nifty and
                        SGX Nifty, Dow/S&P500, USD/INR, Crude Oil

All methods are stateless; inputs are passed in and results returned as
pandas Series / DataFrames.  Nothing is fetched from the network – callers
supply the raw data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroSignalExtractor:
    """Extract macro features from pre-loaded price/flow series.

    All public methods are pure functions of their arguments.
    """

    # ------------------------------------------------------------------
    # VIX Regime
    # ------------------------------------------------------------------

    #: Bucket boundaries and labels (inclusive lower bound)
    VIX_BUCKETS = [
        (0.0, 12.0, "ultra_low"),      # complacency / stealth danger
        (12.0, 16.0, "normal"),
        (16.0, 20.0, "elevated"),       # caution
        (20.0, 25.0, "high_fear"),
        (25.0, float("inf"), "extreme_fear"),
    ]

    #: Ordered list of regime labels (used for one-hot encoding columns)
    VIX_REGIME_LABELS = [b[2] for b in VIX_BUCKETS]

    def fetch_india_vix_regime(self, vix_series: pd.Series) -> pd.DataFrame:
        """Map India VIX levels to named regimes and produce one-hot features.

        Parameters
        ----------
        vix_series : pd.Series
            India VIX time series (index = DatetimeIndex, values = float).

        Returns
        -------
        pd.DataFrame
            Columns:
              vix_regime        – string label (see VIX_BUCKETS)
              vix_regime_code   – integer code 0–4 (ordinal)
              vix_ultra_low     – one-hot: 1 when regime == "ultra_low"
              vix_normal        – one-hot
              vix_elevated      – one-hot
              vix_high_fear     – one-hot
              vix_extreme_fear  – one-hot
              vix_change_1d     – 1-period % change
              vix_z_score_21d   – z-score over trailing 21 periods
        """
        if vix_series.empty:
            raise ValueError("vix_series is empty.")

        def _label(v: float) -> str:
            for lo, hi, label in self.VIX_BUCKETS:
                if lo <= v < hi:
                    return label
            return "extreme_fear"

        regime = vix_series.apply(_label)
        code = regime.map({lbl: i for i, lbl in enumerate(self.VIX_REGIME_LABELS)})

        df = pd.DataFrame(index=vix_series.index)
        df["vix_regime"] = regime
        df["vix_regime_code"] = code.astype(int)

        # One-hot columns
        for lbl in self.VIX_REGIME_LABELS:
            df[f"vix_{lbl}"] = (regime == lbl).astype(int)

        # Change and z-score
        df["vix_change_1d"] = vix_series.pct_change(fill_method=None).round(4)
        rolling = vix_series.rolling(21, min_periods=5)
        df["vix_z_score_21d"] = (
            (vix_series - rolling.mean()) / rolling.std().replace(0, np.nan)
        ).round(4)

        return df

    # ------------------------------------------------------------------
    # FII Flow Momentum
    # ------------------------------------------------------------------

    def compute_fii_momentum(
        self, fii_df: pd.DataFrame, window: int = 5
    ) -> pd.DataFrame:
        """Compute institutional flow momentum features.

        Expected columns in *fii_df*:
          fii_net   – daily net FII flow (positive = buying, INR crore)
          dii_net   – daily net DII flow (positive = buying, INR crore)
          close     – Nifty 50 closing price on the same date

        Parameters
        ----------
        fii_df : pd.DataFrame
            Must contain at least ``fii_net``; ``dii_net`` and ``close`` are
            used when present.
        window : int
            Rolling window in trading days (default 5 = 1 week).

        Returns
        -------
        pd.DataFrame
            Original DataFrame plus new columns:

            fii_5d_sum
                Rolling *window*-day cumulative FII net flow.
                Positive → sustained buying pressure.
            fii_flow_acceleration
                First difference of fii_5d_sum: rate of change of momentum.
                Positive → flow increasing (accelerating into market).
            fii_flow_normalised
                fii_5d_sum normalised by its trailing 252-day standard
                deviation.  Comparable across time.
            fii_vs_price_divergence
                1 if FII 5-day sum < 0 AND Nifty 5-day return > 0
                (FII selling but price rising = distribution / bearish warning).
                -1 if FII buying but price falling = accumulation / support.
                0 otherwise.  Requires ``close`` column.
            dii_defense_signal
                1 when DII net > 2× its trailing 252-day mean AND
                FII net < 0 simultaneously.  Signals a support floor.
                Requires ``dii_net`` and ``fii_net``.
        """
        out = fii_df.copy()

        if "fii_net" not in out.columns:
            raise KeyError("fii_df must contain a 'fii_net' column.")

        fii = out["fii_net"]

        # Rolling sum
        out["fii_5d_sum"] = fii.rolling(window, min_periods=1).sum()

        # Flow acceleration (d(momentum)/dt)
        out["fii_flow_acceleration"] = out["fii_5d_sum"].diff()

        # Normalised by 252-day rolling std
        std_252 = fii.rolling(252, min_periods=20).std().replace(0, np.nan)
        out["fii_flow_normalised"] = (out["fii_5d_sum"] / std_252).round(4)

        # FII vs price divergence
        if "close" in out.columns:
            price_ret_5d = out["close"].pct_change(window, fill_method=None)
            fii_neg = out["fii_5d_sum"] < 0
            price_up = price_ret_5d > 0
            fii_pos = out["fii_5d_sum"] > 0
            price_down = price_ret_5d < 0

            conditions = [
                fii_neg & price_up,   # distribution
                fii_pos & price_down, # accumulation
            ]
            choices = [-1, 1]
            out["fii_vs_price_divergence"] = np.select(
                conditions, choices, default=0
            )
        else:
            logger.debug("'close' column absent – skipping fii_vs_price_divergence.")

        # DII defence signal
        if "dii_net" in out.columns:
            dii = out["dii_net"]
            dii_mean_252 = dii.rolling(252, min_periods=20).mean()
            dii_defence = (dii > 2 * dii_mean_252) & (fii < 0)
            out["dii_defense_signal"] = dii_defence.astype(int)
        else:
            logger.debug("'dii_net' column absent – skipping dii_defense_signal.")

        return out

    # ------------------------------------------------------------------
    # Options Market Signals
    # ------------------------------------------------------------------

    def compute_options_signals(
        self,
        pcr: float,
        max_pain: float,
        spot: float,
        vix: float,
        put_iv: Optional[float] = None,
        call_iv: Optional[float] = None,
        vix_history: Optional[pd.Series] = None,
    ) -> dict:
        """Derive options-market signals for a single point in time.

        Parameters
        ----------
        pcr : float
            Put-Call Ratio (open interest basis).
        max_pain : float
            Max-pain strike price (option-writer friendly level).
        spot : float
            Nifty 50 spot price.
        vix : float
            India VIX value.
        put_iv : float, optional
            ATM put implied volatility (annualised %).
        call_iv : float, optional
            ATM call implied volatility (annualised %).
        vix_history : pd.Series, optional
            Trailing 252 trading-day VIX series (including today) to compute
            the percentile rank.

        Returns
        -------
        dict
            Keys:
              max_pain_pull      – (max_pain - spot) / spot × 100  (%)
                                   Positive → spot below max pain (upward pull)
              pcr_regime         – "bearish" (<0.7) | "neutral" (0.7–1.1) | "bullish" (>1.1)
              pcr_value          – raw PCR
              iv_skew            – put_iv / call_iv; > 1 = fear of downside
              vix_percentile     – percentile rank of current VIX vs history (0–100)
              expected_move_pct  – ±1σ daily move = (VIX / √252) %
              expected_move_abs  – expected_move_pct × spot / 100  (index points)
        """
        # Max-pain magnetic pull
        max_pain_pull = round((max_pain - spot) / spot * 100, 4) if spot != 0 else 0.0

        # PCR regime
        if pcr < 0.7:
            pcr_regime = "bearish"
        elif pcr <= 1.1:
            pcr_regime = "neutral"
        else:
            pcr_regime = "bullish"

        # IV skew
        if put_iv is not None and call_iv is not None and call_iv != 0:
            iv_skew = round(put_iv / call_iv, 4)
        else:
            iv_skew = None

        # VIX percentile
        if vix_history is not None and len(vix_history) > 0:
            vix_pct = round(
                float((vix_history <= vix).mean() * 100), 2
            )
        else:
            vix_pct = None

        # Expected daily move
        trading_days = 252
        em_pct = round(vix / np.sqrt(trading_days), 4)  # 1σ daily % move
        em_abs = round(em_pct * spot / 100, 2)

        result = {
            "max_pain_pull": max_pain_pull,
            "pcr_regime": pcr_regime,
            "pcr_value": round(pcr, 4),
            "iv_skew": iv_skew,
            "vix_percentile": vix_pct,
            "expected_move_pct": em_pct,
            "expected_move_abs": em_abs,
        }
        return result

    # ------------------------------------------------------------------
    # Global Correlation
    # ------------------------------------------------------------------

    def compute_global_correlation(
        self,
        nifty: pd.Series,
        sgx_nifty: Optional[pd.Series] = None,
        dow: Optional[pd.Series] = None,
        dxy: Optional[pd.Series] = None,
        crude: Optional[pd.Series] = None,
        usdinr: Optional[pd.Series] = None,
        window: int = 21,
    ) -> pd.DataFrame:
        """Rolling correlation between Nifty 50 returns and global proxies.

        Correlations are computed on log-returns to minimise scale effects.
        A positive correlation with USD/INR (USDINR) is expected to be
        slightly negative in practice – weak rupee → outflows → lower Nifty.

        Parameters
        ----------
        nifty : pd.Series
            Nifty 50 price series (DatetimeIndex).
        sgx_nifty : pd.Series, optional
            SGX Nifty futures (pre-market Indian session signal).
        dow : pd.Series, optional
            DJIA or S&P 500 closing price.
        dxy : pd.Series, optional
            US Dollar Index.
        crude : pd.Series, optional
            Crude oil front-month futures (Brent or WTI, USD/barrel).
        usdinr : pd.Series, optional
            USD/INR spot rate.
        window : int
            Rolling window in trading days (default 21 ≈ 1 month).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex aligned to *nifty*. Columns added for each non-None input:
              corr_sgx_nifty  – rolling corr(nifty_ret, sgx_ret)
              corr_dow        – rolling corr(nifty_ret, dow_ret)
              corr_dxy        – rolling corr(nifty_ret, dxy_ret)
              corr_crude      – rolling corr(nifty_ret, crude_ret)
              corr_usdinr     – rolling corr(nifty_ret, usdinr_ret)
              global_risk_on  – composite score in [-1, 1]:
                                average of available correlations weighted by
                                sign-expected direction (positive for risk-on
                                assets, negative for haven/dollar).
        """
        if nifty.empty:
            raise ValueError("nifty series is empty.")

        nifty_ret = np.log(nifty / nifty.shift(1))
        df = pd.DataFrame(index=nifty.index)

        series_map = {
            "sgx_nifty": (sgx_nifty, +1),  # expected +corr
            "dow": (dow, +1),               # expected +corr
            "dxy": (dxy, -1),               # expected –corr (USD strength → outflows)
            "crude": (crude, +1),           # mild +corr (risk-on)
            "usdinr": (usdinr, -1),         # expected –corr
        }

        composite_parts: list = []
        composite_weights: list = []

        for col_suffix, (series, direction) in series_map.items():
            if series is None:
                continue
            # Align to nifty index
            aligned = series.reindex(nifty.index, method="ffill")
            ret = np.log(aligned / aligned.shift(1))
            col = f"corr_{col_suffix}"
            df[col] = nifty_ret.rolling(window, min_periods=max(5, window // 2)).corr(ret)
            df[col] = df[col].round(4)
            # For composite: flip sign for inverse-expected assets
            composite_parts.append(df[col] * direction)
            composite_weights.append(1)

        # Global risk-on composite
        if composite_parts:
            composite_df = pd.concat(composite_parts, axis=1)
            df["global_risk_on"] = composite_df.mean(axis=1).round(4)
        else:
            df["global_risk_on"] = np.nan

        # Additional derived: rolling beta to Dow
        if dow is not None:
            dow_ret = np.log(dow.reindex(nifty.index, method="ffill") /
                             dow.reindex(nifty.index, method="ffill").shift(1))
            dow_var = dow_ret.rolling(window, min_periods=max(5, window // 2)).var()
            nifty_dow_cov = nifty_ret.rolling(window, min_periods=max(5, window // 2)).cov(dow_ret)
            df["nifty_beta_dow"] = (nifty_dow_cov / dow_var.replace(0, np.nan)).round(4)

        return df

    # ------------------------------------------------------------------
    # Utility: build full macro feature matrix
    # ------------------------------------------------------------------

    def build_macro_feature_matrix(
        self,
        price_df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        fii_df: Optional[pd.DataFrame] = None,
        sgx_nifty: Optional[pd.Series] = None,
        dow: Optional[pd.Series] = None,
        dxy: Optional[pd.Series] = None,
        crude: Optional[pd.Series] = None,
        usdinr: Optional[pd.Series] = None,
        corr_window: int = 21,
        fii_window: int = 5,
    ) -> pd.DataFrame:
        """Convenience method: compute all available macro features and join them.

        Parameters
        ----------
        price_df : pd.DataFrame
            Must have ``close`` column and DatetimeIndex.
        vix_series : pd.Series, optional
        fii_df : pd.DataFrame, optional
            Must have at least ``fii_net``; optionally ``dii_net``.
            Index must be a DatetimeIndex.
        sgx_nifty, dow, dxy, crude, usdinr : pd.Series, optional
            Global price series passed to :meth:`compute_global_correlation`.
        corr_window : int
            Rolling window for correlation computation.
        fii_window : int
            Rolling window for FII momentum.

        Returns
        -------
        pd.DataFrame
            All macro features left-joined onto *price_df*'s index.
        """
        if "close" not in price_df.columns:
            raise KeyError("price_df must have a 'close' column.")

        result = price_df.copy()

        # 1. VIX regimes
        if vix_series is not None:
            try:
                vix_aligned = vix_series.reindex(price_df.index, method="ffill")
                vix_features = self.fetch_india_vix_regime(vix_aligned)
                result = result.join(vix_features, how="left", rsuffix="_vix")
            except Exception as exc:  # noqa: BLE001
                logger.warning("VIX regime computation failed: %s", exc)

        # 2. FII momentum
        if fii_df is not None:
            try:
                fii_aligned = fii_df.reindex(price_df.index, method="ffill")
                if "close" not in fii_aligned.columns:
                    fii_aligned["close"] = price_df["close"]
                fii_features = self.compute_fii_momentum(fii_aligned, window=fii_window)
                fii_cols = [
                    c for c in fii_features.columns
                    if c not in result.columns and c not in fii_df.columns
                ]
                result = result.join(fii_features[fii_cols], how="left")
            except Exception as exc:  # noqa: BLE001
                logger.warning("FII momentum computation failed: %s", exc)

        # 3. Global correlation
        try:
            corr_features = self.compute_global_correlation(
                nifty=price_df["close"],
                sgx_nifty=sgx_nifty,
                dow=dow,
                dxy=dxy,
                crude=crude,
                usdinr=usdinr,
                window=corr_window,
            )
            result = result.join(corr_features, how="left", rsuffix="_corr")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Global correlation computation failed: %s", exc)

        return result
