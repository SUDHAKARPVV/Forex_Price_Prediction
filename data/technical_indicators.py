"""
Technical indicator stream (Section 3.1.1, first bullet).

Computes RSI, MACD, Bollinger Bands, and rolling-volume features directly
from OHLC price history, using pandas/numpy only (no extra heavy TA
dependency needed, keeping the project self-contained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid.bfill(), width.fillna(0.0)


def stochastic_k(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic oscillator %K: where the close sits inside the trailing
    high-low range -- a bounded momentum/mean-reversion gauge that
    complements RSI (RSI looks at close-to-close changes, %K at range
    position)."""
    low_min = ohlc["low"].rolling(period, min_periods=period).min()
    high_max = ohlc["high"].rolling(period, min_periods=period).max()
    k = (ohlc["close"] - low_min) / (high_max - low_min + 1e-12)
    return k.clip(0, 1).fillna(0.5)


def compute_technical_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Return a 12-column technical feature block:
    [open, high, low, close (log-returns), RSI, MACD hist, BB width,
     volume z-score, ATR%, ROC-10, Stochastic %K, EMA12/26 ratio]

    Uses log-returns rather than simple/arithmetic returns (pct_change) for
    the OHLC-derived columns, for consistency with the model's prediction
    target, which is a cumulative LOG-return (see data/dataset.py). Log and
    arithmetic returns are nearly identical at small magnitudes, but using
    the same transform on both sides keeps the input and target on a
    theoretically consistent (additive, time-summable) scale.

    The last four indicators were added after the daily-scale round to
    strengthen momentum/volatility coverage at intraday resolution:
      ATR%      -- Average True Range normalised by price: pure volatility
                   level, scale-free across price regimes.
      ROC-10    -- 10-bar rate of change: direct momentum, the quantity a
                   drift-following baseline implicitly exploits.
      Stoch %K  -- close's position inside the 14-bar high-low range.
      EMA ratio -- log(EMA12/EMA26): smoothed trend direction/strength,
                   the state variable behind the MACD histogram.
    """
    close = ohlc["close"]

    ret_o = np.log(ohlc["open"] / ohlc["open"].shift(1)).fillna(0.0)
    ret_h = np.log(ohlc["high"] / ohlc["high"].shift(1)).fillna(0.0)
    ret_l = np.log(ohlc["low"] / ohlc["low"].shift(1)).fillna(0.0)
    ret_c = np.log(close / close.shift(1)).fillna(0.0)

    rsi_vals = rsi(close) / 100.0  # scale to [0,1]
    _, _, macd_hist = macd(close)
    macd_hist = (macd_hist / close).fillna(0.0)  # normalise by price level
    _, bb_width = bollinger_bands(close)

    vol = ohlc["volume"]
    vol_z = (vol - vol.rolling(20, min_periods=1).mean()) / (vol.rolling(20, min_periods=1).std() + 1e-6)
    vol_z = vol_z.fillna(0.0)

    atr_pct = (average_true_range(ohlc) / close.replace(0, np.nan)).fillna(0.0)
    roc_10 = np.log(close / close.shift(10)).fillna(0.0)
    stoch = stochastic_k(ohlc)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_ratio = np.log(ema12 / ema26).fillna(0.0)

    feats = pd.DataFrame(
        {
            "ret_open": ret_o,
            "ret_high": ret_h,
            "ret_low": ret_l,
            "ret_close": ret_c,
            "rsi": rsi_vals,
            "macd_hist": macd_hist,
            "bb_width": bb_width,
            "volume_z": vol_z,
            "atr_pct": atr_pct,
            "roc_10": roc_10,
            "stoch_k": stoch,
            "ema_ratio": ema_ratio,
        },
        index=ohlc.index,
    )
    return feats


def realized_volatility(close: pd.Series, window: int = 10) -> pd.Series:
    """Rolling realised volatility, used by the regime detector (Section 3.1.5)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=1).std().fillna(0.0)


def average_true_range(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()
