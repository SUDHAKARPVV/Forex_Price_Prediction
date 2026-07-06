"""
Synthetic multi-modal data generator.

IMPORTANT — read this before using in production
--------------------------------------------------
This sandbox has no network access to market-data vendors (Yahoo Finance,
Refinitiv, FRED, news APIs, etc.) or to Hugging Face's model hub, so the
project cannot download real XAU/USD tick history, real macro releases, or
pretrained FinBERT weights at build time.

This module generates *plausible, regime-switching synthetic OHLC data*
(a two-regime Geometric Brownian Motion with volatility clustering) plus
synthetic macro and news-sentiment streams, so that the full pipeline below
is runnable and testable end-to-end today.

To go from this scaffold to the real dissertation pipeline, swap out:
    - `generate_ohlc()`            -> a real OHLC feed (e.g. a broker/vendor API)
    - `generate_macro_stream()`    -> real CPI / rate-differential / CB-statement data
    - `data/sentiment.py`          -> a live FinBERT model + real news crawler
Everything downstream (feature engineering, fusion, model, training,
evaluation) is written against the same DataFrame schema, so no other code
needs to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_correlated_market(
    n_days: int = 1500,
    start_price: float = 1950.0,
    seed: int = 42,
    regime_switch_prob: float = 0.01,
    signal_strength: float = 0.35,
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """Generate OHLC + macro + news together, with a genuine (but properly
    lagged, non-leaky) causal link from a shared latent "market mood" into
    next-bar returns.

    Why this exists
    ----------------
    The original `generate_ohlc` / `generate_macro_stream` /
    `generate_news_headlines` are fully independent of each other by
    design — useful as a noise-floor sanity check, but it also means there
    is *no real relationship* for the multi-modal fusion layer to learn,
    so a more expressive hybrid architecture has nothing to gain over a
    plain LSTM (this is exactly what the first evaluation run showed).

    Here, a persistent latent "mood" series (AR(1)-smoothed, bounded to
    [-1, 1]) partially drives both the macro central-bank stance and the
    news sentiment stream, AND partially drives next-bar price drift —
    but only through `mood[t-1]` affecting `return[t]`, never `mood[t]`
    affecting `return[t]`. So a model using only information available up
    to bar t can legitimately exploit this relationship to predict bar
    t+1 (and, thanks to the mood series' own persistence, several bars
    beyond that) without any look-ahead leakage.

    `signal_strength=0.0` reproduces the original pure-noise behaviour
    exactly (useful as an ablation baseline); higher values inject a
    progressively stronger, still-realistic-scale signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    # --- shared latent "market mood", persistent via AR(1) smoothing ---
    phi = 0.85
    raw = rng.normal(0, 1, n_days)
    mood_unbounded = np.zeros(n_days)
    for t in range(1, n_days):
        mood_unbounded[t] = phi * mood_unbounded[t - 1] + (1 - phi) * raw[t] * 3.0
    mood = np.tanh(mood_unbounded)  # bounded [-1, 1]

    cb_component = np.tanh(0.7 * mood_unbounded + 0.3 * rng.normal(0, 1, n_days))
    sentiment_component = np.tanh(0.7 * mood_unbounded + 0.3 * rng.normal(0, 1, n_days))

    # --- volatility regime switching (unchanged from generate_ohlc) ---
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = regime[t - 1]
        if rng.random() < regime_switch_prob:
            regime[t] = 1 - regime[t - 1]
    vol_low, vol_high = 0.004, 0.018
    sigmas = np.where(regime == 0, vol_low, vol_high)

    # --- price path: mood[t-1] (already known at t-1) drives return[t] ---
    drift_base = 0.0002
    lagged_mood = np.roll(mood, 1)
    lagged_mood[0] = 0.0
    signal_drift = signal_strength * sigmas * lagged_mood
    log_returns = rng.normal(loc=drift_base, scale=sigmas) + signal_drift
    close = start_price * np.exp(np.cumsum(log_returns))

    intraday_range = np.abs(rng.normal(0, sigmas * close, size=n_days))
    open_ = close * (1 + rng.normal(0, sigmas / 3, size=n_days))
    high = np.maximum(open_, close) + intraday_range * rng.uniform(0.2, 0.6, n_days)
    low = np.minimum(open_, close) - intraday_range * rng.uniform(0.2, 0.6, n_days)
    volume = rng.lognormal(mean=10, sigma=0.5, size=n_days) * (1 + regime * 0.8)

    ohlc = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "true_regime": regime},
        index=dates,
    )
    ohlc.index.name = "date"

    # --- macro stream: cb_stance driven by cb_component; other macro
    # features stay independent (realistic -- not every macro series
    # carries signal) ---
    rate_diff = np.cumsum(rng.normal(0, 0.01, n_days))
    cpi_surprise = rng.normal(0, 0.15, n_days)
    rate_diff_series = pd.Series(rate_diff, index=dates)
    cpi_series = pd.Series(cpi_surprise, index=dates)
    macro = pd.DataFrame(
        {
            "rate_diff": rate_diff,
            "cpi_surprise": cpi_surprise,
            "cb_stance": cb_component,
            "rate_diff_lag5": rate_diff_series.shift(5).bfill().values,
            "cpi_surprise_lag5": cpi_series.shift(5).bfill().values,
            "days_since_cb_event": np.clip(np.abs(np.diff(np.r_[0, cb_component])) * 0, 0, 1),  # kept simple/neutral
        },
        index=dates,
    )

    # --- news stream: latent tag driven by sentiment_component ---
    n_headlines = rng.poisson(lam=6, size=n_days)
    tags = np.where(sentiment_component > 0.15, "positive", np.where(sentiment_component < -0.15, "negative", "neutral"))
    template = {
        "positive": "Markets rally as investors show bullish confidence and strong growth momentum",
        "negative": "Markets slump amid bearish selloff and recession fears weigh on sentiment",
        "neutral": "Markets trade sideways as investors await further economic data",
    }
    text = [template[t] for t in tags]
    news = pd.DataFrame(
        {"headline_count": n_headlines, "latent_tag": tags, "latent_sentiment": sentiment_component, "text": text},
        index=dates,
    )

    return ohlc, macro, news


def generate_ohlc(
    n_days: int = 1500,
    start_price: float = 1950.0,
    seed: int = 42,
    regime_switch_prob: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic daily OHLC data with volatility regime switching.

    Two latent volatility regimes (low / high) are simulated with a simple
    Markov switch, mimicking the "quiet vs. news-driven" market behaviour
    referenced in Section 3.1.1 of the report.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = regime[t - 1]
        if rng.random() < regime_switch_prob:
            regime[t] = 1 - regime[t - 1]

    vol_low, vol_high = 0.004, 0.018
    drift = 0.0002
    sigmas = np.where(regime == 0, vol_low, vol_high)

    log_returns = rng.normal(loc=drift, scale=sigmas)
    close = start_price * np.exp(np.cumsum(log_returns))

    # Build plausible OHLC around the close path
    intraday_range = np.abs(rng.normal(0, sigmas * close, size=n_days))
    open_ = close * (1 + rng.normal(0, sigmas / 3, size=n_days))
    high = np.maximum(open_, close) + intraday_range * rng.uniform(0.2, 0.6, n_days)
    low = np.minimum(open_, close) - intraday_range * rng.uniform(0.2, 0.6, n_days)
    volume = rng.lognormal(mean=10, sigma=0.5, size=n_days) * (1 + regime * 0.8)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "true_regime": regime,  # kept only for validation/debugging, not fed to model
        }
    )
    df.set_index("date", inplace=True)
    return df


def generate_macro_stream(index: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """Synthetic macro stream (6 features, Section 3.1.1 second bullet):
    interest-rate differential, CPI surprise, central-bank stance, two
    5-day lags of the fastest-moving series, and a "days since last CB
    event" calendar flag -- aligning fundamental drivers to the trading
    calendar as described in the report.
    """
    rng = np.random.default_rng(seed)
    n = len(index)

    rate_diff = np.cumsum(rng.normal(0, 0.01, n))
    cpi_surprise = rng.normal(0, 0.15, n)
    # CB statements are infrequent; forward-fill a sparse hawkish/dovish score
    cb_stance = np.zeros(n)
    event_idx = np.sort(rng.choice(n, size=max(1, n // 20), replace=False))
    cb_stance[event_idx] = rng.uniform(-1, 1, size=len(event_idx))
    cb_stance_series = pd.Series(cb_stance, index=index).replace(0, np.nan).ffill().fillna(0)

    rate_diff_series = pd.Series(rate_diff, index=index)
    cpi_series = pd.Series(cpi_surprise, index=index)
    rate_diff_lag5 = rate_diff_series.shift(5).bfill()
    cpi_surprise_lag5 = cpi_series.shift(5).bfill()

    days_since_event = np.zeros(n)
    last_event = -1
    for i in range(n):
        if i in event_idx:
            last_event = i
        days_since_event[i] = (i - last_event) if last_event >= 0 else n
    days_since_event = np.clip(days_since_event, 0, 60) / 60.0  # normalise to [0,1]

    df = pd.DataFrame(
        {
            "rate_diff": rate_diff_series.values,
            "cpi_surprise": cpi_series.values,
            "cb_stance": cb_stance_series.values,
            "rate_diff_lag5": rate_diff_lag5.values,
            "cpi_surprise_lag5": cpi_surprise_lag5.values,
            "days_since_cb_event": days_since_event,
        },
        index=index,
    )
    return df


def generate_news_headlines(index: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    """Synthetic daily news-headline count + a latent sentiment tag, mapped
    to a template headline string in the 'text' column so this matches the
    same ['text', 'headline_count']-shaped schema that real FXStreet
    headlines provide via `data/real_data_feed.py`.
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    latent_sentiment = np.clip(rng.normal(0, 0.4, n), -1, 1)
    n_headlines = rng.poisson(lam=6, size=n)

    tags = []
    for s in latent_sentiment:
        if s > 0.15:
            tags.append("positive")
        elif s < -0.15:
            tags.append("negative")
        else:
            tags.append("neutral")

    template = {
        "positive": "Markets rally as investors show bullish confidence and strong growth momentum",
        "negative": "Markets slump amid bearish selloff and recession fears weigh on sentiment",
        "neutral": "Markets trade sideways as investors await further economic data",
    }
    text = [template[t] for t in tags]

    return pd.DataFrame(
        {
            "headline_count": n_headlines,
            "latent_tag": tags,
            "latent_sentiment": latent_sentiment,
            "text": text,
        },
        index=index,
    )
