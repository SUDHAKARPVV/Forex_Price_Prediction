"""
Builds the fused 26-feature panel (technical + macro + sentiment streams,
Section 3.1.1) and slices it into sliding windows of length T=60 with a
k=10 multi-step-ahead target, for a single currency pair.

Supports two data sources:
    source="synthetic" (default) -- signal-linked synthetic OHLC/macro/news
        (see data/synthetic_data.py:generate_correlated_market). Use
        signal_strength=0.0 to reproduce the original pure-noise ablation.
    source="real" -- live XAU/USD 5-minute candles (Yahoo Finance) + FXStreet
        news (data/real_data_feed.py). Falls back to synthetic automatically,
        with a printed warning, if either feed is unreachable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_CFG
from data.synthetic_data import generate_correlated_market
from data.technical_indicators import compute_technical_features, realized_volatility, average_true_range
from data.sentiment import FinBERTSentimentScorer, build_sentiment_features
from data.real_data_feed import try_fetch_real_panel
from data.synthetic_data import generate_macro_stream as _synthetic_macro_stream


@dataclass
class FXPanel:
    features: np.ndarray       # (N, 26) fused feature matrix (RAW, not normalised -- see time_split)
    close: np.ndarray          # (N,) raw close price, for target construction
    realized_vol: np.ndarray   # (N,) rolling realised volatility, for regime labels
    atr: np.ndarray            # (N,) average true range, for regime labels
    dates: pd.DatetimeIndex
    feature_names: list
    source: str = "synthetic"  # "synthetic" or "real", for reporting/labelling


def _assemble_panel(ohlc: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame, source: str) -> FXPanel:
    tech = compute_technical_features(ohlc)
    scorer = FinBERTSentimentScorer()
    sentiment = build_sentiment_features(news, scorer)

    assert tech.shape[1] == DATA_CFG.n_technical_features
    assert macro.shape[1] == DATA_CFG.n_macro_features
    assert sentiment.shape[1] == DATA_CFG.n_sentiment_features

    fused = pd.concat([tech, macro, sentiment], axis=1)
    assert fused.shape[1] == DATA_CFG.n_total_features, (
        f"Expected {DATA_CFG.n_total_features} fused features, got {fused.shape[1]}"
    )
    fused = fused.fillna(0.0)

    rv = realized_volatility(ohlc["close"])
    atr = average_true_range(ohlc)

    return FXPanel(
        features=fused.values.astype(np.float32),  # RAW -- normalised later, train-only, in time_split
        close=ohlc["close"].values.astype(np.float32),
        realized_vol=rv.values.astype(np.float32),
        atr=atr.values.astype(np.float32),
        dates=ohlc.index,
        feature_names=list(fused.columns),
        source=source,
    )


def build_fx_panel(
    pair: str = "XAU/USD",
    n_days: int = 1500,
    seed: int = 42,
    source: str = "synthetic",
    signal_strength: float = 0.35,
    real_ticker: str = "GC=F",
    real_interval: str = "5m",
    real_count: int = 1000,
) -> FXPanel:
    """Assemble the full multi-modal panel for one currency pair.

    source="real": tries live Yahoo Finance / FXStreet feeds first; on any
    failure (as will happen in a network-restricted sandbox), prints a
    clear warning and falls back to the signal-linked synthetic generator
    so the pipeline always returns something runnable.
    """
    if source == "real":
        real = try_fetch_real_panel(ticker_symbol=real_ticker, interval=real_interval, count=real_count)
        if real is not None:
            ohlc = real["ohlc"]
            # No live macro feed was supplied alongside fxratefeed/fxnewsfeed,
            # so macro stays synthetic, aligned to the real price index.
            macro = _synthetic_macro_stream(ohlc.index, seed=seed + 1)
            news = real["news_aligned"]
            print(f"[data] Using LIVE data: {len(ohlc)} candles from Yahoo Finance, "
                  f"{real['n_raw_headlines']} raw headlines from FXStreet.")
            return _assemble_panel(ohlc, macro, news, source="real")
        else:
            warnings.warn(
                "Live rate/news feeds were unreachable (see warnings above) -- "
                "falling back to signal-linked synthetic data. This is expected "
                "in network-restricted environments; run on a machine with open "
                "internet access to use real data.",
                stacklevel=2,
            )
            print("[data] Falling back to SYNTHETIC data (live feeds unreachable).")

    ohlc, macro, news = generate_correlated_market(n_days=n_days, seed=seed, signal_strength=signal_strength)
    return _assemble_panel(ohlc, macro, news, source="synthetic")


class FXWindowDataset(Dataset):
    """Sliding-window dataset producing (X, y, regime_context) tuples.

    X: (T, 26)   input window of fused features
    y: (k,)      k-step-ahead *log-returns* of close price (multi-horizon target)
    regime_ctx: (2,) [realised_vol_at_origin, atr_at_origin] used to help
                supervise / sanity-check the regime detector
    """

    def __init__(self, panel: FXPanel, lookback: int = None, horizon: int = None, start: int = 0, end: int = None):
        self.lookback = lookback or DATA_CFG.lookback
        self.horizon = horizon or DATA_CFG.horizon
        self.panel = panel

        n = len(panel.close)
        end = end if end is not None else n
        log_close = np.log(panel.close)

        self.indices = []
        # origin t is the last index of the input window; target is t+1..t+k
        for t in range(max(start, self.lookback - 1), min(end, n - self.horizon) - 1):
            self.indices.append(t)

        self._log_close = log_close

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.panel.features[t - self.lookback + 1 : t + 1]  # (T, 26)
        future = self._log_close[t + 1 : t + 1 + self.horizon]
        y = future - self._log_close[t]  # cumulative log-return targets, (k,)
        regime_ctx = np.array([self.panel.realized_vol[t], self.panel.atr[t]], dtype=np.float32)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(regime_ctx),
        )


def time_split(panel: FXPanel):
    """Chronological train/val/test split (no shuffling, to avoid leakage),
    with feature normalisation fit on the TRAIN split only and then applied
    to the full series -- avoids the val/test statistics leaking into the
    training distribution.
    """
    n = len(panel.close)
    train_end = int(n * DATA_CFG.train_frac)
    val_end = int(n * (DATA_CFG.train_frac + DATA_CFG.val_frac))

    train_mean = panel.features[:train_end].mean(axis=0)
    train_std = panel.features[:train_end].std(axis=0)
    # Guard for (near-)constant columns -- e.g. a one-hot trading-signal
    # class that never fires in the train split. Dividing by its ~0 std
    # would explode the feature to ~1e8 the first time it appears in
    # val/test; leaving such columns unscaled is the safe behaviour.
    train_std[train_std < 1e-6] = 1.0
    normalized = (panel.features - train_mean) / train_std

    norm_panel = FXPanel(
        features=normalized.astype(np.float32),
        close=panel.close,
        realized_vol=panel.realized_vol,
        atr=panel.atr,
        dates=panel.dates,
        feature_names=panel.feature_names,
        source=panel.source,
    )

    train_ds = FXWindowDataset(norm_panel, start=0, end=train_end)
    val_ds = FXWindowDataset(norm_panel, start=train_end, end=val_end)
    test_ds = FXWindowDataset(norm_panel, start=val_end, end=n)
    return train_ds, val_ds, test_ds
