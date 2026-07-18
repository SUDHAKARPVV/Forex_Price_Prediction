"""Candidate-indicator scan for gold H1 -- run on TRAIN+VAL ONLY (test untouched).

Question: beyond the 37 shipped features, do standard additional indicators
carry directional information worth wiring in? Candidates: time-of-day and
day-of-week seasonality (the model has no clock), ADX-14 trend strength,
candle shape (range %, close position in range).

MEASURED VERDICT (2026-07-19, 52.7k train+val bars): none clears noise level.
    candidate            |spearman|   quintile P(up) spread   (base 0.516)
    hour_sin/cos          0.009-0.011      0.017-0.021
    dow_sin/cos           0.014-0.016      0.023-0.029
    range_pct             0.005            0.010
    close_pos_in_range    0.014            0.027
    adx14                 0.013            0.010
For reference the strongest shipped candidate, env_dev20, measured ~0.07
spearman with a ~1.5pp extreme-quintile deviation -- itself weak. Adding these
would be feature bloat without evidence; the binding constraint at H1 remains
the absence of extractable directional signal, not feature coverage.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_CFG

df = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv", parse_dates=["date"])
b = pd.read_csv("exports/pairs/XAUUSD/XAUUSD_H1.CSV", sep="\t")
b.columns = [c.strip("<>").lower() for c in b.columns]
b["dt"] = pd.to_datetime(b["date"] + " " + b["time"], format="%Y.%m.%d %H:%M:%S")
b = b.set_index("dt").reindex(pd.DatetimeIndex(df["date"]))
close, high, low = b["close"].to_numpy(float), b["high"].to_numpy(float), b["low"].to_numpy(float)
logc = np.log(close)
K = DATA_CFG.horizon
fut = np.full(len(df), np.nan)
fut[:-K] = logc[K:] - logc[:-K]
cut = int(len(df) * (DATA_CFG.train_frac + DATA_CFG.val_frac))

hr = pd.DatetimeIndex(df["date"]).hour.to_numpy()
dow = pd.DatetimeIndex(df["date"]).dayofweek.to_numpy()
rng = (high - low) / close
close_pos = np.where(high > low, (close - low) / (high - low), 0.5)
tr_ = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)


def _sm(x, n=14):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


pdi = 100 * _sm(plus_dm) / (_sm(tr_) + 1e-9)
mdi = 100 * _sm(minus_dm) / (_sm(tr_) + 1e-9)
adx = pd.Series(100 * abs(pdi - mdi) / (pdi + mdi + 1e-9)).rolling(14, min_periods=14).mean().to_numpy()

cands = {"hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
         "dow_sin": np.sin(2 * np.pi * dow / 5), "dow_cos": np.cos(2 * np.pi * dow / 5),
         "range_pct": rng, "close_pos_in_range": close_pos, "adx14": adx}
base = float((fut[:cut] > 0).mean())
print(f"candidate scan, TRAIN+VAL only ({cut:,} bars, base P(up) {base:.3f})")
print(f"{'candidate':20s} {'|spearman|':>10s} {'quintile P(up) spread':>22s}")
for n_, x in cands.items():
    m = np.isfinite(x[:cut]) & np.isfinite(fut[:cut])
    rho = abs(spearmanr(x[:cut][m], fut[:cut][m]).statistic)
    q = pd.qcut(pd.Series(x[:cut][m]), 5, labels=False, duplicates="drop")
    pu = pd.Series((fut[:cut][m] > 0)).groupby(q).mean()
    print(f"{n_:20s} {rho:10.4f} {pu.max() - pu.min():18.3f}")
