"""HAR-RV (Heterogeneous AutoRegressive Realized Volatility) pre-check for gold
H1 -- scanned on TRAIN+VAL ONLY before any wiring (same discipline as COT/FVG).

HAR-RV is the classical multi-timescale volatility benchmark (Corsi 2009): it
decomposes volatility into daily / weekly / monthly realized-vol components, so
it captures the LONG-MEMORY vol structure that the model's short-window atr_pct
(14 bars) misses. Because the modality ablation showed price/technicals already
carry the magnitude signal, the real question is INCREMENTAL: do the longer HAR
timescales add magnitude skill BEYOND atr_pct? (partial correlation, the same
decider used for COT.)

Causal features at bar t (rolling realized vol over trailing windows, r known at
t): har_rv_d (1 day, 24 bars), har_rv_w (1 week, 120), har_rv_m (~1 month, 480),
plus short/long regime ratios. Target = |cumulative 10-bar move| (the model's
magnitude target). Also reports direction, expected to be dead.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from config import DATA_CFG

df = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
close = df["close"].to_numpy(float)
logc = np.log(close)
r = np.diff(logc, prepend=logc[0])
r2 = r ** 2
N = len(df)


def rv(w):
    return np.sqrt(pd.Series(r2).rolling(w, min_periods=max(2, w // 4)).sum().to_numpy())


har_d, har_w, har_m = rv(24), rv(120), rv(480)
feats = {
    "har_rv_d": har_d,
    "har_rv_w": har_w,
    "har_rv_m": har_m,
    "har_ratio_d_m": har_d / np.where(har_m > 0, har_m, np.nan),   # short/long regime
    "har_ratio_w_m": har_w / np.where(har_m > 0, har_m, np.nan),
}
atr = df["atr_pct"].to_numpy(float) if "atr_pct" in df.columns else None

K = DATA_CFG.horizon
fut = np.full(N, np.nan); fut[:-K] = logc[K:] - logc[:-K]
mag = np.abs(fut)
cut = int(N * (DATA_CFG.train_frac + DATA_CFG.val_frac))
base = float((fut[:cut] > 0).mean())


def partial_spearman(u, v, ctrl):
    ru, rv_, rc = rankdata(u), rankdata(v), rankdata(ctrl)
    def resid(t, c):
        c1 = np.c_[np.ones_like(c), c]
        return t - c1 @ np.linalg.lstsq(c1, t, rcond=None)[0]
    return float(np.corrcoef(resid(ru, rc), resid(rv_, rc))[0, 1])


print(f"HAR-RV pre-check on gold H1 -- TRAIN+VAL only ({cut:,} of {N:,} bars), base P(up)={base:.3f}")
if atr is not None:
    m = np.isfinite(atr[:cut]) & np.isfinite(mag[:cut])
    print(f"reference: atr_pct vs |move| rho = {spearmanr(atr[:cut][m], mag[:cut][m]).statistic:+.3f}\n")
print(f"{'feature':16s}{'|rho| DIR':>11s}{'rho MAG':>10s}{'MAG | atr_pct':>15s}{'quint |mv| rel':>15s}")
for name, x in feats.items():
    md = np.isfinite(x[:cut]) & np.isfinite(fut[:cut])
    rho_dir = abs(spearmanr(x[:cut][md], fut[:cut][md]).statistic)
    rho_mag = spearmanr(x[:cut][md], mag[:cut][md]).statistic
    part = partial_spearman(x[:cut][md], mag[:cut][md], atr[:cut][md]) if atr is not None else float("nan")
    try:
        q = pd.qcut(pd.Series(x[:cut][md]), 5, labels=False, duplicates="drop")
        mv = pd.Series(mag[:cut][md]).groupby(q).mean()
        rel = float((mv.max() - mv.min()) / mag[:cut][md].mean())
    except Exception:
        rel = float("nan")
    print(f"{name:16s}{rho_dir:>11.3f}{rho_mag:>+10.3f}{part:>+15.3f}{rel:>15.2f}")

print("\nread: MAG rho = standalone magnitude signal; 'MAG | atr_pct' = INCREMENTAL")
print("signal beyond the atr_pct the model already has (the decider). A HAR")
print("timescale earns a wiring test only if its partial rho clears ~0.05 -- i.e.")
print("it adds long-memory vol structure atr_pct misses. DIR is expected ~noise.")
