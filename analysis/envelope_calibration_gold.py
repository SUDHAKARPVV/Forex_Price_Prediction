"""Two cheap, leak-free pre-checks on gold H1 (no torch, no retrain):

A) ENVELOPE deviation -- the user trades with MA envelopes (MA +/- k%). The
   panel has bb_width (band WIDTH = volatility) but no PRICE-POSITION-vs-band
   feature, so envelope deviation dev = close/SMA20 - 1 is genuinely new
   directional information *if* it predicts. Test: on the TEST split, the
   conditional future 10-bar return per train-calibrated dev quintile, plus a
   base-rate-controlled accuracy of the classic fade rule at the extremes.

B) DRIFT CALIBRATION -- the clean Hybrid scores 0.5126 vs always-up 0.534:
   BELOW the unconditional drift. That is a calibration problem, not (only) a
   signal problem: the model's sign threshold ignores that gold drifted up.
   Test: add the TRAIN-split mean cumulative return (a constant, leak-free)
   to the saved test predictions and re-score. If DirAcc rises to >= base
   rate, "model >= naive" is recoverable without any retraining.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd

from config import DATA_CFG

K = DATA_CFG.horizon
panel = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv", parse_dates=["date"])
close = panel["close"].to_numpy(float)
logc = np.log(close)
N = len(panel)
tr_end = int(N * DATA_CFG.train_frac)
te_start = int(N * (DATA_CFG.train_frac + DATA_CFG.val_frac))

# future K-bar cumulative log-return at each origin
fut = np.full(N, np.nan)
fut[: N - K] = logc[K:] - logc[: N - K]

# ---------------- A) envelope deviation ----------------
sma20 = pd.Series(close).rolling(20).mean().to_numpy()
dev = close / sma20 - 1.0                      # % above/below the 20-bar MA

test = np.zeros(N, bool); test[te_start: N - K] = True
m = test & np.isfinite(dev) & np.isfinite(fut)
qs = np.nanquantile(dev[:tr_end], [0.2, 0.4, 0.6, 0.8])   # train-calibrated bins
bins = np.digitize(dev, qs)

print("A) ENVELOPE deviation (close/SMA20 - 1) -> future 10-bar return, TEST split")
print(f"   {'quintile':>9s} {'n':>6s} {'mean fut ret':>13s} {'P(up)':>7s}")
for q in range(5):
    mq = m & (bins == q)
    print(f"   {q + 1:>9d} {mq.sum():6d} {np.nanmean(fut[mq]):+13.6f} {(fut[mq] > 0).mean():7.3f}")
base_up = (fut[m] > 0).mean()
lo, hi = m & (bins == 0), m & (bins == 4)
fade_lo = (fut[lo] > 0).mean()          # below lower envelope -> fade = expect UP
fade_hi = (fut[hi] < 0).mean()          # above upper envelope -> fade = expect DOWN
print(f"   test base rate P(up) = {base_up:.3f}")
print(f"   fade rule: below-envelope P(up) {fade_lo:.3f} (vs {base_up:.3f}) | "
      f"above-envelope P(down) {fade_hi:.3f} (vs {1 - base_up:.3f})")
print(f"   corr(dev, fut) on test: {np.corrcoef(dev[m], fut[m])[0, 1]:+.4f}")

# ---------------- B) drift calibration of the saved predictions ----------------
print("\nB) DRIFT CALIBRATION of the saved Hybrid test predictions (no retrain)")
dfp = pd.read_csv("results/predictions_h1_XAUUSD.csv")
A = dfp[[f"actual_h{h}" for h in range(1, K + 1)]].to_numpy()
P = dfp[[f"pred_h{h}" for h in range(1, K + 1)]].to_numpy()
# train-split mean cumulative return per horizon step (leak-free constants)
drift = np.array([np.nanmean(logc[h: tr_end + h] - logc[:tr_end]) for h in range(1, K + 1)])
print(f"   train drift per horizon (1e-4): {np.round(drift * 1e4, 2)}")
base = max((A > 0).mean(), 1 - (A > 0).mean())
for lbl, Q in (("raw preds", P),
               ("+0.5x drift", P + 0.5 * drift),
               ("+1.0x drift", P + 1.0 * drift),
               ("+2.0x drift", P + 2.0 * drift)):
    da = (np.sign(Q) == np.sign(A)).mean()
    print(f"   {lbl:12s} DirAcc {da:.4f}   (vs always-up {base:.4f}, edge {(da - base) * 100:+.1f}pp)")
print("   read: if calibrated DirAcc >= base rate, the 'model below naive' gap was "
      "a sign-threshold bias, recoverable honestly (train-derived constant).")
