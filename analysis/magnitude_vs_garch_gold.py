"""FAST PRE-CHECK: does GARCH's conditional sigma beat the trained magnitude
model (and atr_pct) at forecasting |cumulative 10-bar return| on gold H1?

The magnitude 3-seed sweep showed the Hybrid beating the SIMPLE indicator
(atr_pct) robustly. GARCH is the classical VOLATILITY model, so beating it on
this target would be the real "Hybrid > GARCH" claim -- on the axis where skill
actually exists (magnitude), not direction (where GARCH's drift wins a coin
flip). This reuses the committed seed-9 magnitude checkpoint (no retraining),
adds GARCH-sigma as a third forecaster, and rank-scores all three. GARCH-sigma
is computed on STRIDED test origins (rolling fit_window) to stay fast -- this is
a go/no-go pre-check, not the final 3-seed run.

Leak-free: every forecast at origin t uses only data <= t; the actual |move|
spans [t+1, t+10] and never overlaps the features.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

# MUST precede dataset construction: flip the window target to |cumulative return|
_os.environ["FOREX_TARGET"] = "magnitude"

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xgboost  # noqa: F401  (before torch)
import torch
import joblib

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir, panel_csv_path
from baselines.xgboost_baseline import XGBAugmentedDataset, XGBoostForexModel, walk_forward_expert_preds
from baselines.garch_baseline import garch_sigma_forecast
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model
from main import _load_garch_expert

PAIR, SLUG = "XAU/USD", "XAUUSD"
K = DATA_CFG.horizon
GARCH_STRIDE = int(os.environ.get("PRECHECK_GARCH_STRIDE", "10"))
GARCH_FIT_WINDOW = int(os.environ.get("PRECHECK_GARCH_FITWIN", "5000"))
CKPT = os.path.join(checkpoint_dir(PAIR), "magnitude", "seed9")
if not os.path.exists(os.path.join(CKPT, "hybrid.pt")):
    CKPT = os.path.join(checkpoint_dir(PAIR), "magnitude")   # fall back to the single-run ckpt

# ---- reconstruct the seed-9 magnitude model's test predictions ----
panel = build_fx_panel(pair=PAIR, source="panel", panel_csv=panel_csv_path(PAIR))
tr, va, te = time_split(panel)
tr.indices = tr.indices[::3]; va.indices = va.indices[::3]     # match the sweep's stride
ckpt_base = checkpoint_dir(PAIR)
garch_by = _load_garch_expert(panel, os.path.join(ckpt_base, "garch_expert_preds.npz"))
_zero = np.zeros(K, dtype="float32")
xgb = XGBoostForexModel(); xgb.model = joblib.load(os.path.join(CKPT, "xgb.pkl"))
wf = walk_forward_expert_preds(tr, va, te, refit_every=100)
te_x = XGBAugmentedDataset(te, xgb, preds=wf,
                           garch_preds=np.stack([np.asarray(garch_by.get(t, _zero), "float32") for t in te.indices]))
hybrid = HybridCNNLSTMTransformer()
hybrid.load_state_dict(torch.load(os.path.join(CKPT, "hybrid.pt"), map_location="cpu", weights_only=True))
rep, y_true, y_pred, band = evaluate_deep_model(hybrid, te_x, "mag", device="cpu")

n = len(y_true)
origins = np.array(te.indices)[:n]
actual = y_true[:, -1]            # actual |cumulative 10-bar move|
model_mag = y_pred[:, -1]         # model's magnitude forecast
names = list(panel.feature_names)
atr_pct = panel.features[:, names.index("atr_pct")][origins]

# reproduction guard: this must match the committed seed-9 magnitude spearman
_pub = 0.328726
_repro = spearmanr(model_mag, actual).statistic
assert abs(_repro - _pub) < 0.02, f"repro failed: {_repro:.4f} vs published {_pub:.4f}"
print(f"reproduction OK (model spearman {_repro:.4f} ~= committed seed-9 {_pub:.4f})")

# ---- GARCH conditional sigma on strided test origins ----
close = np.asarray(panel.close, dtype=np.float64)
garch_sigma = np.full(n, np.nan, dtype=np.float64)
idx = np.arange(0, n, GARCH_STRIDE)
print(f"computing GARCH-sigma at {len(idx)} strided test origins "
      f"(stride {GARCH_STRIDE}, fit_window {GARCH_FIT_WINDOW}) ...")
done = 0
for i in idx:
    t = int(origins[i])
    try:
        garch_sigma[i] = float(garch_sigma_forecast(close[: t + 1], K, fit_window=GARCH_FIT_WINDOW)[-1])
    except Exception:
        continue
    done += 1
    if done % 100 == 0:
        print(f"  {done}/{len(idx)} GARCH fits done")

# ---- rank-score all three on the COMMON strided origins (fair comparison) ----
ok = np.isfinite(garch_sigma)
print(f"\ngold H1 MOVE-MAGNITUDE forecasting -- {int(ok.sum()):,} common test origins")


def score(name, x):
    xo, ao = x[ok], actual[ok]
    rho = spearmanr(xo, ao).statistic
    thr = pd.Series(xo).rolling(500, min_periods=100).median().to_numpy()
    m = np.isfinite(thr)
    hi_thr = pd.Series(ao).rolling(500, min_periods=100).median().to_numpy()
    acc = float(((xo[m] > thr[m]) == (ao[m] > hi_thr[m])).mean())
    print(f"  {name:22s} spearman {rho:+.3f} | large-move acc {acc:.3f}")
    return rho, acc


m_rho, m_acc = score("model (magnitude)", model_mag)
a_rho, a_acc = score("atr_pct", atr_pct)
g_rho, g_acc = score("GARCH-sigma", garch_sigma)

beats_garch = (m_rho > g_rho) and (m_acc > g_acc)
beats_atr = (m_rho > a_rho) and (m_acc > a_acc)
print("\n=== PRE-CHECK VERDICT ===")
print(f"  model vs atr_pct : {'model ahead' if beats_atr else 'atr_pct ahead'} "
      f"(rho {m_rho - a_rho:+.3f}, acc {(m_acc - a_acc) * 100:+.1f}pp)")
print(f"  model vs GARCH   : {'MODEL BEATS GARCH -> full 3-seed is worth it' if beats_garch else 'GARCH ahead -> reconsider before the 7h run'} "
      f"(rho {m_rho - g_rho:+.3f}, acc {(m_acc - g_acc) * 100:+.1f}pp)")
print("\nnote: GARCH-sigma on strided origins is a go/no-go signal, not the final "
      "number; the full run scores every origin across 3 seeds.")
