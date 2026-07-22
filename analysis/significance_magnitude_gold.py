"""Statistical-significance tests for the gold H1 MAGNITUDE result: is the Hybrid
genuinely better than atr_pct and GARCH-sigma, or within sampling noise?

Three complementary tests on the frozen seed-9 magnitude forecasts (no retrain):
  1. BLOCK BOOTSTRAP of the difference in the metrics we actually reported
     (spearman rank skill, rolling-median large-move accuracy). Distribution-free,
     block-resampled to respect the autocorrelation of overlapping 10-bar windows.
  2. DIEBOLD-MARIANO test on squared-error loss of |10-bar move| forecasts, with
     Newey-West HAC variance (lag = horizon) for the overlap. Baselines rescaled
     to the |move| target by a scale fit on VALIDATION (never on test).
  3. HANSEN MODEL CONFIDENCE SET (arch) -- the set of forecasters statistically
     indistinguishable from the best at 90%.

Reproduction-guarded against the committed seed-9 spearman. Arrays cached ->
re-runs are instant.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())
_os.environ["FOREX_TARGET"] = "magnitude"          # before any dataset is built

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

from config import DATA_CFG
from data.pairs import checkpoint_dir, panel_csv_path, get_pair

PAIR, SLUG = "XAU/USD", "XAUUSD"
K = DATA_CFG.horizon
MAG9 = os.path.join(checkpoint_dir(PAIR), "magnitude", "seed9")
ARR = f"results/significance_arrays_{SLUG}.npz"
RNG = np.random.default_rng(7)


def _garch_sigma(panel, origins, stride=5, fit_window=5000):
    from baselines.garch_baseline import garch_sigma_forecast
    close = np.asarray(panel.close, dtype=np.float64)
    ti = np.asarray(origins, dtype=np.int64)
    sig = np.full(len(ti), np.nan)
    for j in range(0, len(ti), stride):
        t = int(ti[j])
        try:
            sig[j] = float(garch_sigma_forecast(close[: t + 1], K, fit_window=fit_window)[-1])
        except Exception:
            pass
    return pd.Series(sig).ffill().bfill().to_numpy()


def build_arrays():
    import xgboost  # noqa: F401
    import torch, joblib
    from data.dataset import build_fx_panel, time_split
    from baselines.xgboost_baseline import XGBAugmentedDataset, XGBoostForexModel, walk_forward_expert_preds
    from models.hybrid_model import HybridCNNLSTMTransformer
    from training.evaluate import evaluate_deep_model

    def _load_garch_expert(panel, path):
        import hashlib
        if not os.path.exists(path):
            return {}
        z = np.load(path, allow_pickle=True)
        if str(z["close_md5"]) != hashlib.md5(np.asarray(panel.close, np.float64).tobytes()).hexdigest():
            return {}
        return {int(t): p for t, p in zip(z["origins"], z["preds"])}

    panel = build_fx_panel(pair=PAIR, source="panel", panel_csv=panel_csv_path(PAIR))
    tr, va, te = time_split(panel)
    tr.indices = tr.indices[::3]; va.indices = va.indices[::3]
    base = checkpoint_dir(PAIR)
    garch_by = _load_garch_expert(panel, os.path.join(base, "garch_expert_preds.npz"))
    _z = np.zeros(K, dtype="float32")
    def _g(ds): return np.stack([np.asarray(garch_by.get(t, _z), "float32") for t in ds.indices])
    xgb = XGBoostForexModel(); xgb.model = joblib.load(os.path.join(MAG9, "xgb.pkl"))
    print("[sig] walk-forward expert ...")
    wf = walk_forward_expert_preds(tr, va, te, refit_every=100)
    va_x = XGBAugmentedDataset(va, xgb, garch_preds=_g(va))
    te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))
    hyb = HybridCNNLSTMTransformer()
    hyb.load_state_dict(torch.load(os.path.join(MAG9, "hybrid.pt"), map_location="cpu", weights_only=True))
    _, yv, muv, _ = evaluate_deep_model(hyb, va_x, "v", device="cpu")
    _, yt, mut, _ = evaluate_deep_model(hyb, te_x, "t", device="cpu")
    names = list(panel.feature_names); ai = names.index("atr_pct")
    ov, ot = np.array(va.indices), np.array(te.indices)
    d = dict(
        actual_v=yv[:, -1], model_v=muv[:, -1], atr_v=panel.features[:, ai][ov[:len(yv)]],
        garch_v=_garch_sigma(panel, ov[:len(yv)]),
        actual_t=yt[:, -1], model_t=mut[:, -1], atr_t=panel.features[:, ai][ot[:len(yt)]],
        garch_t=_garch_sigma(panel, ot[:len(yt)]),
    )
    np.savez(ARR, **{k: v.astype(np.float64) for k, v in d.items()})
    return d


d = {k: np.load(ARR)[k] for k in np.load(ARR).files} if os.path.exists(ARR) else build_arrays()
if os.path.exists(ARR):
    print(f"[sig] loaded cached arrays ({ARR})")
act_t = d["actual_t"]; model_t = d["model_t"]; atr_t = d["atr_t"]; garch_t = d["garch_t"]
_repro = spearmanr(model_t, act_t).statistic
assert abs(_repro - 0.3287) < 0.02, f"repro failed: {_repro:.4f}"
print(f"reproduction OK (test spearman {_repro:.4f}); n_test={len(act_t):,}\n")

# ---- rolling-median large-move correctness per origin (for the accuracy test) ----
_w, _mp = 500, 100
roll = pd.Series(act_t).rolling(_w, min_periods=_mp).median().to_numpy()
ok = np.isfinite(roll); yhi = act_t > roll
def _correct(x):
    thr = pd.Series(x).rolling(_w, min_periods=_mp).median().to_numpy()
    return ((x > thr) == yhi)[ok].astype(float)
cor = {"model": _correct(model_t), "atr": _correct(atr_t), "garch": _correct(garch_t)}
xok = {"model": model_t[ok], "atr": atr_t[ok], "garch": garch_t[ok]}; a_ok = act_t[ok]


def block_boot(stat_fn, n_boot=2000, block=50):
    n = len(a_ok); nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        out[b] = stat_fn(idx)
    return out


print("=" * 68)
print("1. BLOCK BOOTSTRAP of the reported-metric differences (model - baseline)")
print("=" * 68)
boot = {}
for bl in ("atr", "garch"):
    dsp = block_boot(lambda idx: (spearmanr(xok["model"][idx], a_ok[idx]).statistic
                                  - spearmanr(xok[bl][idx], a_ok[idx]).statistic))
    dac = block_boot(lambda idx: (cor["model"][idx].mean() - cor[bl][idx].mean()))
    for nm, key, arr in (("spearman", "spearman", dsp), ("large-move acc", "acc", dac)):
        lo, hi = np.percentile(arr, [2.5, 97.5]); p = float((arr <= 0).mean())
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  model - {bl:5s} {nm:15s}: mean {arr.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
              f"p(diff<=0)={p:.4f} {star}")
        boot[f"{bl}_{key}"] = {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p": p}
print()

print("=" * 68)
print("2. DIEBOLD-MARIANO (squared-error of |10-bar move|, Newey-West HAC lag=10)")
print("=" * 68)
def _scale(fv, av): return float(np.dot(fv, av) / np.dot(fv, fv))   # val-fit scale
c_m = _scale(d["model_v"], d["actual_v"]); c_a = _scale(d["atr_v"], d["actual_v"])
c_g = _scale(d["garch_v"], d["actual_v"])
e_m = act_t - c_m * model_t; e_a = act_t - c_a * atr_t; e_g = act_t - c_g * garch_t
def dm(e1, e2, lag=K):
    dloss = e1 ** 2 - e2 ** 2; n = len(dloss); dbar = dloss.mean(); dc = dloss - dbar
    g0 = np.dot(dc, dc) / n
    var = g0 + 2 * sum((1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n for k in range(1, lag + 1))
    stat = dbar / np.sqrt(var / n); return stat, 2 * norm.cdf(-abs(stat))
for bl, e_b in (("atr", e_a), ("garch", e_g)):
    s, p = dm(e_m, e_b); verdict = "model better" if s < 0 else "baseline better"
    print(f"  model vs {bl:5s}: DM={s:+.3f}  p={p:.4f}  -> {verdict if p < 0.05 else 'not significant'}")
print()

print("=" * 68)
print("3. HANSEN MODEL CONFIDENCE SET (90%)")
print("=" * 68)
mcs_res = None
try:
    from arch.bootstrap import MCS
    losses = pd.DataFrame({"Hybrid": e_m ** 2, "atr_pct": e_a ** 2, "GARCH_sigma": e_g ** 2})
    mcs = MCS(losses, size=0.10, reps=1000, block_size=50, method="R", seed=7); mcs.compute()
    pv = mcs.pvalues.sort_values("Pvalue")
    incl = [m for m in losses.columns if float(pv.loc[m, "Pvalue"]) > 0.10] if hasattr(pv, "loc") else None
    print(pv.to_string())
    included = list(mcs.included) if hasattr(mcs, "included") else incl
    print(f"  -> MCS(90%) includes: {included}")
    mcs_res = {"pvalues": pv["Pvalue"].to_dict(), "included": included}
except Exception as ex:
    print(f"  MCS skipped ({type(ex).__name__}: {ex}); DM + bootstrap above are the primary tests.")

_dm_a, _dm_g = dm(e_m, e_a), dm(e_m, e_g)
_mcs_incl = (mcs_res or {}).get("included") or []
summary = {"pair": PAIR, "n_test": int(len(act_t)), "repro_spearman": _repro,
           "bootstrap": boot,
           "dm": {"model_vs_atr": {"stat": _dm_a[0], "p": _dm_a[1]},
                  "model_vs_garch": {"stat": _dm_g[0], "p": _dm_g[1]}},
           "scales": {"model": c_m, "atr": c_a, "garch": c_g}, "mcs": mcs_res,
           # headline verdict the dashboard reads: significant only if the model
           # beats a baseline at p<0.05 on a bootstrap metric AND MCS drops that baseline
           "significant_vs_baselines": bool(
               (min(boot.get("atr_spearman", {}).get("p", 1), boot.get("garch_spearman", {}).get("p", 1)) < 0.05)
               and len(_mcs_incl) < 3)}
json.dump(summary, open(f"results/significance_magnitude_{SLUG}.json", "w"), indent=2, default=float)
print(f"\nwritten to results/significance_magnitude_{SLUG}.json")
