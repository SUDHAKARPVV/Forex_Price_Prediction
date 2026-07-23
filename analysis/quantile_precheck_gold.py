"""Quantile-regression PRE-CHECK: would pinball-loss quantile heads improve
interval calibration BEYOND the Gaussian sigma head + conformal we already have,
before spending a ~2h retrain? Runs on the cached direction-model forecasts
(results/conformal_arrays_XAUUSD.npz) -- no retraining.

The deep model's sigma head already gives a CONDITIONAL scale, and split/ACI
conformal already fixes the global multiplier. Quantile heads would add
CONDITIONAL SHAPE -- i.e. the number of sigmas per quantile varying by regime.
So the decisive test is: does a sigma-CONDITIONAL quantile of the standardized
residual z=(y-mu)/sigma beat the single SCALAR conformal quantile? If z's tails
are the same across sigma regimes (conditional ~= scalar), the sigma head already
captured it and quantile heads are redundant (the HAR-RV lesson). If conditional
is clearly closer to target, quantile heads are worth wiring.

Calibrate on VAL, evaluate coverage on TEST (temporal, leak-free), per horizon,
at 80/90/95%. Compares: Gaussian (z_tau) | scalar conformal | sigma-conditional.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import json
import numpy as np
from scipy.stats import norm

SLUG = "XAUUSD"
LEVELS = [0.80, 0.90, 0.95]
NBINS = 5
z = np.load(f"results/conformal_arrays_{SLUG}.npz")
yv, muv, sv, yt, mut, st = (z["yv"], z["muv"], z["sv"], z["yt"], z["mut"], z["st"])
K = yv.shape[1]
_eps = 1e-8
sv = np.clip(sv, _eps, None); st = np.clip(st, _eps, None)
zv = (yv - muv) / sv                       # standardized residuals (val)
zt = (yt - mut) / st                       # standardized residuals (test)

print(f"quantile-regression pre-check -- calib(val) n={len(yv):,}, test n={len(yt):,}, horizons={K}\n")
print(f"{'level':>6s} | {'Gaussian':>9s} {'scalar-conf':>12s} {'sigma-cond':>11s}  (test coverage; target=level)")
print("-" * 60)
summary = {"slug": SLUG, "levels": {}}
for lvl in LEVELS:
    zc = norm.ppf(1.0 - (1.0 - lvl) / 2.0)         # Gaussian half-width in sigma units
    cov_g, cov_s, cov_c = [], [], []
    for h in range(K):
        av = np.abs(zv[:, h]); at = np.abs(zt[:, h])
        # 1) Gaussian: |z| <= z_tau
        cov_g.append(float((at <= zc).mean()))
        # 2) scalar conformal: single val quantile of |z|
        qs = np.quantile(av, min(1.0, lvl * (1 + 1 / len(av))), method="higher")
        cov_s.append(float((at <= qs).mean()))
        # 3) sigma-conditional: per-sigma-bin val quantile of |z| (bins from val sigma)
        edges = np.quantile(sv[:, h], np.linspace(0, 1, NBINS + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        vb = np.digitize(sv[:, h], edges[1:-1]); tb = np.digitize(st[:, h], edges[1:-1])
        qbin = np.array([np.quantile(av[vb == b], min(1.0, lvl * (1 + 1 / max((vb == b).sum(), 1))),
                                     method="higher") if (vb == b).sum() >= 30 else qs
                         for b in range(NBINS)])
        cov_c.append(float((at <= qbin[tb]).mean()))
    g, s, c = float(np.mean(cov_g)), float(np.mean(cov_s)), float(np.mean(cov_c))
    print(f"{lvl*100:>5.0f}% | {g*100:>8.1f}% {s*100:>11.1f}% {c*100:>10.1f}%")
    summary["levels"][f"{lvl:.2f}"] = {"target": lvl, "gaussian": g, "scalar_conformal": s,
                                       "sigma_conditional": c}

# verdict: does sigma-conditional close the gap to target vs scalar conformal?
gaps_s = [abs(v["scalar_conformal"] - v["target"]) for v in summary["levels"].values()]
gaps_c = [abs(v["sigma_conditional"] - v["target"]) for v in summary["levels"].values()]
impr = float(np.mean(gaps_s) - np.mean(gaps_c))       # >0 => conditional is closer to target
summary["mean_abs_gap_scalar"] = float(np.mean(gaps_s))
summary["mean_abs_gap_conditional"] = float(np.mean(gaps_c))
summary["conditional_improvement_pp"] = round(impr * 100, 2)
summary["promising"] = bool(impr > 0.02)              # >2pp average coverage-gap improvement
_os.makedirs("results", exist_ok=True)
json.dump(summary, open(f"results/quantile_precheck_{SLUG}.json", "w"), indent=2, default=float)

print(f"\nmean |coverage-target| gap: scalar-conformal {np.mean(gaps_s)*100:.1f}pp -> "
      f"sigma-conditional {np.mean(gaps_c)*100:.1f}pp  (improvement {impr*100:+.1f}pp)")
print("read: sigma-conditional quantiles ~= the shape gain quantile heads would add over the")
print("sigma head + scalar conformal. If it barely improves the gap, the sigma head already")
print(f"captured the conditional uncertainty -> quantile heads redundant. VERDICT: "
      f"{'PROMISING -- wire + retrain' if summary['promising'] else 'NOT promising -- sigma head + conformal suffice'}")
print(f"written to results/quantile_precheck_{SLUG}.json")
