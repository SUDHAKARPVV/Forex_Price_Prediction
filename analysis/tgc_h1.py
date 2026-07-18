"""Trend-Gated Committee (TGC) at H1 -- the selective-accuracy analysis.

The clean H1 runs showed NO unfiltered directional edge (nothing beats the
always-up base rate on metals; euro ~coin-flip). The project's genuine edge on
daily data was SELECTIVE accuracy: trade only when the Hybrid and the GARCH
expert AGREE on direction AND the trend-quality gate is open (|drift_tstat| at
the origin >= the TRAIN-split top-tercile threshold -- calibrated on train,
nothing tuned on test). This script ports that committee to the H1 checkpoints.

train_pairs.py persists only aggregate metrics, so the per-window test
predictions are RECOMPUTED here, reproducing the training-time eval exactly:
same frozen panel, same train/val stride (3), same walk-forward XGBoost expert
schedule (refit_every=100), GARCH experts from the pair's cached npz. The
recomputed unfiltered DirAcc is asserted against results/pair_metrics/<slug>.json
to prove the reproduction before any committee numbers are trusted.

Outputs results/tgc_h1.json and results/predictions_h1_<slug>.csv.
Run from repo root:  python analysis/tgc_h1.py [--pairs "XAU/USD" ...]
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import argparse
import json
import os

import xgboost  # noqa: F401  (before torch)
import numpy as np
import torch

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import get_pair, checkpoint_dir, panel_csv_path
from baselines.xgboost_baseline import (XGBAugmentedDataset, XGBoostForexModel,
                                        walk_forward_expert_preds)
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model
from main import _load_garch_expert

ALL_PAIRS = ["XAU/USD", "XAG/USD", "EUR/USD"]
TRAIN_STRIDE = 3          # must match the clean H1 run
REFIT_EVERY = 100         # must match the clean H1 run
SEED = 9


def recompute_test_predictions(pair: str):
    """Reproduce the clean-run eval: returns (origin_dates, y_true, hybrid_pred,
    garch_pred, drift_tstat_at_origin, train_tercile_threshold)."""
    cfg = get_pair(pair)
    panel = build_fx_panel(pair=pair, source="panel", panel_csv=panel_csv_path(pair))
    tr, va, te = time_split(panel)
    tr.indices = tr.indices[::TRAIN_STRIDE]
    va.indices = va.indices[::TRAIN_STRIDE]

    ckpt = checkpoint_dir(pair)
    garch_by = _load_garch_expert(panel, os.path.join(ckpt, "garch_expert_preds.npz"))
    if garch_by is None:
        raise SystemExit(f"[tgc] {cfg.slug}: garch npz missing/stale -- rerun training first.")
    _zero = np.zeros(DATA_CFG.horizon, dtype="float32")

    def _g(ds):
        return np.stack([np.asarray(garch_by.get(t, _zero), dtype="float32") for t in ds.indices])

    import joblib
    xgb = XGBoostForexModel()
    xgb.model = joblib.load(os.path.join(ckpt, "xgb.pkl"))
    print(f"[tgc] {cfg.slug}: walk-forward expert preds (refit_every={REFIT_EVERY}) ...")
    wf = walk_forward_expert_preds(tr, va, te, refit_every=REFIT_EVERY)
    te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))

    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(os.path.join(ckpt, "hybrid.pt"),
                                      map_location="cpu", weights_only=True))
    print(f"[tgc] {cfg.slug}: evaluating Hybrid over {len(te)} test windows ...")
    rep, y_true, y_pred, _ = evaluate_deep_model(hybrid, te_x, f"TGC_{cfg.slug}", device="cpu")

    # assert the reproduction matches the published run before trusting anything
    pub = json.load(open(f"results/pair_metrics/{cfg.slug}.json"))
    da = rep["overall"]["DirectionalAccuracy"]
    if abs(da - pub["hybrid"]["DirAcc"]) > 1e-3:
        raise SystemExit(f"[tgc] {cfg.slug}: recomputed DirAcc {da:.4f} != published "
                         f"{pub['hybrid']['DirAcc']:.4f} -- reproduction failed, aborting.")
    print(f"[tgc] {cfg.slug}: reproduction OK (DirAcc {da:.4f} == published)")

    garch_pred = np.stack([garch_by.get(t, _zero) for t in te.indices])[: len(y_true)]
    names = list(panel.feature_names)
    dt_col = panel.features[:, names.index("drift_tstat")]
    drift = np.array([dt_col[t] for t in te.indices])[: len(y_true)]
    thr = float(np.quantile(np.abs(dt_col[: int(len(panel.close) * DATA_CFG.train_frac)]), 2 / 3))
    dates = [str(panel.dates[t]) for t in te.indices][: len(y_true)]
    return cfg, dates, y_true, y_pred, garch_pred, drift, thr


def committee(y_true, hyb, gch, drift, thr):
    """Origin rule: h1 sign agreement + open drift gate; score ALL horizons of
    the selected origins. Also a per-horizon variant. Split-half robustness."""
    strong = np.abs(drift) >= thr
    agree1 = np.sign(hyb[:, 0]) == np.sign(gch[:, 0])
    sel = strong & agree1

    def allh(mask):
        if mask.sum() == 0:
            return float("nan"), 0
        h = np.sign(gch[mask]) == np.sign(y_true[mask])
        return float(h.mean()), int(mask.sum())

    da_o, n_o = allh(sel)
    # per-horizon committee
    hits, tot = [], 0
    for h in range(y_true.shape[1]):
        m = strong & (np.sign(hyb[:, h]) == np.sign(gch[:, h]))
        hits.append(np.sign(gch[m, h]) == np.sign(y_true[m, h]))
        tot += int(m.sum())
    ph = np.concatenate(hits) if hits else np.array([])
    half = len(sel) // 2
    m1 = sel.copy(); m1[half:] = False
    m2 = sel.copy(); m2[:half] = False
    # THE DECISIVE CONTROL: compare against the best naive directional rule on
    # the SAME selected origins. The drift gate deliberately picks trending
    # periods, so the subset's up-fraction is not ~50% -- any committee number
    # must beat THIS, not the global base rate, to demonstrate skill.
    up = float((y_true[sel] > 0).mean()) if sel.sum() else float("nan")
    naive = float(max(up, 1 - up)) if sel.sum() else float("nan")
    return {
        "unfiltered_diracc": float((np.sign(hyb) == np.sign(y_true)).mean()),
        "origin_rule": {"diracc": da_o, "n_origins": n_o,
                        "coverage": float(sel.mean()),
                        "diracc_half1": allh(m1)[0], "diracc_half2": allh(m2)[0]},
        "per_horizon_committee": {"diracc": float(ph.mean()) if len(ph) else float("nan"),
                                  "n_pairs": int(tot),
                                  "pair_coverage": float(tot / y_true.size)},
        "gate_threshold_abs_drift_tstat": thr,
        "selected_subset": {"up_fraction": up, "best_naive_diracc": naive,
                            "tgc_edge_vs_naive_pp": round((da_o - naive) * 100, 2)
                            if sel.sum() else float("nan")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=ALL_PAIRS)
    args = ap.parse_args()

    out = {}
    for pair in args.pairs:
        cfg, dates, y_true, hyb, gch, drift, thr = recompute_test_predictions(pair)
        # persist per-window predictions (train_pairs never saved these)
        import pandas as pd
        dfp = pd.DataFrame({"origin": dates})
        for h in range(y_true.shape[1]):
            dfp[f"actual_h{h+1}"] = y_true[:, h]
            dfp[f"pred_h{h+1}"] = hyb[:, h]
            dfp[f"garch_h{h+1}"] = gch[:, h]
        dfp["drift_tstat"] = drift
        os.makedirs("results", exist_ok=True)
        dfp.to_csv(f"results/predictions_h1_{cfg.slug}.csv", index=False)

        res = committee(y_true, hyb, gch, drift, thr)
        out[cfg.slug] = res
        o = res["origin_rule"]
        print(f"[tgc] {cfg.slug}: unfiltered {res['unfiltered_diracc']:.4f} | "
              f"ORIGIN RULE {o['diracc']:.4f} @ {o['coverage']*100:.1f}% coverage "
              f"(halves {o['diracc_half1']:.3f}/{o['diracc_half2']:.3f}) | "
              f"per-horizon {res['per_horizon_committee']['diracc']:.4f} @ "
              f"{res['per_horizon_committee']['pair_coverage']*100:.1f}%")

    json.dump(out, open("results/tgc_h1.json", "w"), indent=2)
    print("\n[tgc] wrote results/tgc_h1.json")


if __name__ == "__main__":
    main()
