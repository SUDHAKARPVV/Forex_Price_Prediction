"""
Per-pair FINE-TUNING -- the fix for the failed euro zero-shot transfer.

For each target pair (XAG/USD, EUR/USD):
  1. Build the pair's news-less 35-feature panel (offline mode) + 70/15/15 split.
  2. Compute the pair's OWN walk-forward GARCH expert forecasts for every
     origin (parallel pool; leakage-free per-origin fits) and the pair's own
     static XGBoost expert -- the same dual-expert input stack as gold.
  3. Load the GOLD dual-expert checkpoint and evaluate ZERO-SHOT (baseline).
  4. FINE-TUNE the full model on the pair's train split (short, low LR,
     early-stopped on the pair's validation split).
  5. Re-evaluate on the pair's test windows with the walk-forward XGBoost
     expert (refit_every=14, the gold operating point).

Output: results/cross_pair_finetune.json + printed table.
Usage:  FOREX_OFFLINE_NEWS=1 python analysis_finetune_pairs.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("FOREX_OFFLINE_NEWS", "1")

import xgboost  # noqa: F401  (before torch)
import numpy as np
import torch

from config import DATA_CFG, TRAIN_CFG
from data.dataset import build_fx_panel, time_split
from baselines.xgboost_baseline import (XGBAugmentedDataset, XGBoostForexModel,
                                        walk_forward_expert_preds)
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model
from training.train import train_model

PAIRS = ["XAG/USD", "EUR/USD"]
CKPT = "exports/dashboard/hybrid.pt"
MIN_HISTORY = 250


def _garch_one(args):
    t, close, horizon = args
    from baselines.garch_baseline import garch_multistep_forecast
    try:
        return t, garch_multistep_forecast(close[: t + 1], horizon)
    except Exception:
        return t, np.zeros(horizon, dtype="float32")


def garch_expert_for(panel, datasets, workers=6):
    """Walk-forward GARCH forecasts for every origin of the given datasets."""
    close = np.asarray(panel.close, dtype=np.float64)
    origins = sorted(set(t for ds in datasets for t in ds.indices))
    todo = [(t, close, DATA_CFG.horizon) for t in origins if t + 1 >= MIN_HISTORY]
    by = {t: np.zeros(DATA_CFG.horizon, dtype="float32") for t in origins}
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for t, p in pool.imap_unordered(_garch_one, todo, chunksize=16):
            by[t] = np.asarray(p, dtype="float32")
    return by


def main():
    out = {"checkpoint": CKPT, "pairs": {}}
    for pair in PAIRS:
        print(f"\n===== {pair} =====")
        panel = build_fx_panel(pair=pair, n_days=10000, seed=9, source="real", real_interval="1d")
        tr, va, te = time_split(panel)
        print(f"[finetune] {pair}: {len(panel.close)} bars, train={len(tr)} val={len(va)} test={len(te)}")

        garch_by = garch_expert_for(panel, [tr, va, te])
        print(f"[finetune] {pair}: {len(garch_by)} walk-forward GARCH expert forecasts computed")

        def _g(ds):
            return np.stack([garch_by[t] for t in ds.indices])

        xgb = XGBoostForexModel()
        xgb.fit(tr, va)
        tr_x = XGBAugmentedDataset(tr, xgb, garch_preds=_g(tr))
        va_x = XGBAugmentedDataset(va, xgb, garch_preds=_g(va))
        wf = walk_forward_expert_preds(tr, va, te, refit_every=14)
        te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))

        # zero-shot baseline (gold dual-expert weights, frozen)
        hybrid = HybridCNNLSTMTransformer()
        hybrid.load_state_dict(torch.load(CKPT, map_location="cpu"))
        rep0, *_ = evaluate_deep_model(hybrid, te_x, f"zero_shot_{pair}", device="cpu")
        print(f"[finetune] {pair} zero-shot: DirAcc {rep0['overall']['DirectionalAccuracy']:.4f}")

        # fine-tune the full model on the pair's train split (short, low LR)
        torch.manual_seed(9)
        hybrid, _ = train_model(hybrid, tr_x, va_x, epochs=10,
                                lr=TRAIN_CFG.lr * 0.125, device="cpu", seed=9)
        rep1, *_ = evaluate_deep_model(hybrid, te_x, f"finetuned_{pair}", device="cpu")
        print(f"[finetune] {pair} fine-tuned: DirAcc {rep1['overall']['DirectionalAccuracy']:.4f}")

        # pair's own GARCH baseline on the same test windows
        logc = np.log(np.asarray(panel.close, dtype=np.float64))
        yt, gp = [], []
        for t in te.indices:
            if t + DATA_CFG.horizon < len(logc):
                yt.append(logc[t + 1: t + 1 + DATA_CFG.horizon] - logc[t])
                gp.append(garch_by[t])
        yt, gp = np.array(yt), np.array(gp)
        g_da = float((np.sign(gp) == np.sign(yt)).mean())

        out["pairs"][pair] = {
            "bars": int(len(panel.close)), "test_windows": int(len(te)),
            "zero_shot_diracc": rep0["overall"]["DirectionalAccuracy"],
            "finetuned_diracc": rep1["overall"]["DirectionalAccuracy"],
            "finetuned_mae": rep1["overall"]["MAE"],
            "own_garch_diracc": g_da,
        }
        print(f"[finetune] {pair}: zero-shot {rep0['overall']['DirectionalAccuracy']:.4f} -> "
              f"fine-tuned {rep1['overall']['DirectionalAccuracy']:.4f} | own GARCH {g_da:.4f}")

    json.dump(out, open("results/cross_pair_finetune.json", "w"), indent=2, default=float)
    print("\n[finetune] written results/cross_pair_finetune.json")


if __name__ == "__main__":
    main()
