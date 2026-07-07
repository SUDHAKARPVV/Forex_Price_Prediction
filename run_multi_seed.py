"""
Multi-seed evaluation: runs the full pipeline across several random seeds
and reports mean +/- std for each model/metric, instead of relying on a
single run. With a test set of only ~200-450 samples, a single run's
directional-accuracy difference of a few percentage points is within
normal sampling noise (standard error ~ sqrt(0.5*0.5/n) ~ 0.03-0.04 for
n=200-450) -- this script exists to give a statistically honest answer to
"does the Hybrid model actually beat the baselines" rather than reporting
whichever single seed looks best.

Usage:
    python run_multi_seed.py --seeds 42 7 123 --epochs 25
"""
from __future__ import annotations

import argparse
import json

# Must precede any torch import -- see the note at the top of main.py
# (xgboost/torch OpenMP-runtime clash on macOS/conda).
import xgboost  # noqa: F401

import numpy as np

from main import run as run_pipeline


def build_seed_ensemble(seeds, model="Hybrid_CNN_LSTM_Transformer"):
    """Roadmap item 6 -- seed ensembling: average the per-seed test
    forecasts of the Hybrid (the market data is identical across seeds --
    only training RNG differs -- so the actuals must match exactly; if
    they don't, the export files came from different fetches and the
    ensemble is skipped). Returns (metrics_dict, n_windows) or None.
    """
    import os

    import pandas as pd

    from utils.metrics import summarize

    frames = []
    for s in seeds:
        path = f"exports/predictions_test_{model}_seed{s}.csv"
        if not os.path.exists(path):
            return None
        frames.append(pd.read_csv(path))
    act_cols = [f"actual_h{h}" for h in range(1, 11)]
    pred_cols = [f"pred_h{h}" for h in range(1, 11)]
    base = frames[0][act_cols].values
    for f in frames[1:]:
        if f.shape != frames[0].shape or not np.allclose(f[act_cols].values, base):
            print("[ensemble] per-seed actuals differ (stale export files?) -- skipping seed ensemble")
            return None
    mean_pred = np.mean([f[pred_cols].values for f in frames], axis=0)
    out = frames[0][["origin"]].copy()
    for h in range(10):
        out[f"actual_h{h+1}"] = base[:, h]
        out[f"pred_h{h+1}"] = mean_pred[:, h]
    out.to_csv(f"exports/predictions_test_{model}_seed_ensemble.csv", index=False)
    return summarize(base, mean_pred), len(base)


def multi_seed_evaluation(seeds, **run_kwargs):
    all_runs = []
    for seed in seeds:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        reports = run_pipeline(seed=seed, report_dir=f"report/report_seed_{seed}", **run_kwargs)
        all_runs.append(reports)

    model_names = list(all_runs[0].keys())
    metrics = ["MAE", "RMSE", "DirectionalAccuracy"]

    summary = {}
    for model in model_names:
        summary[model] = {}
        for metric in metrics:
            values = [run[model]["overall"][metric] for run in all_runs if model in run]
            summary[model][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    print(f"\n{'='*70}\nMULTI-SEED SUMMARY ({len(seeds)} seeds: {seeds})\n{'='*70}")
    header = f"{'Model':30s}" + "".join(f"{m:>22s}" for m in metrics)
    print(header)
    for model in model_names:
        row = f"{model:30s}"
        for metric in metrics:
            s = summary[model][metric]
            row += f"{s['mean']:>10.5f} +/-{s['std']:.5f}"
        print(row)

    with open("multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nFull multi-seed summary written to multi_seed_summary.json")

    # --- Roadmap extras: seed ensemble + event-window + calibrated abstention ---
    roadmap = {"seeds": list(seeds)}

    ens = build_seed_ensemble(seeds)
    if ens:
        ens_metrics, n_windows = ens
        roadmap["seed_ensemble"] = {"metrics": ens_metrics, "n_test_windows": n_windows}
        print(f"[ensemble] Hybrid seed-ensemble ({len(seeds)} seeds averaged): "
              f"DirAcc={ens_metrics['DirectionalAccuracy']:.4f}  "
              f"@20%={ens_metrics['DirAcc@20pctCoverage']:.4f}  "
              f"@10%={ens_metrics['DirAcc@10pctCoverage']:.4f}  MAE={ens_metrics['MAE']:.5f}")

    hybrid_key = "Hybrid_CNN_LSTM_Transformer"
    roadmap["event_window"] = {
        model: [run[model].get("event_window") for run in all_runs if model in run]
        for model in model_names
    }
    roadmap["calibrated_abstention"] = [
        run[hybrid_key].get("calibrated_abstention") for run in all_runs if hybrid_key in run
    ]
    roadmap["backtest"] = [
        run[hybrid_key].get("backtest") for run in all_runs if hybrid_key in run
    ]
    roadmap["feature_importance"] = (
        all_runs[0][hybrid_key].get("xgb_feature_importance") if hybrid_key in all_runs[0] else None
    )

    with open("roadmap_summary.json", "w") as f:
        json.dump(roadmap, f, indent=2, default=float)
    print("Roadmap extras (ensemble / event-window / abstention) written to roadmap_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[9, 36, 99])
    parser.add_argument("--pair", type=str, default="XAU/USD")
    parser.add_argument("--n_days", type=int, default=10000,
                        help="Max bars to keep from the live fetch. Daily interval fetches the "
                             "FULL listed history (GC=F reaches back to 2000), so the default "
                             "no longer caps at 5,000.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source", type=str, default="real", choices=["synthetic", "real"])
    parser.add_argument("--interval", type=str, default="1d",
                        help="'1d' daily bars (default: full 25-year history, natural alignment "
                             "with the daily/monthly macro cadence -- the hourly round showed "
                             "intraday scale favours the AR baselines and starves the macro "
                             "features), '1h' hourly (730-day cap), minute bars (60-day cap). "
                             "Yahoo has no native 12h interval; daily dominates a resampled "
                             "12h on bar count (6,485 vs ~1,460) and on macro alignment.")
    parser.add_argument("--signal_strength", type=float, default=None)
    args = parser.parse_args()

    multi_seed_evaluation(
        args.seeds,
        pair=args.pair,
        n_days=args.n_days,
        epochs=args.epochs,
        source=args.source,
        interval=args.interval,
        signal_strength=args.signal_strength,
    )
