"""
Evaluation layer (Figure 7): multi-horizon error metrics + regime-segmented
benchmarking against ARIMA / VanillaLSTM / SimplifiedTFT baselines
(Section 4 / Section 1.3).

Reports TWO directional-accuracy numbers side by side for every deep model:
    - DirectionalAccuracy: derived from the regression forecast's sign
      (what the original evaluation used)
    - ClassifierDirectionalAccuracy: from the auxiliary classification
      head, trained directly on sign-agreement (see training/train.py)
Both are shown, deliberately, rather than only reporting whichever is
higher -- the point is to make the effect of the auxiliary head visible,
not to quietly swap in a better-looking number.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import (
    summarize,
    per_horizon_metrics,
    regime_segmented_metrics,
    classifier_directional_accuracy,
    per_horizon_classifier_accuracy,
    regime_segmented_classifier_accuracy,
)
from utils.regime_detector import label_regimes
from baselines.arima_baseline import rolling_arima_evaluation


def collect_predictions(model, dataset, device: str = "cpu"):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    all_true, all_pred, all_logits, all_regime_ctx = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, regime_ctx, xgb_pred = batch
                xgb_pred = xgb_pred.to(device)
            else:
                x, y, regime_ctx = batch
                xgb_pred = None
            x, regime_ctx = x.to(device), regime_ctx.to(device)
            out = model(x, regime_ctx, xgb_pred)
            pred = out["forecast"] if isinstance(out, dict) else out
            logits = out.get("direction_logits") if isinstance(out, dict) else None
            all_true.append(y.numpy())
            all_pred.append(pred.cpu().numpy())
            all_regime_ctx.append(regime_ctx.cpu().numpy())
            if logits is not None:
                all_logits.append(logits.cpu().numpy())
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else None
    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
        logits_array,
        np.concatenate(all_regime_ctx, axis=0),
    )


def evaluate_deep_model(model, dataset, model_name: str, device: str = "cpu"):
    y_true, y_pred, direction_logits, regime_ctx = collect_predictions(model, dataset, device=device)
    regime_labels = label_regimes(regime_ctx[:, 0])  # realised vol column

    overall = summarize(y_true, y_pred)
    per_horizon = per_horizon_metrics(y_true, y_pred)
    regime_segmented = regime_segmented_metrics(y_true, y_pred, regime_labels)

    if direction_logits is not None:
        overall["ClassifierDirectionalAccuracy"] = classifier_directional_accuracy(y_true, direction_logits)
        per_horizon["classifier_directional_accuracy"] = per_horizon_classifier_accuracy(y_true, direction_logits)
        for regime_name, acc in regime_segmented_classifier_accuracy(y_true, direction_logits, regime_labels).items():
            regime_segmented.setdefault(regime_name, {})["classifier_directional_accuracy"] = acc

    report = {
        "model": model_name,
        "overall": overall,
        "per_horizon": per_horizon,
        "regime_segmented": regime_segmented,
    }
    return report, y_true, y_pred, regime_labels


def evaluate_arima(panel, test_ds, horizon: int, max_origins: int = 40, order=(2, 1, 2)):
    """ARIMA is refit at every forecast origin (walk-forward), which is
    expensive; we subsample test origins for tractability in this demo
    (`max_origins`), matching common practice for classical-baseline
    comparison runs. Increase `max_origins` for a fuller benchmark.
    """
    close = panel.close
    origins = test_ds.indices
    if len(origins) > max_origins:
        step = max(1, len(origins) // max_origins)
        origins = origins[::step][:max_origins]

    preds, valid_origins = rolling_arima_evaluation(close, origins, horizon, order=order)
    if len(valid_origins) == 0:
        return None

    log_close = np.log(close)
    y_true = np.array([
        (log_close[t + 1 : t + 1 + horizon] - log_close[t]) for t in valid_origins
    ])
    regime_labels = label_regimes(panel.realized_vol[valid_origins])

    report = {
        "model": "ARIMA",
        "overall": summarize(y_true, preds),
        "per_horizon": per_horizon_metrics(y_true, preds),
        "regime_segmented": regime_segmented_metrics(y_true, preds, regime_labels),
        "n_origins_evaluated": len(valid_origins),
    }
    return report
