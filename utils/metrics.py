"""
Multi-horizon evaluation metrics (Section 4 / evaluation layer of Figure 7).
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Paper 1 (Dave et al., 2025) treats
    this alongside RMSE as a primary metric, so it's reported here too.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where predicted sign of return matches actual sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def classifier_directional_accuracy(y_true: np.ndarray, direction_logits: np.ndarray) -> float:
    """Directional accuracy using the auxiliary classification head's
    predicted sign (sigmoid(logits) > 0.5) instead of the regression
    forecast's sign. Kept as a separate metric from `directional_accuracy`
    so a report can show both and make the comparison explicit.
    """
    predicted_sign = np.where(direction_logits > 0, 1.0, -1.0)
    return float(np.mean(np.sign(y_true) == predicted_sign))


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: (N, k). Returns MAE/RMSE/DA at each of the k horizons."""
    k = y_true.shape[1]
    out = {"mae": [], "rmse": [], "directional_accuracy": []}
    for h in range(k):
        out["mae"].append(mae(y_true[:, h], y_pred[:, h]))
        out["rmse"].append(rmse(y_true[:, h], y_pred[:, h]))
        out["directional_accuracy"].append(directional_accuracy(y_true[:, h], y_pred[:, h]))
    return out


def regime_segmented_metrics(y_true: np.ndarray, y_pred: np.ndarray, regime_labels: np.ndarray) -> dict:
    """Compute MAE/RMSE/directional accuracy separately for each regime label
    (0=low/stable, 1=high-volatility), as required by Section 3.1.5 / Section 4.
    """
    results = {}
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        name = "high_volatility" if regime == 1 else "stable"
        results[name] = {
            "n_samples": int(mask.sum()),
            "mae": mae(y_true[mask], y_pred[mask]),
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "directional_accuracy": directional_accuracy(y_true[mask], y_pred[mask]),
        }
    return results


def per_horizon_classifier_accuracy(y_true: np.ndarray, direction_logits: np.ndarray) -> list:
    k = y_true.shape[1]
    return [classifier_directional_accuracy(y_true[:, h], direction_logits[:, h]) for h in range(k)]


def regime_segmented_classifier_accuracy(y_true: np.ndarray, direction_logits: np.ndarray, regime_labels: np.ndarray) -> dict:
    results = {}
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        name = "high_volatility" if regime == 1 else "stable"
        results[name] = classifier_directional_accuracy(y_true[mask], direction_logits[mask])
    return results


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
        "DirectionalAccuracy": directional_accuracy(y_true, y_pred),
    }
