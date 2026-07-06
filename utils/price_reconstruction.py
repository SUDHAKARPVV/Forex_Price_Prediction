"""
Reconstructs actual price-level predictions from the model's cumulative
log-return forecasts, so predicted prices can be plotted directly against
actual prices over the test period -- what Paper 1 does throughout (their
Tables II-IV report price-level MAE/RMSE) and what you asked for
explicitly: "predict the forex price and compare with actual values in
train & test runs and plot the performance results."

The model itself still predicts in log-return space (Section 3's
multi-step forecast is defined that way, and log-returns are the correct
space to train in -- they're stationary, unlike raw price, which Paper 1's
own ADF test confirms is non-stationary). This module only exists to
convert back to price level *for reporting*.
"""
from __future__ import annotations

import numpy as np


def reconstruct_prices(y_log_return: np.ndarray, origin_close: np.ndarray, horizon_idx: int = 0) -> np.ndarray:
    """y_log_return: (N, k) cumulative log-returns from each origin.
    origin_close: (N,) the close price at each forecast origin.
    Returns: (N,) reconstructed price at horizon_idx+1 steps ahead.

    price_{t+h} = close_t * exp(cumulative_log_return_h)
    """
    return origin_close * np.exp(y_log_return[:, horizon_idx])


def get_price_level_series(dataset, y_true: np.ndarray, y_pred: np.ndarray, panel, horizon_idx: int = 0):
    """Given a dataset's y_true/y_pred (from evaluate_deep_model) and the
    FXPanel they came from, reconstruct aligned (dates, actual_price,
    predicted_price) arrays at a given horizon step for plotting.
    """
    origin_indices = np.array(dataset.indices)
    origin_close = panel.close[origin_indices]
    dates = panel.dates[origin_indices + horizon_idx + 1]

    actual_price = reconstruct_prices(y_true, origin_close, horizon_idx)
    predicted_price = reconstruct_prices(y_pred, origin_close, horizon_idx)
    return dates, actual_price, predicted_price


def price_level_metrics(actual_price: np.ndarray, predicted_price: np.ndarray) -> dict:
    """MAE/RMSE/MAPE/R2 computed directly on price level (not log-return),
    matching how Paper 1 reports its Tables II-IV.
    """
    from utils.metrics import summarize

    return summarize(actual_price, predicted_price)
