"""
Random Walk with Drift baseline (Section III.D of Dave et al. 2025).

Paper 1 runs an Augmented Dickey-Fuller test on the price series, finds a
unit root (non-stationarity), and argues Random Walk with Drift is
therefore the appropriate naive baseline: P_t = P_{t-1} + mu + eps_t, where
mu is a constant drift estimated from historical returns. All of their
ML/DL models are shown to beat this baseline by a wide margin (their
Figure 3: e.g. 0.0064 RMSE for their best model vs 0.0598 for Random Walk
on EUR/USD) -- that comparison is itself a useful sanity check to include
here: if the Hybrid model (or any model) couldn't beat Random Walk with
Drift, that would be a much bigger problem than not beating ARIMA.
"""
from __future__ import annotations

import numpy as np


def adf_test(close: np.ndarray) -> dict:
    """Augmented Dickey-Fuller test for a unit root, matching the
    methodology in Paper 1 Section III.D. Requires statsmodels (already a
    core dependency).
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(close, regression="ct")  # constant + trend, matching the paper's equation
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < 0.05),
    }


def random_walk_drift_forecast(train_close: np.ndarray, horizon: int) -> np.ndarray:
    """Estimate drift mu as the mean log-return over the training series,
    then forecast cumulative log-return at each horizon step as h * mu
    (the expected value of a Random Walk with Drift h steps ahead).
    """
    log_returns = np.diff(np.log(train_close))
    mu = log_returns.mean()
    return np.array([mu * h for h in range(1, horizon + 1)], dtype=np.float32)


def evaluate_random_walk(panel, test_ds, horizon: int) -> dict:
    """Applies the SAME drift estimate (from the training portion of the
    series) at every test origin -- consistent with how Random Walk with
    Drift is meant to be used: it doesn't refit at every point, it just
    projects a constant historical drift forward.
    """
    train_end = test_ds.indices[0] - test_ds.lookback + 1 if test_ds.indices else 0
    # Use all data up to the first test origin as the "training" drift estimate
    train_close = panel.close[: max(train_end, 2)]
    forecast_template = random_walk_drift_forecast(train_close, horizon)

    log_close = np.log(panel.close)
    y_true, y_pred = [], []
    for t in test_ds.indices:
        y_true.append(log_close[t + 1 : t + 1 + horizon] - log_close[t])
        y_pred.append(forecast_template)

    return np.array(y_true), np.array(y_pred)
