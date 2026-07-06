"""ARIMA baseline (Section 1.3 comparative benchmarks)."""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_multistep_forecast(train_close: np.ndarray, horizon: int, order=(2, 1, 2)) -> np.ndarray:
    """Fit ARIMA on a close-price history and return a k-step-ahead log-return forecast."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_close, order=order)
        fit = model.fit()
        forecast_prices = fit.forecast(steps=horizon)
    last_price = train_close[-1]
    log_returns = np.log(forecast_prices / last_price)
    return np.asarray(log_returns, dtype=np.float32)


def rolling_arima_evaluation(close: np.ndarray, origins: list, horizon: int, order=(2, 1, 2), min_history: int = 120):
    """Run ARIMA at each forecast origin using only data available up to
    that point (walk-forward), matching how the deep model is evaluated."""
    preds = []
    valid_origins = []
    for t in origins:
        if t < min_history:
            continue
        try:
            history = close[: t + 1]
            preds.append(arima_multistep_forecast(history, horizon, order=order))
            valid_origins.append(t)
        except Exception:
            continue
    return np.array(preds), valid_origins
