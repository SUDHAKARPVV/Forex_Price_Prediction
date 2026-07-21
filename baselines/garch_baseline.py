"""GARCH baseline (dissertation Section 1.3 / Tools: statsmodels-arch).

An AR(1) conditional-mean + GARCH(1,1) conditional-variance model fit on
log-returns -- the canonical econometric benchmark for volatile financial
series (Bollerslev 1986), and the second of the two classical baselines
(alongside ARIMA) the Hybrid model is compared against.

The mean equation supplies the k-step point forecast (cumulative
log-returns, matching the dataset target); the variance equation is what
GARCH exists for and is exposed as a forecast band, mirroring the Hybrid
model's uncertainty output. Returns are scaled by 100 before fitting
(standard practice for GARCH numerical stability) and the forecast is
scaled back.
"""
from __future__ import annotations

import warnings

import numpy as np


def garch_multistep_forecast(train_close: np.ndarray, horizon: int) -> np.ndarray:
    """Fit AR(1)-GARCH(1,1) on the log-return history and return the
    k-step-ahead CUMULATIVE log-return point forecast."""
    from arch import arch_model

    log_returns = np.diff(np.log(train_close)) * 100.0  # % scale for stability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(log_returns, mean="AR", lags=1, vol="GARCH", p=1, q=1, rescale=False)
        fit = model.fit(disp="off", show_warning=False)
        fc = fit.forecast(horizon=horizon, reindex=False)
    step_returns = np.asarray(fc.mean.values[-1], dtype=np.float64) / 100.0
    return np.cumsum(step_returns).astype(np.float32)


def garch_sigma_forecast(train_close: np.ndarray, horizon: int,
                         fit_window: int | None = None) -> np.ndarray:
    """Conditional SIGMA of the k-step CUMULATIVE log-return -- the volatility
    forecast GARCH actually exists for (the magnitude analog of the mean point
    forecast). Returns sigma at each horizon 1..k, i.e. sqrt of the summed
    per-step conditional variances (Gaussian cumulative-return approximation;
    scale-consistent for rank scoring). Optional rolling fit_window caps the
    history per fit for speed (mirrors the ARIMA baseline's fit_window)."""
    from arch import arch_model

    log_returns = np.diff(np.log(train_close)) * 100.0  # % scale for stability
    if fit_window:
        log_returns = log_returns[-fit_window:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(log_returns, mean="AR", lags=1, vol="GARCH", p=1, q=1, rescale=False)
        fit = model.fit(disp="off", show_warning=False)
        fc = fit.forecast(horizon=horizon, reindex=False)
    var_steps = np.asarray(fc.variance.values[-1], dtype=np.float64) / (100.0 ** 2)
    return np.sqrt(np.cumsum(var_steps)).astype(np.float32)


def rolling_garch_evaluation(close: np.ndarray, origins: list, horizon: int, min_history: int = 250):
    """Walk-forward GARCH at each forecast origin using only data available
    up to that point, matching rolling_arima_evaluation's contract."""
    preds = []
    valid_origins = []
    for t in origins:
        if t < min_history:
            continue
        try:
            history = close[: t + 1]
            preds.append(garch_multistep_forecast(history, horizon))
            valid_origins.append(t)
        except Exception:
            continue
    return np.array(preds), valid_origins
