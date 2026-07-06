"""
Prophet baseline (Section 1.3). Prophet is an optional, heavy dependency
(pulls in cmdstanpy / a Stan toolchain), so it is imported lazily and the
function degrades gracefully with an informative message if it is not
installed, rather than breaking the rest of the project.

    pip install prophet --break-system-packages
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401

        return True
    except Exception:
        return False


def prophet_multistep_forecast(dates: pd.DatetimeIndex, close: np.ndarray, horizon: int) -> np.ndarray:
    if not prophet_available():
        raise ImportError(
            "prophet is not installed in this environment. "
            "Run: pip install prophet --break-system-packages"
        )
    from prophet import Prophet

    df = pd.DataFrame({"ds": dates, "y": close})
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="B")
    fc = m.predict(future)
    forecast_prices = fc["yhat"].values[-horizon:]
    last_price = close[-1]
    return np.log(forecast_prices / last_price).astype(np.float32)
