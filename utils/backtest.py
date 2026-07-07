"""
Costed conviction backtest (improvement-roadmap item: "accuracy is a
proxy; P&L with costs is the decision-grade metric").

Converts the Hybrid model's validation-calibrated abstention rule into a
bar-by-bar trading simulation on the test set:

    - at each test origin, if |1-step forecast| >= tau (the threshold
      calibrated on VALIDATION data, no test tuning), take a +/-1 position
      in the direction of the forecast; otherwise stay flat;
    - each position change pays `cost_bps` basis points (round-trip =
      2 changes), a realistic friction for COMEX gold futures/spot XAU;
    - P&L compounds in log-return space and is compared against
      buy-and-hold over the identical window.

Consecutive test origins are consecutive bars (stride-1 windows), so the
1-step-ahead column of the prediction matrix forms a proper sequential
strategy without overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conviction_backtest(
    y_true_h1: np.ndarray,
    y_pred_h1: np.ndarray,
    dates,
    tau: float,
    cost_bps: float = 2.0,
) -> dict:
    """Simulate the abstention-gated strategy. Returns summary statistics
    plus the equity curves (for charting)."""
    dates = pd.DatetimeIndex(dates)
    position = np.where(np.abs(y_pred_h1) >= tau, np.sign(y_pred_h1), 0.0)
    gross = position * y_true_h1
    trades = np.abs(np.diff(np.r_[0.0, position]))  # each unit change = one transaction
    costs = trades * cost_bps * 1e-4
    net = gross - costs

    equity = np.cumsum(net)                 # strategy log-equity
    buy_hold = np.cumsum(y_true_h1)         # long-only benchmark, same window

    span_years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    bars_per_year = len(net) / span_years
    ann = np.sqrt(bars_per_year)

    def max_drawdown(curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(curve)
        return float((curve - peak).min())

    in_market = position != 0
    return {
        "tau": float(tau),
        "cost_bps_per_change": float(cost_bps),
        "n_bars": int(len(net)),
        "time_in_market": float(in_market.mean()),
        "n_transactions": int(trades.sum()),
        "hit_rate_when_in": float((np.sign(y_pred_h1[in_market]) == np.sign(y_true_h1[in_market])).mean()) if in_market.any() else None,
        "total_return_log": float(equity[-1]),
        "total_return_pct": float(np.expm1(equity[-1]) * 100),
        "buy_hold_return_pct": float(np.expm1(buy_hold[-1]) * 100),
        "annualised_sharpe": float(net.mean() / (net.std() + 1e-12) * ann),
        "buy_hold_sharpe": float(y_true_h1.mean() / (y_true_h1.std() + 1e-12) * ann),
        "max_drawdown_log": max_drawdown(equity),
        "equity_curve": equity.tolist(),
        "buy_hold_curve": buy_hold.tolist(),
        "dates": [str(d) for d in dates],
    }
