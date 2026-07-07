"""
Scheduled macro-event calendar for event-window evaluation.

Two recurring US releases dominate gold's scheduled-event volatility:

- NFP (non-farm payrolls): first Friday of each month -- an exact rule,
  computed on the fly, no data needed.
- FOMC rate decisions: 8 scheduled meetings/year on irregular dates, so
  the decision dates must be tabulated. The table below covers 2022-2026,
  which spans the daily benchmark's test window (the most recent ~15% of a
  25-year history). Dates are the SECOND day of each two-day meeting (the
  statement/decision day), from the Federal Reserve's published schedule.

Both are exposed as boolean masks over an arbitrary bar-date index so the
event-window metric (main.py) and any future event-conditioned logic can
share one definition.
"""
from __future__ import annotations

import pandas as pd

# FOMC statement (decision) dates, 2022-2026. Source: Federal Reserve
# published meeting calendars. 2026 dates are the Fed's announced schedule.
FOMC_DECISION_DATES = pd.to_datetime([
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (announced schedule)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
])


def nfp_mask(dates: pd.DatetimeIndex) -> "pd.Series":
    """True on first-Friday-of-month bars (NFP release days)."""
    d = pd.DatetimeIndex(dates)
    return (d.dayofweek == 4) & (d.day <= 7)


def fomc_mask(dates: pd.DatetimeIndex, window_days: int = 1) -> "pd.Series":
    """True on FOMC decision days and the `window_days` trading days after
    (the decision's impact spills into the following session)."""
    d = pd.DatetimeIndex(dates).normalize()
    fomc = set()
    for base in FOMC_DECISION_DATES:
        for k in range(window_days + 1):
            fomc.add((base + pd.Timedelta(days=k)).normalize())
    return pd.Series(d.isin(fomc), index=range(len(d))).values


def scheduled_event_mask(dates: pd.DatetimeIndex) -> "pd.Series":
    """Union of NFP and FOMC scheduled-event days."""
    import numpy as np

    return np.asarray(nfp_mask(dates)) | np.asarray(fomc_mask(dates))
