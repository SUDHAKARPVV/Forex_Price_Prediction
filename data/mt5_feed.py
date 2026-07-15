"""
MetaTrader 5 (MT5) rate adapter -- CSV import.

WHY A CSV ADAPTER (and not the live MT5 API): the official `MetaTrader5`
Python package is **Windows-only** -- it talks to a running MT5 terminal over
a Windows-local IPC bridge, so it cannot be imported or connected from macOS
or Linux. This project runs on macOS, so live MT5 pulls are not possible in
this environment.

Instead, this adapter ingests MT5-EXPORTED CSVs, per pair, and the data feed
prefers them over yfinance when present. To produce them, on any Windows box
(or a Windows VM) with MT5 + a (dummy) account, run either:

    * MT5 GUI:  right-click a symbol chart -> "Save As" / export bars to CSV, or
    * a 3-line script using the MetaTrader5 package:
        import MetaTrader5 as mt5, pandas as pd
        mt5.initialize()
        r = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1, 0, 100000)
        pd.DataFrame(r).to_csv("XAUUSD.csv", index=False)

Drop the file at  exports/mt5/<SLUG>.csv  (e.g. exports/mt5/XAUUSD.csv,
XAGUSD.csv, EURUSD.csv) or exports/mt5/<SLUG>_<interval>.csv for a specific
interval. The loader is tolerant of the common MT5 column spellings
(time/date, open/high/low/close, tick_volume/real_volume/volume) and returns
the SAME schema fetch_gold_candles produces: columns [open, high, low, close,
volume], a tz-naive DatetimeIndex named 'date'.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd


def mt5_csv_path(pair: str, interval: str = "1d", exports_dir: str = "exports") -> "str | None":
    """Return the MT5 CSV path for this pair/interval if one exists, else None.
    Checks the interval-specific name first, then the pair-only name."""
    from data.pairs import get_pair
    slug = get_pair(pair).slug
    candidates = [
        os.path.join(exports_dir, "mt5", f"{slug}_{interval}.csv"),
        os.path.join(exports_dir, "mt5", f"{slug}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


_TIME_COLS = ("time", "date", "datetime", "<DATE>", "<TIME>", "timestamp")
_COL_ALIASES = {
    "open": ("open", "<OPEN>", "o"),
    "high": ("high", "<HIGH>", "h"),
    "low": ("low", "<LOW>", "l"),
    "close": ("close", "<CLOSE>", "c", "price"),
    "volume": ("tick_volume", "real_volume", "volume", "<TICKVOL>", "<VOL>", "vol"),
}


def _find(colmap: dict, aliases) -> "str | None":
    for a in aliases:
        if a.lower() in colmap:
            return colmap[a.lower()]
    return None


def load_mt5_ohlc(pair: str, interval: str = "1d", exports_dir: str = "exports") -> "pd.DataFrame | None":
    """Load an MT5-exported CSV for `pair` into the canonical OHLC schema.
    Returns None (never raises) if no file is present or it can't be parsed,
    so the caller falls through to yfinance transparently."""
    path = mt5_csv_path(pair, interval, exports_dir)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        colmap = {c.lower(): c for c in df.columns}

        # Time: either a single datetime column, or MT5's split <DATE>+<TIME>.
        tcol = _find(colmap, _TIME_COLS)
        if tcol is None and "<date>" in colmap:
            tcol = colmap["<date>"]
        if tcol is None:
            warnings.warn(f"[mt5_feed] {path}: no recognisable time column; ignoring MT5 file.")
            return None
        if "<date>" in colmap and "<time>" in colmap:
            ts = pd.to_datetime(df[colmap["<date>"]].astype(str) + " " + df[colmap["<time>"]].astype(str),
                                errors="coerce")
        else:
            ts = pd.to_datetime(df[tcol], errors="coerce")

        out = pd.DataFrame(index=pd.DatetimeIndex(ts))
        for canon, aliases in _COL_ALIASES.items():
            src = _find(colmap, aliases)
            if src is not None:
                out[canon] = pd.to_numeric(df[src].values, errors="coerce")
            elif canon == "volume":
                out[canon] = 0.0
        missing = [c for c in ("open", "high", "low", "close") if c not in out.columns]
        if missing:
            warnings.warn(f"[mt5_feed] {path}: missing OHLC columns {missing}; ignoring MT5 file.")
            return None

        out = out[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        # tz-naive, de-duplicated, chronological
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        print(f"[mt5_feed] using MT5 rates for {pair}: {len(out):,} bars from {path} "
              f"({out.index.min().date()} -> {out.index.max().date()}).")
        return out
    except Exception as e:
        warnings.warn(f"[mt5_feed] failed to parse {path} ({type(e).__name__}: {e}); "
                      f"falling back to yfinance.")
        return None
