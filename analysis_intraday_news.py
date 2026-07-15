"""
Intraday news event study -- the honest test of news alpha at publication-hour
resolution.

The daily-scale falsification showed headline sentiment adds no next-day
directional accuracy: by the daily close, news is priced in. The remaining
honest question: does sentiment predict returns in the HOURS after
publication? This event study aligns every directional FinBERT-scored
headline (|polarity| >= 0.15) to the first hourly XAU/USD bar strictly after
its timestamp and measures sign-aligned forward returns at +1h, +3h, +6h,
+24h -- plus the same-day close-to-close as the "priced-in" control.

No model, no fitting -- a pure event study. Output: per-horizon event
directional accuracy vs the 0.5 base rate with binomial standard errors.

Usage:  python analysis_intraday_news.py
Writes: exports/intraday_event_study.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd


def main():
    import yfinance as yf

    print("[intraday] fetching hourly GC=F bars (max ~730 days) ...")
    bars = yf.Ticker("GC=F").history(period="730d", interval="1h")
    if bars.empty:
        sys.exit("hourly fetch failed")
    bars.index = pd.to_datetime(bars.index, utc=True).tz_localize(None)  # UTC-naive
    close = bars["Close"].astype(float)
    times = close.index.values.astype("datetime64[ns]")
    logc = np.log(close.values)
    print(f"[intraday] {len(bars)} hourly bars {close.index[0]} -> {close.index[-1]}")

    news = pd.read_csv("exports/archive/news_GCF.csv", parse_dates=["timestamp"])
    news = news.dropna(subset=["polarity"])
    news = news[news["timestamp"] >= close.index[0]]
    directional = news[np.abs(news["polarity"]) >= 0.15].copy()
    print(f"[intraday] {len(directional)} directional headlines inside the hourly window")

    horizons = {"+1h": 1, "+3h": 3, "+6h": 6, "+24h": 24}
    results = {}
    rows = []
    for _, r in directional.iterrows():
        ts = np.datetime64(r["timestamp"])
        i = int(np.searchsorted(times, ts, side="right"))       # first bar AFTER publication
        if i < 1 or i + 24 >= len(logc):
            continue
        entry = logc[i]                                          # first tradeable price after the news
        row = {"polarity": float(r["polarity"]), "sign": float(np.sign(r["polarity"]))}
        for name, h in horizons.items():
            row[name] = float(logc[i + h] - entry)
        rows.append(row)
    ev = pd.DataFrame(rows)
    print(f"[intraday] {len(ev)} events with full forward windows")

    out = {"n_events": int(len(ev)), "bars": int(len(bars)),
           "window": [str(close.index[0]), str(close.index[-1])], "horizons": {}}
    for name in horizons:
        hits = (np.sign(ev[name]) == ev["sign"])
        n = int(hits.notna().sum()); acc = float(hits.mean())
        se = float(np.sqrt(0.25 / n))
        mean_ret = float((ev[name] * ev["sign"]).mean())         # signed (aligned) mean return
        out["horizons"][name] = {"diracc": acc, "se": se, "n": n,
                                 "signed_mean_return_bps": mean_ret * 1e4,
                                 "z_vs_coinflip": (acc - 0.5) / se}
        print(f"  {name:>4s}: event DirAcc {acc:.4f} (±{se:.4f}, z={out['horizons'][name]['z_vs_coinflip']:+.2f}) "
              f"signed mean {mean_ret*1e4:+.1f}bps  n={n}")

    # strong-signal subset
    strong = ev[np.abs(ev["polarity"]) >= 0.5]
    if len(strong) > 50:
        out["strong_subset"] = {}
        for name in horizons:
            hits = (np.sign(strong[name]) == strong["sign"])
            acc = float(hits.mean()); n = int(len(strong))
            out["strong_subset"][name] = {"diracc": acc, "n": n}
        print("  strong (|pol|>=0.5):", {k: round(v['diracc'], 4) for k, v in out["strong_subset"].items()})

    json.dump(out, open("exports/intraday_event_study.json", "w"), indent=2)
    print("[intraday] written exports/intraday_event_study.json")


if __name__ == "__main__":
    main()
