"""Cross-harvest news between per-pair archives (zero fetching, zero scoring).

The archives were fetched with per-pair query lanes, but relevance overlaps:
gold's lanes catch joint metals stories ("Gold and silver rally as...") that
silver's thinner lanes missed -- 1,388 such headlines at audit time, already
FinBERT-scored. This script copies rows from SIBLING archives into a target
pair's archive when the title matches the target's OWN relevance filter
(filter_relevant_news, so foreign-exclusion/co-mention rules apply), dedupes
by title (the archive convention), and re-saves. Scores travel with the rows,
so FinBERT is never invoked.

Run from repo root:
    python scripts/cross_harvest_news.py                 # silver + euro
    python scripts/cross_harvest_news.py --pairs XAG/USD
"""
from __future__ import annotations

# Resolve project imports when run as `python scripts/<this>.py`.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import argparse

import pandas as pd

from data.pairs import PAIRS, get_pair
from data.real_data_feed import filter_relevant_news, news_archive_path

# harvest sources per target: every OTHER pair's archive
ALL_TICKERS = {name: cfg.ticker for name, cfg in PAIRS.items()}


def harvest(target: str) -> None:
    cfg = get_pair(target)
    tpath = news_archive_path(cfg.ticker)
    tgt = pd.read_csv(tpath, parse_dates=["timestamp"])
    have = set(tgt["title"].astype(str))
    before_years = tgt["timestamp"].dt.year.value_counts().sort_index()

    pools = []
    for name, ticker in ALL_TICKERS.items():
        if name == target:
            continue
        spath = news_archive_path(ticker)
        if not _os.path.exists(spath):
            continue
        src = pd.read_csv(spath, parse_dates=["timestamp"])
        src = src[~src["title"].astype(str).isin(have)]
        pools.append(src)
    if not pools:
        print(f"[harvest] {cfg.slug}: no sibling archives found")
        return
    pool = pd.concat(pools, ignore_index=True).drop_duplicates(subset=["title"])

    # the target pair's OWN relevance filter decides what counts (this applies
    # the asset/macro patterns AND the foreign-exclusion/co-mention rules)
    kept = filter_relevant_news(pool, pair=cfg)
    kept = kept[kept["polarity"].notna()] if "polarity" in kept.columns else kept
    if kept.empty:
        print(f"[harvest] {cfg.slug}: nothing new to add")
        return

    merged = (pd.concat([tgt, kept], ignore_index=True)
              .dropna(subset=["title"]).drop_duplicates(subset=["title"])
              .sort_values("timestamp").reset_index(drop=True))
    merged.to_csv(tpath, index=False)

    gained = merged["timestamp"].dt.year.value_counts().sort_index() - before_years.reindex(
        merged["timestamp"].dt.year.unique(), fill_value=0).sort_index().fillna(0)
    print(f"[harvest] {cfg.slug}: {len(tgt):,} -> {len(merged):,} headlines "
          f"(+{len(merged) - len(tgt):,}, all pre-scored)")
    print("  per-year gains: " + ", ".join(f"{int(y)}:+{int(g)}" for y, g in gained.items() if g > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["XAG/USD", "EUR/USD"])
    args = ap.parse_args()
    for p in args.pairs:
        harvest(p)


if __name__ == "__main__":
    main()
