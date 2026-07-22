"""CFTC Commitments-of-Traders (COT) pre-check for gold H1 -- scanned on TRAIN+VAL
ONLY before any wiring (the same discipline that kept envelope honest and kept
FVG / order-book out). COT is the one order-flow-type source that is both FREE
and genuinely HISTORICAL (weekly COMEX gold large-trader positioning, back to
2015+), so it can actually feed the fusion model's training -- unlike the
retail/live-only OANDA book.

Source: CFTC public Socrata API (Legacy Futures-Only), contract 088691.
Look-ahead: COT is "as of" Tuesday but released ~Friday, so a report is marked
available report_date + 4 days (Saturday) and merged BACKWARD onto each H1 bar
-- no bar ever sees a report before it was public.

Signal is weekly positioning, so it plausibly relates to WEEKLY direction/vol,
not the next 10 hours. We therefore scan the model's own 10-H1-bar targets
(what we'd wire it for) AND a ~1-week (120-bar) target to give COT its best
shot. Reported for BOTH direction (base-rate-controlled quintile P(up) spread)
and magnitude (spearman + quintile |move| spread).
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import json
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_CFG

COT_CSV = "exports/pairs/XAUUSD/cot_gold.csv"


def fetch_cot():
    base = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        "$where": "cftc_contract_market_code='088691' and "
                  "report_date_as_yyyy_mm_dd>='2014-06-01T00:00:00.000'",
        "$select": "report_date_as_yyyy_mm_dd,open_interest_all,"
                   "noncomm_positions_long_all,noncomm_positions_short_all,"
                   "comm_positions_long_all,comm_positions_short_all",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "2000",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    rows = json.load(urllib.request.urlopen(req, timeout=90))
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "report_date_as_yyyy_mm_dd":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    _os.makedirs(_os.path.dirname(COT_CSV), exist_ok=True)
    df.to_csv(COT_CSV, index=False)
    print(f"[cot] fetched {len(df)} weekly rows {df['report_date'].min().date()} -> "
          f"{df['report_date'].max().date()} -> {COT_CSV}")
    return df


def build_cot_features(df):
    df = df.sort_values("report_date").reset_index(drop=True)
    oi = df["open_interest_all"].replace(0, np.nan)
    spec_net = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    comm_net = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    feats = pd.DataFrame({"report_date": df["report_date"]})
    feats["cot_spec_net_pct_oi"] = (spec_net / oi).astype(float)        # spec positioning level
    feats["cot_spec_long_ratio"] = (df["noncomm_positions_long_all"] /
                                    (df["noncomm_positions_long_all"] +
                                     df["noncomm_positions_short_all"]).replace(0, np.nan)).astype(float)
    feats["cot_comm_net_pct_oi"] = (comm_net / oi).astype(float)        # hedgers ("smart money")
    feats["cot_spec_net_chg"] = spec_net.diff().astype(float)          # weekly positioning momentum
    feats["cot_oi_chg_pct"] = oi.pct_change().astype(float)            # open-interest change
    # positioning extremity (|z| over trailing 52w) -- mean-reversion / vol hypothesis
    m = (spec_net / oi)
    feats["cot_spec_extremity"] = (m - m.rolling(52, min_periods=12).mean()).abs() / \
        m.rolling(52, min_periods=12).std()
    # released ~Friday (report is Tuesday) -> available report_date + 4 days (Sat)
    feats["available_date"] = df["report_date"] + pd.Timedelta(days=4)
    return feats.dropna(subset=["available_date"]).sort_values("available_date")


FEATURES = ["cot_spec_net_pct_oi", "cot_spec_long_ratio", "cot_comm_net_pct_oi",
            "cot_spec_net_chg", "cot_oi_chg_pct", "cot_spec_extremity"]


def scan():
    cot = pd.read_csv(COT_CSV) if _os.path.exists(COT_CSV) else fetch_cot()
    if isinstance(cot, pd.DataFrame) and "report_date" in cot.columns and cot["report_date"].dtype == object:
        cot["report_date"] = pd.to_datetime(cot["report_date"])
    feats = build_cot_features(cot if "noncomm_positions_long_all" in cot.columns else fetch_cot())

    panel = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv", parse_dates=["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    # leak-free merge: each bar sees only the most recent PUBLIC report
    merged = pd.merge_asof(panel[["date", "close"]], feats,
                           left_on="date", right_on="available_date", direction="backward")

    logc = np.log(merged["close"].to_numpy())
    N = len(merged)
    K = DATA_CFG.horizon                                    # 10 H1 bars (the model's target)
    KW = 120                                                # ~1 trading week of H1 bars
    cut = int(N * (DATA_CFG.train_frac + DATA_CFG.val_frac))  # TRAIN+VAL only

    def targets(k):
        fut = np.full(N, np.nan)
        fut[:-k] = logc[k:] - logc[:-k]
        return fut

    print(f"\nCOT pre-check on gold H1 -- TRAIN+VAL only ({cut:,} of {N:,} bars, "
          f"through {merged['date'].iloc[cut].date()})")
    print(f"COT coverage on train+val: "
          f"{merged[FEATURES].iloc[:cut].notna().all(axis=1).mean()*100:.1f}% of bars\n")

    for k, lbl in ((K, f"{K}-bar (model target)"), (KW, f"{KW}-bar (~1 week)")):
        fut = targets(k)
        base = float((fut[:cut] > 0).mean())
        print(f"=== horizon {lbl} | base P(up)={base:.3f} ===")
        print(f"{'feature':22s}{'|rho| DIR':>11s}{'quint P(up)':>13s}{'rho MAG':>10s}{'quint|mv|':>11s}")
        for f in FEATURES:
            x = merged[f].to_numpy()[:cut]
            y = fut[:cut]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 500:
                continue
            rho_dir = spearmanr(x[m], y[m]).statistic          # signed-return rank corr
            rho_mag = spearmanr(x[m], np.abs(y[m])).statistic   # magnitude rank corr
            try:
                q = pd.qcut(pd.Series(x[m]), 5, labels=False, duplicates="drop")
                pup = pd.Series((y[m] > 0)).groupby(q).mean()
                dir_spread = float(pup.max() - pup.min())
                mv = pd.Series(np.abs(y[m])).groupby(q).mean()
                mag_spread = float((mv.max() - mv.min()) / np.abs(y[m]).mean())  # rel. spread
            except Exception:
                dir_spread = mag_spread = float("nan")
            print(f"{f:22s}{abs(rho_dir):>11.3f}{dir_spread:>13.3f}{rho_mag:>+10.3f}{mag_spread:>11.2f}")
        print()

    print("read: DIR |rho|/quint-spread test directional signal (base-rate-controlled); "
          "MAG rho / quint|mv| test volatility-magnitude signal. As with FVG, a feature "
          "earns wiring only if it clears noise on train+val (|rho|>~0.05 or a quintile "
          "spread well above the base rate).")


if __name__ == "__main__":
    scan()
