"""Feature-group ablation on the MAGNITUDE target -- does the multimodal fusion
(news + macro) actually drive the move-magnitude edge, or do price-only
technicals suffice? This is the dissertation's core thesis test, and it is only
meaningful on a target that HAS signal (magnitude); on direction, where
everything is noise, an ablation would compare noise to noise.

Trains the Hybrid on the magnitude target under four feature sets (seed 9), by
zeroing feature-group columns in the shared normalized panel (FOREX_FEATURE_
GROUPS, handled in train_pairs.run_pair -- masks the datasets AND the XGBoost
expert consistently; GARCH stays price-only):

    tech         technical only (price/OHLC/indicators, 18)
    tech_macro   + macro (rates, CPI, dollar; 24)
    tech_news    + FinBERT sentiment (31)
    all          everything (37)   <- reused from the committed seed-9 result

The contribution of each modality is (config - tech) on model skill (spearman
rank + large-move accuracy) vs the SAME actual |10-bar move|. atr_pct / GARCH-
sigma baselines are identical across configs (they never use the masked cols),
so the model's own skill is the only moving part.

~2h/config x 3 trained configs (all is reused) => ~6h. Single seed: the
magnitude result is extraordinarily seed-stable (sd ~5e-5 on spearman), so
cross-config deltas above ~1e-3 are real, not RNG.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.getcwd())

import json

import xgboost  # noqa: F401  (before torch)

from scripts.train_pairs import run_pair
from data.pairs import get_pair

PAIR = "XAU/USD"
SLUG = get_pair(PAIR).slug
SEED = 9
TRAIN_CONFIGS = ["tech", "tech_macro", "tech_news"]   # "all" reused from committed


def _score_of(meta):
    mm = meta["magnitude_vs_atr"]
    return {
        "model_spearman": mm["model_spearman"],
        "model_large_move_acc": mm["model_large_move_acc"],
        "atr_pct_spearman": mm["atr_pct_spearman"],
        "atr_pct_large_move_acc": mm["atr_pct_large_move_acc"],
        "garch_sigma_spearman": mm.get("garch_sigma_spearman"),
        "garch_sigma_large_move_acc": mm.get("garch_sigma_large_move_acc"),
    }


rows = {}

# --- all-features baseline: reuse the committed seed-9 magnitude result ---
allf = json.load(open(f"results/pair_metrics/{SLUG}_magnitude_seed{SEED}.json"))
rows["all"] = _score_of(allf)
print(f"[ablation] all (reused): model spearman {rows['all']['model_spearman']:.4f} "
      f"| acc {rows['all']['model_large_move_acc']:.4f}")

# --- train the three ablated feature sets ---
for cfg in TRAIN_CONFIGS:
    os.environ["FOREX_FEATURE_GROUPS"] = cfg
    print(f"\n{'='*70}\nABLATION CONFIG: {cfg}\n{'='*70}")
    meta = run_pair(PAIR, interval="1h", source="panel",
                    train_stride=3, refit_every=100, target="magnitude", seed=SEED)
    rows[cfg] = _score_of(meta)
os.environ.pop("FOREX_FEATURE_GROUPS", None)

# --- contributions (config - tech) on model skill ---
tech = rows["tech"]
contrib = {}
for cfg in ("tech_macro", "tech_news", "all"):
    contrib[cfg] = {
        "d_spearman": rows[cfg]["model_spearman"] - tech["model_spearman"],
        "d_acc": rows[cfg]["model_large_move_acc"] - tech["model_large_move_acc"],
    }

summary = {
    "pair": PAIR, "slug": SLUG, "seed": SEED, "target": "magnitude", "interval": "1h",
    "configs": rows,
    "contributions_vs_tech": contrib,
    "macro_adds": contrib["tech_macro"],
    "news_adds": contrib["tech_news"],
    "both_add": contrib["all"],
}
out = f"results/magnitude_ablation_{SLUG}.json"
json.dump(summary, open(out, "w"), indent=2, default=float)

print(f"\n{'='*70}\nMAGNITUDE FEATURE-GROUP ABLATION (seed {SEED})\n{'='*70}")
print(f"{'config':14s}{'model spearman':>16s}{'model acc':>12s}")
for cfg in ("tech", "tech_macro", "tech_news", "all"):
    r = rows[cfg]
    print(f"{cfg:14s}{r['model_spearman']:>16.4f}{r['model_large_move_acc']:>12.4f}")
print(f"\ncontribution vs price-only (tech):")
print(f"  + macro   : spearman {contrib['tech_macro']['d_spearman']:+.4f} | acc {contrib['tech_macro']['d_acc']*100:+.2f}pp")
print(f"  + news    : spearman {contrib['tech_news']['d_spearman']:+.4f} | acc {contrib['tech_news']['d_acc']*100:+.2f}pp")
print(f"  + both    : spearman {contrib['all']['d_spearman']:+.4f} | acc {contrib['all']['d_acc']*100:+.2f}pp")
_thr = 0.003
_helps = [m for m, k in (("macro", "tech_macro"), ("news", "tech_news"))
          if contrib[k]["d_spearman"] > _thr or contrib[k]["d_acc"] > _thr]
print(f"\nread: modalities that add > {_thr:.3f} to model skill over price-only: "
      f"{_helps if _helps else 'NONE -- price-only technicals suffice; multimodal fusion does not help magnitude'}")
print(f"written to {out}")
