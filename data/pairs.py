"""
Single source of truth for the currency pairs the project supports.

Everything that used to be hard-coded to gold (the Yahoo ticker, the news
query lanes, the FinBERT relevance filter, the on-disk file paths) is now
keyed off a PairConfig here, so building/analysing XAG/USD (silver) or
EUR/USD (euro) NEVER overwrites gold's raw files -- each pair owns its own
prices, news, macro, sentiment and feature-panel artifacts.

File layout (per pair, under exports/):
    feature panel   XAU/USD -> feature_panel.csv               (legacy path, gold)
                    others  -> pairs/<slug>/feature_panel.csv
    intermediates   XAU/USD -> fx_prices_yfinance.csv, ...      (legacy top-level)
                    others  -> pairs/<slug>/<name>.csv
    news archive    -> archive/news_<tickersafe>.csv            (already per-ticker)

Gold keeps its historical top-level paths so existing checkpoints, the
committed panel and older tooling keep working unchanged; silver and euro
are cleanly separated. `panel_csv_path()` / `intermediate_path()` are the
helpers the data layer and dashboard call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PairConfig:
    name: str            # canonical "XAU/USD"
    ticker: str          # yfinance ticker, e.g. "GC=F"
    slug: str            # filesystem-safe id, e.g. "XAUUSD"
    label: str           # human label for the UI, e.g. "Gold (XAU/USD)"
    emoji: str
    # ---- news / relevance (per pair) ----
    asset_pattern: str   # regex: headline is ON this asset
    macro_pattern: str   # regex: shared US-macro drivers (COMMON across pairs)
    foreign_pattern: str # regex: OTHER-asset chatter to exclude (pair-specific)
    gnews_lanes: dict    # {lane_name: google-news query}
    gdelt_query: str     # GDELT DOC query for this pair

    @property
    def ticker_safe(self) -> str:
        return (self.ticker.replace("=", "").replace("^", "")
                .replace("-", "").replace(".", ""))


# ---- shared US-macro drivers (Fed policy, inflation, dollar, real yields).
# These move every dollar-denominated pair, so the macro stream + this macro
# relevance clause are COMMON across pairs (per the requirement that macro
# news is shared). ----
_US_MACRO = (
    r"federal reserve|\bfomc\b|\bpowell\b|\bfed\b|rate (?:cut|hike|decision)"
    r"|interest rate decision|\bcpi\b|us inflation|\bpce\b|dollar index|\bdxy\b"
    r"|treasury yield|real yield|safe.?haven"
)

# Generic single-currency / cross-asset chatter. Each pair excludes the OTHER
# currencies (its own terms are removed from its foreign_pattern below).
_FX_MAJORS_EM = (
    r"new zealand dollar|australian dollar|canadian dollar|swiss franc"
    r"|british pound|japanese yen|\bnzd\b|\baud\b|\bcad\b|\bchf\b"
    r"|\bgbp\b|\bjpy\b|aud/usd|gbp/usd|usd/jpy|nzd/usd|usd/cad|usd/chf"
    r"|rbnz|\brba\b|\bboe\b|\bboj\b"
    r"|swedish krona|norwegian krone|danish krone|polish zloty|hungarian forint"
    r"|czech koruna|mexican peso|south african rand|turkish lira|indian rupee"
    r"|chinese yuan|renminbi|singapore dollar|hong kong dollar|brazilian real"
    r"|\bsek\b|\bnok\b|\bpln\b|\bhuf\b|\bczk\b|\bmxn\b|\bzar\b|\btry\b|\binr\b"
    r"|\bcny\b|\bsgd\b|\bhkd\b|\bbrl\b|\bkrw\b|krona|krone|zloty|forint|renminbi"
    r"|\bnbp\b|riksbank|norges bank|banxico"
)


PAIRS: "dict[str, PairConfig]" = {
    "XAU/USD": PairConfig(
        name="XAU/USD", ticker="GC=F", slug="XAUUSD", label="Gold (XAU/USD)", emoji="🥇",
        asset_pattern=r"\bgold\b|xau|bullion|precious metal|comex gold|gold price|spot gold",
        macro_pattern=_US_MACRO,
        # exclude foreign-FX (incl. euro) UNLESS the headline also names gold
        foreign_pattern=_FX_MAJORS_EM + r"|\beuro\b|eur/usd|\becb\b",
        gnews_lanes={
            "reuters": '(gold OR bullion OR "gold price") site:reuters.com',
            "general": '"gold price" OR "gold prices" OR "gold market" OR bullion OR "spot gold"',
        },
        gdelt_query='("gold price" OR "gold prices" OR "gold market" OR "gold rally" OR bullion)',
    ),
    "XAG/USD": PairConfig(
        name="XAG/USD", ticker="SI=F", slug="XAGUSD", label="Silver (XAG/USD)", emoji="🥈",
        asset_pattern=r"\bsilver\b|xag|silver price|spot silver|comex silver|precious metal|bullion",
        macro_pattern=_US_MACRO,
        foreign_pattern=_FX_MAJORS_EM + r"|\beuro\b|eur/usd|\becb\b",
        gnews_lanes={
            "reuters": '(silver OR "silver price" OR bullion) site:reuters.com',
            "general": '"silver price" OR "silver prices" OR "silver market" OR "spot silver"',
        },
        gdelt_query='("silver price" OR "silver prices" OR "silver market" OR "spot silver" OR bullion)',
    ),
    "EUR/USD": PairConfig(
        name="EUR/USD", ticker="EURUSD=X", slug="EURUSD", label="Euro (EUR/USD)", emoji="💶",
        asset_pattern=(r"eur/usd|euro.?dollar|\beuro\b|\beur\b|european central bank"
                       r"|\becb\b|euro ?zone|euro area|\blagarde\b"),
        macro_pattern=_US_MACRO + r"|european central bank|\becb\b|\blagarde\b|euro ?zone",
        # exclude the OTHER currencies but NOT the euro itself
        foreign_pattern=_FX_MAJORS_EM + r"|\bgold\b|xau|bullion|\bsilver\b|xag",
        gnews_lanes={
            "reuters": '("EUR/USD" OR euro OR "European Central Bank") site:reuters.com',
            "general": ('"EUR/USD" OR "euro dollar" OR "euro rises" OR "euro falls" '
                        'OR "European Central Bank" OR eurozone'),
        },
        gdelt_query=('("EUR/USD" OR "euro dollar" OR "euro rises" OR "euro falls" '
                     'OR "European Central Bank" OR eurozone)'),
    ),
}

# Reverse lookup by ticker (the data feed only knows the ticker at fetch time).
_BY_TICKER = {c.ticker: c for c in PAIRS.values()}

DEFAULT_PAIR = "XAU/USD"


def get_pair(pair) -> PairConfig:
    # Accept a PairConfig directly (it's unhashable due to the dict field, so
    # this must come before any dict-membership test).
    if isinstance(pair, PairConfig):
        return pair
    if pair in PAIRS:
        return PAIRS[pair]
    if pair in _BY_TICKER:
        return _BY_TICKER[pair]
    raise KeyError(f"Unknown pair/ticker {pair!r}; known: {list(PAIRS)}")


def pair_for_ticker(ticker: str) -> "PairConfig | None":
    return _BY_TICKER.get(ticker)


def pair_slug(pair: str) -> str:
    return get_pair(pair).slug


def pair_dir(pair: str, exports_dir: str = "exports") -> str:
    """Per-pair artifact directory (created on demand)."""
    d = os.path.join(exports_dir, "pairs", get_pair(pair).slug)
    os.makedirs(d, exist_ok=True)
    return d


def panel_csv_path(pair: str, exports_dir: str = "exports") -> str:
    """Feature-panel path for a pair. Gold keeps the legacy top-level path
    (backward-compat with the committed panel + existing checkpoints); other
    pairs live under exports/pairs/<slug>/."""
    if get_pair(pair).name == DEFAULT_PAIR:
        return os.path.join(exports_dir, "feature_panel.csv")
    return os.path.join(pair_dir(pair, exports_dir), "feature_panel.csv")


def intermediate_path(pair: str, name: str, exports_dir: str = "exports") -> str:
    """Path for an intermediate CSV (prices/news/macro/sentiment). Gold keeps
    its legacy top-level filename; other pairs are namespaced under their
    per-pair directory so they never clobber gold's raw extracts."""
    if get_pair(pair).name == DEFAULT_PAIR:
        return os.path.join(exports_dir, name)
    return os.path.join(pair_dir(pair, exports_dir), name)


def checkpoint_dir(pair: str, base: str = "exports/dashboard") -> str:
    """Per-pair saved-model directory. Gold keeps exports/dashboard/ (legacy);
    others get exports/dashboard/<slug>/."""
    if get_pair(pair).name == DEFAULT_PAIR:
        return base
    d = os.path.join(base, get_pair(pair).slug)
    os.makedirs(d, exist_ok=True)
    return d
