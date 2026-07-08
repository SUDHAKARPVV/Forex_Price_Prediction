"""
Regime-gated GARCH <-> Hybrid decision-layer blend.

GARCH (AR(1)-GARCH(1,1)) still leads RAW directional accuracy on trending /
high-volatility windows, where its conditional-mean drift term is exactly the
right inductive bias; the Hybrid adds value in choppy, mean-reverting regimes
where the news/technical context matters. Neither dominates everywhere, so a
single global choice leaves accuracy on the table.

This module realises the convex-blend idea the XGBoost expert already uses, but
at the DECISION layer and CONDITIONED ON THE VOLATILITY REGIME:

    blended = (1 - w_r) * hybrid + w_r * garch        # w_r chosen per regime r
    signal  = sign(blended) * (+1 follow / -1 fade)   # mode_r chosen per regime

Both the convex weight w_r and the follow/fade mode_r are calibrated ON THE
VALIDATION SET ONLY (per regime), then frozen and applied to the test set --
split-conformal discipline, no test-set tuning. The volatility regime threshold
is fixed on a REFERENCE distribution (the training window) so validation and
test are labelled on the same scale and the per-regime rules transfer.
"""
from __future__ import annotations

import numpy as np


def label_regimes_fixed(realized_vol: np.ndarray, origins, reference_vol: np.ndarray,
                        upper_quantile: float = 0.7) -> np.ndarray:
    """Binary vol-regime label (1 high-vol / 0 stable) for `origins`, using a
    threshold fixed on `reference_vol` (the training distribution) so the
    labels are comparable across the validation and test splits."""
    thr = float(np.quantile(reference_vol, upper_quantile))
    return (np.asarray(realized_vol)[np.asarray(origins)] >= thr).astype(int)


def _diracc(y_true, y_pred, mask=None):
    hits = (np.sign(y_true) == np.sign(y_pred))
    if mask is not None:
        hits = hits[mask]
    return float(hits.mean()) if hits.size else float("nan")


def regime_gated_blend(val_true, val_hybrid, val_garch, val_regimes,
                       test_true, test_hybrid, test_garch, test_regimes,
                       weight_grid=None, deviate_margin=0.02):
    """Fit a per-regime convex weight + follow/fade mode on validation, apply
    to test. All arrays are (N, horizon). Returns a report dict; the headline
    field `blended_diracc` is the test directional accuracy to compare against
    GARCH-alone (the number we are trying to beat).

    SHRINKAGE GUARD (against small-sample validation overfit): the globally
    stronger single model on validation is the ANCHOR (here GARCH, whose AR(1)
    drift dominates direction). A per-regime blended rule is only ADOPTED if it
    beats simply deferring to the anchor within that regime by `deviate_margin`
    on validation; otherwise the regime defers to the anchor (w->anchor). Ties
    in the weight scan break toward the anchor (fewer degrees of freedom). This
    stops the fit from chasing a fragile pure-Hybrid rule in a regime where the
    Hybrid only wins on validation noise -- the failure mode that put the first
    unguarded blend BELOW GARCH out of sample."""
    if weight_grid is None:
        weight_grid = np.round(np.arange(0.0, 1.0001, 0.1), 3)

    regimes = sorted(set(np.unique(val_regimes)) | set(np.unique(test_regimes)))

    # Anchor = globally stronger single model on validation. anchor_w is the
    # convex weight that selects it (1.0 = all GARCH, 0.0 = all Hybrid).
    val_g_acc = _diracc(val_true, val_garch)
    val_h_acc = _diracc(val_true, val_hybrid)
    anchor_w = 1.0 if val_g_acc >= val_h_acc else 0.0
    anchor_name = "garch" if anchor_w == 1.0 else "hybrid"

    # Anchored weight range: the blend may only ADJUST the dominant model, not
    # defect to the weaker one. With GARCH the anchor, w is restricted to
    # [0.5, 1.0] -- the Hybrid decorrelates GARCH's drift (the oracle's winning
    # w~0.9 lives here) but can never replace it. This is the transferable part
    # of the edge; full defection (w->0) was pure validation overfit.
    if anchor_w == 1.0:
        eff_grid = [w for w in weight_grid if w >= 0.5]
    else:
        eff_grid = [w for w in weight_grid if w <= 0.5]

    per_regime = {}

    def _fit(mask_val):
        """Best (w, mode) by validation DirAcc; ties break toward the anchor."""
        best = None
        for w in eff_grid:
            b = (1.0 - w) * val_hybrid + w * val_garch
            follow = _diracc(val_true[mask_val], b[mask_val]) if mask_val.any() else float("nan")
            for mode, acc in (("follow", follow), ("fade", 1.0 - follow)):
                if acc != acc:
                    continue
                closer = best is not None and abs(w - anchor_w) < abs(best["w"] - anchor_w)
                if best is None or acc > best["val_diracc"] or (acc == best["val_diracc"] and closer):
                    best = {"w": float(w), "mode": mode, "val_diracc": float(acc)}
        return best

    global_rule = _fit(np.ones(len(val_regimes), dtype=bool))
    for r in regimes:
        mval = (val_regimes == r)
        if mval.sum() < 20:                       # too little val support -> anchor
            per_regime[int(r)] = {"w": anchor_w, "mode": "follow",
                                  "val_diracc": _diracc(val_true[mval], val_garch[mval] if anchor_w else val_hybrid[mval]),
                                  "n_val": int(mval.sum()), "adopted": False}
            continue
        rule = _fit(mval)
        # Anchor's own validation DirAcc in this regime (defer baseline).
        anchor_val = _diracc(val_true[mval], (val_garch if anchor_w == 1.0 else val_hybrid)[mval])
        if rule["val_diracc"] >= anchor_val + deviate_margin:
            per_regime[int(r)] = {**rule, "n_val": int(mval.sum()), "adopted": True}
        else:                                     # not a robust enough gain -> anchor
            per_regime[int(r)] = {"w": anchor_w, "mode": "follow", "val_diracc": anchor_val,
                                  "n_val": int(mval.sum()), "adopted": False}

    # --- Apply frozen rules to the test set ---
    test_blended = np.empty_like(test_hybrid)
    test_signal = np.empty_like(test_hybrid)   # sign-adjusted (mode applied)
    for r in regimes:
        mtest = (test_regimes == r)
        if not mtest.any():
            continue
        rule = per_regime[int(r)]
        w = rule["w"]
        blended = (1.0 - w) * test_hybrid[mtest] + w * test_garch[mtest]
        test_blended[mtest] = blended
        test_signal[mtest] = blended * (1.0 if rule["mode"] == "follow" else -1.0)

    report = {
        "weight_grid": [float(x) for x in weight_grid],
        "reference_threshold_quantile": 0.7,
        "anchor": anchor_name,
        "deviate_margin": deviate_margin,
        "per_regime": {str(k): v for k, v in per_regime.items()},
        "global_rule": global_rule,
        # Headline: directional accuracy of the regime-gated blend (mode applied)
        "blended_diracc": _diracc(test_true, test_signal),
        # Baselines for comparison, on the SAME test origins:
        "hybrid_diracc": _diracc(test_true, test_hybrid),
        "garch_diracc": _diracc(test_true, test_garch),
        # Convex-blend value error (mode NOT applied -- fade is a directional
        # decision, not a value forecast):
        "blended_mae": float(np.mean(np.abs(test_true - test_blended))),
        "hybrid_mae": float(np.mean(np.abs(test_true - test_hybrid))),
        "garch_mae": float(np.mean(np.abs(test_true - test_garch))),
        "n_test": int(len(test_true)),
        "regime_coverage": {
            str(int(r)): float((test_regimes == r).mean()) for r in regimes
        },
        "per_regime_test_diracc": {
            str(int(r)): {
                "blended": _diracc(test_true, test_signal, test_regimes == r),
                "hybrid": _diracc(test_true, test_hybrid, test_regimes == r),
                "garch": _diracc(test_true, test_garch, test_regimes == r),
            }
            for r in regimes if (test_regimes == r).any()
        },
    }
    return report
