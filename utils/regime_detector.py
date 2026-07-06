"""
Rule-based volatility regime labelling used ONLY for evaluation segmentation
(Section 3.1.5's rolling-realised-volatility / ATR thresholding). This is
independent of the model's own learned soft regime gate — it exists so we
can report regime-segmented metrics against an externally-defined regime
label, as Section 4 describes.
"""
from __future__ import annotations

import numpy as np


def label_regimes(realized_vol: np.ndarray, upper_quantile: float = 0.7) -> np.ndarray:
    """Label each timestep 1 (high-volatility) if realised vol exceeds the
    given quantile of its own distribution, else 0 (stable)."""
    threshold = np.quantile(realized_vol, upper_quantile)
    return (realized_vol >= threshold).astype(int)
