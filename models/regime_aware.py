"""
Regime-aware output layer (Section 3.1.5, Figure 6).

A lightweight volatility-regime detector consumes the pooled global context
vector (plus rolling realised-volatility / ATR side-inputs) and produces a
soft gate in [0, 1] over two decoder heads:
    - stable-regime head:      standard multi-step decoder, tight bands
    - high-volatility head:    trained emphasis on directional accuracy

Rather than a hard routing decision (which would be non-differentiable),
we use a *soft* gate (Figure 6's routing logic implemented as a learned
convex combination) so the whole network remains end-to-end trainable,
consistent with the "soft-gated dual MLP decoder heads" description.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import MODEL_CFG


class VolatilityRegimeDetector(nn.Module):
    """Maps the pooled context vector (+ 2 side features: realised vol, ATR)
    to a scalar high-volatility gate in [0, 1]."""

    def __init__(self, context_dim: int = MODEL_CFG.transformer_d_model, hidden: int = MODEL_CFG.regime_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor, regime_ctx: torch.Tensor) -> torch.Tensor:
        """context: (B, d_model), regime_ctx: (B, 2) -> gate: (B, 1) in [0,1]"""
        inp = torch.cat([context, regime_ctx], dim=-1)
        return torch.sigmoid(self.net(inp))


class DecoderHead(nn.Module):
    """Standard multi-step MLP decoder producing k point forecasts."""

    def __init__(self, context_dim: int = MODEL_CFG.transformer_d_model, hidden: int = MODEL_CFG.regime_hidden, horizon: int = MODEL_CFG.horizon):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class RegimeAwareOutputLayer(nn.Module):
    def __init__(
        self,
        context_dim: int = MODEL_CFG.transformer_d_model + MODEL_CFG.skip_embed_dim,
        hidden: int = MODEL_CFG.regime_hidden,
        horizon: int = MODEL_CFG.horizon,
    ):
        super().__init__()
        self.regime_detector = VolatilityRegimeDetector(context_dim, hidden)
        self.stable_head = DecoderHead(context_dim, hidden, horizon)
        self.high_vol_head = DecoderHead(context_dim, hidden, horizon)
        # Learnable per-horizon widening factor applied to the high-vol head's
        # implicit uncertainty (used only when reporting bands, not in the
        # point-forecast loss).
        self.uncertainty_scale = nn.Parameter(torch.ones(horizon) * 1.5)

    def forward(self, context: torch.Tensor, regime_ctx: torch.Tensor):
        """
        context:    (B, d_model) pooled global context vector
        regime_ctx: (B, 2) [realised_vol, atr] at the forecast origin

        Returns:
            forecast: (B, k) final regime-conditioned point forecast
            gate:     (B, 1) high-volatility gate weight (for diagnostics/XAI)
            band:     (B, k) widened uncertainty band estimate
        """
        gate = self.regime_detector(context, regime_ctx)          # (B, 1)
        stable_out = self.stable_head(context)                     # (B, k)
        high_vol_out = self.high_vol_head(context)                 # (B, k)

        forecast = (1 - gate) * stable_out + gate * high_vol_out   # soft routing
        base_band = torch.abs(high_vol_out - stable_out) + 1e-4
        band = base_band * (1 + gate * (self.uncertainty_scale - 1))
        return forecast, gate, band
