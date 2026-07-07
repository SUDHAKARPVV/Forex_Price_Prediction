"""
Regime-aware output layer (Section 3.1.5, Figure 6), revised to a
PROBABILISTIC distribution head (GARCH emulation).

A lightweight volatility-regime detector consumes the pooled global context
vector (plus rolling realised-volatility / ATR side-inputs) and produces a
soft gate in [0, 1] over two decoder heads (stable-regime / high-vol),
exactly as before -- but each head now parameterises a full Gaussian per
horizon step instead of a point estimate:

    head output (2k):  mu_h  and  log sigma^2_h   for h = 1..k

Why: GARCH's advantage is that it models conditional VARIANCE explicitly;
a point-MSE network on fat-tailed returns is pulled toward predicting a
conservative ~0 mean, destroying directional conviction. Trained under
Gaussian negative log-likelihood (training/train.py), the network must
output its own per-step variance -- learning volatility clustering the way
GARCH does, and giving a principled conviction measure |mu|/sigma (a
t-statistic) that calibrates the abstention rule dynamically.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import MODEL_CFG, TRAIN_CFG


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
    """Multi-step probabilistic decoder: k means + k log-variances.

    log-variance is expressed in return_scale units (i.e. the variance of
    (y - mu)/return_scale), so a zero-initialised head starts at sigma =
    one representative return magnitude -- a sane prior."""

    def __init__(self, context_dim: int = MODEL_CFG.transformer_d_model, hidden: int = MODEL_CFG.regime_hidden, horizon: int = MODEL_CFG.horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(self, context: torch.Tensor):
        out = self.net(context)
        mu, log_var = out[:, : self.horizon], out[:, self.horizon :]
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)  # numerical safety for exp()
        return mu, log_var


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

    def forward(self, context: torch.Tensor, regime_ctx: torch.Tensor):
        """
        context:    (B, d_model) pooled global context vector
        regime_ctx: (B, 2) [realised_vol, atr] at the forecast origin

        Returns:
            forecast: (B, k) regime-blended mean forecast (log-return units)
            gate:     (B, 1) high-volatility gate weight (for diagnostics/XAI)
            band:     (B, k) predicted sigma in RAW log-return units
            log_var:  (B, k) predicted log-variance in return_scale units
                      (consumed by the Gaussian NLL loss)
        """
        gate = self.regime_detector(context, regime_ctx)            # (B, 1)
        mu_s, lv_s = self.stable_head(context)                      # (B, k) each
        mu_h, lv_h = self.high_vol_head(context)

        forecast = (1 - gate) * mu_s + gate * mu_h                  # soft routing
        log_var = (1 - gate) * lv_s + gate * lv_h
        band = torch.exp(0.5 * log_var) * TRAIN_CFG.return_scale    # sigma, raw units
        return forecast, gate, band, log_var
