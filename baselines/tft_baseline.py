"""
Simplified Temporal Fusion Transformer (TFT) reference baseline (Section 1.3
/ Section 3.3). A full TFT includes variable-selection networks, static
covariate encoders, gated residual networks, and quantile outputs; here we
implement a compact version that captures its core idea (LSTM encoder +
self-attention + gated skip connections) so it serves as a meaningful
single-architecture comparison point without the full engineering overhead
of the reference TFT paper implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DATA_CFG


class GatedResidualNetwork(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class SimplifiedTFT(nn.Module):
    def __init__(self, input_size: int = DATA_CFG.n_total_features, d_model: int = 128, horizon: int = DATA_CFG.horizon):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.grn = GatedResidualNetwork(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.1
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
        self.direction_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x: torch.Tensor, regime_ctx: torch.Tensor = None, xgb_pred: torch.Tensor = None) -> dict:
        h = self.input_proj(x)
        h, _ = self.lstm_encoder(h)
        h = self.grn(h)
        h = self.attn(h)
        pooled = h.mean(dim=1)
        forecast = self.head(pooled)
        direction_logits = self.direction_head(pooled)
        return {"forecast": forecast, "direction_logits": direction_logits}
