"""Vanilla (unidirectional, single-stream) LSTM baseline (Section 1.3)."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DATA_CFG


class VanillaLSTM(nn.Module):
    """A plain LSTM operating directly on the fused 22-feature input,
    with no CNN, no attention, and no regime-awareness -- the natural
    single-architecture baseline the hybrid model is compared against.

    Includes the same auxiliary directional-classification head as the
    Hybrid model (see models/hybrid_model.py), so the comparison isolates
    the effect of architecture, not "one model got a classification head
    and the other didn't".
    """

    def __init__(self, input_size: int = DATA_CFG.n_total_features, hidden_size: int = 128, horizon: int = DATA_CFG.horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x_quant: torch.Tensor, x_text: torch.Tensor = None, regime_ctx: torch.Tensor = None, xgb_pred: torch.Tensor = None) -> dict:
        # Single-stream baseline: re-fuse the two modality tensors it is
        # handed by the dual-tower DataLoader into one (B, T, 30) input.
        x = x_quant if x_text is None else torch.cat([x_quant, x_text], dim=-1)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, hidden)
        forecast = self.head(last_hidden)
        direction_logits = self.direction_head(last_hidden)
        return {"forecast": forecast, "direction_logits": direction_logits}
