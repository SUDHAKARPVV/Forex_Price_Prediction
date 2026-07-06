"""
Multi-model feature-fusion layer (Section 3.1.1, Figure 2).

Projects the concatenated [technical | macro | sentiment] tensor (22 dims)
into a 64-dim fused representation, with a small learned cross-modal
weighting (gating) network that lets the model up-weight the sentiment
block during news-driven moves and down-weight it during quiet trading.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DATA_CFG, MODEL_CFG


class FeatureFusion(nn.Module):
    def __init__(
        self,
        n_technical: int = DATA_CFG.n_technical_features,
        n_macro: int = DATA_CFG.n_macro_features,
        n_sentiment: int = DATA_CFG.n_sentiment_features,
        fusion_out: int = MODEL_CFG.fusion_out,
    ):
        super().__init__()
        self.n_technical = n_technical
        self.n_macro = n_macro
        self.n_sentiment = n_sentiment
        n_total = n_technical + n_macro + n_sentiment

        # Per-modality projections (kept modest so ablations can zero one out)
        modality_dim = fusion_out // 2
        self.tech_proj = nn.Linear(n_technical, modality_dim)
        self.macro_proj = nn.Linear(n_macro, modality_dim)
        self.sent_proj = nn.Linear(n_sentiment, modality_dim)

        # Cross-modal gate: from the raw concatenated input, predict a
        # 3-way softmax weighting over {technical, macro, sentiment}.
        self.gate = nn.Sequential(
            nn.Linear(n_total, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

        self.fuse_out = nn.Linear(modality_dim * 3, fusion_out)
        self.norm = nn.LayerNorm(fusion_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 22) -> (B, T, 64)"""
        tech = x[..., : self.n_technical]
        macro = x[..., self.n_technical : self.n_technical + self.n_macro]
        sent = x[..., self.n_technical + self.n_macro :]

        gate_weights = torch.softmax(self.gate(x), dim=-1)  # (B, T, 3)
        w_tech, w_macro, w_sent = gate_weights.unbind(dim=-1)

        h_tech = self.tech_proj(tech) * w_tech.unsqueeze(-1)
        h_macro = self.macro_proj(macro) * w_macro.unsqueeze(-1)
        h_sent = self.sent_proj(sent) * w_sent.unsqueeze(-1)

        fused = torch.cat([h_tech, h_macro, h_sent], dim=-1)
        out = self.fuse_out(fused)
        out = self.norm(out)
        return self.act(out)
