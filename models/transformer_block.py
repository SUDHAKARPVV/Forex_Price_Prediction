"""
Transformer block (Section 3.1.4): 4 layers, 8 heads, d_model=256, FFN=1024,
with residual + layer-norm and a positional encoding so the self-attention
mechanism has access to sequence order.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import MODEL_CFG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerContextBlock(nn.Module):
    def __init__(
        self,
        d_model: int = MODEL_CFG.transformer_d_model,
        n_heads: int = MODEL_CFG.transformer_heads,
        n_layers: int = MODEL_CFG.transformer_layers,
        ffn_dim: int = MODEL_CFG.transformer_ffn,
        dropout: float = MODEL_CFG.transformer_dropout,
        causal: bool = MODEL_CFG.transformer_causal,
    ):
        super().__init__()
        self.causal = causal
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T', 256) -> (B, T', 256) context sequence.

        With `causal=True` (the default -- see Section 3.1.4 update below),
        a look-ahead mask restricts each position to attend only to itself
        and earlier positions, matching the decoder-only autoregressive
        Transformer that Dave et al. 2025 found outperforms a full
        encoder-decoder Transformer for FX price prediction (their
        TFM_DE_TM/FM/TM_FM beat TFM_EN_DE_TM/FM/TM_FM in every reported
        RMSE comparison). The original bidirectional (non-causal) version
        can still be selected via `causal=False` for comparison.
        """
        x = self.pos_enc(x)
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        return self.encoder(x, mask=mask, is_causal=self.causal)
