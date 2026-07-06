"""
Bi-LSTM layer (Section 3.1.3): 2-layer stacked, bidirectional,
H=128/direction -> 256-dim concatenated temporal representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import MODEL_CFG


class BiLSTMTemporalLayer(nn.Module):
    def __init__(
        self,
        input_size: int = MODEL_CFG.cnn_out_channels,
        hidden_size: int = MODEL_CFG.lstm_hidden,
        num_layers: int = MODEL_CFG.lstm_layers,
        dropout: float = MODEL_CFG.lstm_dropout,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T', 128) -> (B, T', 256)"""
        out, _ = self.lstm(x)
        return out
