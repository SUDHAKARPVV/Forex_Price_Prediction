"""
Recurrent temporal layers (Section 3.1.3):

- BiLSTMTemporalLayer -- 2-layer stacked bidirectional LSTM,
  H=128/direction -> 256-dim concatenated temporal representation.
- BiGRUTemporalLayer -- the same geometry built on GRU cells. GRUs use two
  gates instead of the LSTM's three and carry no separate cell state,
  which makes them faster to adapt to regime shifts on noisy intraday
  series; the Hybrid model runs BOTH in parallel and blends them with a
  learned per-sample gate (models/hybrid_model.py), so the network can
  lean on whichever recurrent inductive bias fits the current window.
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


class BiGRUTemporalLayer(nn.Module):
    def __init__(
        self,
        input_size: int = MODEL_CFG.cnn_out_channels,
        hidden_size: int = MODEL_CFG.lstm_hidden,
        num_layers: int = MODEL_CFG.lstm_layers,
        dropout: float = MODEL_CFG.lstm_dropout,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T', 128) -> (B, T', 256)"""
        out, _ = self.gru(x)
        return out
