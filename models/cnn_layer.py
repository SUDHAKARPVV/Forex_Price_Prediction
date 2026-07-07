"""
CNN layer (Section 3.1.2), revised to CAUSAL DILATED convolutions.

The original design (two Conv1D blocks + MaxPool halving 60 -> 30) blurred
consecutive bars together: after pooling, the downstream recurrent and
attention stages could no longer tell whether a volatility spike happened
at t-1 or t-2 -- exactly the lag-1/lag-2 microstructure that gives ARIMA
and GARCH their edge. This revision:

- removes pooling entirely (sequence stays at full T=60 resolution), and
- uses stacked CAUSAL dilated Conv1D blocks (dilations 1, 2, 4 -- WaveNet
  /TCN style): the receptive field grows to 15 bars without discarding
  temporal resolution, and left-only padding guarantees position t only
  ever sees positions <= t (no intra-window look-ahead in the local
  feature extractor).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CFG


class CausalConvBlock(nn.Module):
    """Conv1d with left-only (causal) padding + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C_out, T), length preserved via causal padding."""
        x = F.pad(x, (self.left_pad, 0))
        return self.act(self.bn(self.conv(x)))


class CNNLocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = MODEL_CFG.cnn_in_channels,
        out_channels: int = MODEL_CFG.cnn_out_channels,
        kernel_size: int = MODEL_CFG.cnn_kernel_size,
    ):
        super().__init__()
        self.block1 = CausalConvBlock(in_channels, out_channels, kernel_size, dilation=1)
        self.block2 = CausalConvBlock(out_channels, out_channels, kernel_size, dilation=2)
        self.block3 = CausalConvBlock(out_channels, out_channels, kernel_size, dilation=4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_in) -> (B, T, C_out) -- full temporal resolution."""
        x = x.transpose(1, 2)          # (B, C, T) -- Conv1d expects channels-first
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        return x.transpose(1, 2)       # (B, T, C_out)
