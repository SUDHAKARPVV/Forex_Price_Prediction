"""
CNN layer (Section 3.1.2): Conv1D(kernel=3, ReLU) -> BatchNorm -> MaxPool.
Maps the fused (B, T=60, 64) tensor to (B, T'=30, 128) local feature maps.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import MODEL_CFG


class CNNLocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = MODEL_CFG.cnn_in_channels,
        out_channels: int = MODEL_CFG.cnn_out_channels,
        kernel_size: int = MODEL_CFG.cnn_kernel_size,
        pool_kernel: int = MODEL_CFG.cnn_pool_kernel,
    ):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding along the time axis
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)  # T=60 -> T'=30
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 64) -> (B, T/2, 128)"""
        x = x.transpose(1, 2)          # (B, 64, T)  -- Conv1d expects channels-first
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)               # (B, 128, T/2)
        return x.transpose(1, 2)       # (B, T/2, 128)
