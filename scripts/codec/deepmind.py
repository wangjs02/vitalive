from __future__ import annotations

import torch
from torch import nn

from blocks.resnet import ResidualBlock2D as ResBlock


class DeepMindEncoder(nn.Module):
    """External DeepMind-style Conv2d encoder."""

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2 * n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * n_hid, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
        )
        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMindDecoder(nn.Module):
    """External DeepMind-style Conv2d decoder."""

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            nn.ConvTranspose2d(2 * n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "DeepMindDecoder",
    "DeepMindEncoder",
]
