from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .presets import ModelSize, resolve_depth


class ResidualBlock1D(nn.Module):
    """Simple same-channel Conv1d residual block."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or channels)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + x)


class ResidualBlock2D(nn.Module):
    """Simple same-channel Conv2d residual block."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + x)


def build_residual_stack_1d(
    channels: int,
    *,
    hidden_channels: int | None = None,
    num_layers: int | None = None,
    size: ModelSize = "small",
) -> nn.Sequential:
    """Build a stack of same-channel 1D residual blocks."""

    depth = resolve_depth(size=size, num_layers=num_layers, min_layers=1)
    return nn.Sequential(
        *[
            ResidualBlock1D(channels, hidden_channels=hidden_channels)
            for _ in range(depth)
        ]
    )


def build_residual_stack_2d(
    channels: int,
    *,
    hidden_channels: int | None = None,
    num_layers: int | None = None,
    size: ModelSize = "small",
) -> nn.Sequential:
    """Build a stack of same-channel 2D residual blocks."""

    depth = resolve_depth(size=size, num_layers=num_layers, min_layers=1)
    return nn.Sequential(
        *[
            ResidualBlock2D(channels, hidden_channels=hidden_channels)
            for _ in range(depth)
        ]
    )


__all__ = [
    "ResidualBlock1D",
    "ResidualBlock2D",
    "build_residual_stack_1d",
    "build_residual_stack_2d",
]
