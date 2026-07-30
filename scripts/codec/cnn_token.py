from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.config import KwargsConfig


@dataclass
class CNNTokenConfig(KwargsConfig):
    """Configuration for CNNToken encoder and decoder modules."""

    input_dim: int = 4
    hidden_dim: int = 32
    token_length: int = 16
    embedding_dim: int = 8
    time_length: int = 30
    temporal_align_dim: int = 16


class CNNTokenEncoder(nn.Module):
    """Standalone VGG-style Conv1d encoder that emits physiological tokens."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        token_length: int = 64,
        embedding_dim: int = 8,
        time_length: int = 300,
        temporal_align_dim: int = 16,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_length = int(token_length)
        self.embedding_dim = int(embedding_dim)
        self.time_length = int(time_length)
        self.temporal_align_dim = int(temporal_align_dim)
        self.feature_length = (self.time_length + 3) // 4
        self.output_channels = self.embedding_dim

        stem_dim, mid_dim = self.channel_widths(self.hidden_dim)
        self.temporal_downsample_conv1 = self.conv_block(self.input_dim, stem_dim, kernel_size=7)
        self.temporal_downsample_conv2 = self.downsample_block(stem_dim, mid_dim)
        self.temporal_downsample_conv3 = self.downsample_block(mid_dim, self.hidden_dim)
        self.temporal_downsample_conv4 = self.conv_block(self.hidden_dim, self.hidden_dim)
        self.temporal_align_pool1 = nn.AdaptiveAvgPool1d(self.temporal_align_dim)
        self.temporal_align_conv1 = self.conv_block(self.temporal_align_dim, self.embedding_dim, kernel_size=1, activ=False)
        self.token_align_conv1 = self.conv_block(self.hidden_dim, self.token_length, kernel_size=1, activ=False)

    @staticmethod
    def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, activ: bool = True) -> nn.Sequential:
        """Build one same-length Conv1d block."""
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        ]
        if activ:
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    @staticmethod
    def downsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build one temporal downsample block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B, C, T]` into `[B, token_length, embedding_dim]`."""
        x = self.temporal_downsample(x)
        x = self.temporal_alignment(x)
        return self.token_alignment(x)

    def temporal_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Conv1d temporal downsample trunk."""
        x = self.temporal_downsample_conv1(x)
        x = self.temporal_downsample_conv2(x)
        x = self.temporal_downsample_conv3(x)
        return self.temporal_downsample_conv4(x)

    def temporal_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Pool temporal features and project them into embedding width."""
        x = self.temporal_align_pool1(x)
        if self.temporal_align_dim == self.embedding_dim:
            return x
        return self.temporal_align_conv1(x.transpose(1, 2)).transpose(1, 2)

    def token_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Align hidden feature channels to token positions."""
        if self.hidden_dim == self.token_length:
            return x
        return self.token_align_conv1(x)

    @staticmethod
    def channel_widths(hidden_dim: int) -> tuple[int, int]:
        """Return stem and mid widths for the encoder trunk."""
        stem_dim = max(8, hidden_dim // 4)
        mid_dim = max(16, hidden_dim // 2)
        return stem_dim, mid_dim


class CNNTokenDecoder(nn.Module):
    """Standalone VGG-style Conv1d decoder for physiological tokens."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        token_length: int = 64,
        embedding_dim: int = 8,
        time_length: int = 300,
        temporal_align_dim: int = 16,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_length = int(token_length)
        self.embedding_dim = int(embedding_dim)
        self.time_length = int(time_length)
        self.temporal_align_dim = int(temporal_align_dim)
        self.feature_length = (self.time_length + 3) // 4

        stem_dim, mid_dim = self.channel_widths(self.hidden_dim)
        self.temporal_align_conv1 = self.conv_block(self.embedding_dim, self.temporal_align_dim, kernel_size=1)
        self.token_align_conv1 = self.conv_block(self.token_length, self.hidden_dim, kernel_size=1)
        self.temporal_upsample_conv1 = self.conv_block(self.hidden_dim, self.hidden_dim)
        self.temporal_upsample_conv2 = self.upsample_block(self.hidden_dim, mid_dim)
        self.temporal_upsample_conv3 = self.upsample_block(mid_dim, stem_dim)
        self.temporal_upsample_conv4 = self.conv_block(stem_dim, self.input_dim, kernel_size=3, activ=False)

    @staticmethod
    def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, activ: bool = True) -> nn.Sequential:
        """Build one same-length Conv1d block."""
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        ]
        if activ:
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    @staticmethod
    def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build one temporal upsample block."""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode `[B, token_length, embedding_dim]` into `[B, C, time_length]`."""
        x = self.token_alignment(z)
        x = self.temporal_alignment(x)
        return self.temporal_upsample(x)

    def temporal_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Project token embeddings back to the temporal alignment dimension."""
        if self.embedding_dim == self.temporal_align_dim:
            return x
        return self.temporal_align_conv1(x.transpose(1, 2)).transpose(1, 2)

    def token_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Align token positions back to hidden feature channels."""
        if self.token_length == self.hidden_dim:
            return x
        return self.token_align_conv1(x)

    def temporal_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample hidden temporal features back to signal length."""
        x = F.interpolate(x, size=self.feature_length, mode="linear", align_corners=False)
        x = self.temporal_upsample_conv1(x)
        x = self.temporal_upsample_conv2(x)
        x = self.temporal_upsample_conv3(x)
        x = self.temporal_upsample_conv4(x)
        return x[:, :, : self.time_length]

    @staticmethod
    def channel_widths(hidden_dim: int) -> tuple[int, int]:
        """Return stem and mid widths for the decoder trunk."""
        stem_dim = max(8, hidden_dim // 4)
        mid_dim = max(16, hidden_dim // 2)
        return stem_dim, mid_dim


__all__ = [
    "CNNTokenConfig",
    "CNNTokenDecoder",
    "CNNTokenEncoder",
]
