from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from blocks.transformer import (
    build_transformer_blocks,
    interpolate_token_positions,
)
from utils.config import KwargsConfig


@dataclass
class ViTConfig(KwargsConfig):
    """Configuration for ViT encoder and decoder modules."""

    input_dim: int = 4
    hidden_dim: int = 32
    patch_size: int = 10
    embedding_dim: int = 8
    time_length: int = 30
    token_length: int = 16
    transformer_layers: int = 2
    transformer_heads: int = 4


class ViTEncoder(nn.Module):
    """ViT-style patch tokenizer plus Transformer encoder."""

    def __init__(
        self,
        input_dim: int = 4,
        patch_size: int = 1,
        embedding_dim: int = 8,
        time_length: int = 300,
        token_length: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if transformer_heads < 1:
            raise ValueError("transformer_heads must be >= 1")
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be >= 1")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_channels = self.embedding_dim
        self.patch_size = patch_size
        self.time_length = time_length
        self.token_length = token_length
        self.hidden_dim = hidden_dim

        patch_channels = max(self.hidden_dim, self.embedding_dim)
        self.patch_tokenizer_conv1 = nn.Conv1d(self.input_dim, patch_channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_tokenizer_activ1 = nn.GELU()
        self.patch_tokenizer_conv2 = nn.Conv1d(patch_channels, self.embedding_dim, kernel_size=1)
        self.time_pos_embedding = nn.Parameter(torch.zeros(1, self.token_length, self.embedding_dim))
        self.temporal_model_blocks = build_transformer_blocks(
            embedding_dim=self.embedding_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B, input_dim, T]` into `[B, N, embedding_dim]`."""
        x = self.patch_tokenizer(x)
        return self.temporal_model(x)

    def patch_tokenizer(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize temporal patches with Conv1d layers."""
        x = F.pad(x, (0, (-x.size(-1)) % self.patch_size))
        x = self.patch_tokenizer_conv1(x)
        x = self.patch_tokenizer_activ1(x)
        x = self.patch_tokenizer_conv2(x)
        return x.transpose(1, 2)

    def temporal_model(self, x: torch.Tensor) -> torch.Tensor:
        """Run learned position embedding plus Transformer temporal blocks."""
        pos = interpolate_token_positions(self.time_pos_embedding, x.size(1))
        x = x + pos
        for block in self.temporal_model_blocks:
            x = block(x)
        return x


class ViTDecoder(nn.Module):
    """ViT-style token decoder that reconstructs each patch then concatenates."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        patch_size: int = 1,
        time_length: int = 300,
        token_length: int = 16,
        embedding_dim: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if transformer_heads < 1:
            raise ValueError("transformer_heads must be >= 1")
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be >= 1")

        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size)
        self.time_length = int(time_length)
        self.token_length = int(token_length)
        self.embedding_dim = int(embedding_dim)
        self.token_pos_embedding = nn.Parameter(
            torch.zeros(1, self.token_length, self.embedding_dim)
        )
        self.temporal_model_blocks = build_transformer_blocks(
            embedding_dim=self.embedding_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )
        self.patch_reconstruct_linear1 = nn.Linear(self.embedding_dim, self.input_dim * self.patch_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode `[B, N, embedding_dim]` into `[B, input_dim, time_length]`."""
        z = self.temporal_model(z)
        return self.patch_reconstruction(z)

    def temporal_model(self, z: torch.Tensor) -> torch.Tensor:
        """Run learned position embedding plus Transformer temporal blocks."""
        pos = interpolate_token_positions(self.token_pos_embedding, z.size(1))
        z = z + pos
        for block in self.temporal_model_blocks:
            z = block(z)
        return z

    def patch_reconstruction(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal patches and concatenate them on the time axis."""
        x = self.patch_reconstruct_linear1(z)
        if self.patch_size == 1:
            x = x.transpose(1, 2)
        else:
            x = x.view(x.size(0), x.size(1), self.input_dim, self.patch_size)
            x = x.permute(0, 2, 1, 3).reshape(x.size(0), self.input_dim, -1)

        if x.size(-1) != self.time_length:
            x = F.interpolate(
                x,
                size=self.time_length,
                mode="linear",
                align_corners=False,
            )

        return x


__all__ = [
    "ViTConfig",
    "ViTDecoder",
    "ViTEncoder",
]
