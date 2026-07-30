from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .presets import ModelSize, resolve_depth


class AttentionBlock(nn.Module):
    """Pre-norm self-attention block for token sequences."""

    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_ratio * embedding_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block."""

    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.context_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_ratio * embedding_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embedding_dim, embedding_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_norm = self.context_norm(context)
        attn_out, _ = self.attn(
            self.query_norm(query),
            context_norm,
            context_norm,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.mlp(self.mlp_norm(query))
        return query


def build_transformer_blocks(
    embedding_dim: int,
    num_heads: int,
    *,
    num_layers: int | None = None,
    size: ModelSize = "small",
    mlp_ratio: int = 4,
) -> nn.ModuleList:
    """Build a ModuleList of pre-norm Transformer blocks."""

    depth = resolve_depth(size=size, num_layers=num_layers, min_layers=1)
    return nn.ModuleList(
        [
            AttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ]
    )


def build_patch_tokenizer_1d(
    input_channels: int,
    patch_size: int,
    embedding_dim: int,
    *,
    hidden_dim: int,
) -> nn.Sequential:
    """Build a Conv1d patch tokenizer returned as a plain Sequential."""

    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    patch_channels = max(hidden_dim, embedding_dim)
    return nn.Sequential(
        nn.Conv1d(
            input_channels,
            patch_channels,
            kernel_size=patch_size,
            stride=patch_size,
        ),
        nn.GELU(),
        nn.Conv1d(patch_channels, embedding_dim, kernel_size=1),
    )


def interpolate_token_positions(pos: torch.Tensor, token_length: int) -> torch.Tensor:
    """Interpolate learned absolute positional embeddings to a token length."""

    if pos.size(1) == token_length:
        return pos
    return F.interpolate(
        pos.transpose(1, 2),
        size=token_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class PatchTokenizer1D(nn.Module):
    """Conv1d patch tokenizer class for callers that do not need legacy names."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        embedding_dim: int,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.net = build_patch_tokenizer_1d(
            input_channels,
            patch_size,
            embedding_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(F.pad(x, (0, (-x.size(-1)) % self.patch_size)))
        return x.transpose(1, 2)


__all__ = [
    "AttentionBlock",
    "CrossAttentionBlock",
    "PatchTokenizer1D",
    "build_patch_tokenizer_1d",
    "build_transformer_blocks",
    "interpolate_token_positions",
]
