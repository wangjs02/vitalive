from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation={name!r}.")


def _norm_1d(name: str | None, dim: int) -> nn.Module | None:
    if name is None:
        return None
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm={name!r}.")


class MLPBlock(nn.Module):
    """Linear hidden block with optional normalization, activation, dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str | None = None,
        norm_position: str = "pre_activation",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if norm_position not in {"pre_linear", "pre_activation", "post_activation"}:
            raise ValueError(f"Unsupported norm_position={norm_position!r}.")

        norm_dim = int(input_dim) if norm_position == "pre_linear" else int(output_dim)
        self.norm_position = norm_position
        self.norm = _norm_1d(norm, norm_dim)
        self.linear = nn.Linear(int(input_dim), int(output_dim), bias=bias)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None and self.norm_position == "pre_linear":
            x = self.norm(x)
        x = self.linear(x)
        if self.norm is not None and self.norm_position == "pre_activation":
            x = self.norm(x)
        x = self.activation(x)
        if self.norm is not None and self.norm_position == "post_activation":
            x = self.norm(x)
        return self.dropout(x)


def build_mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int | None = None,
    num_blocks: int = 1,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str | None = None,
    norm_position: str = "pre_activation",
    bias: bool = True,
) -> nn.Sequential:
    """Build an MLP from repeated MLPBlock modules plus a final Linear layer."""

    blocks = int(num_blocks)
    if blocks < 0:
        raise ValueError("num_blocks must be >= 0.")
    hidden = int(hidden_dim or input_dim)
    modules: list[nn.Module] = []
    current_dim = int(input_dim)

    for _ in range(blocks):
        modules.append(
            MLPBlock(
                current_dim,
                hidden,
                activation=activation,
                dropout=dropout,
                norm=norm,
                norm_position=norm_position,
                bias=bias,
            )
        )
        current_dim = hidden
    modules.append(nn.Linear(current_dim, int(output_dim), bias=bias))
    return nn.Sequential(*modules)


__all__ = [
    "MLPBlock",
    "build_mlp",
]
