from __future__ import annotations

from torch import nn

from .mlp import build_mlp


def build_linear_projection(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int | None = None,
    num_blocks: int = 0,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str | None = None,
    norm_position: str = "pre_activation",
    flatten: bool = False,
    bias: bool = True,
    identity_if_same: bool = False,
) -> nn.Module:
    """Build a Linear/MLP projection for feature vectors or pooled tensors."""

    blocks = int(num_blocks)
    if hidden_dim is not None and blocks == 0:
        blocks = 1

    if (
        identity_if_same
        and int(input_dim) == int(output_dim)
        and hidden_dim is None
        and blocks == 0
        and not flatten
        and norm is None
        and dropout == 0
    ):
        return nn.Identity()

    if blocks == 0 and hidden_dim is None and norm is None and dropout == 0:
        projection: nn.Module = nn.Linear(int(input_dim), int(output_dim), bias=bias)
    else:
        projection = build_mlp(
            input_dim,
            output_dim,
            hidden_dim=hidden_dim,
            num_blocks=blocks,
            activation=activation,
            dropout=dropout,
            norm=norm,
            norm_position=norm_position,
            bias=bias,
        )

    if flatten:
        return nn.Sequential(nn.Flatten(), projection)
    return projection


def build_cnn_projection(
    input_channels: int,
    output_channels: int,
    *,
    hidden_channels: int | None = None,
    num_blocks: int = 1,
    dropout: float = 0.10,
) -> nn.Sequential:
    """Build a Conv1d projection for sequence features `[B, C, T]`."""
    try:
        from .cnn import CNNBlock
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "build_cnn_projection requires blocks.cnn, which is not present in this workspace."
        ) from exc

    blocks = int(num_blocks)
    if blocks < 0:
        raise ValueError("num_blocks must be >= 0.")

    hidden = int(hidden_channels or input_channels)
    modules: list[nn.Module] = []
    current = int(input_channels)

    for _ in range(blocks):
        modules.append(CNNBlock(current, hidden))
        current = hidden

    if dropout > 0:
        modules.append(nn.Dropout(dropout))
    modules.append(
        CNNBlock(
            current,
            int(output_channels),
            mode="same",
            kernel_size=1,
            activation=None,
        )
    )
    return nn.Sequential(*modules)


__all__ = [
    "build_cnn_projection",
    "build_linear_projection",
]
