from __future__ import annotations

from typing import Literal

ModelSize = Literal["small", "medium", "big"]

DEPTH_PRESETS: dict[ModelSize, int] = {
    "small": 2,
    "medium": 4,
    "big": 8,
}


def resolve_depth(
    size: ModelSize = "small",
    num_layers: int | None = None,
    *,
    min_layers: int = 1,
) -> int:
    """Resolve explicit layer count or named model-size preset."""

    depth = int(num_layers) if num_layers is not None else DEPTH_PRESETS[size]
    if depth < min_layers:
        raise ValueError(f"Expected at least {min_layers} layers, got {depth}.")
    return depth


__all__ = ["DEPTH_PRESETS", "ModelSize", "resolve_depth"]
