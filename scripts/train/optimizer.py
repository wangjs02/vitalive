from __future__ import annotations

import torch
from torch import nn

def build_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build the project-standard AdamW optimizer for model training."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


__all__ = [
    "build_adamw_optimizer",
    "build_weight_decay_parameter_groups",
]
