"""Reusable loss functions for pretrain and transfer workflows."""

import torch
import torch.nn.functional as F


class Normal:
    """Reconstruction loss for normalized VitalDB streams."""

    data_variance = 1.0

    @classmethod
    def inmap(cls, x):
        return x

    @classmethod
    def unmap(cls, x):
        return x

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu) ** 2).mean() / (2 * cls.data_variance)


def token_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy for token code logits shaped [B, tokens, codes]."""

    if logits.dim() != 3:
        raise ValueError(f"Expected logits [B, tokens, codes], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets [B, tokens], got {tuple(targets.shape)}")
    return F.cross_entropy(logits.transpose(1, 2).contiguous(), targets.long())


def code_embedding_commitment_loss(
    z_pred: torch.Tensor,
    targets: torch.Tensor,
    decoder_model,
    commitment_cost: float = 0.25,
    kld_scale: float = 10.0,
) -> torch.Tensor:
    """VQ commitment-style loss from predicted code vectors to teacher codebook vectors."""

    if z_pred.dim() != 4:
        raise ValueError(f"Expected z_pred [B, N, K, D], got {tuple(z_pred.shape)}")
    B, N, K, D = z_pred.shape
    if targets.dim() != 2:
        raise ValueError(f"Expected targets [B, N*K], got {tuple(targets.shape)}")
    if targets.shape != (B, N * K):
        raise ValueError(
            "targets must match flattened predicted token count "
            f"({targets.shape=} expected={(B, N * K)})"
        )

    codebook = _effective_codebook(decoder_model).to(device=z_pred.device, dtype=z_pred.dtype)
    if codebook.size(-1) != D:
        raise ValueError(f"Expected code_dim={codebook.size(-1)}, got predicted D={D}.")

    z_q_target = F.embedding(targets.long(), codebook).reshape(B, N, K, D)
    return commitment_cost * (z_q_target.detach() - z_pred).pow(2).mean() * kld_scale


def _effective_codebook(model) -> torch.Tensor:
    """Return code vectors addressed by teacher code targets."""

    quantizer = model.quantizer
    if hasattr(quantizer, "embed") and torch.is_tensor(quantizer.embed):
        embed = quantizer.embed
        if bool(getattr(quantizer, "rotation_matching", False)):
            dim = int(embed.size(1))
            return torch.stack(
                [torch.roll(embed, shifts=shift, dims=1) for shift in range(dim)],
                dim=1,
            ).reshape(int(embed.size(0)) * dim, dim)
        return embed
    if hasattr(quantizer, "embed") and hasattr(quantizer.embed, "weight"):
        return quantizer.embed.weight
    raise TypeError(f"Unsupported quantizer type: {type(quantizer)!r}")

__all__ = [
    "Normal",
    "code_embedding_commitment_loss",
    "token_cross_entropy_loss",
]
