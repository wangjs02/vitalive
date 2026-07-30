from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2

from utils.config import KwargsConfig

@dataclass
class SequenceEMAQuantizerConfig(KwargsConfig):
    """Configuration for `SequenceEMAQuantize`."""

    n_embed: int = 64
    embedding_dim: int = 8
    decay: float = 0.99
    eps: float = 1e-5
    commitment_cost: float = 0.25
    kld_scale: float = 10.0
    rotation_matching: bool = False
    fourier_matching_dim: int | None = None


class FixedFourierProjection(nn.Module):
    """Fixed Fourier-basis projection with a pseudo-inverse reverse map."""

    def __init__(self, input_dim: int = 8, basis_dim: int = 32) -> None:
        super().__init__()
        if basis_dim < input_dim:
            raise ValueError("basis_dim must be >= input_dim for stable reverse projection.")

        basis = self._make_basis(input_dim, basis_dim)
        reverse_basis = torch.linalg.pinv(basis)

        self.input_dim = input_dim
        self.basis_dim = basis_dim
        self.register_buffer("basis", basis)
        self.register_buffer("reverse_basis", reverse_basis)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {z.size(-1)}")
        return z @ self.basis

    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) != self.basis_dim:
            raise ValueError(f"Expected last dim {self.basis_dim}, got {z.size(-1)}")
        return z @ self.reverse_basis

    @staticmethod
    def _make_basis(input_dim: int, basis_dim: int) -> torch.Tensor:
        t = torch.arange(input_dim, dtype=torch.float64) / input_dim
        columns = []
        freq = 0
        while len(columns) < basis_dim:
            columns.append(torch.cos(2 * math.pi * freq * t))
            if len(columns) < basis_dim:
                columns.append(torch.sin(2 * math.pi * freq * t))
            freq += 1

        basis = torch.stack(columns, dim=1)
        basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-8)
        return basis.to(dtype=torch.float32)


class VQVAEQuantize(nn.Module):
    """Copied from external deep-vector-quantization VQVAEQuantize.

    Kept as Conv2d / [B, C, H, W] quantizer to match the external architecture.
    Only minor device-safe copying is used for k-means initialization.
    """

    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.kld_scale = 10.0

        self.proj = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.register_buffer("data_initialized", torch.zeros(1))

    def forward(self, z):
        B, C, H, W = z.size()

        z_e = self.proj(z)
        z_e = z_e.permute(0, 2, 3, 1)
        flatten = z_e.reshape(-1, self.embedding_dim)

        if self.training and self.data_initialized.item() == 0:
            print("running kmeans!!")
            rp = torch.randperm(flatten.size(0), device=flatten.device)
            sample = flatten[rp[: min(20000, flatten.size(0))]].detach()
            if sample.size(0) >= self.n_embed:
                kd = kmeans2(sample.cpu().numpy(), self.n_embed, minit="points")
                centers = torch.from_numpy(kd[0]).to(
                    device=self.embed.weight.device,
                    dtype=self.embed.weight.dtype,
                )
            else:
                extra = torch.randint(
                    sample.size(0),
                    (self.n_embed - sample.size(0),),
                    device=sample.device,
                )
                centers = torch.cat([sample, sample[extra]], dim=0)
                centers = centers + 1e-4 * torch.randn_like(centers)
                centers = centers.to(device=self.embed.weight.device, dtype=self.embed.weight.dtype)
            self.embed.weight.data.copy_(centers)
            self.data_initialized.fill_(1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.reshape(B, H, W)

        z_q = self.embed_code(ind)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (
            z_q - z_e.detach()
        ).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class SequenceEMAQuantize(nn.Module):
    """Sequence VQ quantizer with k-means initialization and EMA codebook update.

    Input:  [B, token_length, embedding_dim]
    Output: quantized tokens with the same shape, commitment loss, and
    indices [B, token_length].
    """

    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        kld_scale: float = 10.0,
        rotation_matching: bool = False,
        fourier_matching_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.kld_scale = kld_scale
        self.rotation_matching = rotation_matching
        self.last_rotation_shifts = None
        self.matching_projection = (
            FixedFourierProjection(input_dim=embedding_dim, basis_dim=fourier_matching_dim)
            if fourier_matching_dim is not None
            else None
        )

        embed = torch.randn(n_embed, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("data_initialized", torch.zeros(1))

    def forward(self, z):
        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got shape {tuple(z.shape)}")
        if z.size(-1) != self.embedding_dim:
            raise ValueError(f"Expected D={self.embedding_dim}, got {z.size(-1)}")

        B, T, D = z.shape
        flatten = z.reshape(-1, D)

        if self.training and self.data_initialized.item() == 0:
            print("running sequence kmeans!!")
            sample = flatten.detach()
            if sample.size(0) >= self.n_embed:
                rp = torch.randperm(sample.size(0), device=sample.device)
                sample = sample[rp[: min(20000, sample.size(0))]]
                kd = kmeans2(sample.cpu().numpy(), self.n_embed, minit="points")
                centers = torch.from_numpy(kd[0]).to(device=z.device, dtype=z.dtype)
            else:
                extra = torch.randint(
                    sample.size(0),
                    (self.n_embed - sample.size(0),),
                    device=sample.device,
                )
                centers = torch.cat([sample, sample[extra]], dim=0)
                centers = centers + 1e-4 * torch.randn_like(centers)
            self.embed.copy_(centers)
            self.embed_avg.copy_(centers)
            self.cluster_size.fill_(1.0)
            self.data_initialized.fill_(1)

        if self.rotation_matching:
            z_q, indices, shifts, canonical_flatten = self._match_rotated(flatten, B, T, D)
            self.last_rotation_shifts = shifts.reshape(B, T)
            ema_flatten = canonical_flatten
        else:
            dist = self._distance(flatten, self.embed)
            indices = torch.argmin(dist, dim=1)
            z_q = F.embedding(indices, self.embed).reshape(B, T, D)
            self.last_rotation_shifts = None
            ema_flatten = flatten.detach()

        if self.training:
            encodings = F.one_hot(indices, self.n_embed).type(flatten.dtype)
            cluster_size = encodings.sum(0)
            embed_sum = encodings.t() @ ema_flatten

            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.n_embed * self.eps)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.copy_(embed_normalized)

        loss = self.commitment_cost * (z_q.detach() - z).pow(2).mean()
        loss = loss * self.kld_scale
        z_q = z + (z_q - z).detach()
        return z_q, loss, indices.reshape(B, T)

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed)

    def _match_rotated(self, flatten, batch_size: int, token_length: int, dim: int):
        rotated_embed = torch.stack(
            [torch.roll(self.embed, shifts=shift, dims=1) for shift in range(dim)],
            dim=1,
        )
        candidates = rotated_embed.reshape(self.n_embed * dim, dim)
        dist = self._distance(flatten, candidates)
        candidate_indices = torch.argmin(dist, dim=1)
        indices = torch.div(candidate_indices, dim, rounding_mode="floor")
        shifts = candidate_indices.remainder(dim)
        z_q = F.embedding(candidate_indices, candidates).reshape(batch_size, token_length, dim)
        canonical_flatten = self._roll_rows(flatten.detach(), -shifts)
        return z_q, indices, shifts, canonical_flatten

    def _distance(self, x, embed):
        x_match = self._to_matching_space(x)
        embed_match = self._to_matching_space(embed)
        return (
            x_match.pow(2).sum(1, keepdim=True)
            - 2 * x_match @ embed_match.t()
            + embed_match.pow(2).sum(1, keepdim=True).t()
        )

    def _to_matching_space(self, z):
        if self.matching_projection is None:
            return z
        return self.matching_projection(z)

    @staticmethod
    def _roll_rows(x, shifts):
        dim = x.size(1)
        positions = torch.arange(dim, device=x.device).unsqueeze(0)
        source = (positions - shifts.unsqueeze(1)) % dim
        return x.gather(1, source)


def codebook_perplexity(indices: torch.Tensor, num_embeddings: int) -> tuple[torch.Tensor, torch.Tensor]:
    encodings = F.one_hot(indices, num_embeddings).float().reshape(-1, num_embeddings)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use
