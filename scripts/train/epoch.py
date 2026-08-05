from __future__ import annotations

from typing import Any

import torch
from torch import nn

def run_epoch(
    model: nn.Module,
    loader,
    optimizer=None,
    device: torch.device | str = "cpu",
    loss_fn: nn.Module | None = None):
    """Run one epoch and return averaged training diagnostics."""
    
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y) if loss_fn is not None else out["loss"]
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(x)
                loss = loss_fn(out, y) if loss_fn is not None else out["loss"]

    return loss.item()

def eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device | str = "cpu",
    eval_fn: nn.Module | None = None,):
    """Run one evaluation epoch and return averaged diagnostics."""
    totals = totals
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            eval = eval_fn(out, y) if eval_fn is not None else out["loss"]
            totals = totals or {}
            for key, value in eval.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * x.size(0)
    return totals

def run_vqvae_epoch(
    model: nn.Module,
    loader,
    optimizer=None,
    device: torch.device | str = "cpu",
    grad_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Run one VQ-VAE epoch and return averaged training diagnostics."""

    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "commitment_loss": 0.0,
        "perplexity": 0.0,
        "cluster_use": 0.0,
        "x_std_mean": 0.0,
        "x_recon_std_mean": 0.0,
    }
    n_samples = 0

    for batch in loader:
        x = batch["x"] if isinstance(batch, dict) else batch
        x = x.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            out["loss"].backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(x)
        batch_size = x.size(0)
        n_samples += batch_size
        totals["loss"] += float(out["loss"].detach().cpu()) * batch_size
        totals["recon_loss"] += float(out["recon_loss"].detach().cpu()) * batch_size
        totals["commitment_loss"] += float(out["vq_loss"].detach().cpu()) * batch_size
        totals["perplexity"] += float(out["perplexity"].detach().cpu()) * batch_size
        totals["cluster_use"] += float(out["cluster_use"].detach().cpu()) * batch_size
        totals["x_std_mean"] += float(x.std(dim=(0, 2)).mean().detach().cpu()) * batch_size
        totals["x_recon_std_mean"] += (
            float(out["x_recon"].std(dim=(0, 2)).mean().detach().cpu()) * batch_size
        )

    return {key: value / max(n_samples, 1) for key, value in totals.items()}


__all__ = [
    "run_vqvae_epoch",
]
