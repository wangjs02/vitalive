from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

History = Sequence[Mapping[str, Any]]


def _history_dataframe(history: History) -> pd.DataFrame:
    if not history:
        raise ValueError("history must contain at least one epoch.")

    rows = []
    for item in history:
        if "epoch" not in item or "train" not in item or "val" not in item:
            raise KeyError("Each history row must contain epoch, train, and val.")
        row = {"epoch": int(item["epoch"])}
        for split in ("train", "val"):
            metrics = item[split]
            if not isinstance(metrics, Mapping):
                raise TypeError(f"history {split} metrics must be a mapping.")
            for name, value in metrics.items():
                if np.isscalar(value):
                    row[f"{split}_{name}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    required = {
        "train_loss",
        "val_loss",
        "train_recon_loss",
        "val_recon_loss",
        "train_commitment_loss",
        "val_commitment_loss",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"History is missing loss metrics: {missing}")
    return frame


def _best_epoch(frame: pd.DataFrame) -> int:
    finite = frame[np.isfinite(frame["val_recon_loss"].to_numpy(dtype=float))]
    if finite.empty:
        raise ValueError("History contains no finite validation reconstruction loss.")
    return int(finite.loc[finite["val_recon_loss"].idxmin(), "epoch"])


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _plot_split_metric(
    axis,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    best_epoch: int,
) -> None:
    axis.plot(frame["epoch"], frame[f"train_{metric}"], label="Train", linewidth=1.8)
    axis.plot(frame["epoch"], frame[f"val_{metric}"], label="Validation", linewidth=1.8)
    axis.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch {best_epoch}",
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()


def eval_loss_history(
    history: History,
    output_path: str | Path | None = None,
) -> Path | None:
    """Display loss curves, or save them as PNG when a path is provided."""
    frame = _history_dataframe(history)
    best_epoch = _best_epoch(frame)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    _plot_split_metric(axes[0], frame, "loss", "Total Loss", "Loss", best_epoch)
    _plot_split_metric(
        axes[1], frame, "recon_loss", "Reconstruction Loss", "MSE", best_epoch
    )
    _plot_split_metric(
        axes[2],
        frame,
        "commitment_loss",
        "Commitment Loss",
        "Loss",
        best_epoch,
    )
    if output_path is None:
        plt.show()
        return None

    path = _prepare_output_path(output_path)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def eval_quantizer_history(
    history: History,
    output_path: str | Path | None = None,
) -> Path | None:
    """Display quantizer curves, or save them when a path is provided."""
    frame = _history_dataframe(history)
    required = {
        "train_perplexity",
        "val_perplexity",
        "train_cluster_use",
        "val_cluster_use",
        "train_x_std_mean",
        "val_x_std_mean",
        "train_x_recon_std_mean",
        "val_x_recon_std_mean",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"History is missing quantizer metrics: {missing}")

    eps = 1e-8
    frame = frame.copy()
    frame["train_std_ratio"] = frame["train_x_recon_std_mean"] / frame["train_x_std_mean"].clip(lower=eps)
    frame["val_std_ratio"] = frame["val_x_recon_std_mean"] / frame["val_x_std_mean"].clip(lower=eps)
    best_epoch = _best_epoch(frame)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    _plot_split_metric(
        axes[0], frame, "perplexity", "Codebook Perplexity", "Perplexity", best_epoch
    )
    _plot_split_metric(
        axes[1], frame, "cluster_use", "Codebook Cluster Use", "Used Codes", best_epoch
    )
    _plot_split_metric(
        axes[2],
        frame,
        "std_ratio",
        "Reconstruction Std Ratio",
        "Recon Std / Input Std",
        best_epoch,
    )
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    if output_path is None:
        plt.show()
        return None

    path = _prepare_output_path(output_path)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def eval_history(
    history: History,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]] | pd.DataFrame:
    """Display history plots, or save CSV/PNGs when a directory is provided."""
    frame = _history_dataframe(history)
    if output_dir is None:
        eval_loss_history(history)
        eval_quantizer_history(history)
        return frame

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "history": output_root / "history.csv",
        "loss_history": output_root / "loss_history.png",
        "quantizer_history": output_root / "quantizer_history.png",
    }
    frame.to_csv(paths["history"], index=False)
    eval_loss_history(history, paths["loss_history"])
    eval_quantizer_history(history, paths["quantizer_history"])
    return frame, paths


def eval_recon(
    model: nn.Module,
    test_batch: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    transforms: Any | None = None,
    vital_signs: Sequence[str] | None = None,
    num_samples: int = 4,
    device: str | torch.device | None = None,
    save_dir: str | Path | None = None,
) -> tuple[plt.Figure, Path | None]:
    """Display reconstructions, or save them when a directory is provided."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    x = test_batch["x"] if isinstance(test_batch, Mapping) else test_batch
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        shape = getattr(x, "shape", None)
        raise ValueError(f"Expected test batch [B, C, T], got shape {shape}.")

    if device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = x.device
    else:
        model_device = torch.device(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            target = x.to(model_device)
            output = model(target)
            if not isinstance(output, Mapping) or "x_recon" not in output:
                raise KeyError("Model output must contain x_recon.")
            reconstruction = output["x_recon"]
    finally:
        model.train(was_training)

    if transforms is not None:
        inverse = getattr(transforms, "inverse_transform", None)
        if not callable(inverse):
            raise TypeError("transforms must provide inverse_transform(data).")
        target = inverse(target)
        reconstruction = inverse(reconstruction)

    target = target.detach().cpu()
    reconstruction = reconstruction.detach().cpu()
    sample_count = min(num_samples, target.shape[0])
    channel_count = target.shape[1]
    if sample_count == 0 or channel_count == 0:
        raise ValueError("test_batch must contain at least one sample and channel.")

    if vital_signs is None:
        roles = [f"channel {index}" for index in range(channel_count)]
    else:
        roles = list(vital_signs)
        if len(roles) != channel_count:
            raise ValueError(
                f"Expected {channel_count} vital-sign labels, got {len(roles)}."
            )

    figure, axes = plt.subplots(
        sample_count,
        channel_count,
        figsize=(4 * channel_count, 2.8 * sample_count),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    time_axis = np.arange(target.shape[-1])
    for sample_index in range(sample_count):
        for channel_index, role in enumerate(roles):
            axis = axes[sample_index, channel_index]
            axis.plot(
                time_axis,
                target[sample_index, channel_index].numpy(),
                label="Target",
                color="black",
                linewidth=1.5,
            )
            axis.plot(
                time_axis,
                reconstruction[sample_index, channel_index].numpy(),
                label="Reconstruction",
                color="tab:red",
                linewidth=1.2,
            )
            axis.set_title(f"Sample {sample_index} | {role}")
            axis.grid(alpha=0.25)
            if sample_index == sample_count - 1:
                axis.set_xlabel("Time")
            if channel_index == 0:
                axis.set_ylabel("Value")
    axes[0, 0].legend()

    if save_dir is None:
        plt.show()
        return figure, None

    saved_path = _prepare_output_path(
        Path(save_dir) / "reconstruction_samples.png"
    )
    figure.savefig(saved_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return figure, saved_path


__all__ = [
    "eval_history",
    "eval_loss_history",
    "eval_quantizer_history",
    "eval_recon",
]
