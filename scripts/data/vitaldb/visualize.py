from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .dataset import VitalDBDataset, read_data_by_id


def plot_case(
    dataset: VitalDBDataset,
    case_id: int | None = None,
    vital_signs: Sequence[str] | None = None,
    figsize: tuple[float, float] = (14, 2.5),
):
    """Plot all clean segments available for one case."""
    if case_id is None:
        case_id = int(dataset.id_list[0][0])
    sample_ids = sorted(item for item in dataset.id_list if int(item[0]) == case_id)
    if not sample_ids:
        raise KeyError(f"Case {case_id} is not in the dataset.")

    arrays = [read_data_by_id(dataset.vitaldb_data, sample_id) for sample_id in sample_ids]
    x = np.concatenate(arrays, axis=1)
    roles, indices = _resolve_vital_signs(dataset, vital_signs)
    plt = _require_matplotlib()
    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(figsize[0], figsize[1] * len(indices)),
        sharex=True,
        squeeze=False,
    )
    time_sec = np.arange(x.shape[1]) * dataset.vitaldb_data.interval_sec
    for ax, role, channel in zip(axes[:, 0], roles, indices):
        ax.plot(time_sec, x[channel], linewidth=1.2)
        ax.set_ylabel(role)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time (sec)")
    fig.suptitle(f"VitalDB clean segments: case {case_id}", y=1.0)
    fig.tight_layout()
    return fig, axes[:, 0]


def plot_segment(
    dataset: VitalDBDataset,
    segment_index: int = 0,
    vital_signs: Sequence[str] | None = None,
    figsize: tuple[float, float] = (12, 2.5),
):
    """Plot one clean segment from the dataset."""
    sample_id = dataset.id_list[segment_index]
    x = read_data_by_id(dataset.vitaldb_data, sample_id)
    segments = pd.read_csv(dataset.vitaldb_data._case_dir(sample_id[0]) / "segments.csv")
    row = segments.loc[segments["segment_id"] == sample_id[1]].iloc[0]
    roles, indices = _resolve_vital_signs(dataset, vital_signs)
    plt = _require_matplotlib()
    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(figsize[0], figsize[1] * len(indices)),
        sharex=True,
        squeeze=False,
    )
    time_sec = np.arange(x.shape[1]) * dataset.vitaldb_data.interval_sec
    for ax, role, channel in zip(axes[:, 0], roles, indices):
        ax.plot(time_sec, x[channel], linewidth=1.2)
        ax.set_ylabel(role)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time (sec)")
    fig.suptitle(
        f"Case {sample_id[0]}, segment {sample_id[1]} "
        f"({int(row['start_sec'])}-{int(row['end_sec'])} sec)",
        y=1.0,
    )
    fig.tight_layout()
    return fig, axes[:, 0]


def plot_datapoint(
    dataset: VitalDBDataset,
    datapoint_index: int = 0,
    vital_signs: Sequence[str] | None = None,
    figsize: tuple[float, float] = (14, 2.5),
    show_segment_boundaries: bool = True,
):
    """Plot one assembled dataset datapoint, including multi-segment windows.

    The datapoint is loaded through ``dataset[datapoint_index]``, so the plot
    includes segment concatenation and all configured dataset transforms. The
    x-axis always spans ``dataset.time_length`` seconds, regardless of the
    stored or transformed sample frequency.
    """
    x = np.asarray(dataset[datapoint_index], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected datapoint shaped [C, T], got {tuple(x.shape)}.")
    if x.shape[1] == 0:
        raise ValueError("Cannot plot a datapoint with an empty time axis.")

    window_ids = dataset.sample_ids[datapoint_index]
    roles, indices = _resolve_vital_signs(dataset, vital_signs)
    plt = _require_matplotlib()
    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(figsize[0], figsize[1] * len(indices)),
        sharex=True,
        squeeze=False,
    )
    time_sec = np.linspace(
        0.0,
        float(dataset.time_length),
        num=x.shape[1],
        endpoint=False,
    )
    for ax, role, channel in zip(axes[:, 0], roles, indices):
        ax.plot(time_sec, x[channel], linewidth=1.2)
        if show_segment_boundaries and len(window_ids) > 1:
            for boundary_sec in range(60, dataset.time_length, 60):
                ax.axvline(boundary_sec, color="tab:gray", alpha=0.25, linewidth=0.8)
        ax.set_ylabel(role)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time (sec)")

    first_id = window_ids[0]
    last_id = window_ids[-1]
    fig.suptitle(
        f"VitalDB datapoint {datapoint_index}: case {first_id[0]}, "
        f"segments {first_id[1]}-{last_id[1]} "
        f"({dataset.time_length} sec)",
        y=1.0,
    )
    fig.tight_layout()
    return fig, axes[:, 0]


def plot_segment_counts(
    dataset: VitalDBDataset,
    top_n: int = 30,
    figsize: tuple[float, float] = (14, 4),
):
    """Plot clean segment counts per case."""
    counts = pd.Series([item[0] for item in dataset.id_list]).value_counts().head(top_n)
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("case_id")
    ax.set_ylabel("clean segments")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def segment_summary(dataset: VitalDBDataset) -> pd.DataFrame:
    """Return clean segment counts per case."""
    return (
        pd.DataFrame(dataset.id_list, columns=["case_id", "segment_id"])
        .groupby("case_id")
        .size()
        .rename("clean_segments")
        .reset_index()
        .sort_values("clean_segments", ascending=False)
    )


def _resolve_vital_signs(
    dataset: VitalDBDataset,
    vital_signs: Sequence[str] | None,
) -> tuple[list[str], list[int]]:
    available = list(dataset.vitaldb_data.vital_signs)
    selected = available if vital_signs is None else list(vital_signs)
    missing = [role for role in selected if role not in available]
    if missing:
        raise KeyError(f"Vital signs not found: {missing}")
    return selected, [available.index(role) for role in selected]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization functions.") from exc
    return plt
