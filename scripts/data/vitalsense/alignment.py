from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VITALDB_CHANNELS = ("HR", "SpO2", "RR", "BT", "SBP", "DBP", "MBP")


@dataclass(frozen=True)
class VitalSenseAlignmentConfig:
    """Configuration for VitalSense-to-VitalDB sequence alignment."""

    channel_order: tuple[str, ...] = DEFAULT_VITALDB_CHANNELS
    source_frequency_hz: float = 1.0
    target_frequency_hz: float = 1.0
    source_duration_min: float = 2.0
    target_duration_min: float = 2.0
    spo2_fill_value: float = 99.0
    temperature_fill_value: float = 36.5

    @property
    def target_length(self) -> int:
        return int(round(self.target_duration_min * 60 * self.target_frequency_hz))

    @property
    def source_aligned_length(self) -> int:
        return int(round(self.source_duration_min * 60 * self.target_frequency_hz))

    @property
    def repeat_count(self) -> int:
        if self.source_aligned_length <= 0:
            raise ValueError("source_aligned_length must be positive.")
        return int(np.ceil(self.target_length / self.source_aligned_length))


def resample_sequence(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    """Resample a 1-D sequence from source_hz to target_hz using linear interpolation."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if source_hz == target_hz:
        return x
    n_out = int(round(len(x) * target_hz / source_hz))
    src_times = np.arange(len(x)) / source_hz
    dst_times = np.arange(n_out) / target_hz
    return np.interp(dst_times, src_times, x)


def repeat_or_trim(values: np.ndarray, target_length: int) -> np.ndarray:
    """Repeat a short sequence and trim to target_length."""

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot repeat an empty sequence.")
    repeat_count = int(np.ceil(target_length / x.size))
    return np.tile(x, repeat_count)[:target_length]


def align_vitalsense_record_to_vitaldb(
    record: pd.Series | dict,
    config: VitalSenseAlignmentConfig | None = None,
) -> dict:
    """Convert one VitalSense record into VitalDB-style aligned channels.

    Output `x_vitaldb_aligned` has shape [channels, 300] by default.
    """

    cfg = config or VitalSenseAlignmentConfig()
    hr = resample_sequence(np.asarray(record["HR"], dtype=float), cfg.source_frequency_hz, cfg.target_frequency_hz)
    rr = resample_sequence(np.asarray(record["RR"], dtype=float), cfg.source_frequency_hz, cfg.target_frequency_hz)
    hr = repeat_or_trim(hr, cfg.target_length)
    rr = repeat_or_trim(rr, cfg.target_length)

    observed = {
        "HR": hr,
        "RR": rr,
        "BPS": np.full(cfg.target_length, float(record["BPS"]), dtype=float),
        "BPM": np.full(cfg.target_length, float(record["BPM"]), dtype=float),
        "BPD": np.full(cfg.target_length, float(record["BPD"]), dtype=float),
        "SBP": np.full(cfg.target_length, float(record["BPS"]), dtype=float),
        "MBP": np.full(cfg.target_length, float(record["BPM"]), dtype=float),
        "DBP": np.full(cfg.target_length, float(record["BPD"]), dtype=float),
    }
    placeholders = {
        "SpO2": np.full(cfg.target_length, cfg.spo2_fill_value, dtype=float),
        "BT": np.full(cfg.target_length, cfg.temperature_fill_value, dtype=float),
    }

    channels = []
    channel_mask = []
    for channel in cfg.channel_order:
        if channel in observed:
            channels.append(observed[channel])
            channel_mask.append(True)
        elif channel in placeholders:
            channels.append(placeholders[channel])
            channel_mask.append(False)
        else:
            raise KeyError(f"Unsupported VitalDB channel for alignment: {channel}")

    x = np.stack(channels, axis=0).astype(np.float32)
    mask = np.asarray(channel_mask, dtype=bool)
    if x.shape != (len(cfg.channel_order), cfg.target_length):
        raise ValueError(
            f"Expected aligned shape {(len(cfg.channel_order), cfg.target_length)}, got {x.shape}"
        )

    metadata = {
        "subject_id": record.get("subject_id") if hasattr(record, "get") else None,
        "scenario": record.get("scenario") if hasattr(record, "get") else None,
        "channel_order": list(cfg.channel_order),
        "source_frequency_hz": cfg.source_frequency_hz,
        "target_frequency_hz": cfg.target_frequency_hz,
        "source_duration_min": cfg.source_duration_min,
        "target_duration_min": cfg.target_duration_min,
        "source_aligned_length": cfg.source_aligned_length,
        "target_length": cfg.target_length,
        "repeat_count": cfg.repeat_count,
        "missing_channels": [
            channel for channel, is_observed in zip(cfg.channel_order, mask) if not is_observed
        ],
        "bp_policy": "broadcast_recording_scalar_to_constant_sequence",
        "length_policy": "use_1hz_sequence_directly_trim_or_repeat_to_target_length",
    }
    return {
        "x_vitaldb_aligned": x,
        "channel_mask": mask,
        "alignment_metadata": metadata,
    }


def align_vitalsense_dataset_to_vitaldb(
    dataset: pd.DataFrame,
    config: VitalSenseAlignmentConfig | None = None,
) -> pd.DataFrame:
    """Apply VitalSense-to-VitalDB alignment to every record in a DataFrame."""

    rows = []
    for row in dataset.to_dict(orient="records"):
        aligned = align_vitalsense_record_to_vitaldb(row, config=config)
        rows.append({**row, **aligned})
    return pd.DataFrame(rows)


def plot_vitalsense_vitaldb_alignment(aligned_record: pd.Series | dict):
    """Plot one aligned VitalSense record in VitalDB channel order."""

    x = np.asarray(aligned_record["x_vitaldb_aligned"], dtype=float)
    mask = np.asarray(aligned_record["channel_mask"], dtype=bool)
    metadata = aligned_record["alignment_metadata"]
    channel_order = metadata["channel_order"]
    time_min = np.arange(x.shape[1]) / metadata["target_frequency_hz"] / 60.0

    fig, axes = plt.subplots(len(channel_order), 1, figsize=(12, 10), sharex=True)
    title = "VitalSense aligned to VitalDB format"
    if metadata.get("subject_id") and metadata.get("scenario"):
        title += f": {metadata['subject_id']} {metadata['scenario']}"
    fig.suptitle(title, y=0.995)

    for idx, (ax, channel) in enumerate(zip(axes, channel_order)):
        color = "tab:blue" if mask[idx] else "tab:gray"
        ax.plot(time_min, x[idx], color=color, linewidth=1.4)
        suffix = "observed" if mask[idx] else "missing placeholder"
        ax.set_ylabel(f"{channel}\n{suffix}")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    plt.tight_layout()
    return fig, axes
