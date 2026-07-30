from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, welch
from torch.utils.data import Dataset
import torch


DEFAULT_AUGMENT = {
    "noise_std": 0.03,
    "gain_std": 0.10,
    "offset_std": 0.05,
    "trend_std": 0.03,
    "mask_prob": 0.35,
    "mask_width": 6,
    "drop_prob": 0.10,
}


def augment_trace(
    trace: np.ndarray,
    config: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Apply label-preserving perturbations to a normalized RGB trace."""

    cfg = dict(DEFAULT_AUGMENT)
    if config:
        cfg.update({key: value for key, value in config.items() if value is not None})
    x = np.asarray(trace, dtype=np.float32).copy()
    if x.ndim != 2:
        raise ValueError(f"Expected trace [C, T], got {x.shape}.")
    channels, steps = x.shape
    if channels <= 0 or steps <= 0:
        return x

    gain_std = float(cfg.get("gain_std", 0.0))
    if gain_std > 0:
        gain = 1.0 + np.random.normal(0.0, gain_std, size=(channels, 1)).astype(np.float32)
        x *= gain

    offset_std = float(cfg.get("offset_std", 0.0))
    if offset_std > 0:
        offset = np.random.normal(0.0, offset_std, size=(channels, 1)).astype(np.float32)
        x += offset

    trend_std = float(cfg.get("trend_std", 0.0))
    if trend_std > 0:
        trend = np.linspace(-1.0, 1.0, steps, dtype=np.float32)[None, :]
        slope = np.random.normal(0.0, trend_std, size=(channels, 1)).astype(np.float32)
        x += slope * trend

    noise_std = float(cfg.get("noise_std", 0.0))
    if noise_std > 0:
        x += np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    drop_prob = float(cfg.get("drop_prob", 0.0))
    if drop_prob > 0 and channels > 1 and np.random.random() < drop_prob:
        channel = int(np.random.randint(0, channels))
        x[channel] = float(np.mean(x[channel]))

    mask_prob = float(cfg.get("mask_prob", 0.0))
    mask_width = int(cfg.get("mask_width", 0))
    if mask_prob > 0 and mask_width > 0 and steps > 1 and np.random.random() < mask_prob:
        width = int(np.random.randint(1, min(mask_width, steps) + 1))
        start = int(np.random.randint(0, steps - width + 1))
        fill = np.mean(x, axis=1, keepdims=True)
        x[:, start : start + width] = fill

    return x.astype(np.float32)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "COHFACE reference targets require h5py. Use an environment with "
            "h5py installed, such as /Users/wjs/miniconda3/envs/comvision."
        ) from exc
    return h5py


def _optional_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _require_ffmpeg() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise ImportError(
            "COHFACE video feature extraction requires either opencv-python/cv2 "
            "or system ffmpeg/ffprobe on PATH."
        )
    return ffmpeg, ffprobe


def _decode_attr(value):
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, np.generic):
            return item.item()
        return item
    if arr.size == 1:
        return _decode_attr(arr.reshape(-1)[0])
    return [_decode_attr(item) for item in arr]


def read_ref(path: str | Path) -> dict:
    """Read one COHFACE HDF5 reference file.

    Parameters
    ----------
    path:
        Path to `data.hdf5`.

    Returns
    -------
    dict
        Attributes, pulse, respiration, time, and sampling-rate metadata.
    """

    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as h5:
        attrs = {key: _decode_attr(value) for key, value in h5.attrs.items()}
        pulse = np.asarray(h5["pulse"], dtype=np.float32)
        respiration = np.asarray(h5["respiration"], dtype=np.float32)
        time = np.asarray(h5["time"], dtype=np.float32)

    fs = float(attrs.get("sample-rate-hz") or 1.0 / np.median(np.diff(time)))
    return {
        "attrs": attrs,
        "pulse": pulse,
        "respiration": respiration,
        "time": time,
        "sample_rate_hz": fs,
        "duration_sec": float(time[-1] - time[0]) if time.size > 1 else 0.0,
    }


def video_meta(path: str | Path) -> dict:
    """Return basic metadata for one video file.

    OpenCV is used when available. If `cv2` is unavailable, the function falls
    back to `ffprobe`, which is available on the local macOS workspace.
    """

    path = Path(path)
    cv2 = _optional_cv2()
    if cv2 is None:
        _, ffprobe = _require_ffmpeg()
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams") or []
        if not streams:
            raise OSError(f"Could not read video metadata: {path}")
        stream = streams[0]
        rate_text = stream.get("avg_frame_rate") or "0/1"
        num, den = rate_text.split("/")
        fps = float(num) / max(float(den), 1.0)
        duration = float(stream.get("duration") or 0.0)
        frames = int(stream.get("nb_frames") or round(duration * fps))
        return {
            "video_fps": fps,
            "video_frames": frames,
            "video_width": int(stream.get("width") or 0),
            "video_height": int(stream.get("height") or 0),
            "video_duration_sec": float(frames / fps) if fps > 0 else duration,
            "video_backend": "ffprobe",
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "video_fps": fps,
        "video_frames": frames,
        "video_width": width,
        "video_height": height,
        "video_duration_sec": float(frames / fps) if fps > 0 else np.nan,
        "video_backend": "opencv",
    }


def band_rate(
    values: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    min_seconds: float = 8.0,
) -> tuple[float, float]:
    """Estimate dominant frequency in a physiological band.

    Returns rate in cycles per minute and a simple spectral concentration score.
    The score is peak band power divided by total band power.
    """

    x = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x)
    if finite.sum() < int(sample_rate_hz * min_seconds):
        raise ValueError("Signal is too short or invalid for frequency estimation.")
    x = x.copy()
    x[~finite] = float(np.nanmedian(x[finite]))
    x = detrend(x)
    nperseg = min(x.size, max(256, int(round(sample_rate_hz * 16))))
    freqs, power = welch(x, fs=sample_rate_hz, nperseg=nperseg)
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        raise ValueError(f"No Welch bins inside band {band_hz}.")
    band_freqs = freqs[mask]
    band_power = power[mask]
    peak_idx = int(np.argmax(band_power))
    total_power = float(np.sum(band_power))
    quality = float(band_power[peak_idx] / max(total_power, 1e-12))
    return float(band_freqs[peak_idx] * 60.0), quality


def ref_targets(path: str | Path) -> dict:
    """Extract scalar HR and breathing-rate targets from one COHFACE reference."""

    ref = read_ref(path)
    fs = float(ref["sample_rate_hz"])
    hr_bpm, hr_quality = band_rate(ref["pulse"], fs, (0.7, 3.0))
    br_bpm, br_quality = band_rate(ref["respiration"], fs, (0.08, 0.7))
    return {
        "hr_bpm": hr_bpm,
        "br_bpm": br_bpm,
        "hr_quality": hr_quality,
        "br_quality": br_quality,
        "ref_len": int(ref["time"].size),
        "ref_duration_sec": ref["duration_sec"],
        "ref_sample_rate_hz": fs,
        "illumination": ref["attrs"].get("illumination"),
    }


def _fill_series(values: np.ndarray, fallback: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    finite = np.isfinite(x)
    if not finite.any():
        x[:] = float(fallback)
        return x
    if finite.all():
        return x
    idx = np.arange(x.size)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x.astype(np.float32)


def _pad_series(values: np.ndarray, length: int, fallback: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if length <= 0:
        raise ValueError("length must be positive.")
    if x.size == 0:
        return np.full(length, float(fallback), dtype=np.float32)
    if x.size >= length:
        return x[:length].astype(np.float32)
    out = np.empty(length, dtype=np.float32)
    out[: x.size] = x
    out[x.size :] = x[-1]
    return out


def _window_bounds(
    center_sec: float,
    duration_sec: float,
    window_sec: float,
    min_sec: float,
) -> tuple[float, float]:
    half = float(window_sec) / 2.0
    start = max(0.0, center_sec - half)
    end = min(float(duration_sec), center_sec + half)
    if end - start < min_sec:
        need = min_sec - (end - start)
        start = max(0.0, start - need / 2.0)
        end = min(float(duration_sec), end + need / 2.0)
    if end - start < min_sec and start <= 0.0:
        end = min(float(duration_sec), min_sec)
    if end - start < min_sec and end >= duration_sec:
        start = max(0.0, float(duration_sec) - min_sec)
    return start, end


def rate_series(
    values: np.ndarray,
    time: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    length: int = 60,
    target_rate_hz: float = 1.0,
    window_sec: float = 12.0,
    min_sec: float = 8.0,
    fallback_rate_bpm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a 1 Hz local rate series from a reference waveform.

    The returned series is padded to `length`. `mask` marks seconds covered by
    the original reference recording and should be used for evaluation.
    """

    x = np.asarray(values, dtype=np.float32).reshape(-1)
    t = np.asarray(time, dtype=np.float32).reshape(-1)
    if x.size != t.size:
        raise ValueError(f"values/time length mismatch: {x.size=} {t.size=}.")
    if x.size < 2:
        raise ValueError("Reference signal is too short.")
    duration_sec = float(t[-1] - t[0])
    if duration_sec <= 0:
        raise ValueError("Reference duration must be positive.")

    observed_len = int(min(length, max(1, np.floor(duration_sec * target_rate_hz))))
    target_time = np.arange(observed_len, dtype=np.float32) / float(target_rate_hz)
    target_time = target_time + 0.5 / float(target_rate_hz)

    rates = []
    for center in target_time:
        start, end = _window_bounds(center, duration_sec, window_sec, min_sec)
        mask = ((t - t[0]) >= start) & ((t - t[0]) < end)
        try:
            rate, _ = band_rate(x[mask], sample_rate_hz, band_hz, min_seconds=min_sec)
        except Exception:
            rate = np.nan
        rates.append(rate)

    if fallback_rate_bpm is None:
        fallback_rate_bpm, _ = band_rate(x, sample_rate_hz, band_hz, min_seconds=min_sec)
    observed = _fill_series(np.asarray(rates, dtype=np.float32), float(fallback_rate_bpm))
    padded = _pad_series(observed, length=length, fallback=float(fallback_rate_bpm))
    eval_mask = np.zeros(length, dtype=np.float32)
    eval_mask[:observed_len] = 1.0
    return padded.astype(np.float32), eval_mask


def ref_series(path: str | Path, length: int = 60) -> dict:
    """Extract 1 Hz HR/RR sequences from one COHFACE HDF5 reference."""

    ref = read_ref(path)
    fs = float(ref["sample_rate_hz"])
    scalar = ref_targets(path)
    hr_seq, hr_mask = rate_series(
        ref["pulse"],
        ref["time"],
        fs,
        (0.7, 3.0),
        length=length,
        fallback_rate_bpm=float(scalar["hr_bpm"]),
    )
    br_seq, br_mask = rate_series(
        ref["respiration"],
        ref["time"],
        fs,
        (0.08, 0.7),
        length=length,
        fallback_rate_bpm=float(scalar["br_bpm"]),
    )
    mask = np.minimum(hr_mask, br_mask).astype(np.float32)
    return {
        "hr_bpm_seq": hr_seq,
        "br_bpm_seq": br_seq,
        "target_mask": mask,
        "observed_length": int(mask.sum()),
        "sequence_length": int(length),
        "ref_duration_sec": ref["duration_sec"],
        "ref_sample_rate_hz": fs,
    }


def _protocol_rows(root: Path, protocol: str) -> list[dict]:
    rows = []
    protocol_dir = root / "protocols" / protocol
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Missing COHFACE protocol directory: {protocol_dir}")
    for split in ("train", "dev", "test"):
        path = protocol_dir / f"{split}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            item = line.strip()
            if not item:
                continue
            parts = item.split("/")
            if len(parts) < 3:
                raise ValueError(f"Unexpected protocol entry: {item}")
            rows.append(
                {
                    "subject_id": parts[0],
                    "session_id": parts[1],
                    "record_id": item,
                    "split": split,
                    "protocol": protocol,
                    "video_path": root / f"{item}.avi",
                    "hdf5_path": root / f"{item}.hdf5",
                }
            )
    return rows


def build_manifest(
    root: str | Path,
    protocol: str = "all",
    inspect_video: bool = True,
    inspect_ref: bool = True,
) -> pd.DataFrame:
    """Build a COHFACE recording manifest from official protocol files."""

    root = Path(root)
    rows = _protocol_rows(root, protocol)
    frame = pd.DataFrame(rows)
    frame["video_exists"] = frame["video_path"].map(lambda p: Path(p).exists())
    frame["hdf5_exists"] = frame["hdf5_path"].map(lambda p: Path(p).exists())

    if inspect_video:
        meta_rows = [video_meta(path) for path in frame["video_path"]]
        frame = pd.concat([frame, pd.DataFrame(meta_rows)], axis=1)

    if inspect_ref:
        target_rows = [ref_targets(path) for path in frame["hdf5_path"]]
        frame = pd.concat([frame, pd.DataFrame(target_rows)], axis=1)

    return frame


def video_trace(
    path: str | Path,
    every: int = 2,
    crop: tuple[float, float, float, float] | None = (0.20, 0.10, 0.80, 0.85),
    max_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    """Extract a mean RGB trace from a COHFACE video.

    The returned trace has shape `[3, T]`; channels are RGB in `[0, 1]`.
    """

    if every <= 0:
        raise ValueError("every must be positive.")
    path = Path(path)
    cv2 = _optional_cv2()
    if cv2 is None:
        return _video_trace_ffmpeg(path, every=every, crop=crop, max_frames=max_frames)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    rows = []
    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every == 0:
            if crop is not None:
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = crop
                frame = frame[
                    int(round(y0 * h)) : int(round(y1 * h)),
                    int(round(x0 * w)) : int(round(x1 * w)),
                ]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rows.append(rgb.mean(axis=(0, 1)))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not rows:
        raise ValueError(f"No frames read from {path}")
    trace = np.asarray(rows, dtype=np.float32).T
    return trace, fps / every


def _video_trace_ffmpeg(
    path: Path,
    every: int,
    crop: tuple[float, float, float, float] | None,
    max_frames: int | None,
    output_size: tuple[int, int] = (64, 48),
) -> tuple[np.ndarray, float]:
    """Extract RGB trace through ffmpeg when OpenCV is unavailable."""

    ffmpeg, _ = _require_ffmpeg()
    meta = video_meta(path)
    source_fps = float(meta["video_fps"])
    if source_fps <= 0:
        raise ValueError(f"Cannot infer video FPS for {path}")
    target_fps = source_fps / every
    width, height = output_size
    filters = []
    if crop is not None:
        x0, y0, x1, y1 = crop
        filters.append(
            f"crop=iw*{x1 - x0:.8f}:ih*{y1 - y0:.8f}:iw*{x0:.8f}:ih*{y0:.8f}"
        )
    filters.extend([f"fps={target_fps:.8f}", f"scale={width}:{height}"])
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        ",".join(filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.append("-")
    result = subprocess.run(cmd, check=True, capture_output=True)
    frame_size = width * height * 3
    if len(result.stdout) < frame_size:
        raise ValueError(f"No frames read from {path} with ffmpeg.")
    frame_count = len(result.stdout) // frame_size
    raw = np.frombuffer(result.stdout[: frame_count * frame_size], dtype=np.uint8)
    frames = raw.reshape(frame_count, height, width, 3).astype(np.float32) / 255.0
    trace = frames.mean(axis=(1, 2)).T.astype(np.float32)
    return trace, target_fps


def resample_trace(trace: np.ndarray, length: int) -> np.ndarray:
    """Resample `[C, T]` trace to a fixed temporal length."""

    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected trace [C, T], got {x.shape}.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if x.shape[1] == length:
        return x.astype(np.float32)
    source = np.linspace(0.0, 1.0, num=x.shape[1])
    target = np.linspace(0.0, 1.0, num=length)
    out = np.vstack([np.interp(target, source, row) for row in x])
    return out.astype(np.float32)


def resample_trace_time(
    trace: np.ndarray,
    source_rate_hz: float,
    length: int = 60,
    target_rate_hz: float = 1.0,
) -> np.ndarray:
    """Resample `[C, T]` trace onto a fixed 1 Hz time grid with edge padding."""

    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected trace [C, T], got {x.shape}.")
    if source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be positive.")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")
    source_time = np.arange(x.shape[1], dtype=np.float32) / float(source_rate_hz)
    target_time = np.arange(length, dtype=np.float32) / float(target_rate_hz)
    out = np.vstack(
        [
            np.interp(target_time, source_time, row, left=float(row[0]), right=float(row[-1]))
            for row in x
        ]
    )
    return out.astype(np.float32)


def add_video_features(
    frame: pd.DataFrame,
    length: int = 300,
    every: int = 2,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Attach fixed-length RGB traces to a manifest frame."""

    out = frame.copy()
    features = []
    rates = []
    for idx, row in out.iterrows():
        if max_records is not None and len(features) >= max_records:
            features.append(np.full((3, length), np.nan, dtype=np.float32))
            rates.append(np.nan)
            continue
        trace, rate = video_trace(row["video_path"], every=every)
        features.append(resample_trace(trace, length))
        rates.append(rate)
    out["video_rgb"] = features
    out["video_trace_rate_hz"] = rates
    return out


def add_video_series(
    frame: pd.DataFrame,
    length: int = 60,
    target_rate_hz: float = 1.0,
    every: int = 4,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Attach 1 Hz, fixed-length RGB traces aligned to the target sequence."""

    out = frame.copy()
    features = []
    rates = []
    for _, row in out.iterrows():
        if max_records is not None and len(features) >= max_records:
            features.append(np.full((3, length), np.nan, dtype=np.float32))
            rates.append(np.nan)
            continue
        trace, rate = video_trace(row["video_path"], every=every)
        features.append(resample_trace_time(trace, rate, length=length, target_rate_hz=target_rate_hz))
        rates.append(float(target_rate_hz))
    out["video_rgb"] = features
    out["video_trace_rate_hz"] = rates
    return out


def add_vital_series(frame: pd.DataFrame, length: int = 60) -> pd.DataFrame:
    """Attach 1 Hz HR/RR target sequences and evaluation masks."""

    out = frame.copy()
    rows = [ref_series(path, length=length) for path in out["hdf5_path"]]
    series_frame = pd.DataFrame(rows)
    return pd.concat([out.reset_index(drop=True), series_frame], axis=1)


def fit_feature_stats(frame: pd.DataFrame, column: str = "video_rgb") -> dict:
    """Fit train-only channel z-score stats for video trace features."""

    arrays = [np.asarray(value, dtype=np.float32) for value in frame[column]]
    stacked = np.concatenate(arrays, axis=1)
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def add_feature_norm(
    frame: pd.DataFrame,
    stats: Mapping,
    source: str = "video_rgb",
    target: str = "video_rgb_norm",
) -> pd.DataFrame:
    """Apply channel z-score normalization to fixed-length video traces."""

    out = frame.copy()
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None]
    out[target] = [
        ((np.asarray(value, dtype=np.float32) - mean) / std).astype(np.float32)
        for value in out[source]
    ]
    return out


def fit_target_stats(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    """Fit scalar target z-score stats from the training split."""

    stats = {}
    for column in columns:
        values = frame[column].astype(float).to_numpy()
        std = float(np.std(values))
        stats[column] = {
            "mean": float(np.mean(values)),
            "std": std if std >= 1e-6 else 1.0,
        }
    return stats


def fit_sequence_stats(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = ("hr_bpm_seq", "br_bpm_seq"),
    mask_col: str = "target_mask",
) -> dict:
    """Fit z-score stats from observed sequence positions in the training split."""

    stats = {}
    for column in columns:
        values = []
        for _, row in frame.iterrows():
            seq = np.asarray(row[column], dtype=np.float32)
            mask = np.asarray(row[mask_col], dtype=bool)
            values.append(seq[mask])
        flat = np.concatenate(values) if values else np.asarray([], dtype=np.float32)
        if flat.size == 0:
            raise ValueError(f"No observed values for {column}.")
        std = float(np.std(flat))
        stats[column] = {
            "mean": float(np.mean(flat)),
            "std": std if std >= 1e-6 else 1.0,
        }
    return stats


def add_target_norm(
    frame: pd.DataFrame,
    stats: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Add normalized scalar target columns with `_norm` suffix."""

    out = frame.copy()
    for column, item in stats.items():
        out[f"{column}_norm"] = (
            out[column].astype(float) - float(item["mean"])
        ) / float(item["std"])
    return out


def add_sequence_norm(
    frame: pd.DataFrame,
    stats: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Add normalized sequence target columns with `_norm` suffix."""

    out = frame.copy()
    for column, item in stats.items():
        mean = float(item["mean"])
        std = float(item["std"])
        out[f"{column}_norm"] = [
            ((np.asarray(value, dtype=np.float32) - mean) / std).astype(np.float32)
            for value in out[column]
        ]
    return out


class CohfaceVitalDataset(Dataset):
    """COHFACE video trace input with HR/RR time-series targets."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_col: str = "video_rgb_norm",
        target_cols: tuple[str, str] = ("hr_bpm_seq_norm", "br_bpm_seq_norm"),
        mask_col: str = "target_mask",
        augment: bool = False,
        augment_config: Mapping[str, float] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_col = feature_col
        self.target_cols = target_cols
        self.mask_col = mask_col
        self.augment = bool(augment)
        self.augment_config = dict(augment_config or {})

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        x = np.asarray(row[self.feature_col], dtype=np.float32)
        if self.augment:
            x = augment_trace(x, self.augment_config)
        y = np.stack([np.asarray(row[col], dtype=np.float32) for col in self.target_cols], axis=0)
        mask = np.asarray(row[self.mask_col], dtype=np.float32)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
            "condition": str(row.get("illumination", "")),
        }


class CohfaceCodeDataset(Dataset):
    """Patchified COHFACE video trace input with VitalDB teacher code targets."""

    def __init__(
        self,
        frame: pd.DataFrame,
        patch_len: int,
        codes_per_token: int,
        feature_col: str = "video_rgb_norm",
        augment: bool = False,
        augment_config: Mapping[str, float] | None = None,
    ) -> None:
        if patch_len <= 0:
            raise ValueError("patch_len must be positive.")
        if codes_per_token <= 0:
            raise ValueError("codes_per_token must be positive.")
        self.frame = frame.reset_index(drop=True)
        self.patch_len = int(patch_len)
        self.codes_per_token = int(codes_per_token)
        self.feature_col = feature_col
        self.augment = bool(augment)
        self.augment_config = dict(augment_config or {})

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        x = np.asarray(row[self.feature_col], dtype=np.float32)
        if self.augment:
            x = augment_trace(x, self.augment_config)
        if x.ndim != 2:
            raise ValueError(f"Expected video feature [C, T], got {x.shape}.")
        if x.shape[1] % self.patch_len != 0:
            raise ValueError(
                "Video feature length must be divisible by patch_len "
                f"({x.shape[1]=} {self.patch_len=})."
            )
        n_patches = x.shape[1] // self.patch_len
        x_patch = x.reshape(x.shape[0], n_patches, self.patch_len).transpose(1, 0, 2)
        y_raw = np.asarray(row["teacher_code_targets"], dtype=np.int64)
        if y_raw.ndim == 1:
            if y_raw.size % self.codes_per_token != 0:
                raise ValueError("teacher_code_targets length does not match codes_per_token.")
            y_token = y_raw.reshape(-1, self.codes_per_token)
        elif y_raw.ndim == 2:
            y_token = y_raw
        else:
            raise ValueError(f"Unexpected teacher code target shape: {y_raw.shape}")
        if y_token.shape[0] != n_patches:
            raise ValueError(
                "Video patch count must match teacher token count "
                f"({n_patches=} teacher_tokens={y_token.shape[0]})."
            )
        item = {
            "x": torch.from_numpy(x_patch.astype(np.float32)),
            "y": torch.from_numpy(y_token.reshape(-1).astype(np.int64)),
            "y_token": torch.from_numpy(y_token.astype(np.int64)),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
            "condition": str(row.get("illumination", "")),
        }
        if "hr_bpm_seq" in row and "br_bpm_seq" in row:
            vital = np.stack(
                [
                    np.asarray(row["hr_bpm_seq"], dtype=np.float32),
                    np.asarray(row["br_bpm_seq"], dtype=np.float32),
                ],
                axis=0,
            )
            item["vital"] = torch.from_numpy(vital)
        if "target_mask" in row:
            item["mask"] = torch.from_numpy(np.asarray(row["target_mask"], dtype=np.float32))
        return item


__all__ = [
    "CohfaceCodeDataset",
    "CohfaceVitalDataset",
    "DEFAULT_AUGMENT",
    "add_feature_norm",
    "add_target_norm",
    "add_sequence_norm",
    "add_video_series",
    "add_vital_series",
    "add_video_features",
    "augment_trace",
    "band_rate",
    "build_manifest",
    "fit_feature_stats",
    "fit_sequence_stats",
    "fit_target_stats",
    "rate_series",
    "read_ref",
    "ref_series",
    "ref_targets",
    "resample_trace",
    "resample_trace_time",
    "video_meta",
    "video_trace",
]
