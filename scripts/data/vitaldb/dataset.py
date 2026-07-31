from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Dict, Optional


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ArrayTransform = Callable[[np.ndarray], np.ndarray]
ROOT_DIR = '/home/junshi//data/VitalDB'

DEFAULT_VITALSIGN = {
    "HR": "Solar8000/HR",
    "SpO2": "Solar8000/PLETH_SPO2",
    "RR": "Solar8000/VENT_RR",
    "BT": "Solar8000/BT",
    "SBP": "Solar8000/NIBP_SBP",
    "DBP": "Solar8000/NIBP_DBP",
    "MBP": "Solar8000/NIBP_MBP",
}


DEFAULT_RULES = {
    "HR": [45, 120],
    "SpO2": [92, 100],
    "RR": [8, 30],
    "BT": [35, 40],
    "SBP": [90, 150],
    "DBP": [50, 95],
    "MBP": [65, 110],
}


BP_ROLES = {"SBP", "DBP", "MBP"}


def vital_signs_to_tracks(vital_signs: Sequence[str]) -> dict[str, str]:
    selected = tuple(vital_signs)
    if not selected:
        raise ValueError("vital_signs must contain at least one role.")
    unknown = [role for role in selected if role not in DEFAULT_VITALSIGN]
    if unknown:
        raise ValueError(
            f"Unknown vital signs {unknown}. Choose from {list(DEFAULT_VITALSIGN)}."
        )
    if len(set(selected)) != len(selected):
        raise ValueError("vital_signs must not contain duplicates.")
    return {role: DEFAULT_VITALSIGN[role] for role in selected}

def get_case_ids(
    metadata_dir: str | Path = f"{ROOT_DIR}/metadata",
    vital_signs: Sequence[str] = tuple(DEFAULT_VITALSIGN),
) -> list[int]:
    """Return case IDs containing every requested vital-sign track."""
    mapping = vital_signs_to_tracks(vital_signs)
    tracks_path = Path(metadata_dir) / "VitalDB_trks_uncompressed.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Track metadata not found: {tracks_path}")

    tracks = pd.read_csv(tracks_path)
    required = set(mapping.values())
    matched = tracks[tracks["tname"].isin(required)]
    grouped = matched.groupby("caseid")["tname"].agg(set)
    case_ids = grouped[grouped.apply(required.issubset)].index.tolist()
    return [int(case_id) for case_id in case_ids]

def get_vital_file_path_by_id(
    case_id: int,
    data_dir: str | Path = f"{ROOT_DIR}/raw",
) -> Path:
    """Return the local .vital path for one case ID."""
    data_root = Path(data_dir)
    candidates = (
        data_root / f"{int(case_id)}.vital",
        data_root / f"case{int(case_id)}.vital",
        data_root / f"{int(case_id):04d}.vital",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"VitalDB file not found for case {case_id} in {data_root}.")

def _read_raw_data_by_id(
    case_id: int,
    data_dir: str | Path = f"{ROOT_DIR}/raw",
    vital_signs: Sequence[str] = tuple(DEFAULT_VITALSIGN),
    interval_sec: int = 2,
) -> pd.DataFrame:
    """Read one case directly from its .vital file."""
    try:
        import vitaldb  # type: ignore
    except ImportError as exc:
        raise ImportError("The vitaldb package is required to read .vital files.") from exc

    mapping = vital_signs_to_tracks(vital_signs)
    vital_path = get_vital_file_path_by_id(case_id, data_dir)
    frame = vitaldb.VitalFile(str(vital_path)).to_pandas(
        list(mapping.values()),
        interval=interval_sec,
    )
    if frame is None or frame.empty:
        raise ValueError(f"VitalDB case {case_id} contains no requested data.")

    frame = frame.rename(columns={track: role for role, track in mapping.items()})
    frame = frame[list(mapping)]
    frame.index = frame.index.astype(float)
    frame.index.name = "time"
    return frame

def _resample_dense_signals(
    frame: pd.DataFrame,
    dense_signs: Sequence[str],
    interval_sec: int,
) -> pd.DataFrame:
    if frame.empty or not dense_signs:
        return pd.DataFrame()
    start = int(np.floor(frame.index.min() / interval_sec) * interval_sec)
    end = int(np.ceil(frame.index.max() / interval_sec) * interval_sec)
    grid = np.arange(start, end + interval_sec, interval_sec)
    regular = pd.DataFrame(index=pd.Index(grid, name="time"))

    for role in dense_signs:
        series = frame[role].dropna()
        if series.empty:
            regular[role] = np.nan
            continue
        rounded = np.rint(series.index.to_numpy(dtype=float) / interval_sec)
        rounded = rounded.astype(np.int64) * interval_sec
        regular[role] = pd.Series(series.to_numpy(), index=rounded).groupby(level=0).median().reindex(grid)
    return regular

def _is_dense_signal_clean(
    regular: pd.DataFrame,
    role: str,
    segment_start: int,
    segment_end: int,
    interval_sec: int,
    segment_sec: int,
    coverage_threshold: float,
    rules: Mapping[str, Sequence[float]],
) -> tuple[bool, dict[str, Any]]:
    clip = regular.loc[
        (regular.index >= segment_start) & (regular.index < segment_end),
        role,
    ]
    expected_count = segment_sec // interval_sec
    valid_count = int(clip.notna().sum())
    coverage = valid_count / expected_count if expected_count else 0.0
    stats: dict[str, Any] = {
        f"{role}_valid_count": valid_count,
        f"{role}_coverage": coverage,
    }
    if coverage < coverage_threshold:
        return False, stats

    values = clip.ffill().bfill().to_numpy(dtype=float)
    if values.size != expected_count or not np.isfinite(values).all():
        return False, stats
    stats.update(
        {
            f"{role}_min": float(values.min()),
            f"{role}_max": float(values.max()),
            f"{role}_median": float(np.median(values)),
            f"{role}_p05": float(np.percentile(values, 5)),
            f"{role}_p95": float(np.percentile(values, 95)),
        }
    )
    low, high = rules.get(role, (float("-inf"), float("inf")))
    return bool(values.min() >= low and values.max() <= high), stats

def _is_bp_clean(
    frame: pd.DataFrame,
    bp_signals: Sequence[str],
    segment_start: int,
    segment_end: int,
    bp_nearest_sec: int,
    rules: Mapping[str, Sequence[float]],
) -> tuple[bool, dict[str, Any]]:
    if not bp_signals:
        return True, {}
    center = (segment_start + segment_end) / 2
    stats: dict[str, Any] = {}
    clean = True
    for role in bp_signals:
        if role not in frame:
            clean = False
            continue
        series = frame[role].dropna()
        if series.empty:
            clean = False
            continue
        distances = np.abs(series.index.to_numpy(dtype=float) - center)
        position = int(np.argmin(distances))
        value = float(series.iloc[position])
        age_sec = float(distances[position])
        stats[f"{role}_nearest_value"] = value
        stats[f"{role}_nearest_age_sec"] = age_sec
        low, high = rules.get(role, (float("-inf"), float("inf")))
        clean = clean and age_sec <= bp_nearest_sec and low <= value <= high

    if BP_ROLES.issubset(bp_signals):
        sbp = stats.get("SBP_nearest_value")
        dbp = stats.get("DBP_nearest_value")
        mbp = stats.get("MBP_nearest_value")
        consistent = bool(
            sbp is not None
            and dbp is not None
            and mbp is not None
            and sbp > mbp > dbp
        )
        stats["BP_consistent"] = consistent
        clean = clean and consistent
    return bool(clean), stats

class ComposeVitalDB:
    """Apply a sequence of transforms to a VitalDB channel-first array."""

    def __init__(self, transforms: Sequence[ArrayTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32)
        for transform in self.transforms:
            out = transform(out)
        return np.asarray(out, dtype=np.float32)

    def fit(self, data: Iterable[np.ndarray]) -> "ComposeVitalDB":
        """Fit stateful transforms using data after all preceding transforms."""
        fitted_transforms: list[ArrayTransform] = []
        for transform in self.transforms:
            fit = getattr(transform, "fit", None)
            if callable(fit):
                fit(
                    self._apply_transforms(sample, fitted_transforms)
                    for sample in data
                )
            fitted_transforms.append(transform)
        return self

    def get_transform(self, transform_type: type[Any]) -> Any:
        """Return the single transform matching `transform_type`."""
        matches = [
            transform for transform in self.transforms if isinstance(transform, transform_type)
        ]
        if len(matches) != 1:
            raise LookupError(
                f"Expected one {transform_type.__name__}, found {len(matches)}."
            )
        return matches[0]

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Reverse invertible transforms, skipping stages without an inverse."""
        out: np.ndarray | torch.Tensor = x
        for transform in reversed(self.transforms):
            inverse = getattr(transform, "inverse_transform", None)
            if callable(inverse):
                out = inverse(out)
        return out

    @staticmethod
    def _apply_transforms(
        x: np.ndarray,
        transforms: Sequence[ArrayTransform],
    ) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32)
        for transform in transforms:
            out = transform(out)
        return out


class SmoothVitalDB:
    """Apply exponential moving-average smoothing along the time axis."""

    def __init__(self, interval_sec: float, tau_sec: float = 30.0) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive.")
        if tau_sec < 0:
            raise ValueError("tau_sec must be non-negative.")
        self.interval_sec = float(interval_sec)
        self.tau_sec = float(tau_sec)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if self.tau_sec == 0 or values.shape[1] < 2:
            return values.copy()

        alpha = 1.0 - np.exp(-self.interval_sec / self.tau_sec)
        out = np.empty_like(values)
        out[:, 0] = values[:, 0]
        for index in range(1, values.shape[1]):
            out[:, index] = alpha * values[:, index] + (1.0 - alpha) * out[:, index - 1]
        return out


class DifferenceVitalDB:
    """Convert each channel to a fixed-horizon temporal difference."""

    def __init__(self, interval_sec: float, horizon_sec: float = 10.0) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive.")
        if horizon_sec <= 0:
            raise ValueError("horizon_sec must be positive.")
        self.horizon_steps = max(1, int(round(horizon_sec / interval_sec)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        out = np.zeros_like(values)
        if self.horizon_steps < values.shape[1]:
            out[:, self.horizon_steps :] = (
                values[:, self.horizon_steps :] - values[:, : -self.horizon_steps]
            )
        return out


class ResampleVitalDB:
    """Resample a channel-first array to a target temporal frequency."""

    def __init__(
        self,
        input_frequency_hz: float = 0.5,
        target_frequency_hz: float = 1.0,
    ) -> None:
        if input_frequency_hz <= 0:
            raise ValueError("input_frequency_hz must be positive.")
        if target_frequency_hz <= 0:
            raise ValueError("target_frequency_hz must be positive.")
        self.input_frequency_hz = float(input_frequency_hz)
        self.target_frequency_hz = float(target_frequency_hz)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected [C, T] input, got shape {values.shape}.")
        if values.shape[1] == 0:
            raise ValueError("Cannot resample an empty time axis.")
        if self.input_frequency_hz == self.target_frequency_hz:
            return values.copy()

        output_length = max(
            1,
            int(round(values.shape[1] * self.target_frequency_hz / self.input_frequency_hz)),
        )
        source_time = np.arange(values.shape[1], dtype=np.float64) / self.input_frequency_hz
        target_time = np.arange(output_length, dtype=np.float64) / self.target_frequency_hz
        out = np.empty((values.shape[0], output_length), dtype=np.float32)
        for channel in range(values.shape[0]):
            out[channel] = np.interp(target_time, source_time, values[channel]).astype(
                np.float32
            )
        return out


class RandomNoiseVitalDB:
    """Add zero-mean Gaussian noise to a clean VitalDB segment."""

    def __init__(self, std: float = 0.01, seed: int | None = None) -> None:
        if std < 0:
            raise ValueError("std must be non-negative.")
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if self.std == 0:
            return values.copy()
        noise = self.rng.normal(0.0, self.std, size=values.shape).astype(np.float32)
        return values + noise


class NormalizeVitalDB:
    """Fit and apply train-split channel normalization statistics."""

    def __init__(
        self,
        vital_signs: Sequence[str],
        statistics: Mapping[str, tuple[float, float]] | None = None,
        method: str = "z_score",
        clip: bool = True,
        eps: float = 1e-6,
    ) -> None:
        if method not in {"z_score", "min_max"}:
            raise ValueError(f"Unsupported normalization method: {method}")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.vital_signs = tuple(vital_signs)
        if not self.vital_signs:
            raise ValueError("vital_signs must contain at least one role.")
        self.statistics = dict(statistics or {})
        self.method = method
        self.clip = bool(clip)
        self.eps = float(eps)

        if statistics is not None:
            self._validate_statistics()

    @property
    def is_fitted(self) -> bool:
        """Return whether every configured channel has fitted statistics."""
        return all(role in self.statistics for role in self.vital_signs)

    def fit(self, data: Iterable[np.ndarray] | np.ndarray) -> "NormalizeVitalDB":
        """Estimate per-channel statistics from training arrays shaped `[C, T]`."""
        samples: Iterable[np.ndarray]
        if isinstance(data, np.ndarray):
            samples = (data,)
        else:
            samples = data

        count = np.zeros(len(self.vital_signs), dtype=np.int64)
        mean = np.zeros(len(self.vital_signs), dtype=np.float64)
        m2 = np.zeros(len(self.vital_signs), dtype=np.float64)
        minimum = np.full(len(self.vital_signs), np.inf, dtype=np.float64)
        maximum = np.full(len(self.vital_signs), -np.inf, dtype=np.float64)

        for sample in samples:
            values = self._validate_input(sample)
            for channel in range(len(self.vital_signs)):
                channel_values = values[channel].astype(np.float64, copy=False).ravel()
                if not np.isfinite(channel_values).all():
                    role = self.vital_signs[channel]
                    raise ValueError(f"Training data for {role} contains non-finite values.")
                if channel_values.size == 0:
                    continue
                if self.method == "min_max":
                    minimum[channel] = min(minimum[channel], float(channel_values.min()))
                    maximum[channel] = max(maximum[channel], float(channel_values.max()))
                    count[channel] += channel_values.size
                    continue

                batch_count = channel_values.size
                batch_mean = float(channel_values.mean())
                batch_m2 = float(np.square(channel_values - batch_mean).sum())
                delta = batch_mean - mean[channel]
                total_count = count[channel] + batch_count
                mean[channel] += delta * batch_count / total_count
                m2[channel] += (
                    batch_m2
                    + delta * delta * count[channel] * batch_count / total_count
                )
                count[channel] = total_count

        empty_roles = [
            role for role, role_count in zip(self.vital_signs, count) if role_count == 0
        ]
        if empty_roles:
            raise ValueError(f"Cannot fit normalization without data for: {empty_roles}")

        if self.method == "z_score":
            std = np.sqrt(m2 / count)
            self.statistics = {
                role: (float(mean[index]), max(float(std[index]), self.eps))
                for index, role in enumerate(self.vital_signs)
            }
        else:
            self.statistics = {
                role: (
                    float(minimum[index]),
                    max(float(maximum[index]), float(minimum[index]) + self.eps),
                )
                for index, role in enumerate(self.vital_signs)
            }
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize one fitted channel-first array."""
        if not self.is_fitted:
            raise RuntimeError("NormalizeVitalDB must be fitted on training data first.")
        values = self._validate_input(x).copy()

        for channel, role in enumerate(self.vital_signs):
            first, second = self.statistics[role]
            if self.method == "z_score":
                if second <= 0:
                    raise ValueError(f"Standard deviation for {role} must be positive.")
                values[channel] = (values[channel] - first) / second
            else:
                if second <= first:
                    raise ValueError(f"Min/max statistics for {role} are invalid.")
                if self.clip:
                    values[channel] = np.clip(values[channel], first, second)
                values[channel] = (values[channel] - first) / (second - first)
        return values

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit from one array and return its normalized values."""
        return self.fit(data).transform(data)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Restore normalized `[C, T]` or `[B, C, T]` data to original units."""
        if not self.is_fitted:
            raise RuntimeError("NormalizeVitalDB must be fitted on training data first.")
        if isinstance(x, torch.Tensor):
            if x.ndim not in {2, 3} or x.shape[-2] != len(self.vital_signs):
                raise ValueError(
                    f"Expected [C, T] or [B, C, T] input with "
                    f"{len(self.vital_signs)} channels, got shape {tuple(x.shape)}."
                )
            values = x.clone()
        else:
            values = np.asarray(x, dtype=np.float32)
            if values.ndim not in {2, 3} or values.shape[-2] != len(self.vital_signs):
                raise ValueError(
                    f"Expected [C, T] or [B, C, T] input with "
                    f"{len(self.vital_signs)} channels, got shape {values.shape}."
                )
            values = values.copy()
        for channel, role in enumerate(self.vital_signs):
            first, second = self.statistics[role]
            if self.method == "z_score":
                values[..., channel, :] = values[..., channel, :] * second + first
            else:
                values[..., channel, :] = (
                    values[..., channel, :] * (second - first) + first
                )
        return values

    def state_dict(self) -> dict[str, Any]:
        """Export configuration and fitted statistics for checkpoint storage."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export an unfitted NormalizeVitalDB transform.")
        return {
            "vital_signs": self.vital_signs,
            "method": self.method,
            "clip": self.clip,
            "eps": self.eps,
            "statistics": dict(self.statistics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "NormalizeVitalDB":
        """Restore configuration and statistics exported by `state_dict`."""
        state_vital_signs = tuple(state["vital_signs"])
        state_method = str(state["method"])
        if state_vital_signs != self.vital_signs:
            raise ValueError(
                f"Vital-sign mismatch: state={state_vital_signs}, "
                f"transform={self.vital_signs}."
            )
        if state_method != self.method:
            raise ValueError(
                f"Normalization method mismatch: state={state_method}, "
                f"transform={self.method}."
            )
        self.clip = bool(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        self.statistics = {
            role: (float(values[0]), float(values[1]))
            for role, values in state["statistics"].items()
        }
        self._validate_statistics()
        return self

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected [C, T] input, got shape {values.shape}.")
        if values.shape[0] != len(self.vital_signs):
            raise ValueError(
                f"Expected {len(self.vital_signs)} channels, got {values.shape[0]}."
            )
        return values

    def _validate_statistics(self) -> None:
        missing = [role for role in self.vital_signs if role not in self.statistics]
        if missing:
            raise KeyError(f"Missing normalization statistics for roles: {missing}")
        for role in self.vital_signs:
            first, second = self.statistics[role]
            if self.method == "z_score" and second <= 0:
                raise ValueError(f"Standard deviation for {role} must be positive.")
            if self.method == "min_max" and second <= first:
                raise ValueError(f"Min/max statistics for {role} are invalid.")


class VitalDBData:
    """Create and access an incremental disk-backed clean VitalDB store.

    Processed layout:

    data/processed/vitaldb/
      metadata.json
      case_0001/
        segments.csv
        HR/segment_000000.npy
        RR/segment_000000.npy
        SBP/segment_000000.npy
    """

    def __init__(
        self,
        data_dir: str | Path = "data/VitalDB/raw",
        metadata_dir: str | Path = "data/VitalDB/metadata",
        clean_dir: str | Path = "data/vitaldb/processed/pretrain-7vitalsign-v1",
        vital_signs: Sequence[str] = tuple(DEFAULT_VITALSIGN),
        interval_sec: int = 2,
        segment_sec: int = 60,
        coverage: float = 0.9,
        bp_nearest_sec: int = 60,
        rules: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_dir = Path(metadata_dir)
        self.clean_dir = Path(clean_dir)
        self.vital_signs = tuple(vital_signs_to_tracks(vital_signs))
        self.interval_sec = int(interval_sec)
        self.segment_sec = int(segment_sec)
        self.coverage = float(coverage)
        self.bp_nearest_sec = int(bp_nearest_sec)
        self.rules = dict(rules or DEFAULT_RULES)
        self.metadata_path = self.clean_dir / "metadata.json"
        self.ids: list[tuple[int, int]] = []
        if self.metadata_path.exists():
            self._validate_metadata()
            self._refresh_ids()

    def clean(
        self,
        case_ids: Iterable[int] | None = None,
        progress_every: int | None = 10,
    ) -> "VitalDBData":
        """Add only missing vital signs to the disk-backed clean store."""
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self._write_or_validate_metadata()
        if case_ids is None:
            case_ids = get_case_ids(self.metadata_dir, self.vital_signs)
        selected_case_ids = [int(case_id) for case_id in case_ids]
        completed_cases = 0

        for case_index, case_id in enumerate(selected_case_ids, start=1):
            case_dir = self._case_dir(case_id)
            missing_signs = [
                sign for sign in self.vital_signs
                if not (case_dir / sign).exists()
            ]
            if not missing_signs:
                completed_cases += 1
            else:
                try:
                    frame = _read_raw_data_by_id(
                        case_id,
                        data_dir=self.data_dir,
                        vital_signs=missing_signs,
                        interval_sec=self.interval_sec,
                    )
                except (FileNotFoundError, ValueError):
                    frame = None

                if frame is not None:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    segments = self._load_or_create_segments(case_dir, frame)
                    for sign in missing_signs:
                        self._clean_one_signal(case_dir, frame, sign, segments)
                    completed_cases += 1

            if progress_every and (
                case_index == 1
                or case_index == len(selected_case_ids)
                or case_index % int(progress_every) == 0
            ):
                print(
                    f"[VitalDBData.clean] processed {case_index}/{len(selected_case_ids)} "
                    f"cases, completed_cases={completed_cases}",
                    flush=True,
                )

        self._refresh_ids()
        if not self.ids:
            raise ValueError("VitalDB cleaning produced no shared clean segments.")
        return self

    def read(
        self,
        sample_id: tuple[int, int],
    ) -> np.ndarray:
        """Read and stack requested vital signs for one pair ID."""
        case_id, segment_id = int(sample_id[0]), int(sample_id[1])
        case_dir = self._case_dir(case_id)
        arrays = [
            np.load(case_dir / sign / f"segment_{segment_id:06d}.npy").astype(np.float32)
            for sign in self.vital_signs
        ]
        return np.stack(arrays, axis=0)

    def _clean_one_signal(
        self,
        case_dir: Path,
        frame: pd.DataFrame,
        sign: str,
        segments: pd.DataFrame,
    ) -> None:
        sign_dir = case_dir / sign
        sign_dir.mkdir(parents=True, exist_ok=True)

        if sign in BP_ROLES:
            self._process_discrete_signal(frame, segments, sign, sign_dir)
        else:
            self._process_dense_signal(frame, segments, sign, sign_dir)


    def _process_discrete_signal(self, frame, segments, sign, sign_dir):
        for row in segments.to_dict(orient="records"):
            clean, stats = _is_bp_clean(
                frame,
                [sign],
                int(row["start_sec"]),
                int(row["end_sec"]),
                self.bp_nearest_sec,
                self.rules,
            )
            if clean:
                value = stats[f"{sign}_nearest_value"]
                length = self.segment_sec // self.interval_sec
                x = np.full(length, value, dtype=np.float32)
                np.save(sign_dir / f"segment_{int(row['segment_id']):06d}.npy", x)
        return


    def _process_dense_signal(self, frame, segments, sign, sign_dir):
        resampled = _resample_dense_signals(frame, [sign], self.interval_sec)
        for row in segments.to_dict(orient="records"):
            start_sec = int(row["start_sec"])
            end_sec = int(row["end_sec"])
            clean, _stats = _is_dense_signal_clean(
                resampled,
                sign,
                start_sec,
                end_sec,
                self.interval_sec,
                self.segment_sec,
                self.coverage,
                self.rules,
            )
            if clean:
                clip = resampled.loc[
                    (resampled.index >= start_sec) & (resampled.index < end_sec),
                    sign,
                ].ffill().bfill()
                x = clip.to_numpy(dtype=np.float32)
                np.save(sign_dir / f"segment_{int(row['segment_id']):06d}.npy", x)
        return

    def _load_or_create_segments(
        self,
        case_dir: Path,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        path = case_dir / "segments.csv"
        if path.exists():
            return pd.read_csv(path)

        start = int(np.floor(frame.index.min() / self.interval_sec) * self.interval_sec)
        end = int(np.ceil(frame.index.max() / self.interval_sec) * self.interval_sec)
        rows = [
            {
                "segment_id": start_sec // self.segment_sec,
                "start_sec": start_sec,
                "end_sec": start_sec + self.segment_sec,
            }
            for start_sec in range(start, end - self.segment_sec + 1, self.segment_sec)
        ]
        segments = pd.DataFrame(rows)
        segments.to_csv(path, index=False)
        return segments

    def _refresh_ids(self) -> None:
        ids: list[tuple[int, int]] = []
        for case_dir in sorted(self.clean_dir.glob("case_*")):
            try:
                case_id = int(case_dir.name.removeprefix("case_"))
            except ValueError:
                continue
            sign_ids = []
            for sign in self.vital_signs:
                sign_dir = case_dir / sign
                if not sign_dir.exists():
                    sign_ids = []
                    break
                sign_ids.append({
                    int(path.stem.removeprefix("segment_"))
                    for path in sign_dir.glob("segment_*.npy")
                })
            if sign_ids:
                for segment_id in sorted(set.intersection(*sign_ids)):
                    ids.append((case_id, segment_id))
        self.ids = ids

    def _case_dir(self, case_id: int) -> Path:
        return self.clean_dir / f"case_{int(case_id):04d}"

    def _write_or_validate_metadata(self) -> None:
        config = {
            "interval_sec": self.interval_sec,
            "segment_sec": self.segment_sec,
        }
        if self.metadata_path.exists():
            self._validate_metadata()
        else:
            self.metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _validate_metadata(self) -> None:
        config = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected = {
            "interval_sec": self.interval_sec,
            "segment_sec": self.segment_sec,
        }
        if config != expected:
            raise ValueError(
                f"Processed VitalDB config mismatch: found={config}, expected={expected}."
            )


def read_data_by_id(
    vitaldb_data: VitalDBData,
    sample_id: tuple[int, int],
) -> np.ndarray:
    """Read one clean VitalDB segment by (case_id, segment_id)."""
    return vitaldb_data.read(sample_id)


class VitalDBDataset(Dataset):
    """PyTorch access to one or more consecutive clean VitalDB segments.

    Args:
        vitaldb_data: Clean disk-backed VitalDB storage.
        id_list: Candidate ``(case_id, segment_id)`` pairs. IDs are grouped by
            case and sorted by segment ID before consecutive windows are built.
        transforms: Optional transform applied after consecutive segments are
            concatenated along the time axis.
        time_length: Datapoint duration in seconds. It must be a positive
            multiple of 60. The default 60 preserves one-segment samples;
            ``time_length=600`` joins ten consecutive 60-second segments.

    Notes:
        Datapoints never cross case boundaries or gaps in segment IDs. Windows
        slide by one 60-second segment, so adjacent datapoints may overlap.
    """

    def __init__(
        self,
        vitaldb_data: VitalDBData,
        id_list: Sequence[tuple[int, int]] | None = None,
        transforms: ArrayTransform | None = None,
        time_length: int = 60,
    ) -> None:
        if not vitaldb_data.ids:
            raise RuntimeError("Call VitalDBData.clean(...) before creating the dataset.")
        if isinstance(time_length, bool) or not isinstance(time_length, int):
            raise TypeError("time_length must be an integer number of seconds.")
        if time_length <= 0 or time_length % 60 != 0:
            raise ValueError("time_length must be a positive multiple of 60 seconds.")
        if vitaldb_data.segment_sec != 60:
            raise ValueError(
                "VitalDBDataset time grouping requires 60-second clean segments; "
                f"got segment_sec={vitaldb_data.segment_sec}."
            )

        self.vitaldb_data = vitaldb_data
        self.id_list = list(vitaldb_data.ids if id_list is None else id_list)
        if not self.id_list:
            raise ValueError("id_list must contain at least one clean segment ID.")
        self.transforms = transforms
        self.time_length = time_length
        self.segments_per_sample = time_length // vitaldb_data.segment_sec
        self.sample_ids = self._build_sample_ids(self.id_list)
        if not self.sample_ids:
            raise ValueError(
                "id_list contains no sufficiently long consecutive segment run for "
                f"time_length={time_length}."
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> np.ndarray:
        x = self._read_sample(index)
        if self.transforms is not None:
            x = self.transforms(x)
        return np.asarray(x, dtype=np.float32)

    def fit_transforms(self) -> ArrayTransform:
        """Fit this dataset's stateful transforms from its raw training samples."""
        if self.transforms is None:
            raise RuntimeError("Cannot fit transforms because this dataset has none.")
        fit = getattr(self.transforms, "fit", None)
        if not callable(fit):
            raise TypeError("The configured transform does not provide fit(data).")
        fit(self._read_sample(index) for index in range(len(self)))
        return self.transforms

    def _read_sample(self, index: int) -> np.ndarray:
        """Read one untransformed datapoint for transform fitting and access."""
        consecutive_ids = self.sample_ids[int(index)]
        segments = [
            read_data_by_id(self.vitaldb_data, sample_id)
            for sample_id in consecutive_ids
        ]
        return np.concatenate(segments, axis=1).astype(np.float32, copy=False)

    def _build_sample_ids(
        self,
        id_list: Sequence[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], ...]]:
        """Build overlapping, same-case windows of consecutive IDs."""
        ids_by_case: dict[int, set[int]] = {}
        for case_id, segment_id in id_list:
            ids_by_case.setdefault(int(case_id), set()).add(int(segment_id))

        sample_ids: list[tuple[tuple[int, int], ...]] = []
        window_size = self.segments_per_sample
        for case_id in sorted(ids_by_case):
            sorted_segment_ids = sorted(ids_by_case[case_id])
            run: list[int] = []
            for segment_id in sorted_segment_ids:
                if run and segment_id != run[-1] + 1:
                    sample_ids.extend(self._windows_from_run(case_id, run, window_size))
                    run = []
                run.append(segment_id)
            sample_ids.extend(self._windows_from_run(case_id, run, window_size))
        return sample_ids

    @staticmethod
    def _windows_from_run(
        case_id: int,
        run: Sequence[int],
        window_size: int,
    ) -> list[tuple[tuple[int, int], ...]]:
        """Create complete sliding windows with a one-segment stride."""
        return [
            tuple((case_id, segment_id) for segment_id in run[start:end])
            for start in range(0, len(run) - window_size + 1)
            for end in [start + window_size]
        ]

__all__ = [
    "BP_ROLES",
    "ComposeVitalDB",
    "DEFAULT_RULES",
    "DEFAULT_VITALSIGN",
    "DifferenceVitalDB",
    "NormalizeVitalDB",
    "RandomNoiseVitalDB",
    "ResampleVitalDB",
    "SmoothVitalDB",
    "VitalDBData",
    "VitalDBDataset",
]


def main() -> None:
    """Test incremental per-signal clean data through Dataset and DataLoader."""
    test_dir = Path(f"{ROOT_DIR}/processed/test")
    data = VitalDBData(
        clean_dir=test_dir,
        vital_signs=["HR", "RR"],
        interval_sec=2,
        segment_sec=60,
    )
    case_dir = data.clean_dir / "case_0001"
    (case_dir / "HR").mkdir(parents=True, exist_ok=True)
    (case_dir / "RR").mkdir(parents=True, exist_ok=True)
    rows = []
    for segment_id in range(10):
        x = np.arange(60, dtype=np.float32).reshape(2, 30) + segment_id * 60
        np.save(case_dir / "HR" / f"segment_{segment_id:06d}.npy", x[0])
        np.save(case_dir / "RR" / f"segment_{segment_id:06d}.npy", x[1])
        rows.append(
            {
                "case_id": 1,
                "segment_id": segment_id,
                "start_sec": segment_id * 60,
                "end_sec": (segment_id + 1) * 60,
            }
        )
    pd.DataFrame(rows).to_csv(case_dir / "segments.csv", index=False)
    data._write_or_validate_metadata()

    data = VitalDBData(
        clean_dir=test_dir,
        vital_signs=["HR", "RR"],
        interval_sec=2,
        segment_sec=60,
    )
    normalize = NormalizeVitalDB(
        vital_signs=["HR", "RR"],
        method="min_max",
    )
    transforms = ComposeVitalDB(
        [
            ResampleVitalDB(
                input_frequency_hz=0.5,
                target_frequency_hz=1.0,
            ),
            normalize,
            RandomNoiseVitalDB(std=0.01, seed=42),
        ]
    )
    dataset = VitalDBDataset(
        data,
        id_list=[(1, segment_id) for segment_id in range(10)],
        transforms=transforms,
        time_length=600,
    )
    dataset.fit_transforms()
    sample = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=1)))

    assert isinstance(sample, np.ndarray)
    assert dataset.time_length == 600
    assert dataset.segments_per_sample == 10
    assert dataset.sample_ids == [tuple((1, index) for index in range(10))]
    assert normalize.statistics == {"HR": (0.0, 569.0), "RR": (30.0, 599.0)}
    assert tuple(batch.shape) == (1, 2, 600)

    print("VitalDBDataset main test passed")
    print(f"test_dir: {test_dir}")
    print(f"sample: {sample.shape} {sample.dtype}")
    print(f"batch:  {tuple(batch.shape)} {batch.dtype}")


if __name__ == "__main__":
    main()
