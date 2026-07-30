from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


def downsample_radar_signal(
    signal: np.ndarray,
    source_frequency_hz: float = 333.3333333333,
    target_frequency_hz: float = 10.0,
    max_denominator: int = 1000,
) -> np.ndarray:
    """Anti-aliased downsampling for VitalSense radar input."""

    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if target_frequency_hz <= 0:
        raise ValueError("target_frequency_hz must be positive.")
    if source_frequency_hz <= 0:
        raise ValueError("source_frequency_hz must be positive.")
    if target_frequency_hz >= source_frequency_hz:
        return x

    ratio = Fraction(target_frequency_hz / source_frequency_hz).limit_denominator(
        max_denominator
    )
    y = resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    expected_length = int(round(x.size * target_frequency_hz / source_frequency_hz))
    if expected_length <= 0:
        raise ValueError("Downsampled signal would be empty.")
    if y.size > expected_length:
        y = y[:expected_length]
    elif y.size < expected_length:
        y = np.pad(y, (0, expected_length - y.size), mode="edge")
    return y.astype(np.float32)


class VitalSenseCodeDataset(Dataset):
    """Patchified radar input with frozen pretrained teacher discrete code targets."""

    def __init__(
        self,
        frame: pd.DataFrame,
        patch_len: int,
        codes_per_token: int,
        source_frequency_hz: float = 333.3333333333,
        target_frequency_hz: float | None = 10.0,
        augment: bool = False,
        noise_std: float = 0.0,
        amplitude_scale_std: float = 0.0,
        offset_std: float = 0.0,
        mask_probability: float = 0.0,
        mask_fraction: float = 0.0,
        mask_value: float = 0.0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        if patch_len <= 0:
            raise ValueError("patch_len must be positive.")
        if codes_per_token <= 0:
            raise ValueError("codes_per_token must be positive.")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        if amplitude_scale_std < 0:
            raise ValueError("amplitude_scale_std must be non-negative.")
        if offset_std < 0:
            raise ValueError("offset_std must be non-negative.")
        if not 0 <= mask_probability <= 1:
            raise ValueError("mask_probability must be in [0, 1].")
        if not 0 <= mask_fraction <= 1:
            raise ValueError("mask_fraction must be in [0, 1].")
        self.patch_len = int(patch_len)
        self.codes_per_token = int(codes_per_token)
        self.source_frequency_hz = source_frequency_hz
        self.target_frequency_hz = target_frequency_hz
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.amplitude_scale_std = float(amplitude_scale_std)
        self.offset_std = float(offset_std)
        self.mask_probability = float(mask_probability)
        self.mask_fraction = float(mask_fraction)
        self.mask_value = float(mask_value)

    def __len__(self) -> int:
        return len(self.frame)

    def _augment_radar_patches(self, x_patch: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x_patch
        if (
            self.noise_std <= 0
            and self.amplitude_scale_std <= 0
            and self.offset_std <= 0
            and (self.mask_probability <= 0 or self.mask_fraction <= 0)
        ):
            return x_patch

        x_aug = x_patch.copy()
        if self.amplitude_scale_std > 0:
            scale = np.random.normal(1.0, self.amplitude_scale_std)
            x_aug *= np.float32(scale)
        if self.offset_std > 0:
            offset = np.random.normal(0.0, self.offset_std)
            x_aug += np.float32(offset)
        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=x_aug.shape)
            x_aug += noise.astype(np.float32)
        if (
            self.mask_probability > 0
            and self.mask_fraction > 0
            and np.random.random() < self.mask_probability
        ):
            flat = x_aug.reshape(-1)
            mask_len = max(1, int(round(flat.size * self.mask_fraction)))
            start_max = max(1, flat.size - mask_len + 1)
            start = int(np.random.randint(0, start_max))
            flat[start : start + mask_len] = np.float32(self.mask_value)
        return x_aug.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        x = np.asarray(row["radar_signal_norm"], dtype=np.float32)
        if self.target_frequency_hz is not None:
            x = downsample_radar_signal(
                x,
                source_frequency_hz=self.source_frequency_hz,
                target_frequency_hz=self.target_frequency_hz,
            )
        if x.size % self.patch_len != 0:
            raise ValueError(
                "Downsampled radar length must be divisible by patch_len "
                f"({x.size=} {self.patch_len=})."
            )
        n_patches = x.size // self.patch_len
        x_patch = x.reshape(n_patches, 1, self.patch_len)
        x_patch = self._augment_radar_patches(x_patch)
        y_raw = np.asarray(row["teacher_code_targets"], dtype=np.int64)
        if y_raw.ndim == 1:
            if y_raw.size % self.codes_per_token != 0:
                raise ValueError(
                    "Flattened teacher_code_targets length must be divisible by codes_per_token "
                    f"({y_raw.size=} {self.codes_per_token=})."
                )
            y_token = y_raw.reshape(-1, self.codes_per_token)
        elif y_raw.ndim == 2:
            if y_raw.shape[1] != self.codes_per_token:
                raise ValueError(
                    "teacher_code_targets second dimension must equal codes_per_token "
                    f"({y_raw.shape=} {self.codes_per_token=})."
                )
            y_token = y_raw
        else:
            raise ValueError(
                "teacher_code_targets must be [N*codes_per_token] or [N, codes_per_token], "
                f"got shape {y_raw.shape}."
            )
        if y_token.shape[0] != n_patches:
            raise ValueError(
                "Signal patch count must match teacher token count "
                f"({n_patches=} teacher_tokens={y_token.shape[0]})."
            )
        y = y_token.reshape(-1)
        return {
            "x": torch.from_numpy(x_patch),
            "y": torch.from_numpy(y),
            "y_token": torch.from_numpy(y_token),
            "subject_id": row["subject_id"],
            "scenario": row["scenario"],
        }


__all__ = [
    "VitalSenseCodeDataset",
    "downsample_radar_signal",
]
