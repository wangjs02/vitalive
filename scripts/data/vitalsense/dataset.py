from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transfer_dataset import VitalSenseCodeDataset, downsample_radar_signal


class VitalSenseHRRRDataset(Dataset):
    """VitalSense radar input with HR/RR sequence targets."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        x = np.asarray(row["radar_signal_norm"], dtype=np.float32)[None, :]
        y = np.stack(
            [
                np.asarray(row["HR_norm"], dtype=np.float32),
                np.asarray(row["RR_norm"], dtype=np.float32),
            ],
            axis=0,
        )
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "subject_id": row["subject_id"],
            "scenario": row["scenario"],
        }


class VitalSenseBPDataset(Dataset):
    """VitalSense radar input with BPS/BPM/BPD scalar targets."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        x = np.asarray(row["radar_signal_norm"], dtype=np.float32)[None, :]
        y = np.asarray(
            [row["BPS_norm"], row["BPM_norm"], row["BPD_norm"]],
            dtype=np.float32,
        )
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "subject_id": row["subject_id"],
            "scenario": row["scenario"],
        }


__all__ = [
    "VitalSenseBPDataset",
    "VitalSenseCodeDataset",
    "VitalSenseHRRRDataset",
    "downsample_radar_signal",
]
