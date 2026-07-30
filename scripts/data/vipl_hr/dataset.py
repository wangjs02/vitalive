from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def get_feature_path_by_id(person_id: int = 1,
                  scene_id: int = 1,
                  source_id: int = 1,
                  data_dir: str | Path = "data/VIPL-HR/VIPL-HR/data",
                  data_type: str = "video"):
    """Get path for a feature by id from the VIPL-HR dataset."""
    folder = "/Users/wjs/Code/hkuaiot/Vitalive/"
    data_dir = Path(folder) / data_dir

    # Find the subfolder for the person_id
    # The data are divided into group of 5, named as p1-5, p6-10,
    group_id = (person_id - 1) // 5 + 1
    subfolder = f"p{(group_id - 1) * 5 + 1}-{group_id * 5}"
    person_folder = data_dir / subfolder / f"p{person_id}" / f"v{scene_id}" / f"source{source_id}"

    # Check if the folder exists, if not raise an error
    if not person_folder.exists():
        raise FileNotFoundError(f"Folder {person_folder} does not exist.")

    # Each folder contains
    # HR(gt_HR.csv), SpO2(gt_SpO2.csv), BVP(wave.csv), Time For Camera Frame(time.txt), video.avi
    if data_type == "video":
        video_path = person_folder / "video.avi"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        return video_path
    elif data_type == "BVP":
        bvp_path = person_folder / "wave.csv"
        if not bvp_path.exists():
            raise FileNotFoundError(f"BVP file {bvp_path} does not exist.")
        return bvp_path
    elif data_type == "HR":
        hr_path = person_folder / "gt_HR.csv"
        if not hr_path.exists():
            raise FileNotFoundError(f"HR file {hr_path} does not exist.")
        return hr_path
    elif data_type == "SpO2":
        spo2_path = person_folder / "gt_SpO2.csv"
        if not spo2_path.exists():
            raise FileNotFoundError(f"SpO2 file {spo2_path} does not exist.")
        return spo2_path
    elif data_type == "Time":
        time_path = person_folder / "time.txt"
        if not time_path.exists():
            raise FileNotFoundError(f"Time file {time_path} does not exist.")
        return time_path
    
def read_file_by_type(file_path: str | Path):
    """Read a file based on its type (csv, txt, avi)."""
    file_path = Path(file_path)
    # If file is csv or txt, read it as a one-column numeric array.
    if file_path.suffix == ".csv":
        return np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=np.float32).reshape(-1)
    if file_path.suffix == ".txt":
        return np.genfromtxt(file_path, dtype=np.float32).reshape(-1)
    # If file is avi, read it using opencv
    if file_path.suffix == ".avi":
        frames, _fps = read_video_file(file_path)
        return frames


def read_video_file(file_path: str | Path) -> tuple[np.ndarray, float | None]:
    """Read an AVI file and return decoded frames plus metadata FPS."""
    import cv2

    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames), fps


def read_data_by_id(person_id: int = 1,
                    scene_id: int = 1,
                    source_id: int = 1,
                    data_dir: str | Path = "data/VIPL-HR/VIPL-HR/data",
                    data_types: list[str] = ["video", "HR", "SpO2"],
                    transforms: None | callable = None
                ) -> Mapping[str, object]:
    """Read data by id from the VIPL-HR dataset."""
    data = {}
    # First read in HR, SpO2
    for data_type in data_types:
        if data_type in ["HR", "SpO2"]:
            file_path = get_feature_path_by_id(person_id, scene_id, source_id, data_dir, data_type)
            data[data_type] = read_file_by_type(file_path)
    # Then read in video (together with time stamps)
    if "video" in data_types:
        video_path = get_feature_path_by_id(person_id, scene_id, source_id, data_dir, "video")
        video_data, video_fps = read_video_file(video_path)
        try:
            time_path = get_feature_path_by_id(person_id, scene_id, source_id, data_dir, "Time")
            time_data = read_file_by_type(time_path)
        except FileNotFoundError:
            time_data = None
        data["video"] = {"frames": video_data, "time_ms": time_data, "fps": video_fps}

    if transforms is not None:
        data = transforms(data)

    return data

def get_all_data(pid_list: list[int] = list(range(1, 108)),
                 scene_list: list[int] = list(range(1, 10)),
                 source_list: list[int] = list(range(1, 5)),
                 transforms: None | callable = None,
                 resample_hz: float | None = None) -> Mapping[tuple[int, int, int], Mapping[str, object]]:
    """Get all data in the VIPL-HR dataset."""
    if transforms is None and resample_hz is not None:
        transforms = ResampleVIPLHR(resample_hz)
    all_data = {}
    for person_id in pid_list:
        for scene_id in scene_list:
            for source_id in source_list:
                try:
                    data = read_data_by_id(person_id, scene_id, source_id, transforms=transforms)
                except FileNotFoundError:
                    print(f"Data not found for Person {person_id}, Scene {scene_id}, Source {source_id}")
                    continue
                all_data[(person_id, scene_id, source_id)] = data
    return all_data

class ComposeVIPLHR:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
    
class ResampleVIPLHR:
    def __init__(self, resample_hz: float = 1.0):
        self.resample_hz = resample_hz

    def __call__(self, data):
        out = dict(data)
        label_lengths = [
            len(out[data_type])
            for data_type in ["HR", "SpO2"]
            if data_type in out
        ]
        target_length = min(label_lengths) if label_lengths else None
        for data_type in out:
            if data_type in ["HR", "SpO2", "video"]:
                out[data_type] = self.resample_by_type(
                    out[data_type],
                    data_type,
                    target_length=target_length if data_type == "video" else None,
                )
        return out

    def resample_by_type(self, data, data_type: str, target_length: int | None = None):
        """Resample one VIPL-HR value based on its data type."""
        if data_type in ["HR", "SpO2"]:
            return data
        if data_type == "BVP":
            raise NotImplementedError("Resampling for BVP data is not implemented yet.")
        if data_type == "video":
            return self.resample_video(data, target_length=target_length)
        return data

    def resample_video(self, data: Mapping[str, object], target_length: int | None = None) -> np.ndarray:
        """Resample raw video frames to the transform frequency."""
        frames = data["frames"]
        time = data.get("time_ms")
        fps = data.get("fps")
        if len(frames) == 0:
            raise ValueError("Cannot resample an empty video.")

        if target_length is None and time is not None:
            time_ms = np.asarray(time, dtype=np.float32).reshape(-1)
            duration_seconds = max(0.0, (time_ms[-1] - time_ms[0]) / 1000.0)
            target_length = int(np.floor(duration_seconds * self.resample_hz)) + 1
        elif target_length is None and fps is not None:
            duration_seconds = len(frames) / fps
            target_length = int(np.floor(duration_seconds * self.resample_hz)) + 1

        if target_length is None:
            raise ValueError("Cannot infer video target length without time, fps, or HR/SpO2 labels.")
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}.")

        if time is not None:
            time_ms = np.asarray(time, dtype=np.float32).reshape(-1)
            usable_length = min(len(frames), len(time_ms))
            frames = frames[:usable_length]
            time_ms = time_ms[:usable_length]
            target_time_ms = time_ms[0] + np.arange(target_length) * (1000.0 / self.resample_hz)
            frame_indices = self.nearest_time_indices(time_ms, target_time_ms)
        else:
            # Source2-style recordings do not provide per-frame timestamps.
            # HR/SpO2 rows are the 1Hz anchor, so spread one frame per label
            # across the full video instead of trusting often-stale AVI FPS.
            frame_indices = np.linspace(0, len(frames) - 1, target_length)
            frame_indices = np.rint(frame_indices).astype(int)

        return frames[frame_indices]

    @staticmethod
    def nearest_time_indices(time_ms: np.ndarray, target_time_ms: np.ndarray) -> np.ndarray:
        """Map target timestamps to nearest source timestamp indices."""
        right = np.searchsorted(time_ms, target_time_ms, side="left")
        right = np.clip(right, 0, len(time_ms) - 1)
        left = np.clip(right - 1, 0, len(time_ms) - 1)
        use_left = np.abs(time_ms[left] - target_time_ms) <= np.abs(time_ms[right] - target_time_ms)
        return np.where(use_left, left, right)


class ResizeVIPLHR:
    """Resize `data["video"]` frames while preserving the rest of the sample."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def __call__(self, data):
        out = dict(data)
        video = out["video"]
        out["video"] = self.resize_video_frames(video)
        return out

    def resize_video_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize video frames to the configured `(height, width)`."""
        import cv2

        return np.array([cv2.resize(frame, (self.target_size[1], self.target_size[0])) for frame in frames])


class NormalizeVIPLHR:
    """Normalize `data["video"]` frames using RGB channel statistics."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        bgr_to_rgb: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, data):
        out = dict(data)
        video = out["video"]
        out["video"] = self.normalize_video_frames(video)
        return out

    def normalize_video_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize video frames to float32 RGB values with channel z-score stats."""
        x = np.asarray(frames)
        if self.bgr_to_rgb:
            x = x[..., ::-1]
        x = x.astype(np.float32)
        if x.size > 0 and np.nanmax(x) > 1.0:
            x = x / 255.0
        mean_arr = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std_arr = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 1, 3)
        return ((x - mean_arr) / std_arr).astype(np.float32)


class ClipAndPadVIPLHR:
    """Clip or pad video and labels to a fixed temporal length."""

    def __init__(self, target_length: int = 30, random_clip: bool = True, seed: int | None = None):
        if target_length <= 0:
            raise ValueError("target_length must be positive.")
        self.target_length = target_length
        self.random_clip = random_clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, data):
        out = dict(data)
        clip_start = self._get_clip_start(out)
        for key in out:
            out[key] = self.clip_and_pad_array(out[key], start_index=clip_start)
        return out

    def _get_clip_start(self, data: Mapping[str, object]) -> int:
        """Choose one temporal crop start shared by video, HR, and SpO2."""
        lengths = [np.asarray(value).shape[0] for value in data.values()]
        max_start = min(length - self.target_length for length in lengths)
        if max_start <= 0:
            return 0
        if not self.random_clip:
            return 0
        return int(self.rng.integers(0, max_start + 1))

    def clip_and_pad_array(
        self,
        values: np.ndarray,
        start_index: int = 0,
    ) -> np.ndarray:
        """Clip or reflect-pad an array on axis 0."""
        arr = np.asarray(values)
        if arr.shape[0] == 0:
            raise ValueError("Cannot clip/pad an empty array.")
        if arr.shape[0] >= self.target_length:
            max_start = arr.shape[0] - self.target_length
            start_index = int(np.clip(start_index, 0, max_start))
            return arr[start_index:start_index + self.target_length]
        pad_count = self.target_length - arr.shape[0]
        front_pad = pad_count // 2
        back_pad = pad_count - front_pad
        pad_width = [(front_pad, back_pad)] + [(0, 0)] * (arr.ndim - 1)
        if arr.shape[0] == 1:
            return np.pad(arr, pad_width=pad_width, mode="edge")
        return np.pad(arr, pad_width=pad_width, mode="reflect")

class VIPLHRDataset(Dataset):
    # Placeholder for the actual dataset class implementation, Lazy dataset
    # Input: id_list, transforms, data_dir
    def __init__(
        self,
        id_list: list[tuple[int, int, int]],
        transforms: callable | None = None,
        data_dir: str | Path = "data/VIPL-HR/VIPL-HR/data",
    ):
        self.id_list = id_list
        self.transforms = transforms
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx: int):
        person_id, scene_id, source_id = self.id_list[idx]
        data = read_data_by_id(person_id, scene_id, source_id, data_dir=self.data_dir)
        if self.transforms:
            data = self.transforms(data)
        X = data.get("video", None)
        y = None
        if "HR" in data and "SpO2" in data:
            hr = np.asarray(data["HR"], dtype=np.float32).reshape(-1)
            spo2 = np.asarray(data["SpO2"], dtype=np.float32).reshape(-1)
            length = min(hr.size, spo2.size)
            y = np.stack([hr[:length], spo2[:length]], axis=0)
        return X, y

if __name__ == "__main__":

    def _video_shape(video) -> tuple:
        """Return the visible video shape for raw or transformed video values."""
        if isinstance(video, Mapping):
            return video["frames"].shape
        return video.shape


    def _print_data_shapes(prefix: str, data: Mapping[str, object]) -> None:
        """Print VIPL-HR sample shapes for quick manual checks."""
        print(f"{prefix}: Video {_video_shape(data['video'])}, HR {data['HR'].shape}, SpO2 {data['SpO2'].shape}")


    # Example usage 1
    def example_usage_1():
        data = read_data_by_id(
            person_id=1,
            scene_id=1,
            source_id=1,
            data_types=["video", "HR", "SpO2"],
            transforms=ResampleVIPLHR(resample_hz=1.0),
        )
        print(data.keys())
        # Print out Source 1 Dimension: Video shape, HR shape, SpO2 shape
        _print_data_shapes("Source 1", data)
    
    def example_usage_2():
        all_data = get_all_data(pid_list=[1, 2], scene_list=[1, 2], source_list=[1, 2], resample_hz=1.0)
        for key, value in all_data.items():
            _print_data_shapes(f"Person {key[0]}, Scene {key[1]}, Source {key[2]}", value)

    def example_usage_3():
        transforms = ComposeVIPLHR([
            ResampleVIPLHR(resample_hz=1.0),
            ClipAndPadVIPLHR(target_length=30),
            ResizeVIPLHR(target_size=(36, 36)),
            NormalizeVIPLHR(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), bgr_to_rgb=True),
        ])
        train_ids = [(1, 1, 1), (1, 1, 2), (1, 2, 1)]
        train_dataset = VIPLHRDataset(id_list=train_ids, transforms=transforms)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)    

        for X, y in train_loader:
            print("Batch video shape:", X.shape)
            print("Batch target shape:", y.shape)
            break
    
    example_usage_1()
