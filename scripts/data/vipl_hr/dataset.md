# VIPL-HR Dataset Loader

## Purpose

`dataset.py` provides a simple VIPL-HR data loader, preprocessing transform
classes, and a lazy PyTorch `Dataset` wrapper for video-to-vital prediction
experiments.

## Inputs

- VIPL-HR recording ids: `(person_id, scene_id, source_id)`.
- Local data root, defaulting to `data/VIPL-HR/VIPL-HR/data`.
- VIPL-HR files:
  - `video.avi`
  - `gt_HR.csv`
  - `gt_SpO2.csv`
  - optional `time.txt`
- Optional transform chain built with `ComposeVIPLHR`.

## Outputs

- `read_data_by_id`: dictionary containing `video`, `HR`, and `SpO2`.
- `VIPLHRDataset.__getitem__`: `(X, y)` where:
  - `X` is the transformed video array.
  - `y` is stacked HR/SpO2 shaped `[2, T]`.

## Public API

- `ComposeVIPLHR`
- `ResampleVIPLHR`
- `ClipAndPadVIPLHR`
- `ResizeVIPLHR`
- `NormalizeVIPLHR`
- `VIPLHRDataset`

Low-level helpers such as `read_data_by_id`, `read_video_file`,
`read_file_by_type`, `get_feature_path_by_id`, and `get_all_data` remain inside
this module for implementation and debug use, but they are not exported from
`data.vipl_hr`.

## Dependencies

- External: `numpy`, `torch`, `torch.utils.data`
- Optional runtime dependency: `cv2` for reading and resizing videos.
- Internal: N/A

## Used By

- VIPL-HR prediction notebooks under `code/notebooks/prediction/`.
- Future video-based model training scripts such as DeepPhys experiments.

## Responsibilities

This file owns VIPL-HR path lookup, raw file reading, simple temporal
resampling, preprocessing transforms, and a lazy dataset wrapper.

## Boundaries

This file does not own:

- Model definitions.
- Training loops.
- Metrics and plotting.
- Manifest/dataframe-based dataset indexing.
- Large generated caches.

## Architecture

The expected simple training pipeline is:

```text
VIPLHRDataset(id_list, transforms)
  -> read_data_by_id(...)
  -> raw video dict + HR + SpO2
  -> ComposeVIPLHR([
       ResampleVIPLHR(1.0),
       ClipAndPadVIPLHR(target_length=30),
       ResizeVIPLHR((36, 36)),
       NormalizeVIPLHR(...),
     ])
  -> X video and y vital sequence
```

## Tensor Contract

Before transforms:

```text
video: {"frames": [T_raw, H, W, 3], "time_ms": [T_raw] or None, "fps": float}
HR:    [T_label]
SpO2:  [T_label]
```

After the standard transform chain:

```text
video: [target_length, target_height, target_width, 3]
HR:    [target_length]
SpO2:  [target_length]
y:     [2, target_length]
```

The video array remains channel-last. Models that expect channel-first tensors
must transpose it before forward.

## Transform Contract

`ResampleVIPLHR`:

- Keeps HR and SpO2 as 1 Hz labels.
- Resamples video to the label length.
- Uses `time.txt` when available.
- For source folders without `time.txt`, spreads selected frames evenly across
  the full video using HR/SpO2 length as the 1 Hz anchor.
- Owns its implementation through `resample_by_type`, `resample_video`, and
  `nearest_time_indices` methods.

`ClipAndPadVIPLHR`:

- If the sequence is longer than `target_length`, uses one shared contiguous
  crop start for video, HR, and SpO2.
- With `random_clip=True`, the crop start is random.
- If the sequence is shorter than `target_length`, pads both sides using
  reflection along axis 0.
- Reflection excludes the edge item at the boundary. For sequence
  `[0, 1, ..., 19]` padded to length `40`, the result is:

```text
[10, 9, ..., 1, 0, 1, ..., 19, 18, 17, ..., 9]
```

- When the required padding count is odd, the extra item goes to the back. For
  example, padding 19 items gives 9 front items and 10 back items.
- If a sequence contains only one item, reflection is impossible; the function
  falls back to edge padding for that degenerate case.
- Owns its implementation through the `clip_and_pad_array` method.

`ResizeVIPLHR`:

- Resizes each video frame to `(height, width)` with OpenCV.
- Owns its implementation through the `resize_video_frames` method.

`NormalizeVIPLHR`:

- Optionally converts OpenCV BGR frames to RGB.
- Converts values to `float32`.
- Scales images from `[0, 255]` to `[0, 1]` when needed.
- Applies channel z-score normalization.
- Owns its implementation through the `normalize_video_frames` method.

## Debug Checklist

- Confirm `ResampleVIPLHR` runs before `ClipAndPadVIPLHR`.
- Confirm DataLoader batches only after `ClipAndPadVIPLHR`; otherwise sequence
  lengths can differ.
- Confirm video layout before model input. This loader returns channel-last
  video `[T, H, W, 3]`.
- For DeepPhys-style frame-difference models, avoid edge-repeat padding because
  repeated frames create zero motion. Use the default reflect padding.
- Check missing source folders with `FileNotFoundError` rather than silent
  fallback.

## Sample Case

```python
from data.vipl_hr import (
    ClipAndPadVIPLHR,
    ComposeVIPLHR,
    NormalizeVIPLHR,
    ResampleVIPLHR,
    ResizeVIPLHR,
    VIPLHRDataset,
)

transforms = ComposeVIPLHR([
    ResampleVIPLHR(resample_hz=1.0),
    ClipAndPadVIPLHR(target_length=30),
    ResizeVIPLHR(target_size=(36, 36)),
    NormalizeVIPLHR(),
])

dataset = VIPLHRDataset(
    id_list=[(1, 1, 1), (1, 1, 2)],
    transforms=transforms,
)
X, y = dataset[0]
```

Expected shapes:

```text
X: [30, 36, 36, 3]
y: [2, 30]
```

## Failure Modes

- Missing recording folders or files raise `FileNotFoundError`.
- Empty videos or empty arrays raise `ValueError`.
- BVP resampling is not implemented.
- DataLoader stacking fails if fixed-length transforms are not applied before
  batching.
- Channel-last video must be transposed before channel-first model families.
