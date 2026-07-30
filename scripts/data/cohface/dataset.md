# COHFACE Dataset

## Purpose

Provide reusable COHFACE data utilities for the video knowledge-transfer
notebook. The module builds official-protocol manifests, reads synchronized HDF5
reference signals, derives 1 Hz HR/RR time-series targets, extracts lightweight
RGB video traces, normalizes train-split features and targets, and exposes
PyTorch datasets for direct sequence prediction and VitalDB code transfer.

## Inputs

- COHFACE root directory, usually `data/COHFACE/cohface/`.
- Official protocol files under `protocols/{all,clean,natural}/`.
- Per-recording `data.avi` face videos.
- Per-recording `data.hdf5` reference files containing `pulse`,
  `respiration`, and `time`.

## Outputs

- A `pandas.DataFrame` manifest with subject, session, split, protocol, paths,
  video metadata, scalar spectral summaries, and reference metadata.
- Fixed-length RGB trace arrays with shape `[3, 60]` for the current notebook.
- Pulse-derived HR sequences with shape `[60]`.
- Respiration-derived RR sequences with shape `[60]`.
- `target_mask` arrays with shape `[60]`; only masked positions are evaluated.
- `CohfaceVitalDataset` samples for direct video-to-vital sequence regression.
- `CohfaceCodeDataset` samples for VitalDB teacher-code transfer.
- Optional train-only RGB trace augmentation for video baselines and
  video-to-token transfer.

## Public API

- `build_manifest(root, protocol, inspect_video, inspect_ref)`
- `read_ref(path)`
- `ref_targets(path)`
- `rate_series(values, time, sample_rate_hz, band_hz, length, target_rate_hz)`
- `ref_series(path, length)`
- `video_meta(path)`
- `video_trace(path, every, crop, max_frames)`
- `resample_trace(trace, length)`
- `resample_trace_time(trace, source_rate_hz, length, target_rate_hz)`
- `add_video_features(frame, length, every, max_records)`
- `add_video_series(frame, length, target_rate_hz, every, max_records)`
- `add_vital_series(frame, length)`
- `fit_feature_stats(frame, column)`
- `add_feature_norm(frame, stats, source, target)`
- `fit_sequence_stats(frame, columns, mask_col)`
- `add_sequence_norm(frame, stats)`
- `fit_target_stats(frame, columns)`
- `add_target_norm(frame, stats)`
- `DEFAULT_AUGMENT`
- `augment_trace(trace, config)`
- `CohfaceVitalDataset`
- `CohfaceCodeDataset`

## Dependencies

- `numpy`
- `pandas`
- `scipy.signal`
- `torch`
- Optional packages for data extraction:
  - `h5py` for `data.hdf5`
  - `cv2` from OpenCV for `data.avi`
- System fallback for video extraction:
  - `ffmpeg`
  - `ffprobe`

The functions raise explicit `ImportError` messages if `h5py` is missing. Video
metadata and RGB trace extraction use OpenCV when available and fall back to
`ffmpeg`/`ffprobe` when OpenCV is missing.

## Used By

- `code/notebooks/prediction/cohface_knowledge_transfer.ipynb`
- `code/scripts/model/cohface.py`
- `code/scripts/model/vitaldb_cohface.py`

## Sample Case

```python
from pathlib import Path
from data.cohface import (
    build_manifest,
    add_vital_series,
    add_video_series,
)

root = Path("data/COHFACE/cohface")
manifest = build_manifest(root, protocol="clean")
frame = add_vital_series(manifest.head(8), length=60)
frame = add_video_series(frame, length=60, target_rate_hz=1.0, every=4)
```

`CohfaceVitalDataset` sample:

```text
x:    [3, 60] normalized RGB trace
y:    [2, 60] normalized sequence targets [HR bpm, RR breaths/min]
mask: [60] observed COHFACE seconds
```

`CohfaceCodeDataset` sample for the current ViT teacher:

```text
x:       [6, 3, 10]
y:       [96]
y_token: [6, 16]
vital:   [2, 60]
mask:    [60]
```

Train-time augmentation keeps the same tensor shapes and labels:

```text
input trace:        [3, 60]
augmented trace:    [3, 60]
code target:        unchanged [96]
vital target/mask:  unchanged [2, 60] / [60]
```

The default augmentation uses channel gain/offset, Gaussian noise, slow
illumination trend, short temporal masking, and channel dropout. It deliberately
does not apply temporal shifting or warping because teacher code targets are
patch-aligned to 10-second windows.

## Failure Modes

- Missing `h5py` prevents reference target extraction.
- Missing both `cv2` and `ffmpeg`/`ffprobe` prevents video feature extraction.
- Protocol files may cover fewer recordings than the raw directory.
- COHFACE does not provide blood pressure, SpO2, or body temperature labels, so
  these targets must not be evaluated.
- COHFACE recordings are usually about 60 seconds. The current transfer
  workflow loads the VitalDB teacher with `time_length=60`, so HR/RR and video
  traces use the first 60 seconds. Longer clean-protocol recordings are
  truncated to this teacher length.
- Temporal augmentation that changes sample alignment would corrupt
  video-to-token labels unless teacher targets are shifted or regenerated.
