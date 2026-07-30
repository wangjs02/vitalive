# COHFACE Data Package Init

## Purpose

Expose COHFACE dataset loading and preprocessing helpers.

## Inputs

No runtime inputs for package initialization. Submodules accept COHFACE root
paths, protocol files, HDF5 reference files, and video paths.

## Outputs

Package-level exports for COHFACE manifest, reference target extraction, video
trace extraction, normalization helpers, and PyTorch datasets.

## Public API

- `build_manifest`
- `read_ref`
- `ref_targets`
- `ref_series`
- `rate_series`
- `video_meta`
- `video_trace`
- `resample_trace`
- `resample_trace_time`
- `add_video_features`
- `add_video_series`
- `add_vital_series`
- `fit_feature_stats`
- `add_feature_norm`
- `fit_sequence_stats`
- `add_sequence_norm`
- `fit_target_stats`
- `add_target_norm`
- `DEFAULT_AUGMENT`
- `augment_trace`
- `CohfaceVitalDataset`
- `CohfaceCodeDataset`

## Dependencies

- `data.cohface.dataset`
- Optional runtime packages `h5py` and `cv2` for reference/video extraction.

## Responsibilities And Boundaries

This package init owns the stable COHFACE data import surface. It does not own
model architectures, training loops, or VitalDB teacher decoding.

## Used By

COHFACE EDA, baseline, and knowledge-transfer notebooks.

## Sample Case

```python
from data.cohface import build_manifest, add_vital_series, CohfaceVitalDataset
from data.cohface import DEFAULT_AUGMENT, augment_trace
```

## Failure Modes

Import succeeds without reading data. Runtime extraction functions raise clear
errors when required files or optional dependencies are unavailable.
