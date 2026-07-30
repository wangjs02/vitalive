# VitalSense Transfer Dataset

## Purpose

Own radar downsampling, radar patching, and train-only mmWave augmentation for
radar-to-token transfer.

## Inputs

- VitalSense transfer dataframe rows containing:
  - `radar_signal_norm`
  - `teacher_code_targets`
  - `subject_id`
  - `scenario`
- `patch_len` and `codes_per_token` derived from the frozen teacher config.
- Optional radar augmentation parameters for training split only.

## Outputs

- `downsample_radar_signal(signal)` returns a 1D float32 radar sequence at the
  requested target frequency.
- `VitalSenseCodeDataset[index]` returns:
  - `x`: `[N, 1, patch_len]`
  - `y`: `[N * codes_per_token]`
  - `y_token`: `[N, codes_per_token]`
  - `subject_id`, `scenario`

## Public API

- `downsample_radar_signal`
- `VitalSenseCodeDataset`

## Dependencies

- `numpy`
- `pandas`
- `torch`
- `scipy.signal.resample_poly`

## Responsibilities And Boundaries

This module owns data-side radar patch construction and radar-only augmentation
for the radar-to-token transfer task.

It does not own:

- Student mapper model classes, which belong in `model.vitaldb_vitalsense`.
- Teacher code extraction, which belongs in `model.vitaldb`.
- Losses, which belong in `train.loss`.
- Training loops, which belong in `train`.
- Metrics and plots, which belong in `eval.vitalsense`.

## Used By

- VitalSense knowledge transfer notebook.
- `data.vitalsense.dataset` and `data.vitalsense.__init__` re-exports.

## Sample Case

```python
dataset = VitalSenseCodeDataset(
    train_transfer_dataset,
    patch_len=120,
    codes_per_token=16,
    augment=True,
    noise_std=0.02,
    mask_probability=0.2,
    mask_fraction=0.05,
)
sample = dataset[0]
sample["x"].shape       # [N, 1, 120]
sample["y_token"].shape # [N, 16]
```

## Failure Modes

- Downsampled radar length must be divisible by `patch_len`.
- `teacher_code_targets` must be `[N * codes_per_token]` or
  `[N, codes_per_token]`.
- Radar patch count must match teacher token count.
- Augmentation should not be enabled for validation or test datasets.
