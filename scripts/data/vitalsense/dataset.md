# VitalSense Dataset Wrappers

## Purpose

Define supervised VitalSense baseline dataset wrappers and keep the historical
`data.vitalsense.dataset` import path for the transfer dataset.

## Inputs

- VitalSense pandas rows containing:
  - `radar_signal_norm`
  - `HR_norm`, `RR_norm`
  - `BPS_norm`, `BPM_norm`, `BPD_norm`
  - `subject_id`, `scenario`
- Transfer rows are passed through to `data.vitalsense.transfer_dataset`.

## Outputs

- `VitalSenseHRRRDataset` sample:
  - `x`: `[1, T]`
  - `y`: `[2, target_length]`
  - `subject_id`, `scenario`
- `VitalSenseBPDataset` sample:
  - `x`: `[1, T]`
  - `y`: `[3]`
  - `subject_id`, `scenario`
- Re-exported transfer dataset:
  - `VitalSenseCodeDataset`
  - `downsample_radar_signal`

## Public API

- `VitalSenseHRRRDataset`
- `VitalSenseBPDataset`
- `VitalSenseCodeDataset`
- `downsample_radar_signal`

## Dependencies

- `numpy`
- `pandas`
- `torch`
- `data.vitalsense.transfer_dataset`

## Responsibilities And Boundaries

This module owns supervised baseline dataset wrappers. It also re-exports the
transfer dataset from `data.vitalsense.transfer_dataset` so existing notebooks
can keep importing from `data.vitalsense.dataset`.

It does not own model definitions, losses, train loops, metrics, plotting, or
teacher code extraction.

## Used By

- VitalSense knowledge transfer notebook.
- VitalSense baseline experiments.

## Sample Case

```python
hrrr_dataset = VitalSenseHRRRDataset(train_dataset)
sample = hrrr_dataset[0]
sample["x"].shape  # [1, T]
sample["y"].shape  # [2, target_length]
```

## Failure Modes

- Missing normalized target columns raise dataframe key errors.
- Target arrays must already be normalized before dataset construction.
- Radar signal is assumed to be pre-normalized in `radar_signal_norm`.
