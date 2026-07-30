# VitalSense Data Package Init

## Purpose

Expose the stable VitalSense data namespace for alignment and dataset exports.

## Inputs

No runtime inputs for package initialization. Submodules accept VitalSense rows,
radar arrays, target columns, and teacher transfer targets.

## Outputs

Package-level exports for alignment helpers and dataset classes.

## Public API

- `VitalSenseAlignmentConfig`
- `align_vitalsense_record_to_vitaldb`
- `align_vitalsense_dataset_to_vitaldb`
- `VitalSenseHRRRDataset`
- `VitalSenseBPDataset`
- `VitalSenseCodeDataset`
- `downsample_radar_signal`

## Dependencies

- `data.vitalsense.alignment`
- `data.vitalsense.dataset`
- `data.vitalsense.transfer_dataset`

## Responsibilities And Boundaries

This package init owns the stable VitalSense data import surface. It does not
own baseline models, radar-to-token mappers, training losses, checkpoint
loading, metrics, or plots.

## Used By

VitalSense baseline and radar-to-token transfer notebooks.

## Sample Case

```python
from data.vitalsense import VitalSenseCodeDataset
```

## Failure Modes

Import can fail if required dependencies for the exported data modules are
missing from the active environment.
