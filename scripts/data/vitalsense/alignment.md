# VitalSense Alignment

## Purpose

Convert one VitalSense record into VitalDB-style channel order and sequence
length so that pretrained VitalDB encoders can be reused for transfer.

## Inputs

- VitalSense record or dataframe with `HR`, `RR`, `BPS`, `BPM`, and `BPD`.
- `VitalSenseAlignmentConfig` defining channel order, source/target frequency,
  duration, and placeholder values.

## Outputs

- `x_vitaldb_aligned`: float32 array with shape `[channels, target_length]`.
- `channel_mask`: boolean array marking observed versus placeholder channels.
- `alignment_metadata`: reproducibility metadata.

## Public API

- `VitalSenseAlignmentConfig`
- `align_vitalsense_record_to_vitaldb`
- `align_vitalsense_dataset_to_vitaldb`
- `plot_vitalsense_vitaldb_alignment`
- `resample_sequence`
- `repeat_or_trim`

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`

## Responsibilities And Boundaries

This module owns VitalSense-to-VitalDB temporal/channel alignment. It does not
own radar-to-token models, transfer losses, teacher checkpoint loading, or
baseline regression datasets.

## Used By

- VitalSense knowledge transfer notebook
- `data.vitalsense.dataset`
- `model.vitaldb`

## Sample Case

```python
aligned = align_vitalsense_record_to_vitaldb(record)
x = aligned["x_vitaldb_aligned"]
```

## Failure Modes

- Empty sequence cannot be repeated.
- Unsupported channel name in config.
- Output shape differs from the config contract.
