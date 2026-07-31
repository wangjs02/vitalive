# VitalDB Data And Dataset

## Purpose

Separate one-time VitalDB cleaning from PyTorch training-time access.

## Inputs

- Raw `.vital` files and track metadata.
- Selected `vital_signs` from HR, SpO2, RR, BT, SBP, DBP, and MBP.
- Cleaning rules and segment settings.
- Clean sample IDs shaped as `(case_id, segment_id)`.
- Dataset `time_length` in seconds; it must be a positive multiple of 60.
- Training arrays used to fit z-score or min-max normalization statistics.

## Outputs

- Disk-backed clean segment arrays under
  `/home/junshi/data/VitalDB/processed/pretrain-7vitalsign-v1`.
- Per-case `segments.csv` files defining stable time-based segment IDs.
- Per-vital-sign `.npy` files that can be added incrementally.
- `VitalDBDataset` samples as NumPy arrays shaped `[C, T]`, optionally formed
  from multiple consecutive 60-second segments.
- Train-fitted per-role normalization statistics reusable for validation, test,
  checkpoint metadata, and inverse conversion utilities.

## Public API

- `get_case_ids`
- `get_vital_file_path_by_id`
- `read_data_by_id`
- `VitalDBData`
- `VitalDBDataset`
- `ComposeVitalDB`
- `SmoothVitalDB`
- `DifferenceVitalDB`
- `ResampleVitalDB`
- `NormalizeVitalDB`
- `RandomNoiseVitalDB`

## Dependencies

- `numpy`
- `pandas`
- `torch`
- `vitaldb` for the cleaning stage.

## Used By

- `pipeline.vitaldb`
- VitalDB visualization helpers.

## Architecture

```text
VitalDBData(...).clean(case_ids)
  -> read raw .vital one case at a time
  -> skip vital-sign folders that already exist
  -> clean only missing vital signs
  -> save one single-channel .npy per clean segment
  -> save case-level segments.csv

VitalDBDataset(vitaldb_data, id_list)
  -> receive candidate (case_id, segment_id) pairs
  -> group IDs by case and consecutive segment ID
  -> build complete sliding time_length windows with a one-segment stride
  -> load every 60-second segment in the requested window
  -> concatenate segments along time into [C, T]
  -> apply optional transforms to the concatenated datapoint
  -> return NumPy [C, T]
  -> DataLoader collates to Tensor [B, C, T]

train_dataset.fit_transforms()
  -> read raw samples from the training dataset
  -> apply preceding transforms before fitting the next transform
  -> fit NormalizeVitalDB statistics from training values only
  -> return the fitted composition for validation and test data
```

Raw cases and clean arrays are never all retained in RAM.

Current code defaults are split between helper-level absolute paths and
`VitalDBData` constructor-relative paths:

```text
ROOT_DIR = /home/junshi/

Helper defaults:
- metadata_dir = /home/junshi/data/VitalDB/metadata
- data_dir = /home/junshi/data/VitalDB/raw

VitalDBData(...) defaults:
- data_dir = data/VitalDB/raw
- metadata_dir = data/VitalDB/metadata
- clean_dir = data/vitaldb/processed/pretrain-7vitalsign-v1
```

Server path layout used by the helper functions:

```text
/home/junshi/data/VitalDB/
├── metadata/
│   ├── VitalDB_cases_uncompressed.csv
│   ├── VitalDB_trks_uncompressed.csv
│   └── VitalDB_labs_uncompressed.csv
├── raw/
│   ├── 0001.vital
│   ├── 0002.vital
│   └── ...
└── processed/
    └── pretrain-7vitalsign-v1/
```

Processed layout:

```text
/home/junshi/data/VitalDB/processed/pretrain-7vitalsign-v1/
├── metadata.json
└── case_0001/
    ├── segments.csv
    ├── HR/segment_000000.npy
    ├── RR/segment_000000.npy
    └── SBP/segment_000000.npy
```

When a different vital-sign subset is requested, existing channel folders are
reused. `VitalDBData.ids` contains only the `(case_id, segment_id)` intersection
available for every requested vital sign.

## Tensor Contract

For seven vital signs, a 60-second segment, and a two-second interval:

```text
sample ID: (case_id, segment_id)
x: [7, 30], float32
```

For `time_length=600`, ten consecutive 60-second segments are joined:

```text
sample IDs: ((case_id, segment_id), ..., (case_id, segment_id + 9))
x: [7, 300], float32 at the stored 0.5 Hz frequency
```

Datapoints never cross case boundaries or missing segment IDs. Each consecutive
run is converted to sliding windows with a one-segment (60-second) stride, so
adjacent datapoints overlap. A run shorter than the requested `time_length`
produces no datapoint.

For example, 12 consecutive segments with `time_length=600` produce three
datapoints covering segment IDs `0–9`, `1–10`, and `2–11`.

## Sample Case

```python
from data.vitaldb import VitalDBData, VitalDBDataset

data = VitalDBData(vital_signs=["HR", "RR", "SBP"])
data.clean(case_ids=[1, 2, 3])
dataset = VitalDBDataset(data, id_list=data.ids)
sample = dataset[0]
```

By default this uses:

```text
helper data_dir=/home/junshi/data/VitalDB/raw
helper metadata_dir=/home/junshi/data/VitalDB/metadata
constructor clean_dir=data/vitaldb/processed/pretrain-7vitalsign-v1
```

To construct 600-second datapoints without changing the 60-second clean files:

```python
dataset = VitalDBDataset(
    data,
    id_list=data.ids,
    time_length=600,
)

# Ten stored [C, 30] segments become one [C, 300] datapoint at 0.5 Hz.
sample = dataset[0]
```

To resample the default 0.5 Hz clean arrays to 1 Hz at dataset access time:

```python
from data.vitaldb import ComposeVitalDB, ResampleVitalDB, VitalDBDataset

transforms = ComposeVitalDB([ResampleVitalDB()])
dataset = VitalDBDataset(data, id_list=data.ids, transforms=transforms)

# [C, 30] at 0.5 Hz becomes [C, 60] at 1 Hz.
sample = dataset[0]
```

To fit normalization from the training split instead of specifying statistics:

```python
from data.vitaldb import ComposeVitalDB, NormalizeVitalDB, VitalDBDataset

normalize = NormalizeVitalDB(vital_signs=data.vital_signs, method="z_score")
train_dataset = VitalDBDataset(
    data,
    id_list=train_ids,
    transforms=ComposeVitalDB([normalize]),
)
fitted_transforms = train_dataset.fit_transforms()
val_dataset = VitalDBDataset(data, id_list=val_ids, transforms=fitted_transforms)

# Mapping role -> (mean, std). For method="min_max", values are (min, max).
normalization_state = normalize.state_dict()

# Supports NumPy/Tensor [C, T] and [B, C, T].
x_original = fitted_transforms.inverse_transform(x_normalized)
```

`NormalizeVitalDB.fit` streams over `[C, T]` arrays, so fitting does not load
the full training split into memory. Explicit `statistics=` remains supported
when restoring a previously fitted transform from checkpoint metadata.
The transform exposes `fit`, `transform`, `fit_transform`, `inverse_transform`,
and callable (`normalize(x)`) interfaces. Use `state_dict` and
`load_state_dict` to save and restore fitted preprocessing with checkpoints.
`ComposeVitalDB.inverse_transform` walks the composition in reverse and calls
the inverse method of each invertible stage. Stages without a defined inverse,
such as random noise and resampling, are skipped; their effects are not undone.

## Failure Modes

- Raw `.vital` file or `vitaldb` package is missing during cleaning.
- Cleaning produces no accepted segments.
- A requested `(case_id, segment_id)` is unavailable for one selected channel.
- A clean `.npy` file is missing or has an invalid shape.
- `time_length` is not a positive integer multiple of 60.
- Clean storage does not use 60-second base segments.
- No same-case consecutive run is long enough for the requested `time_length`.
- Resampling receives input that is not shaped `[C, T]` or has an empty time axis.
- Normalization is called before `fit`, or fitting data has the wrong channel
  count, an empty channel, or non-finite values.

## Debug Checklist

- Confirm `VitalDBData.clean()` completes before Dataset construction.
- Confirm IDs are `(case_id, segment_id)` pairs.
- Confirm a 600-second sample contains ten consecutive IDs from one case.
- Confirm train and test case IDs are disjoint.
- Confirm normalization is fitted only with the training dataset and that
  validation/test datasets reuse the same transform instance.
- Confirm `normalize.state_dict()` is saved with important checkpoints.
- Confirm one DataLoader batch has shape `[B, C, T]`.
