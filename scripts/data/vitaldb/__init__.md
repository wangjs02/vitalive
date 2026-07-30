# VitalDB Data Package

## Purpose

Expose the VitalDB download, cleaning, transform, dataset, and visualization API.

## Inputs

The package initializer has no runtime inputs. Exported classes accept VitalDB
paths, vital-sign selections, clean-segment rules, and transform settings.
Download helpers accept a production data root and optional case/track
selection.

## Outputs

A concise package-level namespace for the new VitalDB data contract.

## Public API

- `VitalDBDataset`
- `VitalDBData`
- `get_case_ids`
- `get_vital_file_path_by_id`
- `read_data_by_id`
- `download_metadata`
- `download_case_tracks`
- `download_track`
- `find_track_id`
- `load_track_index`
- `ComposeVitalDB`
- `SmoothVitalDB`
- `DifferenceVitalDB`
- `ResampleVitalDB`
- `NormalizeVitalDB`
- `DEFAULT_VITALSIGN`
- `DEFAULT_RULES`
- `BP_ROLES`
- `plot_case`
- `plot_datapoint`
- `plot_segment`
- `plot_segment_counts`
- `segment_summary`

## Dependencies

- `data.vitaldb.dataset`
- `data.vitaldb.download`
- `data.vitaldb.visualize`

## Used By

New VitalDB preparation, pretraining, and model workflows.

## Sample Case

```python
from data.vitaldb import (
    ComposeVitalDB,
    download_metadata,
    NormalizeVitalDB,
    VitalDBDataset,
)
```

## Failure Modes

- Import fails when required numerical or PyTorch dependencies are unavailable.
- Download helpers require network access when files are not already cached.
- Old `VitalDB_Data`, `VitalDB_DataSet`, pretrain augmentation, and SimCLR pair
  APIs are intentionally not exported.
