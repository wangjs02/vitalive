# VitalDB Data Package

## Purpose

Expose the VitalDB cleaning, transform, dataset, and visualization API.

## Inputs

The package initializer has no runtime inputs. Exported classes accept VitalDB
paths, vital-sign selections, clean-segment rules, and transform settings.

## Outputs

A concise package-level namespace for the new VitalDB data contract.

## Public API

- `VitalDBDataset`
- `VitalDBData`
- `get_case_ids`
- `get_vital_file_path_by_id`
- `read_data_by_id`
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
- `data.vitaldb.visualize`

## Used By

New VitalDB preparation, pretraining, and model workflows.

## Sample Case

```python
from data.vitaldb import (
    ComposeVitalDB,
    NormalizeVitalDB,
    VitalDBDataset,
)
```

## Failure Modes

- Import fails when required numerical or PyTorch dependencies are unavailable.
- Old `VitalDB_Data`, `VitalDB_DataSet`, pretrain augmentation, and SimCLR pair
  APIs are intentionally not exported.
