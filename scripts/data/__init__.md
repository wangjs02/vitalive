# Data Package Init

## Purpose

Define `data` as the canonical namespace for dataset loading, preprocessing,
alignment, and dataset-specific PyTorch sample construction.

## Inputs

No runtime inputs.

## Outputs

Python package initialization for `data`.

## Public API

Subpackages provide the public API:

- `data.vitaldb`
- `data.vitalsense`
- `data.cohface`
- `data.guardian`
- `data.vipl_hr`

## Dependencies

None at package import time.

## Responsibilities And Boundaries

This package init owns only the data namespace. Dataset-specific loading and
sample logic belong in subpackages. Models stay in `codec` or `model`, losses
and training loops stay in `train`, and metrics/plots stay in `eval`.

## Used By

Notebooks and training scripts that need stable dataset imports.

## Sample Case

```python
import data.vitaldb
```

## Failure Modes

None expected at package import time.
