# Pipeline Package Init

## Purpose

Expose the `pipeline` package for experiment orchestration entrypoints.

## Inputs

No runtime inputs.

## Outputs

No public re-exports yet. Import concrete pipeline modules directly.

## Public API

N/A

## Dependencies

N/A

## Used By

- Scripted experiment entrypoints.
- Notebooks that call reusable pipelines.

## Responsibilities

Own package-level initialization for `code/scripts/pipeline`.

## Boundaries

This file does not define model, data, train, or eval behavior.

## Architecture

N/A

## Tensor Contract

N/A

## Debug Checklist

- Import pipeline modules directly, such as `pipeline.vitaldb`.

## Sample Case

```python
from pipeline.vitaldb import pretrain
```

## Failure Modes

N/A
