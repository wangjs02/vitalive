# Model Presets

## Purpose

Define shared model-size presets so Transformer and ResNet modules can use
consistent depth names.

## Inputs

- `size`: one of `small`, `medium`, or `big`.
- Optional explicit `num_layers`.
- Optional `min_layers` validation threshold.

## Outputs

- Integer layer depth.
- `DEPTH_PRESETS` mapping:
  - `small`: 2
  - `medium`: 4
  - `big`: 8

## Public API

- `DEPTH_PRESETS`
- `ModelSize`
- `resolve_depth`

## Dependencies

- Standard-library typing.

## Responsibilities And Boundaries

This module owns shared depth presets only. It does not create layers, own
architecture-specific hyperparameters, load checkpoints, train models, or define
codec frameworks.

## Used By

- `blocks.transformer`
- `blocks.resnet`

## Sample Case

```python
depth = resolve_depth(size="medium")  # 4
depth = resolve_depth(size="small", num_layers=3)  # 3
```

## Failure Modes

- Unknown `size` raises a key error through the typed preset mapping.
- Resolved depth below `min_layers` raises `ValueError`.
