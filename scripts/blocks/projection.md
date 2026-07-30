# Projection Components

## Purpose

Provide lightweight projection builders that map one feature representation to
another feature or prediction space.

This module exposes the active Linear/MLP projection builder and keeps a legacy
CNN projection entrypoint only to fail clearly when old code still calls it:

```text
build_linear_projection -> Linear / MLP projection
build_cnn_projection    -> legacy Conv1d sequence projection, requires removed blocks.cnn
```

The internal blocks come from lower-level modules:

- Linear projections use `blocks.mlp.build_mlp`.
- CNN projections used `blocks.cnn.CNNBlock`; `blocks.cnn` has been removed
  from the current workspace.

## Inputs

- `build_linear_projection`: tensors with last dimension `input_dim`; when
  `flatten=True`, pooled tensors such as `[B, input_dim, 1]` are flattened to
  `[B, input_dim]` before projection.
- `build_cnn_projection`: legacy sequence features shaped
  `[B, input_channels, T]`; this path is not available unless `blocks.cnn`
  exists.

## Outputs

- `build_linear_projection`: tensor with last dimension `output_dim`, or
  `[B, output_dim]` when `flatten=True`.
- `build_cnn_projection`: sequence tensor shaped `[B, output_channels, T]`
  when the removed CNN block is restored.

## Public API

- `build_linear_projection`
- `build_cnn_projection` legacy entrypoint

## Dependencies

- `torch.nn`
- `blocks.mlp`
- `blocks.cnn` only when calling the legacy CNN projection path

## Responsibilities And Boundaries

This module owns projection composition only.

It does not own:

- low-level Linear block behavior, which belongs in `blocks.mlp`;
- low-level Conv1d block behavior; current model-specific Conv blocks live in
  their owning codec/model modules;
- Fourier matching for quantizers, which belongs in `blocks.quantizer`;
- encoder-decoder architectures;
- temporal downsampling or upsampling;
- losses or training loops.

## Used By

- `model.vitaldb_vitalsense` for radar-to-token temporal projection.
- `model.vitalsense` for HR/RR CNN sequence projection and BP Linear scalar
  projection.

## Tensor Contract

Single-axis Linear projection:

```text
x:                         [2, 16]
build_linear_projection:    [2, 8]
```

Flattened Linear projection:

```text
x:                         [2, 64, 1]
flatten:                   [2, 64]
MLP projection:             [2, 3]
```

Legacy CNN projection:

```text
features:                  [2, 64, 120]
CNNBlock 64 -> 64:          [2, 64, 120]
CNNBlock 64 -> 2, k1:       [2, 2, 120]
```

## Config Contract

- `build_linear_projection(..., identity_if_same=True)` only returns
  `nn.Identity` for the simple no-flatten, no-hidden-block case.
- `build_linear_projection(..., hidden_dim=...)` implies at least one
  `MLPBlock` before the final Linear layer.
- `build_cnn_projection(..., num_blocks=N)` requires the removed
  `blocks.cnn.CNNBlock` implementation and raises if it is unavailable.

## Sample Case

```python
linear = build_linear_projection(input_dim=16, output_dim=8)
y = linear(x)  # [B, 8]

scalar = build_linear_projection(
    input_dim=64,
    output_dim=3,
    hidden_dim=64,
    num_blocks=1,
    norm="layer",
    norm_position="pre_linear",
    flatten=True,
)
y_scalar = scalar(pooled_features)  # [B, 3]

# Legacy only. Raises unless blocks.cnn is restored.
seq = build_cnn_projection(input_channels=64, output_channels=2)
y_seq = seq(features)  # [B, 2, T]
```

## Failure Modes

- Unsupported MLP activation, normalization, or norm position raises from
  `blocks.mlp`.
- `build_cnn_projection(...)` raises `ModuleNotFoundError` if `blocks.cnn` is
  not present.
- `build_cnn_projection(..., num_blocks < 0)` raises `ValueError` when
  `blocks.cnn` is present and the path can be built.
- Linear projection flattened feature dimension must match `input_dim`.
- CNN projection input channel dimension must match `input_channels`.
