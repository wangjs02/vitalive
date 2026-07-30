# MLP Components

## Purpose

Provide reusable feed-forward blocks and a single MLP builder for codec
frameworks, assembled models, and diagnostics.

The design mirrors the CNN block boundary:

```text
MLPBlock       -> one Linear hidden block
build_mlp(...) -> num_blocks MLPBlock modules + final Linear projection
```

## Inputs

- Feature tensors shaped `[B, D]` or `[B, N, D]`.
- `MLPBlock` constructor:
  - `input_dim`: input feature dimension.
  - `output_dim`: output feature dimension.
  - `activation`: `gelu`, `relu`, or `silu`.
  - `dropout`: dropout after activation and optional normalization.
  - `norm`: optional `batch` or `layer` normalization.
  - `norm_position`: `pre_linear`, `pre_activation`, or `post_activation`.
- `build_mlp` constructor:
  - `input_dim`, `output_dim`, optional `hidden_dim`.
  - `num_blocks`: number of repeated `MLPBlock` hidden blocks before the final
    Linear layer.

## Outputs

- `MLPBlock`: tensor shaped `[... , output_dim]`.
- `build_mlp`: plain `nn.Sequential` module shaped `[... , output_dim]`.

## Public API

- `MLPBlock`
- `build_mlp`

## Dependencies

- `torch`
- `torch.nn`

## Responsibilities And Boundaries

This module owns reusable feed-forward layer construction.

It does not own:

- single-axis projection semantics, which belong in `blocks.projection`;
- CNN projection semantics, which belong in `blocks.projection`;
- domain-specific controller semantics;
- DINO teacher/student logic;
- SimCLR loss;
- dataset loading or training loops.

## Used By

- `blocks.projection.build_linear_projection`.
- Future codec or model components that need generic feed-forward stacks.

## Architecture

`MLPBlock`:

```text
optional norm before Linear
Linear(input_dim -> output_dim)
optional norm before activation
activation
optional norm after activation
optional dropout
```

`build_mlp`:

```text
repeat num_blocks times:
  MLPBlock(current_dim -> hidden_dim)
Linear(hidden_dim or input_dim -> output_dim)
```

When `num_blocks = 0`, `build_mlp` is a single Linear layer.

## Tensor Contract

```text
x:                         [2, 128]
MLPBlock 128 -> 256:        [2, 256]
final Linear 256 -> 32:     [2, 32]
```

Sequence-shaped tensors keep their leading dimensions:

```text
x:                         [2, 10, 128]
build_mlp 128 -> 32:        [2, 10, 32]
```

## Sample Case

```python
block = MLPBlock(128, 256, norm="layer", norm_position="pre_linear")
z_hidden = block(features)

head = build_mlp(128, 32, hidden_dim=256, num_blocks=2, norm="layer")
z = head(features)
```

## Failure Modes

- Unsupported activation name raises `ValueError`.
- Unsupported normalization name raises `ValueError`.
- Unsupported `norm_position` raises `ValueError`.
- `num_blocks < 0` raises `ValueError`.
