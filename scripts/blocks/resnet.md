# ResNet Components

## Purpose

Provide reusable residual blocks and residual-stack builders for 1D and 2D
models.

## Inputs

- 1D feature tensors shaped `[B, C, T]`.
- 2D feature tensors shaped `[B, C, H, W]`.
- Channel counts, optional hidden channel counts, and depth preset settings.

## Outputs

- Residual feature tensors with the same shape as input.
- `nn.Sequential` residual stacks.

## Public API

- `ResidualBlock1D`
- `ResidualBlock2D`
- `build_residual_stack_1d`
- `build_residual_stack_2d`

## Dependencies

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `blocks.presets`

## Responsibilities And Boundaries

This module owns reusable residual blocks. It does not own complete DeepMind
VQ-VAE encoder/decoder frameworks, quantizers, datasets, training loops, or
evaluation helpers.

## Used By

- `codec.deepmind`
- future CNN/ResNet model components

## Sample Case

```python
block = ResidualBlock2D(channels=128, hidden_channels=32)
y = block(x)
```

## Failure Modes

- Input channels must match the configured `channels` value because residual
  addition preserves shape.
- Layer depth below one raises from `blocks.presets.resolve_depth`.
