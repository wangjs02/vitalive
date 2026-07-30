# Blocks Package Init

## Purpose

Expose reusable low-level neural-network components from the `blocks`
namespace.

## Inputs

No runtime inputs.

## Outputs

Package-level exports for common Transformer, MLP, linear projection, residual,
and quantizer components.

## Public API

- Transformer: `AttentionBlock`, `CrossAttentionBlock`, `PatchTokenizer1D`,
  `build_patch_tokenizer_1d`, `build_transformer_blocks`,
  `interpolate_token_positions`
- MLP: `MLPBlock`, `build_mlp`
- ResNet: `ResidualBlock1D`, `ResidualBlock2D`, residual stack builders
- Quantizer/projection: `SequenceEMAQuantize`, `VQVAEQuantize`,
  `codebook_perplexity`, `build_linear_projection`
- Presets: `DEPTH_PRESETS`, `ModelSize`, `resolve_depth`

## Dependencies

- `blocks.transformer`
- `blocks.mlp`
- `blocks.resnet`
- `blocks.projection`
- `blocks.quantizer`
- `blocks.presets`

## Responsibilities And Boundaries

This package init owns the stable public namespace for reusable low-level
components. It does not define complete encoder-decoder frameworks,
dataset-specific assembled models, training loops, dataset loaders, or
evaluation helpers.

## Used By

- Codec frameworks.
- Assembled models.
- Pretrain and transfer code that needs direct access to reusable components.

## Sample Case

```python
from blocks import build_linear_projection, SequenceEMAQuantize
```

## Failure Modes

Import fails if PyTorch or SciPy are not available in the active environment.
CNN blocks are no longer exported from this package; model code that still
imports `blocks.cnn` must be migrated to local model-specific Conv blocks.
