# Transformer Components

## Purpose

Provide reusable Transformer attention blocks, Conv1d patch tokenization, and
learned positional embedding interpolation for codec frameworks.

## Inputs

- Token tensors shaped `[B, N, D]`.
- Context tensors shaped `[B, M, D]` for cross-attention.
- Signal tensors shaped `[B, C, T]` for `PatchTokenizer1D`.
- Learned positional embedding tensors shaped `[1, N, D]`.

## Outputs

- Self-attended token tensors shaped `[B, N, D]`.
- Cross-attended query tensors shaped `[B, Q, D]`.
- Patch-token tensors shaped `[B, ceil(T / patch_size), D]`.
- Interpolated positional embeddings shaped `[1, target_length, D]`.

## Public API

- `AttentionBlock`
- `CrossAttentionBlock`
- `build_transformer_blocks`
- `build_patch_tokenizer_1d`
- `interpolate_token_positions`
- `PatchTokenizer1D`

## Dependencies

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `blocks.presets`

## Responsibilities And Boundaries

This module owns generic Transformer blocks and tokenization helpers. It does
not own ViT encoder/decoder framework semantics, VQ grouping, radar transfer,
checkpoint loading, datasets, losses, or evaluation.

## Used By

- `codec.vit`
- `model.vitaldb_vitalsense`

## Tensor Contract

Patch tokenizer sample:

```text
B = 2, C = 5, T = 600, patch_size = 60, embedding_dim = 128

x:                         [2, 5, 600]
pad to patch multiple:      [2, 5, 600]
Conv1d patch tokenizer:     [2, 128, 10]
PatchTokenizer1D output:    [2, 10, 128]
```

Transformer block sample:

```text
tokens:                    [2, 10, 128]
self-attention output:      [2, 10, 128]
MLP residual output:        [2, 10, 128]
```

## Sample Case

```python
blocks = build_transformer_blocks(embedding_dim=128, num_heads=4, num_layers=4)
pos = interpolate_token_positions(pos_embedding, token_length=x.size(1))
```

## Failure Modes

- `patch_size` below one raises `ValueError`.
- Layer depth below one raises from `blocks.presets.resolve_depth`.
- Multi-head attention follows PyTorch constraints: `embedding_dim` must be
  divisible by `num_heads`.
