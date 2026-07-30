# VitalDB-to-VitalSense Models

## Purpose

Define the VitalDB-to-VitalSense radar-to-token student mapper models used by
knowledge transfer.

## Inputs

- Radar patch tensor shaped `[B, N, 1, patch_len]` for student mappers.

## Outputs

- Student mapper outputs shaped `[B, N, codes_per_token, code_dim]`.

## Public API

- `SignalToTokenMapper`
- `SignalToTokenViTMapper`

## Dependencies

- `codec.vit` for the ViT student encoder reused from the teacher shape.
- `blocks.projection` for single-axis projection builders.

## Responsibilities And Boundaries

This module owns:

- Signal-to-token student mapper classes.

It does not own:

- Radar patch dataset construction or radar downsampling, which belong in
  `data.vitalsense.transfer_dataset`.
- Loss functions, which belong in `train.loss`.
- Epoch/training loops, which belong in `train`.
- Prediction dataframe collection, metrics, or plotting, which belong in
  `eval.vitalsense`.
- Frozen VitalDB teacher loading, codebook matching, and decode helpers, which
  belong in `model.vitaldb`.

## Used By

- `code/notebooks/prediction/vitalsense_knowledge_transfer.ipynb`
- `eval.vitalsense.transfer`
- VitalSense radar-to-token experiments.

## Architecture

Preferred student design:

```text
radar patches [B,N,1,patch_len]
  -> reshape to teacher-style sequence [B,1,N*patch_len]
  -> ViTEncoder copied from teacher token shape
  -> z [B,N,embedding_dim]
  -> reshape [B,N,codes_per_token,code_dim]
  -> fixed teacher codebook match / frozen decoder
```

Older CNN mapper:

```text
radar patch [B*N,1,patch_len]
  -> local Conv1d patch feature stack
  -> AdaptiveAvgPool1d(feature_pool_length)
  -> Linear feature_pool_length -> code_dim via `blocks.projection`
  -> z [B,N,codes_per_token,code_dim]
```

## Tensor Contract

- `VitalSenseCodeDataset` in `data.vitalsense.transfer_dataset` returns:
  - `x`: `[N, 1, patch_len]`
  - `y`: `[N * codes_per_token]`
  - `y_token`: `[N, codes_per_token]`
- `SignalToTokenViTMapper.forward(x)` expects `[B, N, 1, patch_len]`.
- Mapper output is `[B, N, codes_per_token, code_dim]`.

CNN mapper sample with `B = 2`, `N = 10`, `patch_len = 120`,
`codes_per_token = 16`, `code_dim = 8`, and `hidden_dim = 64`:

```text
x:                         [2, 10, 1, 120]
reshape patches:            [20, 1, 120]
CNN stem/downsample:         [20, 64, 30]
CNN code-slot projection:    [20, 16, 30]
adaptive pool to 16:         [20, 16, 16]
temporal_proj 16 -> 8:       [20, 16, 8]
reshape to sequence:         [2, 10, 16, 8]
```

ViT mapper sample with the same teacher token layout:

```text
x:                         [2, 10, 1, 120]
reshape to sequence:         [2, 1, 1200]
ViT patch tokenizer:         [2, 10, 128]
reshape grouped codes:       [2, 10, 16, 8]
```

## Config Contract

- `patch_len` must match the radar patching policy in the data module.
- `token_length` must match the number of radar patches.
- `embedding_dim == codes_per_token * code_dim`.
- Frozen teacher checkpoint config defines the target code shape through
  `model.vitaldb`.
- No fallback checkpoint or codebook behavior is provided.

## Step-by-step Tensor Flow

1. Receive radar patches from the data module.
2. Map radar patches to grouped code embeddings.
3. Train the mapper with losses from `train.loss`.
4. Use `model.vitaldb` and `eval.vitalsense` to decode predictions and compute
   VitalSense metrics.

## Debug Checklist

- Confirm mapper output shape matches teacher target shape.
- Plot code frequency coverage by train/val/test split.
- Compare train and test predicted decode cases.

## Sample Case

```python
mapper = SignalToTokenViTMapper(
    code_dim=8,
    codes_per_token=16,
    patch_len=120,
    token_length=10,
    embedding_dim=128,
)
z_pred = mapper(batch["x"])  # [B, 10, 16, 8]
```

## Failure Modes

- Student token shape does not match teacher target shape.
- Radar patch count does not match teacher token count.
- Predicted code dimension differs from effective teacher codebook dimension.
- Checkpoint normalization stats are missing or incompatible.
