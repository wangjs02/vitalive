# ViT Codec

## Purpose

Define the concrete ViT encoder and decoder used in pretrain v1.2 and reused by
VitalSense radar-to-token transfer.

## Inputs

- Encoder input signal tensor shaped `[B, C, T]`.
- Decoder latent token tensor shaped `[B, token_length, embedding_dim]`.
- Constructor values for input dimension, patch size, token length, embedding
  dimension, transformer depth, heads, and hidden dimension.

## Outputs

- `ViTEncoder`: token embeddings shaped `[B, N, embedding_dim]`, where `N` is
  the number of patch tokens after padding.
- `ViTDecoder`: reconstructed signal shaped `[B, input_dim, time_length]`.

## Public API

- `ViTConfig`
- `ViTEncoder`
- `ViTDecoder`

## Dependencies

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `blocks.transformer`
- `utils.config.KwargsConfig`

## Used By

- `model.vitaldb`
- `model.vitaldb_vitalsense.SignalToTokenViTMapper`
- Pretrain v1.2 ViT experiments.
- `pretrain_vit_pipeline.ipynb`.

## Responsibilities And Boundaries

This module owns the concrete ViT encoder/decoder architecture. It does not own
VQ quantization, grouped-code reshaping, FFT preprocessing, datasets, losses,
training loops, or evaluation.

The VQ-VAE wrapper owns conversion from `[B, N, embedding_dim]` to grouped code
vectors `[B, N, codes_per_token, code_dim]`.

## Architecture

Encoder:

```text
x [B,C,T]
  -> patch_tokenizer:
       right pad to patch multiple
       patch_tokenizer_conv1: Conv1d input_dim -> patch_channels, k=patch_size, stride=patch_size + GELU
       patch_tokenizer_conv2: Conv1d patch_channels -> embedding_dim, k1
       transpose to tokens [B,N,D]
  -> temporal_model:
       additive learned absolute positional embedding
       temporal_model_blocks: Transformer blocks
  -> z [B,N,D]
```

Decoder:

```text
z [B,N,D]
  -> temporal_model:
       additive learned absolute positional embedding
       temporal_model_blocks: Transformer blocks
  -> patch_reconstruction:
       patch_reconstruct_linear1: Linear D -> input_dim * patch_size
       patch concat
       interpolate/crop to time_length
```

The positional embedding is added as `x = x + pos`. This is learned absolute
positional encoding with interpolation for length mismatch. It is not RoPE.

## Tensor Contract

Plain ViT codec sample:

```text
B = 2, C = 5, T = 600
patch_size = 60, token_length = 10, embedding_dim = 128

x:                         [2, 5, 600]
pad to patch multiple:      [2, 5, 600]
patch_tokenizer_conv1:      [2, 128, 10]
patch_tokenizer_conv2:      [2, 128, 10]
transpose to tokens:        [2, 10, 128]
add pos:                    [2, 10, 128]
temporal_model_blocks:      [2, 10, 128]
encoder z:                  [2, 10, 128]

decoder z:                  [2, 10, 128]
add pos:                    [2, 10, 128]
temporal_model_blocks:      [2, 10, 128]
patch_reconstruct_linear1:  [2, 10, 300]
view patches:               [2, 10, 5, 60]
concat patches:             [2, 5, 600]
x_hat:                      [2, 5, 600]
```

Grouped VQ-VAE view, owned by `model.vitaldb`:

```text
encoder tokens:             [2, 10, 128]
grouped tokens:             [2, 10, 16, 8]
quantizer input:            [2, 160, 8]
code indices:               [2, 10, 16]
quantized tokens:           [2, 10, 128]
decoder tokens:             [2, 10, 128]
```

## Config Contract

- `patch_size >= 1`.
- `input_dim` must match the input signal channel count.
- `transformer_heads >= 1`.
- `transformer_layers >= 1`.
- `embedding_dim` must be divisible by `transformer_heads`.
- `time_pos_embedding` and `token_pos_embedding` are initialized at
  construction-time `token_length`; they are interpolated if the actual token
  count differs.
- `ViTConfig` stores encoder/decoder constructor values and can be unpacked
  directly as `ViTEncoder(**ViTConfig())`.
- The VQ-VAE wrapper owns grouped-code reshaping; this module only emits token
  embeddings.

## Step-by-step Tensor Flow

1. Pad the time axis to a multiple of `patch_size`.
2. Tokenize patches with `patch_tokenizer_conv1` and `patch_tokenizer_conv2`.
3. Add learned absolute positional embeddings.
4. Run `temporal_model_blocks`.
5. Decode tokens to per-patch channel/time values with
   `patch_reconstruct_linear1`.
6. Concatenate patches and interpolate/crop to configured time length.

## Debug Checklist

- Verify patch size and token count against the checkpoint.
- Confirm `embedding_dim % transformer_heads == 0`.
- Print encoder output before VQ-VAE grouped reshaping.
- For transfer, verify student output shape equals teacher token shape.
- Do not add FFT unless preprocessing explicitly used FFT.

## Sample Case

```python
encoder = ViTEncoder(
    input_dim=5,
    patch_size=60,
    embedding_dim=128,
    time_length=600,
    token_length=10,
)
z = encoder(torch.randn(2, 5, 600))  # [2, 10, 128]

decoder = ViTDecoder(
    input_dim=5,
    patch_size=60,
    time_length=600,
    token_length=10,
    embedding_dim=128,
)
x_hat = decoder(z)  # [2, 5, 600]
```

## Failure Modes

- Patch/token config mismatch with checkpoint weights.
- Positional embedding interpolation can hide unintended input-length changes;
  print token count when debugging.
- Decoder interpolation can mask an unexpected patch count; verify shapes before
  training.
