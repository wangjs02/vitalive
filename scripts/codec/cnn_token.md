# CNNToken Codec

## Purpose

`cnn_token.py` defines a standalone VGG-style Conv1d token encoder and decoder
for physiological sequences. It no longer depends on `blocks.cnn` or accepts
scale/channel plans; the architecture is fixed inside this codec framework.

## Inputs

- Encoder input tensor: `[B, input_dim, time_length]`.
- Decoder input tensor: `[B, token_length, embedding_dim]`.
- Constructor values:
  - `input_dim`
  - `hidden_dim`
  - `token_length`
  - `embedding_dim`
  - `time_length`
  - `temporal_align_dim`

## Outputs

- `CNNTokenEncoder`: token sequence `[B, token_length, embedding_dim]`.
- `CNNTokenDecoder`: reconstructed signal `[B, input_dim, time_length]`.

## Public API

- `CNNTokenConfig`
- `CNNTokenEncoder`
- `CNNTokenDecoder`

## Dependencies

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `utils.config.KwargsConfig`

## Used By

- `model.vitaldb.VitalDBVQVAE` when `enc_dec="cnn_tokens"`.
- Pretrain notebooks and pipelines that import `codec.CNNTokenEncoder` or
  construct a VitalDB CNN-token VQ-VAE.

## Responsibilities

This module owns the complete CNN-token encoder/decoder framework. It maps
physiological signals to token embeddings and reconstructs signals from token
embeddings.

## Boundaries

This module does not own:

- Quantizer logic.
- VitalDB data preprocessing.
- Training loops and checkpointing.
- Reusable `blocks.cnn` abstractions.
- Configurable scale/channel architecture plans.

## Architecture

Encoder:

```text
x [B,C,T]
  -> temporal_downsample_model:
       Conv1d C -> stem_dim, k7, same length
       Conv1d stem_dim -> mid_dim, k5, stride 2
       Conv1d mid_dim -> hidden_dim, k5, stride 2
       Conv1d hidden_dim -> hidden_dim, k3, same length
  -> temporal_alignment:
       temporal_align_pool1: AdaptiveAvgPool1d(temporal_align_dim)
       temporal_align_conv1: Conv1d temporal_align_dim -> embedding_dim, k1, no activation
  -> token_alignment:
       token_align_conv1: Conv1d hidden_dim -> token_length, k1, no activation
  -> z [B,token_length,embedding_dim]
```

Decoder:

```text
z [B,token_length,embedding_dim]
  -> token_alignment:
       token_align_conv1: Conv1d token_length -> hidden_dim, k1 + GELU
  -> temporal_alignment:
       temporal_align_conv1: Conv1d embedding_dim -> temporal_align_dim, k1 + GELU
  -> temporal_upsample:
       interpolate temporal length to ceil(time_length / 4)
       Conv1d hidden_dim -> hidden_dim
       ConvTranspose1d hidden_dim -> mid_dim, stride 2
       ConvTranspose1d mid_dim -> stem_dim, stride 2
       Conv1d stem_dim -> input_dim, no activation
  -> crop to time_length
```

## Tensor Contract

Sample with `B = 2`, `C = 6`, `T = 300`, `hidden_dim = 64`,
`token_length = 16`, `embedding_dim = 8`, `temporal_align_dim = 16`:

```text
encoder input:             [2, 6, 300]
stem conv:                 [2, 16, 300]
downsample 1:              [2, 32, 150]
downsample 2:              [2, 64, 75]
refine conv:               [2, 64, 75]
temporal_align_pool1:      [2, 64, 16]
temporal_align_conv1:      [2, 64, 8]
token_align_conv1:         [2, 16, 8]

decoder input:             [2, 16, 8]
token_align_conv1:         [2, 64, 16]
temporal_align_conv1:      [2, 64, 16]
interpolate to feature:    [2, 64, 75]
temporal upsample conv1:   [2, 64, 75]
upsample 1:                [2, 32, 150]
upsample 2:                [2, 16, 300]
final projection:          [2, 6, 300]
```

## Config Contract

- `input_dim` must match the signal channel count.
- `time_length` must match the encoder input length and decoder output length.
- `hidden_dim` controls model width.
- `token_length` controls the number of latent tokens.
- `embedding_dim` controls each token width and the sequence quantizer input
  dimension.
- `temporal_align_dim` controls the small temporal alignment dimension between
  convolutional features and token embeddings.
- `CNNTokenConfig` stores these constructor values and can be unpacked directly
  as `CNNTokenEncoder(**CNNTokenConfig())`.
- `conv_block(..., activ=False)` is used for output-style projections that
  should preserve signed values without a final GELU.
- `scale_plan`, `channel_plan`, and decoder variants are intentionally not part
  of the public constructor.

## Step-by-step Tensor Flow

1. Downsample the temporal axis twice with Conv1d stride 2.
2. Pool the feature temporal axis to `temporal_align_dim`.
3. Align temporal bins into token embedding dimension.
4. Align hidden feature channels into token positions.
5. Decode tokens by running token alignment, temporal alignment, then temporal
   upsampling.
6. Interpolate to the encoder feature length and upsample twice to reconstruct
   the original signal length.

## Debug Checklist

- Confirm `CNNTokenEncoder(x).shape == [B, token_length, embedding_dim]`.
- Confirm `CNNTokenDecoder(z).shape == [B, input_dim, time_length]`.
- Check `model.vitaldb.VQVAEConfig.embedding_dim` matches the quantizer
  embedding dimension.
- Old checkpoints trained with the previous `blocks.cnn` implementation will
  not have compatible parameter names.

## Sample Case

```python
import torch
from codec.cnn_token import CNNTokenDecoder, CNNTokenEncoder

encoder = CNNTokenEncoder(
    input_dim=6,
    hidden_dim=64,
    token_length=16,
    embedding_dim=8,
    time_length=300,
)
decoder = CNNTokenDecoder(
    input_dim=6,
    hidden_dim=64,
    token_length=16,
    embedding_dim=8,
    time_length=300,
)

x = torch.randn(2, 6, 300)
z = encoder(x)
x_hat = decoder(z)
```

Expected shapes:

```text
z:     [2, 16, 8]
x_hat: [2, 6, 300]
```

## Failure Modes

- Shape errors surface from PyTorch layers if tensors do not match the expected
  contracts.
- Old CNNToken checkpoints may fail to load because this framework no longer
  uses `blocks.cnn` parameter names.
