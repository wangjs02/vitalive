# VitalDB VQ-VAE

## Purpose

`vitaldb.py` defines one assembled VitalDB VQ-VAE wrapper. The model owns only
three modules:

- `encoder`
- `quantizer`
- `decoder`

It supports two codec families:

- `cnn_tokens`
- `vit`

Losses, reports, checkpoint IO, training loops, and VitalSense transfer logic
belong outside this file.

## Inputs

- `VQVAEConfig`: wrapper config composed from codec and quantizer config
  dataclasses.
- `codec_config`: optional plain dict passed to the selected encoder and
  decoder.
- `quantizer_config`: optional plain dict passed to `SequenceEMAQuantize`.
- `enc_dec`: either `"cnn_tokens"` or `"vit"`.
- Model input tensor: `[B, vital_channels, time_length]`.

## Outputs

- `forward(x)`: VQ-VAE output dictionary with reconstruction, latent tensors,
  loss terms, and codebook diagnostics.
- `encode(x)`: codec latent tokens.
- `quantize(z)`: quantized decoder-input tokens.
- `decode(z)`: reconstructed tensor.
- `embedding_to_code(z)`: convert codec embeddings to quantizer code vectors.
- `code_to_embedding(z)`: convert quantizer code vectors back to codec embeddings.

## Public API

- `VQVAEConfig`
- `VitalDBVQVAE`
- `SUPPORTED_VQVAE_CODECS`
- `cnn_tokens_config`
- `vit_config`
- `quantizer_config`

## Dependencies

- `blocks.quantizer.SequenceEMAQuantize`
- `blocks.quantizer.SequenceEMAQuantizerConfig`
- `codec.cnn_token.CNNTokenConfig`
- `codec.cnn_token.CNNTokenEncoder`
- `codec.cnn_token.CNNTokenDecoder`
- `codec.vit.ViTConfig`
- `codec.vit.ViTEncoder`
- `codec.vit.ViTDecoder`
- `utils.config.KwargsConfig`
- `torch`

## Used By

- VitalDB pretrain code that needs an assembled encoder, quantizer, and decoder.

## Responsibilities

This file owns:

- Selecting the encoder and decoder for `cnn_tokens` or `vit`.
- Building the quantizer.
- Routing tensors through `encode -> quantize -> decode`.
- Reshaping ViT token embeddings into grouped code vectors and back.

## Boundaries

This file does not own:

- Reconstruction loss or VQ loss aggregation.
- Perplexity or codebook reports.
- Checkpoint save/load or registry handling.
- VitalSense transfer, target conversion, or plotting.
- Optimizer or epoch logic.
- Dataset normalization.

## Architecture

```text
x [B,C,T]
  -> encoder
  -> z
  -> embedding_to_code
  -> z_code
  -> quantizer
  -> z_code_q
  -> code_to_embedding
  -> z_q
  -> decoder
  -> x_hat [B,C,T]
```

For `cnn_tokens`, quantizer input and decoder input both use:

```text
[B, token_length, embedding_dim]
```

For `vit`, the default quantizer uses the same last dimension as the ViT token
embedding:

```text
z:                  [B, N, embedding_dim]
quantizer input:    [B, N, embedding_dim]
decoder input:      [B, N, embedding_dim]
```

If a custom `quantizer.embedding_dim` is smaller than the codec `embedding_dim`,
`embedding_to_code` splits the last dimension before quantization and
`code_to_embedding` joins it again before decoding.

## Tensor Contract

CNNToken sample:

```text
B = 2, C = 4, T = 30
token_length = 8, embedding_dim = 4

x:                  [2, 4, 30]
encode:             [2, 8, 4]
quantize:           [2, 8, 4]
decode:             [2, 4, 30]
```

ViT sample:

```text
B = 2, C = 4, T = 30
patch_size = 10, token_length = 16
embedding_dim = 8

x:                  [2, 4, 30]
encode:             [2, 3, 8]
embedding_to_code:  [2, 6, 4] when code_dim = 4
quantize:           [2, 6, 4]
code_to_embedding:  [2, 3, 8]
decode:             [2, 4, 30]
```

## Config Contract

- `codec_config["input_dim"]` is the input channel count.
- `codec_config["time_length"]` is the reconstruction length.
- `enc_dec` is expected to be `"cnn_tokens"` or `"vit"`.
- `CNNTokenConfig` is used by `CNNTokenEncoder` and `CNNTokenDecoder`.
- `ViTConfig` is used by `ViTEncoder` and `ViTDecoder`.
- `SequenceEMAQuantizerConfig` is used by `SequenceEMAQuantize`.
- `VQVAEConfig.use_quantizer=False` bypasses the quantizer.
- `cnn_tokens_config`, `vit_config`, and `quantizer_config` are backward-
  compatible plain-dict aliases built from the dataclass defaults.

## Step-by-step Tensor Flow

1. `encode(x)` runs the selected encoder.
2. `embedding_to_code(z)` reshapes codec embeddings into quantizer code vectors.
3. `quantize(z_code)` runs `SequenceEMAQuantize` and returns quantized code
   vectors.
4. `code_to_embedding(z_code_q)` rebuilds decoder embeddings.
5. `decode(z_q)` runs the selected decoder.
6. `forward(x)` returns a dictionary containing `x_recon`, latent tensors,
   `loss`, `recon_loss`, `vq_loss`, `perplexity`, and `cluster_use`.

## Debug Checklist

- Confirm `model.encoder`, `model.quantizer`, and `model.decoder` are present.
- Confirm `model(x).shape == x.shape`.
- For custom quantizer dimensions, confirm codec `embedding_dim` is divisible
  by quantizer `embedding_dim`.
- For CNNToken, confirm `time_length` matches the input length.

## Sample Case

```python
import torch
from codec.vit import ViTConfig
from model.vitaldb import VQVAEConfig, VitalDBVQVAE

config = VQVAEConfig(enc_dec="vit", codec=ViTConfig())
model = VitalDBVQVAE(config=config)
out = model(torch.randn(2, 4, 30))
```

Expected shape:

```text
out["x_recon"]: [2, 4, 30]
```

## Shape Issues

- `embedding_to_code` leaves tensors unchanged when the last dimension already
  matches the quantizer code dimension.
- `code_to_embedding` leaves tensors unchanged when the last dimension already
  matches `embedding_dim`.
- Shape or config mismatches surface from the underlying codec, reshape, or
  quantizer operations.
