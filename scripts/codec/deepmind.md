# DeepMind Codec

## Purpose

Define the external-style 2D convolution encoder and decoder used as a concrete
VQ-VAE encoder-decoder family for image-like VitalDB inputs.

## Inputs

- Encoder input tensor: `[B, image_input_channels, C, T]`.
- Decoder latent tensor: grid-shaped quantized tensor from `VQVAEQuantize`.
- Configuration values for input channels, hidden dimension, latent dimension,
  and output channels.

## Outputs

- `DeepMindEncoder`: grid latent tensor with `output_channels`.
- `DeepMindDecoder`: reconstructed image-like tensor shaped
  `[B, output_channels, C, T]` before `model.vitaldb` maps it back to `[B,C,T]`.

## Public API

- `DeepMindEncoder`
- `DeepMindDecoder`

## Dependencies

- `torch.nn`
- `blocks.resnet.ResidualStack`

## Used By

- `model.vitaldb`

## Responsibilities And Boundaries

This module owns only the DeepMind-style convolutional encoder/decoder. It does
not own VitalDB sequence padding, VQ-VAE quantizer selection, training loops, or
data preprocessing.

## Architecture

```text
encoder:
  image [B,1,C,T]
  -> strided Conv2d
  -> strided Conv2d
  -> Conv2d projection
  -> residual stack

decoder:
  z grid
  -> Conv2d projection
  -> residual stack
  -> ConvTranspose2d
  -> ConvTranspose2d
  -> image reconstruction
```

## Tensor Contract

For `B = 2`, `image_input_channels = 1`, `C = 6`, `T = 64`,
`embedding_dim = 64`:

```text
DeepMindEncoder(x) -> [2, 64, C', T']
DeepMindDecoder(z) -> [2, 1, C, T] approximately, cropped by model.vitaldb
```

## Config Contract

- `n_hid` controls hidden width.
- `DeepMindEncoder.output_channels` is the quantizer input channel count.
- Final output cropping is handled by `model.vitaldb`, not this module.

## Step-by-step Tensor Flow

1. Convert VitalDB sequence into an image-like tensor in `model.vitaldb`.
2. Encode the image-like tensor with Conv2d and residual blocks.
3. Quantize the resulting grid with `VQVAEQuantize`.
4. Decode the grid back to image-like space.
5. Let `model.vitaldb` crop and squeeze to sequence shape.

## Debug Checklist

- Check the wrapper supplies `[B, 1, C, T]`, not `[B, C, T]`.
- Verify decoder output is large enough for the configured crop.
- Confirm hidden dimension is not accidentally mismatched with checkpoint
  config.

## Sample Case

```python
encoder = DeepMindEncoder(input_channels=1, n_hid=64)
z = encoder(torch.randn(2, 1, 6, 64))
decoder = DeepMindDecoder(n_init=64, n_hid=64, output_channels=1)
x_hat = decoder(z)
```

## Failure Modes

- Input rank mismatch because this codec expects image-like tensors.
- Decoder crop in `model.vitaldb` fails if config time/channel dimensions do not
  match the encoded data.
