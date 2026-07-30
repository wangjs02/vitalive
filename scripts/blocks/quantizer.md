# Quantizer Components

## Purpose

Provide reusable VQ quantizer components used by codec frameworks. These are
model components, not complete encoder-decoder architectures.

## Inputs

- `VQVAEQuantize`: convolutional latent tensor shaped `[B, C, H, W]`.
- `SequenceEMAQuantize`: sequence latent tensor shaped `[B, token_length, D]`.
- Optional rotation matching and Fourier matching settings for sequence tokens.

## Outputs

- Quantized tensor with the same latent structure as input.
- VQ commitment loss scalar.
- Discrete code indices.
- Codebook perplexity and cluster-use diagnostics.

## Public API

- `SequenceEMAQuantizerConfig`
- `VQVAEQuantize`
- `SequenceEMAQuantize`
- `codebook_perplexity`

## Dependencies

- `torch`
- `scipy.cluster.vq.kmeans2`
- `utils.config.KwargsConfig`
- internal `FixedFourierProjection` for optional sequence-token Fourier
  matching.

## Responsibilities And Boundaries

This module owns reusable VQ components and codebook diagnostics. It does not
own full codec frameworks, reconstruction losses, task-specific training loops,
raw data loading, or evaluation visualizations.

## Used By

- `model.vitaldb`
- future codec modules

## Architecture

`VQVAEQuantize` keeps the original grid-style VQ-VAE behavior:

```text
[B, C, H, W] -> 1x1 conv projection -> nearest embedding -> straight-through z_q
```

`SequenceEMAQuantize` works on sequence tokens:

```text
[B, T, D] -> nearest codebook match -> EMA codebook update -> straight-through z_q
```

When `rotation_matching=True`, each code vector is compared against all circular
rotations, but the returned index remains the base code index.

When `fourier_matching_dim` is set, `SequenceEMAQuantize` projects vectors into
a fixed Fourier basis before distance comparison. This projection is internal to
the quantizer because it is only used for codebook matching.

## Tensor Contract

- `VQVAEQuantize.forward(z)` expects `[B, C, H, W]`.
- `SequenceEMAQuantize.forward(z)` expects `[B, T, D]`, where `D` equals
  `embedding_dim`.
- `embed_code(ids)` returns code vectors from the internal codebook.

## Config Contract

- `n_embed`: number of base codebook entries.
- `embedding_dim`: code vector dimension.
- `decay`, `eps`: EMA update parameters.
- `commitment_cost`, `kld_scale`: VQ loss scale.
- `rotation_matching`: compare circularly rotated candidates.
- `fourier_matching_dim`: optional Fourier matching space dimension.
- `SequenceEMAQuantizerConfig` stores these constructor values and can be
  unpacked directly as `SequenceEMAQuantize(**SequenceEMAQuantizerConfig())`.

## Step-by-step Tensor Flow

1. Flatten latent tokens into `[B*T, D]` or grid vectors.
2. Initialize the codebook with k-means on the first training batch.
3. Compute nearest codebook entry by squared Euclidean distance.
4. Build quantized vectors from the selected codebook entries.
5. During training, update EMA codebook statistics.
6. Return straight-through quantized vectors, loss, and indices.

## Debug Checklist

- Confirm input rank and embedding dimension before quantization.
- Check codebook initialization happened once.
- Track codebook perplexity and cluster use.
- For rotation matching, inspect `last_rotation_shifts`.
- Verify `fourier_matching_dim >= embedding_dim` when enabled.

## Sample Case

```python
quantizer = SequenceEMAQuantize(n_embed=64, embedding_dim=8)
z_q, loss, indices = quantizer(z)
```

## Failure Modes

- Input tensor has the wrong rank.
- Last dimension differs from `embedding_dim`.
- SciPy is unavailable for k-means initialization.
- Too few first-batch vectors can lead to duplicated initialization samples.
