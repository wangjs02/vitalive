# Training Losses

## Purpose

Expose reusable loss objects and transfer losses from one training namespace.

## Inputs

- Normalized reconstruction tensors for `Normal.nll`.
- Token logits and discrete code targets for `token_cross_entropy_loss`.
- Predicted code embeddings, teacher targets, and frozen decoder model for
  `code_embedding_commitment_loss`.

## Outputs

- Scalar PyTorch losses.

## Public API

- `Normal`
- `token_cross_entropy_loss`
- `code_embedding_commitment_loss`

## Dependencies

- `torch`
- `torch.nn.functional`

## Responsibilities And Boundaries

This module owns reusable objective functions:

- normalized reconstruction loss
- token cross-entropy loss
- code-embedding commitment loss

It does not own model architecture, teacher loading, decoding, or training
loops.

## Used By

- VQ-VAE pretrain.
- VitalSense radar-to-token transfer.

## Sample Case

```python
loss = token_cross_entropy_loss(logits, targets)
commit = code_embedding_commitment_loss(z_pred, targets, teacher_model)
```

## Failure Modes

- Token logits must be shaped `[B, tokens, codes]`.
- Token targets must be shaped `[B, tokens]`.
- Predicted code embedding dim must match the frozen teacher codebook dim.
