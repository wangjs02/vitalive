# Epoch Training Helpers

## Purpose

Provide reusable epoch-level training/evaluation loops for models whose batch
contract is stable across notebooks and scripts.

## Inputs

- Model returning a VQ-VAE output dictionary.
- Data loader yielding either tensor batches directly or dictionaries with `x`.
- Optional optimizer. If supplied, the loop runs training; otherwise it runs
  evaluation with `torch.no_grad()`.
- Device and optional gradient clipping norm.

## Outputs

- Averaged metric dictionary with loss, reconstruction loss, commitment loss,
  perplexity, cluster use, input standard deviation, and reconstruction
  standard deviation.

## Public API

- `run_vqvae_epoch`

## Dependencies

- `torch`
- `torch.nn`

## Used By

- `pipeline.vitaldb`

## Responsibilities And Boundaries

This module owns the repeated batch loop mechanics for VQ-VAE pretraining. It
does not own data loading, model construction, optimizer construction,
checkpointing, early stopping, or plotting.

## Sample Case

```python
train_metrics = run_vqvae_epoch(model, train_loader, optimizer=optimizer, device=device)
val_metrics = run_vqvae_epoch(model, val_loader, device=device)
```

## Failure Modes

- Batch does not include `x`.
- Model output does not include the expected VQ-VAE keys.
- Device mismatch between model and input tensors.
