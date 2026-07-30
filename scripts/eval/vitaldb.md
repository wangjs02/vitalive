# VitalDB History Evaluation

## Purpose

Convert nested VitalDB VQ-VAE training history into a tabular record and save
loss and quantizer diagnostic plots.

## Inputs

- Training `history`: a sequence of rows containing `epoch`, `train`, and `val`.
- An optional output PNG path for the individual plot functions.
- An optional output directory for the combined evaluation function.
- A VQ-VAE model and test batch shaped `[B, C, T]` for reconstruction review.
- Optional fitted transforms for inverse conversion to physical units.

## Outputs

- `loss_history.png` with total, reconstruction, and commitment losses.
- `quantizer_history.png` with perplexity, cluster use, and std ratio.
- `history.csv` with one flattened row per epoch.
- The combined function returns the DataFrame and artifact path mapping.
- A reconstruction Figure and optional `reconstruction_samples.png` path.

## Public API

- `eval_loss_history`
- `eval_quantizer_history`
- `eval_history`
- `eval_recon`

## Dependencies

- NumPy
- pandas
- Matplotlib
- PyTorch

## Used By

- `code/notebooks/pretrain/pretrain_vit_pipeline.ipynb`
- VitalDB checkpoint artifact generation.

## Responsibilities

Flatten scalar training metrics, infer the best epoch from finite validation
reconstruction loss, save deterministic review artifacts, and compare test
targets with model reconstructions.

## Boundaries

This module does not run training, load checkpoints, compute aggregate test
metrics, or append checkpoint registries.

## Architecture

`eval_history` delegates plot generation to `eval_loss_history` and
`eval_quantizer_history`. With no output directory it displays both figures in
the notebook and returns the flattened DataFrame. With an output directory it
saves the shared DataFrame and both PNGs without displaying them, then returns
the DataFrame and artifact paths.

## Tensor Contract

History inputs are nested scalar metric mappings. Reconstruction inputs are
`[B, C, T]` tensors; model output must contain `x_recon` with the same shape.

## Debug Checklist

- Confirm every row contains `epoch`, `train`, and `val`.
- Confirm validation reconstruction loss contains a finite value.
- Confirm all epoch metrics required by the selected plot are present.
- Confirm the output directory is writable.
- Confirm the model returns an `x_recon` tensor.
- Confirm optional vital-sign labels match the channel count.

## Sample Case

```python
from eval.vitaldb import eval_history

# Notebook review: display plots and do not write files.
history_frame = eval_history(history)

# Checkpoint artifacts: save CSV and PNGs without displaying plots.
history_frame, artifact_paths = eval_history(history, checkpoint_run.run_dir)

figure, reconstruction_path = eval_recon(
    model,
    test_batch,
    transforms=fitted_transforms,
    vital_signs=vital_signs,
    save_dir=checkpoint_run.run_dir,
)
```

## Failure Modes

- Empty history.
- Missing train/validation metrics.
- No finite validation reconstruction loss.
- Output directory cannot be created or written.
- Test batch does not follow `[B, C, T]`.
- Model output does not contain `x_recon`.
- Transform inverse or channel labels do not match the data contract.
