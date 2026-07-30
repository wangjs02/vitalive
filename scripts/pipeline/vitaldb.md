# VitalDB Pipeline

## Purpose

Provide the end-to-end orchestration entrypoint for VitalDB pretraining.

## Inputs

- VitalDB data and metadata directories.
- The default required vital-sign tracks.
- A fixed random seed and train/validation ratios.
- Epoch count, batch size, learning rate, early-stopping patience/minimum
  improvement, and device.

## Outputs

- A trained `VitalDBVQVAE` with its best validation-reconstruction weights.
- Training history and final test metrics.
- Fitted train-only transforms.
- History CSV, diagnostic PNGs, checkpoint payload, and registry record.

## Public API

- `pretrain`

Supporting helpers used by the expanded notebook and tests:

- `train_val_test_split`
- `VitalDBCheckpoint`
- `VitalDBCheckpoint.create_payload`
- `VitalDBCheckpoint.update_registry`

## Dependencies

- `numpy`
- `torch`
- `blocks.quantizer.SequenceEMAQuantizerConfig`
- `codec.vit.ViTConfig`
- `data.vitaldb`
- `model.vitaldb`
- `train.epoch`
- `train.optimizer`
- `eval.vitaldb`
- `utils.checkpoint.Checkpoint`

## Used By

- Future VitalDB pretraining entrypoints and notebooks.

## Responsibilities And Boundaries

`pretrain` is the single end-to-end workflow entrypoint. It loads or generates
the complete disk-backed clean dataset, then calls `train_val_test_split` to
extract the unique case IDs that
actually produced clean segments, shuffle them with a fixed seed, and perform
an 80/10/10 train/validation/test split. Optional limits are applied only after
this clean case-level split. Each `VitalDBDataset` receives
`(case_id, segment_id)` pair lists, and DataLoader reads the processed `.npy`
files during training. Normalization fits on train only, validation drives
model selection, and test is reserved for final evaluation. Early stopping
monitors validation reconstruction loss, requires an improvement larger than
`min_delta`, and stops after `patience` consecutive non-improving epochs. The
best model state is restored before the single final test pass. The workflow
then creates history and reconstruction artifacts, builds the structured
payload, saves `best.pt`, and appends the architecture registry.

The data module owns VitalDB reading and preprocessing. The model module owns
the VQ-VAE implementation, while the train package owns optimizer and epoch
mechanics.

`VitalDBCheckpoint` extends the generic `Checkpoint`. Its `create_payload`
method builds the VitalDB-specific schema without side effects, while its
`update_registry` method owns the VitalDB registry fields. The expanded
notebook can inspect the exact payload without creating another
checkpoint-saving workflow. The parent `Checkpoint` class only supplies run
paths, serialization, loading, and generic registry file mechanics.

## Sample Case

```python
from pipeline.vitaldb import pretrain

result = pretrain(
    project_root=repo_root,
    epochs=100,
    batch_size=8,
    patience=5,
    min_delta=1e-3,
    device="cpu",
)

model = result["model"]
checkpoint = result["checkpoint"]
test_metrics = result["test_metrics"]
```

Run the configured end-to-end workflow from the command line:

```bash
python -m scripts.pipeline.vitaldb
```

## Failure Modes

- VitalDB track metadata is missing.
- No case contains all required tracks.
- Fewer than three clean cases are available.
- Split fractions are invalid or a case limit empties one split.
- Train, validation, and test cases overlap.
- `epochs` or `patience` is non-positive, or `min_delta` is negative.
- Training produces no finite validation checkpoint.
- The fitted transform chain does not contain `NormalizeVitalDB`.
- Required history or test metrics are absent when saving a checkpoint.
- Evaluation artifacts or the checkpoint registry cannot be written.
- Dataset channel/time contracts disagree with the constructed model config.
- The selected device is unavailable.
