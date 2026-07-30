# Scripts Workflow

`code/scripts/` is the importable Python layer for the Vitalive workspace.

## Target Structure

```text
code/scripts/
├── data/
├── blocks/
├── model/
├── codec/
├── train/
├── eval/
└── utils/
```

## Ownership

- `data/`: dataset-specific loading, preprocessing, alignment, transforms, and
  PyTorch dataset classes.
- `blocks/`: reusable low-level neural-network blocks such as CNN blocks,
  transformer blocks, projection layers, positional encodings, pooling, and
  quantizers.
- `model/`: dataset-specific or project-specific assembled models such as
  VitalSense baselines and radar-to-token mappers.
- `codec/`: complete encoder-decoder or tokenizer frameworks such as VQ-VAE,
  CNNToken, ViT, and DeepMind-style codecs.
- `train/`: reusable optimization, epoch loops, early stopping, losses, and
  checkpoint helpers.
- `eval/`: project-specific diagnostics, probes, metrics, and debug views.
- `utils/`: small general helpers for paths, seeds, registries, shape checks, and
  metadata.

## Documentation Rule

Every Python file in `blocks/`, `model/`, `codec/`, `train/`, and `eval/` must
have a same-name Markdown document. The Markdown document records purpose,
inputs, outputs, public API, dependencies, users, sample case, and failure
modes.

For complex model or codec files, the Markdown document must also include the
architecture, tensor contract, config contract, tensor flow, and debug checklist.

## Canonical Import Rule

Final code should import only from the canonical structure:

```text
data/
blocks/
model/
codec/
train/
eval/
utils/
```

Do not keep fallback imports or old-package compatibility shims after notebooks
and scripts are migrated. A broken import should fail explicitly so the missing
dependency or stale path can be fixed at the source.
