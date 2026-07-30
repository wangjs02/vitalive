# Codec Package Init

## Purpose

Expose active complete encoder-decoder and tokenizer frameworks from the
canonical `codec` namespace.

## Inputs

No runtime inputs.

## Outputs

Package-level imports for retained concrete encoder-decoder codec families.

## Public API

- `CNNTokenEncoder`
- `CNNTokenDecoder`
- `ViTEncoder`
- `ViTDecoder`
- `DeepMindEncoder`
- `DeepMindDecoder`

## Dependencies

- `codec.cnn_token`
- `codec.vit`
- `codec.deepmind`

## Responsibilities And Boundaries

This package init owns only the stable codec-level import surface. It does not
define implementation logic beyond importing complete framework classes from
their module files.

Low-level components stay in `blocks`, assembled dataset-specific models such
as `VitalDBVQVAE` stay in `model`, losses stay in `train`, and
project-specific diagnostics stay in `eval`.

## Used By

- Pretrain notebooks.
- VQ-VAE training pipeline through `model.vitaldb`.

## Sample Case

```python
from codec import CNNTokenEncoder, ViTEncoder
```

## Failure Modes

Import fails if the active environment lacks required ML dependencies such as
PyTorch or SciPy.
