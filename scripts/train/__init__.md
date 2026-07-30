# Train Package Init

## Purpose

Define `train` as the canonical namespace for reusable training mechanics.

## Inputs

No runtime inputs.

## Outputs

Python package initialization for `train`.

## Public API

Submodules provide the public API:

- `train.regression`
- `train.epoch`
- `train.optimizer`
- `train.loss`

## Dependencies

None at package import time.

## Responsibilities And Boundaries

This package init owns only the `train` namespace. It does not import heavy
training code at package import time, define model architectures, or load data.

## Used By

Training notebooks, pipeline modules, and future script entrypoints.

## Sample Case

```python
from train.epoch import run_vqvae_epoch
```

## Failure Modes

None expected.
