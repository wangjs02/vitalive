# Eval Package Init

## Purpose

Define `eval` as the canonical namespace for evaluation, probing, and diagnostic
modules.

## Inputs

No runtime inputs.

## Outputs

Python package initialization for `eval`.

## Public API

Subpackages provide public API:

- `eval.pretrain`
- `eval.vitalsense`
- `eval.diagnostics`
- `eval.vitaldb`

## Dependencies

None at package import time.

## Responsibilities And Boundaries

This package init owns only the evaluation namespace. It does not import heavy
plotting/model code at package import time, define models, train models, or load
datasets.

## Used By

Pretrain and prediction notebooks.

## Sample Case

```python
from eval.vitaldb import eval_history

history_frame, artifact_paths = eval_history(history, output_dir)
```

## Failure Modes

None expected.
