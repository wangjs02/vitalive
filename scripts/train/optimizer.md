# Optimizer Helpers

## Purpose

Provide reusable optimizer construction for training scripts, including
project-standard AdamW weight-decay parameter grouping.

## Inputs

- PyTorch model.
- Learning rate, weight decay, Adam betas, and epsilon.

## Outputs

- Parameter groups with decay and no-decay sets.
- `torch.optim.AdamW` optimizer.

## Public API

- `build_weight_decay_parameter_groups`
- `build_adamw_optimizer`

## Dependencies

- `torch`
- `torch.nn`

## Used By

- `pipeline.vitaldb`
- Future supervised and representation-learning training scripts.

## Responsibilities And Boundaries

This module owns optimizer construction policy. It does not own epoch loops,
loss functions, scheduler policy, model definitions, checkpoint IO, or dataset
loading.

## Sample Case

```python
optimizer = build_adamw_optimizer(model, lr=3e-3, weight_decay=1e-4)
```

## Failure Modes

- Parameters that appear in both decay and no-decay groups raise
  `RuntimeError`.
- Frozen parameters are still included if they remain in `model.parameters()`;
  callers should freeze before optimizer construction.
