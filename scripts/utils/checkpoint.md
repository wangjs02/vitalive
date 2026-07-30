# Checkpoint

## Purpose

Provide one generic checkpoint run object that owns run identity, filesystem
paths, payload persistence, and architecture-level registry access.

## Inputs

- Project root, checkpoint architecture subdirectory, and optional run ID.
- A pipeline-prepared checkpoint payload.
- A pipeline-prepared registry summary.

## Outputs

- A timestamped run directory.
- Standard `best.pt` and `last.pt` paths.
- An architecture-level `registry.jsonl` path.
- Saved checkpoint payloads and appended registry records.

## Public API

- `Checkpoint`
- `Checkpoint.create`
- `Checkpoint.save`
- `Checkpoint.update_registry`
- `Checkpoint.load`
- `Checkpoint.load_registry`
- `Checkpoint.best`

## Dependencies

- PyTorch
- standard-library JSON, dataclasses, datetime, and pathlib.

## Used By

- `pipeline.vitaldb.pretrain`
- Checkpoint-loading notebooks and evaluation workflows.

## Responsibilities

Own generic run IDs, paths, serialization, and registry file mechanics.

## Boundaries

This module does not know model architecture, dataset contracts, training
metrics, preprocessing, artifact generation, or pipeline payload schemas.

## Architecture

`Checkpoint.create` establishes a run. Pipeline code prepares payload and
registry dictionaries, then calls `save` and `update_registry`.

## Tensor Contract

N/A.

## Debug Checklist

- Confirm the architecture directory matches the intended task/model family.
- Confirm payload construction happens before `save`.
- Append the registry only after all checkpoint artifacts save successfully.
- Confirm registry artifact paths are relative to the architecture directory.

## Sample Case

```python
from utils.checkpoint import Checkpoint

checkpoint = Checkpoint.create(repo_root, "pretrain/vit")
checkpoint.save(payload)
checkpoint.update_registry(registry_record)
```

## Failure Modes

- Checkpoint directory cannot be created or written.
- Payload contains objects unsupported by `torch.save`.
- Registry record cannot be JSON serialized.
- No registry record contains the requested best-model metric.
