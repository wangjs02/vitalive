# Config Utilities

## Purpose

`config.py` defines a small parent class for dataclass configuration objects
that should be passed into constructors as keyword arguments.

## Inputs

- Dataclass subclasses of `KwargsConfig`.
- Field names must match the target constructor keyword names.

## Outputs

- `dict(config)` returns a plain kwargs dictionary.
- `config.to_kwargs()` returns the same plain kwargs dictionary.
- `SomeModule(**config)` works for config objects that match the constructor.

## Public API

- `KwargsConfig`

## Dependencies

- Python standard library:
  - `collections.abc.Mapping`
  - `dataclasses`
  - `typing`

## Used By

- `codec.cnn_token.CNNTokenConfig`
- `codec.vit.ViTConfig`
- `blocks.quantizer.SequenceEMAQuantizerConfig`
- `model.vitaldb.VQVAEConfig`

## Responsibilities

- Provide one reusable parent for kwargs-style dataclass configs.
- Avoid duplicating `to_kwargs()` in every model or block config class.

## Boundaries

- This file does not validate neural-network shapes.
- This file does not own model defaults.
- This file does not serialize checkpoints.

## Architecture

`KwargsConfig` implements the `Mapping[str, Any]` interface. Python can unpack
mapping objects with `**config`, and callers can also materialize a plain dict
with `dict(config)`.

## Tensor Contract

N/A

## Debug Checklist

- Confirm the subclass is decorated with `@dataclass`.
- Confirm dataclass field names match the target constructor.
- Use `dict(config)` to inspect the exact kwargs before passing them onward.

## Sample Case

```python
from dataclasses import dataclass
from utils.config import KwargsConfig

@dataclass
class LinearConfig(KwargsConfig):
    in_features: int = 8
    out_features: int = 4

kwargs = dict(LinearConfig())
```

Expected `kwargs`:

```text
{"in_features": 8, "out_features": 4}
```

## Failure Modes

- A subclass without `@dataclass` raises `TypeError` in `to_kwargs()`.
- A field name that does not match the target constructor raises a constructor
  `TypeError` when unpacked.
