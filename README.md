# Vitalive Production Code

This repository contains the mature reusable code copied from the Vitalive
research workspace.

## Layout

```text
prod/
├── scripts/      # Reusable Python modules for data, model, codec, train, eval, and utils
├── data/         # Local downloaded datasets and generated data; ignored by git
├── model/        # Local checkpoints, weights, and model outputs; ignored by git except .gitkeep
└── README.md
```

## Data

Download datasets and local generated data into `data/`. Each dataset should
own its own folder, and each dataset folder should use the same internal
structure:

```text
data/
├── VitalDB/
│   ├── docs/     # Dataset notes, metadata tables, API manifests, and paper notes
│   ├── zip/      # Downloaded archives or compressed API responses
│   ├── raw/      # Extracted or API-downloaded raw files
│   └── clean/    # Cleaned, aligned, cached, or training-ready data
├── VitalSense/
│   ├── docs/
│   ├── zip/
│   ├── raw/
│   └── clean/
├── COHFACE/
│   ├── docs/
│   ├── zip/
│   ├── raw/
│   └── clean/
├── Guardian/
│   ├── docs/
│   ├── zip/
│   ├── raw/
│   └── clean/
└── VIPL-HR/
    ├── docs/
    ├── zip/
    ├── raw/
    └── clean/
```

Use `zip/` for original downloaded archives or compressed API responses, `raw/`
for extracted files or direct API downloads, and `clean/` for processed outputs
that are ready for model training or evaluation.

For VitalDB, the Web API metadata should live under `data/VitalDB/docs/` or a
more specific metadata subfolder inside it:

```text
data/VitalDB/docs/
├── VitalDB_cases_uncompressed.csv
├── VitalDB_trks_uncompressed.csv
└── VitalDB_labs_uncompressed.csv
```

The contents of `data/` are intentionally ignored because raw datasets and
generated caches are large and environment-specific.

## Model Artifacts

Store checkpoints, trained weights, and model outputs under `model/`.

Recommended structure:

```text
model/
├── checkpoints/
│   ├── pretrain/
│   │   ├── cnn_token/
│   │   ├── vit/
│   │   └── vqvae/
│   └── prediction/
│       ├── vitalsense/
│       ├── guardian/
│       ├── cohface/
│       └── vipl_hr/
├── runs/
│   ├── pretrain/
│   └── prediction/
└── exports/
```

Use `checkpoints/` for reusable model weights, `runs/` for training outputs and
diagnostics, and `exports/` for packaged artifacts that are intended to be
moved or deployed.

The contents of `model/` are intentionally ignored. The `model/.gitkeep` file
keeps the directory present in a fresh checkout.

## Code

Reusable implementation code lives under `scripts/`.

The current code was copied from the parent Vitalive workspace:

```text
../scripts/ -> scripts/
```

Keep dataset paths, checkpoint paths, and run outputs explicit when adding
production scripts or experiments.
