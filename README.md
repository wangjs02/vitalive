# Vitalive Production Code

This repository contains the mature reusable code copied from the Vitalive
research workspace.

## Layout

```text
prod/
├── scripts/      # Reusable Python modules for data, model, codec, train, eval, and utils
├── model/        # Local checkpoints, weights, and model outputs; ignored by git except .gitkeep
└── README.md
```

## Data

Data does not live inside `prod/`. The production code should read from the
shared data root outside the project folder so multiple projects can reuse the
same datasets.

Server layout:

```text
/
├── disk/
│   └── coconut/
│       └── junshi/
│           └── data/  # Real shared data storage
└── home/
    └── junshi/
        ├── data/      # Symlink to /disk/coconut/junshi/data, also ~/data
        └── vitalive/
            └── prod/  # Production code repo
```

The shared server data root should be stored at:

```text
/disk/coconut/junshi/data
```

Expose it through the stable home-directory path:

```bash
ln -s /disk/coconut/junshi/data /home/junshi/data
```

After that, `~/data` resolves through the symlink and remains the canonical
path scripts should use.

Dataset layout under the shared data root:

```text
/home/junshi/data/
│   ├── VitalDB/
│   ├── VitalSense/
│   ├── COHFACE/
│   ├── Guardian/
│   └── VIPL-HR/
```

Local workspace layout follows the same rule: keep data outside `prod/`.

```text
Vitalive/
├── data/              # Shared local data root
└── code/
    └── prod/          # Production code repo
```

Each dataset owns its own folder inside the shared data root. Keep source
files, documentation, and processed outputs inside the owning dataset folder.

```text
~/data/
├── VitalDB/
│   ├── docs/     # Papers, dataset documentation, and download notes
│   ├── metadata/ # Case, track, and lab metadata tables
│   ├── zip/      # Downloaded archives or compressed API responses
│   ├── raw/      # Extracted files or direct API downloads
│   └── processed/
│       ├── clean_v1/
│       ├── aligned_v1/
│       └── sampled_5s_v1/
├── VitalSense/
│   ├── docs/
│   ├── metadata/
│   ├── zip/
│   ├── raw/
│   └── processed/
├── COHFACE/
│   ├── docs/
│   ├── metadata/
│   ├── zip/
│   ├── raw/
│   └── processed/
├── Guardian/
│   ├── docs/
│   ├── metadata/
│   ├── zip/
│   ├── raw/
│   └── processed/
└── VIPL-HR/
    ├── docs/
    ├── metadata/
    ├── zip/
    ├── raw/
    └── processed/
```

Use `zip/` for original downloaded archives or compressed API responses, `raw/`
for extracted files or direct API downloads, and `<dataset>/processed/<method>/`
for cleaned, aligned, cached, or training-ready outputs. Each processing method
or version should have its own folder so different preprocessing recipes can
coexist without overwriting each other.

For VitalDB, the Web API metadata should live under `~/data/VitalDB/metadata/`:

```text
~/data/VitalDB/metadata/
├── VitalDB_cases_uncompressed.csv
├── VitalDB_trks_uncompressed.csv
└── VitalDB_labs_uncompressed.csv
```

On the server, `~/data` resolves to `/home/junshi/data`, which should be a
symlink to `/disk/coconut/junshi/data`. Scripts should expose an explicit data
root argument or environment variable and default to the external shared data
root, not to a folder inside `prod/`.

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
