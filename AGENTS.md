# Agent Guide

This document explains how agents and collaborators should work inside the
`Vitalive` production repository.

## Document Maintenance

Last updated: 2026-07-30.

This guide is a living project operating manual. Update it when the production
repository structure, server layout, environment setup, or data conventions
change.

The goal is to keep future agents and collaborators from reconstructing project
context from scattered notes. Prefer clear project ownership, reproducible
workflows, and explicit paths.

## Objective

`prod` is the production code repository extracted from the larger `Vitalive`
research workspace.

The production repository should support:

```text
shared data root -> reproducible preprocessing/training code -> model artifacts -> deployment or server-side experiments
```

Do not optimize for ad hoc local experiments. Optimize for a repository that is
easy to run on the server and easy to maintain over time.

## Project Organization

```text
prod/
├── scripts/                      # Importable Python modules and pipeline code
├── model/                        # Local checkpoints and model outputs
├── requirements.txt              # Server Python requirements
├── README.md                     # Human-facing repository guide
└── AGENTS.md                     # Agent/collaborator operating guide
```

Raw and external datasets do not live inside `prod/`. They live under the
shared external data root at `~/data/`.

### `scripts/`

`scripts/` is the importable Python layer for the production repository.

Use it for:

- dataset loading
- preprocessing modules
- model definitions
- training utilities
- evaluation utilities
- deterministic pipeline scripts

Target structure:

```text
scripts/
├── data/
├── blocks/
├── model/
├── codec/
├── pipeline/
├── train/
├── eval/
└── utils/
```

### `model/` And `data/`

`model/` is for checkpoints, run outputs, and model artifacts.

`data/` is the shared external data root and should not be created inside this
repository as the canonical dataset location.

Server layout:

```text
/
├── disk/
│   └── coconut/
│       └── junshi/
│           └── data/             # Real shared data storage
└── home/
    └── junshi/
        ├── data/                 # Symlink to /disk/coconut/junshi/data
        └── vitalive/
            └── prod/
```

The real shared data storage path is:

```text
/disk/coconut/junshi/data
```

Expose it through the stable home-directory path:

```bash
ln -s /disk/coconut/junshi/data /home/junshi/data
```

After that, `~/data` resolves through the symlink and remains the canonical
path that production scripts should use.

Within each dataset folder, keep:

- `docs/` for papers, dataset documentation, and download notes
- `metadata/` for metadata tables
- `zip/` for downloaded archives
- `raw/` for extracted raw files
- `processed/` for processed outputs grouped by method or version

Keep raw datasets, large checkpoints, and generated caches out of git unless the
user explicitly decides otherwise.

## Recommended Workflow

### 1. Read The Repository Context First

Before editing code or docs, read `README.md`, this guide, and the relevant
module documentation under `scripts/`.

### 2. Keep Dataset Roots Explicit

Production scripts should not assume datasets live inside `prod/`.

Use explicit paths, arguments, or configuration that point to the shared data
root under `~/data/`, which should resolve to `/disk/coconut/junshi/data` via
the `/home/junshi/data` symlink.

### 3. Keep Reusable Logic In `scripts/`

Do not leave important reusable logic only in notebooks or shell history.
Extract repeated data loading, model code, loss functions, metrics, and
evaluation helpers into `scripts/`.

### 4. Keep Model Artifacts Under `model/`

Use `model/` for checkpoints, run outputs, exports, and other saved artifacts.

### 5. Keep Environment Setup Reproducible

Use a Python virtual environment named `.vitalive`.

```bash
python3.11 -m venv .vitalive
source .vitalive/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If the server does not provide `python3.11`, install Python 3.11 first or
update `requirements.txt` together with the documented runtime target.

## Gold Practices

- Treat the shared data root as a stable contract, not a temporary convenience.
- Keep dataset roots, checkpoint roots, and output roots explicit.
- Prefer small, reversible changes over broad refactors inside the production repo.
- Keep production documentation aligned with the actual server layout.
- When a workflow stabilizes, encode it in scripts or docs instead of relying on chat history.

## Coding Requirements

### Python Modules

- Use `snake_case` for functions, variables, modules, and local files.
- Public functions and classes should have docstrings that explain inputs,
  outputs, assumptions, and important side effects.
- Keep model/data contracts explicit with shape comments or validation where
  mistakes are likely.
- Avoid broad refactors unless they are necessary for the task.

### Module Markdown Documents

Every Python file under these folders must have a same-name Markdown document:

```text
scripts/model/foo.py  -> scripts/model/foo.md
scripts/blocks/foo.py -> scripts/blocks/foo.md
scripts/codec/foo.py  -> scripts/codec/foo.md
scripts/train/foo.py  -> scripts/train/foo.md
scripts/eval/foo.py   -> scripts/eval/foo.md
```

This is mandatory for:

```text
blocks/
model/
codec/
train/
eval/
```

It is strongly recommended for:

```text
data/
utils/
```

### Reproducibility

- Use deterministic seeds where practical.
- Make data roots, checkpoint paths, and output paths explicit.
- Save or display enough metadata to reproduce important runs.
- Do not add fallback behavior that hides broken paths or missing checkpoints
  unless the user explicitly asks for fallback.

### Data And Checkpoints

- Do not commit raw datasets or large generated artifacts.
- Treat `~/data/` as the shared data root.
- On the server, make `/home/junshi/data` a symlink to `/disk/coconut/junshi/data`.
- Keep checkpoint directories structured by project, model family, and run when possible.
- When changing checkpoint loading, prefer explicit failure over silent fallback.

## Good Practices For Agents

- Start by reading the relevant code and current docs before making changes.
- Use `rg` or `rg --files` for searching.
- Keep summaries concrete: changed files, verification performed, and residual risk.
- Do not revert user changes in a dirty worktree.
- Before final response, verify that path changes landed where expected.
- If a task is only about documentation or environment setup, do not touch unrelated model code.
