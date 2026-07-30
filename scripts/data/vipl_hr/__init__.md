# VIPL-HR Data Package Init

## Purpose

Expose the public VIPL-HR training API: preprocessing transforms and the lazy
PyTorch dataset wrapper.

The package entrypoint intentionally does not expose low-level file/path
helpers such as `read_data_by_id`, `read_video_file`, or `get_all_data`. Those
remain implementation/debug helpers inside `dataset.py`.

## Public API

- `ComposeVIPLHR`
- `ResampleVIPLHR`
- `ClipAndPadVIPLHR`
- `ResizeVIPLHR`
- `NormalizeVIPLHR`
- `VIPLHRDataset`
