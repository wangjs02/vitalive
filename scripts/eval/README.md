# Evaluation Modules

`eval/` owns evaluation, probing, and diagnostic workflows.

Expected subfolders:

- `pretrain/`
- `vitalsense/`
- `diagnostics/`

Top-level modules:

- `vitaldb.py`: save flattened VitalDB training history, loss plots, and
  quantizer diagnostic plots.

Every `.py` file in this folder tree requires a same-name `.md` document.
