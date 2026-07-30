# Data Modules

`data/` owns dataset-specific loading, preprocessing, alignment, transforms, and
dataset classes.

Each dataset should have its own subpackage. Shared dataset-independent helpers
belong in `utils/`, not here.

Expected dataset folders:

- `vitaldb/`
- `vitalsense/`
- `cohface/`
- `guardian/`
- `vipl_hr/`
