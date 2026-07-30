# Utilities

`utils/` owns small dataset-independent and model-independent helpers.

Examples:

- path helpers
- seed helpers
- registry helpers
- JSON helpers
- tensor shape checks
- checkpoint metadata helpers

If a helper grows into dataset, model, training, or evaluation logic, move it to
the owning folder.
