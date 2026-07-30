# VitalDB Visualization

## Purpose

Provide compact plotting and summary helpers for loaded VitalDB cases and
accepted segment ID lists.

## Inputs

- `VitalDBDataset` with loaded raw frames and accepted segments.
- Optional case id, role selection, time range, and figure size.

## Outputs

- Matplotlib figures and axes.
- Segment count summary data frame.

## Public API

- `plot_case`
- `plot_datapoint`
- `plot_segment`
- `plot_segment_counts`
- `segment_summary`

## Dependencies

- `pandas`
- `matplotlib` at call time
- `data.vitaldb.dataset`

## Responsibilities And Boundaries

This module owns lightweight VitalDB exploratory plots and segment summaries.
It does not own dataset splitting, model diagnostics, training curves, or
checkpoint metadata.

## Used By

- VitalDB preparation notebooks
- pretrain diagnostic notebooks

## Sample Case

```python
from data.vitaldb.visualize import plot_datapoint, plot_segment

# Plot one original 60-second clean segment.
fig, axes = plot_segment(dataset, segment_index=0)

# Plot one assembled datapoint through Dataset.__getitem__. For a dataset with
# time_length=600, this plots the full sliding window and applied transforms.
fig, axes = plot_datapoint(dataset, datapoint_index=10)

summary = segment_summary(dataset)
```

`plot_datapoint` builds its time axis from `dataset.time_length`, so both a
stored `[C, 300]` sample at 0.5 Hz and a transformed `[C, 600]` sample at 1 Hz
span 600 seconds. By default, vertical lines mark the underlying 60-second
segment boundaries; pass `show_segment_boundaries=False` to hide them.

## Failure Modes

- No raw data loaded.
- Segment ID list is empty.
- Requested role or case id is absent.
- A datapoint is not shaped `[C, T]` or has an empty time axis.
