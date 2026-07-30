# VitalSense Baseline Models

## Purpose

Define compact supervised VitalSense radar-to-vital baseline model classes.

## Inputs

- Radar tensor shaped `[B, 1, T]`.
- Constructor values for input channels, hidden dimension, output length or
  output dimension, dropout, and optional CNN scale/channel plans.

## Outputs

- `RadarToHRRRCNN`: normalized HR/RR sequence predictions shaped
  `[B, 2, output_length]`.
- `RadarToBPCNN`: normalized BPS/BPM/BPD scalar predictions shaped `[B, 3]`.

## Public API

- `RadarToHRRRCNN`
- `RadarToBPCNN`

## Dependencies

- `torch`
- `torch.nn`
- `blocks.cnn`
- `blocks.projection`

## Used By

- VitalSense prediction notebook.
- `model.__init__` for convenient model imports.

## Responsibilities And Boundaries

This module owns only the supervised baseline model definitions.

It does not own:

- Dataset wrappers, which belong in `data.vitalsense.dataset`.
- Training loops, which belong in `train.regression`.
- Prediction dataframe collection and metrics, which belong in
  `eval.vitalsense.metrics`.
- Plotting, which belongs in `eval.vitalsense.plots`.
- Radar-to-token transfer, which is handled in `model.vitaldb_vitalsense`.

## Architecture

HR/RR sequence baseline:

```text
radar [B,1,T]
  -> blocks.cnn.build_downsample(...)
  -> AdaptiveAvgPool1d(output_length)
  -> blocks.projection.build_cnn_projection(...)
  -> [B,2,output_length]
```

BP scalar baseline:

```text
radar [B,1,T]
  -> blocks.cnn.build_downsample(...)
  -> AdaptiveAvgPool1d(1)
  -> blocks.projection.build_linear_projection(..., flatten=True)
  -> [B,3]
```

## Tensor Contract

For `B = 4`, `T = 40000`, and `hidden_dim = 64`:

```text
x:                         [4, 1, 40000]
CNN stem/downsample:        [4, 64, 10000]
HR/RR adaptive pool:        [4, 64, 120]
HR/RR sequence head:        [4, 2, 120]

x:                         [4, 1, 40000]
CNN stem/downsample:        [4, 64, 10000]
BP adaptive pool:           [4, 64, 1]
linear projection head:     [4, 3]
```

## Config Contract

- `input_channels` must match radar input channel count.
- `output_length` should match HR/RR target length.
- `output_dim` should match scalar BP target count.
- `scale_plan` controls CNN temporal scaling. The default plan is
  `((stem_dim, 1), (mid_dim, 2), (hidden_dim, 2))`.
- `channel_plan` optionally adds same-length channel blocks after the scale
  plan. The default is empty.

## Step-by-step Tensor Flow

1. Validate input rank `[B, C, T]`.
2. Build temporal features with the shared Conv1d block.
3. Pool to the target temporal length or scalar summary length.
4. Apply a lightweight head for sequence or scalar output.

## Debug Checklist

- Print model parameter count relative to VitalSense train size.
- Confirm HR/RR target shape is `[B, 2, output_length]`.
- Confirm BP target shape is `[B, 3]`.
- Compare train/val/test loss to detect overfit.

## Sample Case

```python
model = RadarToHRRRCNN(output_length=120)
y_pred = model(torch.randn(2, 1, 40000))
y_pred.shape  # [2, 2, 120]
```

## Failure Modes

- Forward raises if input rank is not `[B, C, T]`.
- Incorrect output length causes loss-shape mismatch in the training loop.
