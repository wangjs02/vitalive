# DeepPhys Model

## Purpose

`deepphys.py` implements a VGG-style DeepPhys model for video-based pulse
prediction. It reconstructs the two-stream architecture shown in the DeepPhys
diagram: a motion branch processes normalized frame differences, and an
appearance branch produces attention masks from the reference frame.

## Inputs

- Motion tensor: `[B, 3, 36, 36]`, usually
  `(C(t+1) - C(t)) / (C(t+1) + C(t))`.
- Appearance tensor: `[B, 3, 36, 36]`, usually the frame `C(t)`.
- Paired frame convenience input: `[B, 2, 3, 36, 36]`. When `appearance=None`,
  `DeepPhys.forward` treats axis 1 as `[C(t), C(t+1)]` and builds the motion
  tensor internally.

Inputs should already be resized to `36x36` unless `frame_size` is changed in
the constructor. The convenience frame-difference formula assumes non-negative
frame intensities, such as raw `[0, 1]` or `[0, 255]` images.

## Outputs

- Pulse difference prediction shaped `[B, 1]`, corresponding to
  `p(t+1) - p(t)` for each input frame pair.

## Public API

- `DeepPhys`
- `normalized_frame_difference`

## Dependencies

- External: `torch`, `torch.nn`
- Internal: N/A

## Used By

- Future VIPL-HR and COHFACE video prediction notebooks.
- Future training scripts for image-pair pulse prediction.

## Responsibilities

This file owns only the DeepPhys model definition and the small tensor helper
needed to build motion input from frame pairs.

## Boundaries

This file does not own:

- VIPL-HR or COHFACE dataset loading.
- Video resizing or normalization transforms.
- Sequence window creation from full videos.
- Training loops, losses, metrics, or plotting.

## Architecture

```text
motion [B,3,36,36]
  -> conv 3x3, 32, tanh
  -> conv 3x3, 32, tanh
  -> multiply appearance mask 1 [B,32,36,36]
  -> avg pool 2x2
  -> dropout
  -> conv 3x3, 64, tanh
  -> conv 3x3, 64, tanh
  -> multiply appearance mask 2 [B,64,18,18]
  -> avg pool 2x2
  -> dropout
  -> flatten [B,64*9*9]
  -> linear hidden_dim
  -> tanh
  -> dropout
  -> linear 1
  -> [B,1]

appearance [B,3,36,36]
  -> conv 3x3, 32, tanh
  -> conv 3x3, 32, tanh
  -> conv 1x1, 32 + sigmoid + L1 spatial normalization = mask 1
  -> avg pool 2x2
  -> dropout
  -> conv 3x3, 64, tanh
  -> conv 3x3, 64, tanh
  -> conv 1x1, 64 + sigmoid + L1 spatial normalization = mask 2
```

## Tensor Contract

Default config:

```text
B = 2
input_channels = 3
frame_size = 36
hidden_dim = 128

motion:                  [2, 3, 36, 36]
appearance:              [2, 3, 36, 36]
mask 1:                  [2, 32, 36, 36]
motion after pool 1:     [2, 32, 18, 18]
mask 2:                  [2, 64, 18, 18]
motion after pool 2:     [2, 64, 9, 9]
flatten:                 [2, 5184]
output:                  [2, 1]
```

Convenience input:

```text
frames:                  [2, 2, 3, 36, 36]
previous = frames[:, 0]: [2, 3, 36, 36]
current = frames[:, 1]:  [2, 3, 36, 36]
motion:                  [2, 3, 36, 36]
output:                  [2, 1]
```

## Config Contract

- `input_channels` must match the channel dimension of motion and appearance.
- `frame_size` must match both height and width.
- `frame_size` must be at least 4 because the model applies two `2x2` pooling
  operations.
- `hidden_dim` controls the fully connected layer size. Use `128` to match the
  larger diagram variant, or `32` for the compact variant.
- `dropout` is applied after each pooling layer and inside the prediction head.

## Step-by-step Tensor Flow

1. Validate input rank and shape.
2. If only paired frames are provided, compute normalized frame difference and
   use the first frame as appearance.
3. Build two L1-normalized appearance masks.
4. Apply the first mask before the first motion pooling stage.
5. Apply the second mask before the second motion pooling stage.
6. Flatten the final motion feature map and predict one scalar pulse
   difference.

## Debug Checklist

- Confirm videos are converted from channel-last `[B, T, H, W, C]` to
  channel-first frame pairs `[B, 2, C, H, W]` before calling the model.
- Confirm the input frame size matches `frame_size`.
- Confirm frame-difference inputs are finite; zero-valued paired frames can make
  the denominator small.
- Confirm the target is a per-pair pulse difference `[B, 1]`, not a full
  sequence target.
- Use `hidden_dim=32` if the model overfits small datasets.

## Sample Case

```python
import torch
from model.deepphys import DeepPhys

model = DeepPhys(hidden_dim=128)
frames = torch.rand(2, 2, 3, 36, 36)
y = model(frames)
y.shape  # torch.Size([2, 1])
```

Direct branch input:

```python
motion = torch.randn(2, 3, 36, 36)
appearance = torch.rand(2, 3, 36, 36)
y = model(motion, appearance)
```

## Failure Modes

- Raises `ValueError` if input rank is not `[B, 3, H, W]` or
  `[B, 2, 3, H, W]`.
- Raises `ValueError` if spatial size does not match `frame_size`.
- Raises `ValueError` if motion and appearance shapes differ.
- Loss-shape mismatch occurs if training targets are full sequences instead of
  per-pair pulse differences.
