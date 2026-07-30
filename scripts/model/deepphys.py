from __future__ import annotations

import torch
from torch import nn


class DeepPhys(nn.Module):
    """VGG-style DeepPhys model for video-based pulse prediction.

    The model follows the two-branch structure from DeepPhys: a motion branch
    processes normalized frame differences, while an appearance branch generates
    attention masks from the reference frame. The forward pass predicts one
    scalar pulse difference for each input pair.
    """

    def __init__(
        self,
        input_channels: int = 3,
        frame_size: int = 36,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.input_channels = int(input_channels)
        self.frame_size = int(frame_size)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)

        self.motion_conv1 = self._conv_block(input_channels, 32)
        self.motion_conv2 = self._conv_block(32, 32)
        self.motion_conv3 = self._conv_block(32, 64)
        self.motion_conv4 = self._conv_block(64, 64)

        self.appearance_conv1 = self._conv_block(input_channels, 32)
        self.appearance_conv2 = self._conv_block(32, 32)
        self.appearance_mask1 = nn.Conv2d(32, 32, kernel_size=1)
        self.appearance_conv3 = self._conv_block(32, 64)
        self.appearance_conv4 = self._conv_block(64, 64)
        self.appearance_mask2 = nn.Conv2d(64, 64, kernel_size=1)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        pooled_size = frame_size // 4
        if pooled_size <= 0:
            raise ValueError("frame_size must be at least 4 for two pooling layers.")
        flattened_dim = 64 * pooled_size * pooled_size
        self.head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build one VGG-style 3x3 convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        motion: torch.Tensor,
        appearance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict pulse difference from motion and appearance tensors.

        Parameters
        ----------
        motion:
            Either normalized frame difference `[B, C, H, W]`, or paired frames
            `[B, 2, C, H, W]` when `appearance` is omitted.
        appearance:
            Reference appearance frame `[B, C, H, W]`. If omitted, `motion` must
            contain paired frames and this tensor is taken as the first frame.

        Returns
        -------
        torch.Tensor
            Pulse difference prediction shaped `[B, 1]`.
        """
        if appearance is None:
            motion, appearance = self.prepare_motion_and_appearance(motion)
        self._validate_image_tensor(motion, "motion")
        self._validate_image_tensor(appearance, "appearance")
        if motion.shape != appearance.shape:
            raise ValueError(
                "motion and appearance must have the same shape, got "
                f"{tuple(motion.shape)} and {tuple(appearance.shape)}."
            )

        mask1, mask2 = self.appearance_model(appearance)
        z = self.motion_model(motion, mask1=mask1, mask2=mask2)
        return self.head(z.flatten(start_dim=1))

    def motion_model(
        self,
        motion: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        """Run the motion branch with appearance-guided attention masks."""
        x = self.motion_conv1(motion)
        x = self.motion_conv2(x)
        x = x * mask1
        x = self.pool(x)
        x = self.dropout(x)

        x = self.motion_conv3(x)
        x = self.motion_conv4(x)
        x = x * mask2
        x = self.pool(x)
        return self.dropout(x)

    def appearance_model(self, appearance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the appearance branch and return L1-normalized attention masks."""
        x = self.appearance_conv1(appearance)
        x = self.appearance_conv2(x)
        mask1 = self.normalize_attention_mask(torch.sigmoid(self.appearance_mask1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.appearance_conv3(x)
        x = self.appearance_conv4(x)
        mask2 = self.normalize_attention_mask(torch.sigmoid(self.appearance_mask2(x)))
        return mask1, mask2

    def normalize_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """L1-normalize attention so each sample/channel keeps mean weight near one."""
        if mask.dim() != 4:
            raise ValueError(f"Expected attention mask [B, C, H, W], got {tuple(mask.shape)}.")
        spatial_sum = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        scale = mask.shape[2] * mask.shape[3]
        return mask / spatial_sum * scale

    def prepare_motion_and_appearance(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build DeepPhys inputs from paired frames `[B, 2, C, H, W]`."""
        if frames.dim() != 5:
            raise ValueError(f"Expected paired frames [B, 2, C, H, W], got {tuple(frames.shape)}.")
        if frames.shape[1] != 2:
            raise ValueError(f"Expected exactly two frames on axis 1, got {frames.shape[1]}.")
        previous = frames[:, 0]
        current = frames[:, 1]
        motion = normalized_frame_difference(previous, current, eps=self.eps)
        return motion, previous

    def _validate_image_tensor(self, x: torch.Tensor, name: str) -> None:
        if x.dim() != 4:
            raise ValueError(f"Expected {name} [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {name} channel count C={self.input_channels}, got C={x.shape[1]}."
            )
        if x.shape[2] != self.frame_size or x.shape[3] != self.frame_size:
            raise ValueError(
                f"Expected {name} spatial size {self.frame_size}x{self.frame_size}, "
                f"got {x.shape[2]}x{x.shape[3]}."
            )


def normalized_frame_difference(
    previous: torch.Tensor,
    current: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute `(current - previous) / (current + previous)` for frame pairs."""
    if previous.shape != current.shape:
        raise ValueError(
            "previous and current frames must have the same shape, got "
            f"{tuple(previous.shape)} and {tuple(current.shape)}."
        )
    return (current - previous) / (current + previous).clamp_min(eps)


__all__ = [
    "DeepPhys",
    "normalized_frame_difference",
]
