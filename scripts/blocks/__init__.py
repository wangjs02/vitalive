"""Reusable neural-network building blocks."""

from .mlp import MLPBlock, build_mlp
from .presets import DEPTH_PRESETS, ModelSize, resolve_depth
from .projection import build_linear_projection
from .quantizer import SequenceEMAQuantize, VQVAEQuantize, codebook_perplexity
from .resnet import (
    ResidualBlock1D,
    ResidualBlock2D,
    build_residual_stack_1d,
    build_residual_stack_2d,
)
from .transformer import (
    AttentionBlock,
    CrossAttentionBlock,
    PatchTokenizer1D,
    build_patch_tokenizer_1d,
    build_transformer_blocks,
    interpolate_token_positions,
)

__all__ = [
    "AttentionBlock",
    "CrossAttentionBlock",
    "DEPTH_PRESETS",
    "MLPBlock",
    "ModelSize",
    "PatchTokenizer1D",
    "ResidualBlock1D",
    "ResidualBlock2D",
    "SequenceEMAQuantize",
    "VQVAEQuantize",
    "build_linear_projection",
    "build_mlp",
    "build_patch_tokenizer_1d",
    "build_residual_stack_1d",
    "build_residual_stack_2d",
    "build_transformer_blocks",
    "codebook_perplexity",
    "interpolate_token_positions",
    "resolve_depth",
]
