"""Lightweight PEFT helpers for rail depth-domain adaptation."""

from .depth_affine import DepthAffinePEFT, DepthEncoderWithPEFT
from .feature_stats import compute_depth_feature_stats, fdm_loss

__all__ = [
    "DepthAffinePEFT",
    "DepthEncoderWithPEFT",
    "compute_depth_feature_stats",
    "fdm_loss",
]
