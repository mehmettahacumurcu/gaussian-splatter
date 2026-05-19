"""Seed step: depth + deproject + initialize GaussianModel from a single image.

Spec §3 steps [a]-[c].

Camera convention:
  - +X right, +Y up, +Z forward (right-handed)
  - Pixel (0, 0) is top-left; row index increases downward
  - Depth values are positive distance along +Z
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .intrinsics import CameraIntrinsics


def deproject_depth(
    depth: np.ndarray,
    K: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (H, W) depth map into (H*W, 3) camera-space points.

    Returns (points, valid_mask) where:
      - points: (H*W, 3) float32, NaN for invalid pixels
      - valid_mask: (H*W,) bool, True where depth > 0
    """
    H, W = depth.shape
    assert H == K.height and W == K.width, (
        f"depth shape {depth.shape} disagrees with intrinsics ({K.height}, {K.width})"
    )

    u = np.arange(W, dtype=np.float32).reshape(1, W).repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32).reshape(H, 1).repeat(W, axis=1)

    z = depth.astype(np.float32)
    x = (u - K.cx) * z / K.fx
    y = -(v - K.cy) * z / K.fy  # +Y up convention

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (z > 0).reshape(-1)
    points[~valid] = np.nan
    return points, valid


def image_to_pointcloud(
    image_path: Path,
    depth: np.ndarray,
    K: CameraIntrinsics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deproject all valid pixels of an image into (points, colors)."""
    points_np, valid = deproject_depth(depth, K)
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rgb_flat = rgb.reshape(-1, 3)

    points_np = points_np[valid]
    colors_np = rgb_flat[valid]
    points = torch.from_numpy(points_np).float()
    colors = torch.from_numpy(colors_np).float()
    return points, colors
