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


def normalize_seed_depth(
    depth: np.ndarray,
    target_near: float = 0.5,
    target_far: float = 5.0,
    clip_low_pct: float = 2.0,
    clip_high_pct: float = 98.0,
) -> np.ndarray:
    """Rescale MiDaS-style scale-invariant depth into a usable metric range.

    MiDaS produces inverse depth with arbitrary scale; the upstream
    `estimate_depth()` converts it via `1/x` with only a max-depth clamp,
    leaving a few near pixels at ~0.001 units and far pixels at 100+ units.
    Deprojecting that produces a vertical "plume" (most points clumped near
    the camera origin, a few stretched to infinity).

    This helper clips outliers via percentiles and linear-rescales the kept
    range to [target_near, target_far] metres — a plausible indoor-room scale.
    The depth ORDERING is preserved; only the scale changes.
    """
    valid = depth > 0
    if not valid.any():
        return depth.astype(np.float32)
    vals = depth[valid]
    lo = float(np.percentile(vals, clip_low_pct))
    hi = float(np.percentile(vals, clip_high_pct))
    if hi - lo < 1e-6:
        return np.full_like(depth, (target_near + target_far) / 2.0, dtype=np.float32)
    clipped = np.clip(depth, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    scaled = target_near + norm * (target_far - target_near)
    scaled[~valid] = 0.0
    return scaled.astype(np.float32)


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


from backend.model.gaussian_model import GaussianModel


def init_gaussian_model_from_seed(
    points: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int = 0,
    fourier_K: int = 0,
    max_points: int = 1_500_000,
) -> GaussianModel:
    """Build an initial GaussianModel from a seed point cloud.

    For sub-project B static_mode, fourier_K=0 (no motion). sh_degree=0 keeps
    parameter count down; the trainer will progressively unlock higher SH if needed.

    Subsamples uniformly at random if input exceeds max_points (8 GB VRAM budget).
    """
    if points.shape[0] == 0:
        raise ValueError("init_gaussian_model_from_seed: empty point cloud")

    N = points.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        points = points[idx]
        colors = colors[idx]

    return GaussianModel(
        init_points=points,
        init_colors=colors,
        sh_degree=sh_degree,
        fourier_K=fourier_K,
    )
