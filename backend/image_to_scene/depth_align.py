"""Depth alignment for the outpaint loop.

MiDaS depth is scale-invariant inverse depth. To merge a new view's depth
into the existing 3D scene, we fit a per-view (scale, shift) so the new depth
matches the existing scene depth at the visible-from-camera overlap pixels.

Mathematically: minimize ||s · source + b - target||² over (s, b) ∈ R²,
which is a 2-variable linear least squares.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    scale: float
    shift: float
    residual: float
    n_samples: int
    rejected: bool


def solve_scale_shift(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    """Fit s, b so that s*source + b ≈ target. Returns (scale, shift, residual_rms).

    Both inputs are flat 1-D arrays of the same length.
    Residual is RMS in target units, normalized by target range.
    """
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if source.size < 3:
        raise ValueError("need at least 3 samples for least-squares scale+shift")

    A = np.stack([source.astype(np.float64), np.ones_like(source, dtype=np.float64)], axis=1)
    b = target.astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    scale, shift = float(sol[0]), float(sol[1])

    residual_abs = float(np.sqrt(np.mean((scale * source + shift - target) ** 2)))
    target_range = float(np.ptp(target)) or 1.0
    residual_rel = residual_abs / target_range
    return scale, shift, residual_rel


def apply_scale_shift(depth: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """Apply the aligned (scale, shift) and clamp negative values to 0."""
    return np.maximum(scale * depth + shift, 0.0)


def align_new_view(
    new_depth: np.ndarray,
    rendered_depth: np.ndarray,
    overlap_mask: np.ndarray,
    *,
    min_overlap_pixels: int = 500,
    reject_threshold: float = 0.25,
) -> AlignmentResult:
    """Fit alignment using only overlap pixels.

    overlap_mask is a (H, W) bool array — True where rendered_depth is valid AND
    new_depth is valid. Below min_overlap_pixels the view is rejected.
    """
    overlap_idx = np.where(overlap_mask.flatten())[0]
    if overlap_idx.size < min_overlap_pixels:
        return AlignmentResult(scale=1.0, shift=0.0, residual=float("inf"),
                               n_samples=int(overlap_idx.size), rejected=True)

    source = new_depth.flatten()[overlap_idx]
    target = rendered_depth.flatten()[overlap_idx]
    scale, shift, residual = solve_scale_shift(source, target)

    rejected = residual > reject_threshold
    return AlignmentResult(scale=scale, shift=shift, residual=residual,
                           n_samples=int(overlap_idx.size), rejected=rejected)
