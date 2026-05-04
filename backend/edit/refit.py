"""Local refit after Gaussian deletion + inpaint.

Strategy:
  1. Load source ckpt, build GaussianModel.
  2. Drop Gaussians flagged by classifier.
  3. Freeze gradients on all surviving Gaussians outside the affected zone
     (zone defined as a sphere of radius 1.5 * max_deleted_extent around the
     centroid of deleted Gaussians).
  4. Run training loop with inpainted views as new GT for n_iters.
     - RGB + SSIM + LPIPS-512 normal
     - Depth supervision masked off inside the SAM mask
     - Densify only allowed for Gaussians inside the affected zone
  5. Save new ckpt + PLY.

NOTE: Full body wired in Task C2 (which also adds the trainer-side hooks
this needs: filter_in_place, set_freeze_mask, edit_mask_stack support).
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import torch


def compute_affected_zone(
    means: torch.Tensor,            # (N, 3) all gaussians (incl. flagged)
    delete_flags: torch.Tensor,     # (N,) bool
    multiplier: float = 1.5,
) -> tuple[torch.Tensor, float]:
    """Return (centroid: (3,), radius: float) defining the affected zone.

    The radius is multiplier × max distance from the deletion centroid to any
    deleted Gaussian. This bounds the region where surviving Gaussians may be
    re-trained / densified after deletion + inpaint.
    """
    if delete_flags.sum() == 0:
        return torch.zeros(3, device=means.device), 0.0
    deleted = means[delete_flags]
    centroid = deleted.mean(dim=0)
    max_extent = (deleted - centroid).norm(dim=1).max().item()
    radius = multiplier * max_extent
    return centroid, radius


def is_in_zone(
    means: torch.Tensor,            # (N, 3)
    centroid: torch.Tensor,         # (3,)
    radius: float,
) -> torch.Tensor:
    """Bool[N] — True for Gaussians within the affected zone."""
    if radius == 0.0:
        return torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    return (means - centroid).norm(dim=1) <= radius


def run_refit(
    source_ckpt_path: Path,
    inpainted_frames_dir: Path,
    inpainted_mask_stack: torch.Tensor,  # (T, H, W) bool — for depth-loss masking
    delete_flags: torch.Tensor,          # (N,) bool from classifier
    output_ckpt_dir: Path,
    n_iters: int = 5000,
    cancel_check: Callable[[], bool] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Path:
    """Run local refit, save new ckpt to output_ckpt_dir/ckpt.pt. Returns path.

    Wired in Task C2 — depends on trainer-side changes (filter_in_place,
    set_freeze_mask, edit_mask_stack parameter on Trainer4DGS.train()).
    """
    raise NotImplementedError(
        "run_refit body wired in Task C2; this is the C1 skeleton. "
        "C2 adds: GaussianModel.filter_in_place, GaussianModel.set_freeze_mask, "
        "and Trainer4DGS.train(edit_mask_stack=...) support."
    )
