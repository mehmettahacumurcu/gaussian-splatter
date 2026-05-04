"""Classify Gaussians for deletion via projection vote.

A Gaussian is flagged for deletion if, across all training views where it
projects inside the frame, the mask is True at its projected pixel for
>= threshold fraction of those views. Threshold default 0.6 accounts for
slight SAM mask edge bleed without being over-permissive.
"""
from __future__ import annotations
from typing import Sequence

import torch


@torch.no_grad()
def classify_gaussians_for_deletion(
    means: torch.Tensor,            # (N, 3) world coords
    K: torch.Tensor,                # (3, 3) intrinsics at the mask resolution
    w2cs: Sequence[torch.Tensor],   # T of (4, 4)
    masks: torch.Tensor,            # (T, H, W) bool
    threshold: float = 0.6,
    image_size: tuple[int, int] | None = None,  # (W, H)
) -> torch.Tensor:
    """Return bool[N] — True for Gaussians that should be deleted.

    Algorithm:
      1. For each Gaussian and each view, project center to pixel coords.
      2. Skip views where projection lands outside the frame OR behind camera.
      3. Of remaining views, count fraction where the mask pixel is True.
      4. Flag for deletion if fraction >= threshold AND #valid_views >= 1.
    """
    if means.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=means.device)

    N = means.shape[0]
    if image_size is None:
        H, W = masks.shape[1], masks.shape[2]
    else:
        W, H = image_size

    device = means.device
    K_d = K.to(device)
    masks_d = masks.to(device)

    valid_view_count = torch.zeros(N, dtype=torch.long, device=device)
    in_mask_count = torch.zeros(N, dtype=torch.long, device=device)

    homog = torch.cat([means, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)

    for t, w2c in enumerate(w2cs):
        cam_pts = (w2c.to(device) @ homog.T).T[:, :3]  # (N, 3)
        z = cam_pts[:, 2]
        in_front = z > 1e-3

        proj = (K_d @ cam_pts.T).T  # (N, 3)
        u = proj[:, 0] / proj[:, 2].clamp_min(1e-3)
        v = proj[:, 1] / proj[:, 2].clamp_min(1e-3)
        u_int = u.round().long()
        v_int = v.round().long()
        in_frame = (
            in_front
            & (u_int >= 0) & (u_int < W)
            & (v_int >= 0) & (v_int < H)
        )

        valid_view_count = valid_view_count + in_frame.long()

        if in_frame.any():
            u_safe = u_int.clamp(0, W - 1)
            v_safe = v_int.clamp(0, H - 1)
            mask_t = masks_d[t][v_safe, u_safe]  # (N,)
            in_mask_count = in_mask_count + (mask_t & in_frame).long()

    fraction = in_mask_count.float() / valid_view_count.clamp_min(1).float()
    flags = (fraction >= threshold) & (valid_view_count > 0)
    return flags
