"""Depth-warp pixels from one camera to another given known depth in the
target frame. Used to propagate sparse-view inpaints to non-anchor frames
with cross-view consistency that per-frame inpainting cannot give us.

Algorithm:
  1. For each pixel (u_t, v_t) in target, lift to 3D using target depth + K.
  2. Transform 3D point from target camera frame to anchor camera frame.
  3. Project into anchor view to get (u_a, v_a).
  4. Bilinear-sample anchor RGB at (u_a, v_a).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def warp_anchor_to_target(
    anchor_rgb: torch.Tensor,    # (3, H, W) float in [0, 1]
    target_depth: torch.Tensor,  # (H, W) float, depth in target view
    K_anchor: torch.Tensor,      # (3, 3)
    K_target: torch.Tensor,      # (3, 3)
    w2c_anchor: torch.Tensor,    # (4, 4)
    w2c_target: torch.Tensor,    # (4, 4)
) -> torch.Tensor:
    """Warp anchor RGB into target frame. Returns (3, H, W). Out-of-bounds
    samples are zero (caller can detect with a coverage mask if needed).
    """
    device = anchor_rgb.device
    _, H, W = anchor_rgb.shape

    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(us)
    pix_t = torch.stack([us, vs, ones], dim=0)  # (3, H, W)

    K_t_inv = torch.linalg.inv(K_target.to(device))
    rays = K_t_inv @ pix_t.reshape(3, -1)  # (3, H*W)
    pts_target_cam = rays * target_depth.to(device).reshape(1, -1)  # (3, H*W)

    R_t = w2c_target[:3, :3].to(device)
    t_t = w2c_target[:3, 3].to(device).unsqueeze(1)
    pts_world = R_t.T @ (pts_target_cam - t_t)  # (3, H*W)

    R_a = w2c_anchor[:3, :3].to(device)
    t_a = w2c_anchor[:3, 3].to(device).unsqueeze(1)
    pts_anchor_cam = R_a @ pts_world + t_a  # (3, H*W)

    z_a = pts_anchor_cam[2:3, :].clamp_min(1e-6)
    proj_a = K_anchor.to(device) @ pts_anchor_cam
    u_a = (proj_a[0:1, :] / z_a).reshape(H, W)
    v_a = (proj_a[1:2, :] / z_a).reshape(H, W)

    u_n = (u_a / (W - 1)) * 2 - 1
    v_n = (v_a / (H - 1)) * 2 - 1
    grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    sampled = F.grid_sample(
        anchor_rgb.unsqueeze(0), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (1, 3, H, W)
    return sampled.squeeze(0)
