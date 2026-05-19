"""Outpaint loop — spec §3 steps [e]-[f].

Per iteration over a generated camera trajectory:
  1. Render current 3DGS from pose Cᵢ -> (rgb, alpha, depth)
  2. If alpha has uncovered regions, run SD-inpaint on (rgb, ~alpha) -> Iᵢ_completed
  3. Re-estimate depth on Iᵢ_completed
  4. Align new depth to rendered depth on the alpha-covered overlap
  5. Deproject newly-uncovered pixels into world space; append as fresh Gaussians

VRAM strategy:
  - MiDaS stays resident across iterations (it is ~1.5 GB).
  - SDInpainter .load() before the loop, .unload() after.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch

__all__ = [
    "render_pose",
    "visibility_mask_from_alpha",
    "inpaint_with_sd",
    "add_gaussians_from_pixels",
    "OutpaintLoopStats",
    "run_outpaint_loop",
]

from .config import ImageToSceneConfig
from .depth_align import align_new_view, apply_scale_shift, AlignmentResult
from .intrinsics import CameraIntrinsics
from .trajectory import CameraPose
from .seed import deproject_depth
from backend.model.gaussian_model import GaussianModel
from backend.model.renderer import render_view as _render_view_raw


def _w2c_from_pose(pose: CameraPose, device) -> torch.Tensor:
    """Build a (4, 4) world-to-camera matrix for the gsplat rasterizer.

    pose.rotation is world-from-camera (columns = camera axes in world).
    w2c = inverse(pose.as_world_from_camera()). For SE(3): R^T and -R^T t.
    """
    R = pose.rotation  # (3, 3) numpy
    t = pose.position  # (3,) numpy
    R_inv = R.T
    t_inv = -R_inv @ t
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_inv
    w2c[:3, 3] = t_inv
    return torch.from_numpy(w2c).float().to(device)


def render_pose(
    model: GaussianModel,
    pose: CameraPose,
    K: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the current 3DGS from `pose`. Returns (rgb HxWx3, alpha HxW, depth HxW)
    as float32 numpy arrays. Depth is the expected-depth output of gsplat (in world units).
    """
    device = next(model.parameters()).device
    w2c = _w2c_from_pose(pose, device)
    K_t = torch.from_numpy(K.as_matrix()).float().to(device)
    sh_colors = torch.cat([model.sh_dc, model.sh_rest], dim=1)  # (N, num_sh_total, 3)

    rgbd, alpha, _info = _render_view_raw(
        means=model.means,
        quats=model.get_quats,
        scales=model.get_scales,
        opacities=model.get_opacities.squeeze(-1),
        colors=sh_colors,
        K=K_t,
        w2c=w2c,
        width=K.width,
        height=K.height,
        sh_degree=model.sh_degree,
        bg_color=(0.0, 0.0, 0.0),
        with_depth=True,
    )
    # rgbd: (H, W, 4) — channels (R, G, B, expected_depth)
    rgb = rgbd[..., :3].clamp(0, 1).detach().cpu().numpy().astype(np.float32)
    depth = rgbd[..., 3].detach().cpu().numpy().astype(np.float32)
    alpha_arr = alpha.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    return rgb, alpha_arr, depth


def visibility_mask_from_alpha(
    alpha: np.ndarray,
    threshold: float = 0.5,
    dilate_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rendered alpha into (visible, to_fill) masks.

    A pixel is "visible" if alpha > threshold. Eroding visible by `dilate_px`
    grows the to_fill boundary so SD-inpaint produces softer seams.
    """
    visible = alpha > threshold
    if dilate_px > 0:
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        visible = binary_erosion(visible, structure=struct)
    to_fill = ~visible
    return visible, to_fill


from backend.edit.inpainter import InpainterBase


def inpaint_with_sd(
    inpainter: InpainterBase,
    rgb_uint8: np.ndarray,
    fill_mask: np.ndarray,
) -> np.ndarray:
    """Call the existing SDInpainter on the to-fill region. Returns RGB uint8.

    The mask convention matches the existing inpainter: 1 = inpaint here.
    """
    return inpainter.inpaint(rgb_uint8, fill_mask)


from torch import nn
from backend.model.gaussian_model import _rgb_to_sh_dc


def add_gaussians_from_pixels(
    model: GaussianModel,
    rgb_uint8: np.ndarray,           # (H, W, 3)
    depth_aligned: np.ndarray,       # (H, W) in world units
    new_pixel_mask: np.ndarray,      # (H, W) bool — True for pixels we want to add
    pose: CameraPose,
    K: CameraIntrinsics,
    max_new_fraction: float = 0.20,
) -> int:
    """Append new Gaussians sampled from the to-fill pixels.

    Returns the number of Gaussians actually added (may be 0 if mask is empty,
    or capped against max_new_fraction × H*W).
    """
    H, W, _ = rgb_uint8.shape
    flat_mask = new_pixel_mask.flatten().copy()
    n_pixels = int(flat_mask.sum())
    if n_pixels == 0:
        return 0

    cap = int(max_new_fraction * H * W)
    if n_pixels > cap:
        keep_idx = np.where(flat_mask)[0]
        chosen = np.random.default_rng().choice(keep_idx, size=cap, replace=False)
        flat_mask = np.zeros_like(flat_mask)
        flat_mask[chosen] = True
        n_pixels = cap

    # Deproject in camera space.
    points_cam, valid = deproject_depth(depth_aligned, K)
    selected = flat_mask & valid
    points_cam_sel = points_cam[selected]
    colors_sel = (rgb_uint8.astype(np.float32) / 255.0).reshape(-1, 3)[selected]
    M = points_cam_sel.shape[0]
    if M == 0:
        return 0

    # Transform camera-space points to world space using pose.
    R = pose.rotation
    t = pose.position
    points_world = (R @ points_cam_sel.T).T + t

    device = model.means.device
    dtype = model.means.dtype
    new_points = torch.from_numpy(points_world).to(device=device, dtype=dtype)
    new_colors = torch.from_numpy(colors_sel).to(device=device, dtype=dtype)

    with torch.no_grad():
        # means
        model.means = nn.Parameter(torch.cat([model.means.data, new_points], dim=0))
        # scales (log-space): seed from the median of existing log-scales.
        median_log_scale = torch.median(model.scales.data, dim=0).values  # (3,)
        new_scales = median_log_scale.unsqueeze(0).expand(M, -1).clone()
        model.scales = nn.Parameter(torch.cat([model.scales.data, new_scales], dim=0))
        # quats — identity (wxyz = 1, 0, 0, 0)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        new_quats = identity_quat.unsqueeze(0).expand(M, -1).clone()
        model.quats = nn.Parameter(torch.cat([model.quats.data, new_quats], dim=0))
        # opacities — logit-space, -2.0 (matches GaussianModel.__init__ default → sigmoid≈0.119)
        new_op = torch.full((M, 1), -2.0, device=device, dtype=dtype)
        model.opacities = nn.Parameter(torch.cat([model.opacities.data, new_op], dim=0))
        # SH DC from RGB
        new_sh_dc = _rgb_to_sh_dc(new_colors).unsqueeze(1)  # (M, 1, 3)
        model.sh_dc = nn.Parameter(torch.cat([model.sh_dc.data, new_sh_dc], dim=0))
        # SH rest zero
        if model.num_sh_rest > 0:
            new_sh_rest = torch.zeros(
                (M, model.num_sh_rest, 3), device=device, dtype=dtype,
            )
            model.sh_rest = nn.Parameter(torch.cat([model.sh_rest.data, new_sh_rest], dim=0))
        # Fourier coefs (if fourier_K > 0)
        if model.fourier_K > 0 and model.fourier_pos_coeffs is not None:
            new_fourier = torch.zeros(
                (M, model.fourier_K, 2, 3), device=device, dtype=dtype,
            )
            model.fourier_pos_coeffs = nn.Parameter(
                torch.cat([model.fourier_pos_coeffs.data, new_fourier], dim=0)
            )
        # Buffers — re-register so dtype/device stay consistent.
        new_is_static = torch.ones(M, dtype=torch.bool, device=model.is_static.device)
        model.is_static = torch.cat([model.is_static, new_is_static], dim=0)
        new_is_background = torch.zeros(M, dtype=torch.bool, device=model.is_background.device)
        model.is_background = torch.cat([model.is_background, new_is_background], dim=0)
    return M


@dataclass
class OutpaintLoopStats:
    views_processed: int = 0
    views_rejected: int = 0
    gaussians_added: int = 0


def run_outpaint_loop(
    model: GaussianModel,
    poses: list[CameraPose],
    K: CameraIntrinsics,
    inpainter: InpainterBase,
    depth_fn: Callable[[np.ndarray], np.ndarray],
    cfg: ImageToSceneConfig,
    progress_callback: Callable | None = None,
) -> OutpaintLoopStats:
    """Run the iterative outpaint loop over the trajectory.

    Args:
        model: seeded GaussianModel (on GPU).
        poses: list of camera poses; pose[0] should be the input-image viewpoint.
        K: camera intrinsics at the working resolution.
        inpainter: InpainterBase with .load()/.inpaint()/.unload().
        depth_fn: callable that takes (H, W, 3) uint8 RGB and returns (H, W) float32 depth.
    """
    stats = OutpaintLoopStats()
    inpainter.load()
    try:
        # Skip pose[0] — that view is fully covered by the seed point cloud.
        for i, pose in enumerate(poses[1:], start=1):
            if progress_callback:
                progress_callback("outpaint_loop", i / len(poses), f"view {i}/{len(poses)}", {})

            rgb_rendered, alpha, depth_rendered = render_pose(model, pose, K)
            visible, to_fill = visibility_mask_from_alpha(alpha, threshold=0.5, dilate_px=2)
            if to_fill.sum() < 50:
                stats.views_processed += 1
                continue

            rgb_uint8 = (np.clip(rgb_rendered, 0, 1) * 255).astype(np.uint8)
            fill_mask_uint8 = to_fill.astype(np.uint8)
            rgb_filled = inpaint_with_sd(inpainter, rgb_uint8, fill_mask_uint8)

            depth_new = depth_fn(rgb_filled)

            align = align_new_view(
                new_depth=depth_new,
                rendered_depth=depth_rendered,
                overlap_mask=visible,
                min_overlap_pixels=cfg.align_min_overlap_pixels,
                reject_threshold=cfg.align_residual_reject_threshold,
            )
            if align.rejected:
                stats.views_rejected += 1
                stats.views_processed += 1
                continue

            depth_aligned = apply_scale_shift(depth_new, align.scale, align.shift)
            added = add_gaussians_from_pixels(
                model=model,
                rgb_uint8=rgb_filled,
                depth_aligned=depth_aligned,
                new_pixel_mask=to_fill,
                pose=pose,
                K=K,
                max_new_fraction=cfg.max_new_gaussians_per_view_frac,
            )
            stats.gaussians_added += added
            stats.views_processed += 1
    finally:
        inpainter.unload()

    return stats
