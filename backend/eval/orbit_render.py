"""4D Quality v6.1 — Orbit camera render (Madde 1).

Eğitilmiş camera pozlarından smooth Catmull-Rom spline path üretir,
o path boyunca render ve mp4 export eder.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def _catmull_rom(p0, p1, p2, p3, t: float):
    """Catmull-Rom spline interpolation, t∈[0,1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def generate_orbit_w2c(
    train_w2c: list, num_frames: int = 60,
) -> list:
    """Train cam w2c listesinden smooth orbit path üret (Catmull-Rom).

    Args:
      train_w2c: list of (4, 4) torch tensors (training camera w2c'leri)
      num_frames: hedef frame sayisi
    Returns:
      list of (4, 4) torch tensors — interpolated w2c
    """
    if len(train_w2c) < 4:
        # Çok az kamera varsa lineer interpolation
        if len(train_w2c) == 0:
            raise ValueError("train_w2c bos")
        if len(train_w2c) == 1:
            return [train_w2c[0].clone() for _ in range(num_frames)]
        # 2-3 cam: lineer
        out = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            seg = t * (len(train_w2c) - 1)
            seg_idx = int(seg)
            seg_idx_next = min(seg_idx + 1, len(train_w2c) - 1)
            local_t = seg - seg_idx
            w2c = (1 - local_t) * train_w2c[seg_idx] + local_t * train_w2c[seg_idx_next]
            out.append(w2c.clone())
        return out

    # Catmull-Rom — extract cam centers + rotations
    centers = []
    rotations = []
    for w2c in train_w2c:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        # Cam center: -R^T t
        c = -(R.T @ t)
        centers.append(c)
        rotations.append(R)
    centers = torch.stack(centers)  # (N, 3)

    out_w2c = []
    N = len(centers)
    for i in range(num_frames):
        u = i / max(num_frames - 1, 1)
        seg = u * (N - 1)
        seg_idx = int(seg)
        local_t = seg - seg_idx
        # 4 noktalı Catmull-Rom kontrol noktaları
        i0 = max(0, seg_idx - 1)
        i1 = seg_idx
        i2 = min(N - 1, seg_idx + 1)
        i3 = min(N - 1, seg_idx + 2)
        c_interp = _catmull_rom(
            centers[i0], centers[i1], centers[i2], centers[i3], local_t,
        )
        # Rotation interpolation — basit: closest train R'sini kullan (slerp olabilir)
        R_interp = rotations[i1] if local_t < 0.5 else rotations[i2]
        # w2c reconstruct: t = -R c
        t_interp = -(R_interp @ c_interp)
        w2c_new = torch.eye(4, device=c_interp.device, dtype=c_interp.dtype)
        w2c_new[:3, :3] = R_interp
        w2c_new[:3, 3] = t_interp
        out_w2c.append(w2c_new)
    return out_w2c


@torch.no_grad()
def render_orbit_video(
    gs,
    deform,
    K: torch.Tensor,
    train_w2c: list,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 60,
    fps: int = 30,
    static_mode: bool = False,
    scene_extent: float = 1.0,
    device: str = "cuda",
) -> dict:
    """Smooth orbit cam path boyunca render → mp4.

    Args:
      gs: GaussianModel
      deform: DeformationField (None ise static render)
      K: (3,3) intrinsic
      train_w2c: train cam w2c listesi (orbit path source)
      out_path: hedef .mp4
      width/height: render resolution
      num_frames: orbit frame count
      fps: video fps
      static_mode: True = identity deform (her frame ayni geometry)
      scene_extent: deform'un kullandigi extent
      device: cuda|cpu
    Returns:
      dict: {n_frames, path, success, error?}
    """
    from ..model.renderer import render_view

    try:
        path_w2c = generate_orbit_w2c(train_w2c, num_frames=num_frames)
        frames_out = []
        for i, w2c in enumerate(path_w2c):
            t_norm = i / max(num_frames - 1, 1)
            if static_mode or deform is None:
                d_means = gs.means
                d_quats = F.normalize(gs.quats, dim=-1)
                d_scales = gs.get_scales
            else:
                dpos, dquat, dscale = deform(gs.means, t_norm, scene_extent)
                d_means = gs.means + dpos
                d_quats = F.normalize(gs.quats + dquat, dim=-1)
                d_scales = (gs.scales + dscale).exp()
            rgb, _, _ = render_view(
                means=d_means, quats=d_quats, scales=d_scales,
                opacities=gs.get_opacities, colors=gs.get_colors,
                K=K.to(device), w2c=w2c.to(device),
                width=width, height=height,
                sh_degree=gs.sh_degree,
            )
            frames_out.append(rgb.detach().cpu())

        from .video_export import write_video
        write_video(frames_out, out_path, fps=fps, quality=8)
        return {
            "n_frames": len(frames_out),
            "path": str(out_path),
            "success": True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "n_frames": 0,
            "path": str(out_path),
            "success": False,
            "error": str(e),
        }
