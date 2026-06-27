"""4D Quality v6.1 — Novel View Synthesis evaluation (Madde 1+8).

Held-out test camera üzerinde PSNR/SSIM/LPIPS hesapla.
4D temporal hold-out: spesifik frame'leri train'den çıkar, novel-time eval.
Multi-view: cam00 (N3V test cam) tüm timestep'lerde render → metrics.
Single-view: son N frame'i test olarak ayır.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


def _scale_K(K_native: torch.Tensor, w_native: int, h_native: int,
             w_render: int, h_render: int) -> torch.Tensor:
    """Rescale camera intrinsics from native (COLMAP) to render resolution.

    K_first comes from COLMAP at the native frame resolution; eval renders
    at cfg.train.image_resolution. Without this rescale the focal length /
    principal point are off by (render / native), reprojection collapses,
    and PSNR floor-pegs at ~8 dB regardless of model quality.
    """
    sx, sy = w_render / w_native, h_render / h_native
    K_s = K_native.clone()
    K_s[0, 0] *= sx; K_s[0, 2] *= sx
    K_s[1, 1] *= sy; K_s[1, 2] *= sy
    return K_s


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred - gt) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else -10.0 * math.log10(mse)


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Pred / gt: (H, W, 3) float [0, 1]."""
    try:
        from pytorch_msssim import ssim as _ssim_fn
        s = _ssim_fn(
            pred.permute(2, 0, 1).unsqueeze(0),
            gt.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0, size_average=True,
        )
        return float(s.item())
    except ImportError:
        return float("nan")


def _load_lpips(net: str = "vgg", device: str = "cuda"):
    try:
        from ..model.losses_perceptual import LPIPSLoss
        return LPIPSLoss(net=net).to(device)
    except Exception:
        return None


@torch.no_grad()
def eval_held_out_camera(
    gs,
    deform,
    cam_K: torch.Tensor,
    cam_w2c: torch.Tensor,
    frame_paths: list,
    width: int,
    height: int,
    static_mode: bool = False,
    scene_extent: float = 1.0,
    device: str = "cuda",
    skip_lpips: bool = False,
) -> dict:
    """Tek bir held-out cam icin tum frame'lerde render + metric hesapla.

    Args:
      gs, deform: trained model
      cam_K: (3, 3) intrinsic
      cam_w2c: (4, 4) extrinsic — held-out kamera
      frame_paths: ground truth frame yolu listesi (tum timestep'ler)
      width, height: render res
      static_mode: True = identity deform
      scene_extent: deform extent
      device: cuda|cpu
      skip_lpips: True ise LPIPS hesaplama (hizli)
    Returns:
      dict: {psnr_mean, ssim_mean, lpips_mean, n_frames, per_frame: [...]}
    """
    from ..model.renderer import render_view
    from ..model.trainer import load_frame_tensor

    psnr_list, ssim_list, lpips_list = [], [], []
    per_frame = []
    lpips_mod = None if skip_lpips else _load_lpips(device=device)

    T = len(frame_paths)
    cam_K_dev = cam_K.to(device)
    cam_w2c_dev = cam_w2c.to(device)

    for idx, fp in enumerate(frame_paths):
        t_norm = idx / max(T - 1, 1)
        try:
            gt = load_frame_tensor(Path(fp), (width, height)).to(device)
        except Exception:
            continue

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
            K=cam_K_dev, w2c=cam_w2c_dev,
            width=width, height=height,
            sh_degree=gs.sh_degree,
        )
        rgb = rgb.clamp(0, 1)

        psnr_v = _psnr(rgb, gt)
        ssim_v = _ssim(rgb, gt)
        lpips_v = float("nan")
        if lpips_mod is not None:
            try:
                lp = lpips_mod(
                    rgb.permute(2, 0, 1).contiguous(),
                    gt.permute(2, 0, 1).contiguous(),
                )
                lpips_v = float(lp.item())
            except Exception:
                pass

        psnr_list.append(psnr_v)
        ssim_list.append(ssim_v)
        if not math.isnan(lpips_v):
            lpips_list.append(lpips_v)
        per_frame.append({
            "frame": idx, "psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v,
        })

    return {
        "psnr_mean": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "ssim_mean": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "lpips_mean": float(np.mean(lpips_list)) if lpips_list else float("nan"),
        "n_frames": len(psnr_list),
        "per_frame": per_frame,
    }


@torch.no_grad()
def eval_temporal_holdout(
    gs,
    deform,
    cam_K: torch.Tensor,
    cam_w2c_per_frame: list,
    frame_paths: list,
    holdout_indices: list,
    width: int,
    height: int,
    static_mode: bool = False,
    scene_extent: float = 1.0,
    device: str = "cuda",
    skip_lpips: bool = False,
) -> dict:
    """Single-view temporal hold-out: belirli timestep'leri eval olarak işle.

    Madde 8: train'de bu indeksler atlanmis varsayilir. Burada GT ile karsilastirir.
    """
    from ..model.renderer import render_view
    from ..model.trainer import load_frame_tensor

    psnr_list, ssim_list, lpips_list = [], [], []
    per_frame = []
    lpips_mod = None if skip_lpips else _load_lpips(device=device)
    T = len(frame_paths)
    cam_K_dev = cam_K.to(device)

    for idx in holdout_indices:
        if idx < 0 or idx >= T:
            continue
        t_norm = idx / max(T - 1, 1)
        try:
            gt = load_frame_tensor(Path(frame_paths[idx]), (width, height)).to(device)
        except Exception:
            continue
        w2c = cam_w2c_per_frame[idx].to(device)

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
            K=cam_K_dev, w2c=w2c,
            width=width, height=height,
            sh_degree=gs.sh_degree,
        )
        rgb = rgb.clamp(0, 1)

        psnr_v = _psnr(rgb, gt)
        ssim_v = _ssim(rgb, gt)
        lpips_v = float("nan")
        if lpips_mod is not None:
            try:
                lp = lpips_mod(
                    rgb.permute(2, 0, 1).contiguous(),
                    gt.permute(2, 0, 1).contiguous(),
                )
                lpips_v = float(lp.item())
            except Exception:
                pass

        psnr_list.append(psnr_v)
        ssim_list.append(ssim_v)
        if not math.isnan(lpips_v):
            lpips_list.append(lpips_v)
        per_frame.append({
            "frame": idx, "psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v,
        })

    return {
        "psnr_mean": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "ssim_mean": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "lpips_mean": float(np.mean(lpips_list)) if lpips_list else float("nan"),
        "n_frames": len(psnr_list),
        "per_frame": per_frame,
    }


def save_eval_report(report: dict, out_dir: Path) -> Path:
    """Eval report'u JSON'a yaz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nvs_eval.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
