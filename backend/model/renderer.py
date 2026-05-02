"""Faz 5 — gsplat rasterizer wrapper.

gsplat 1.x API'si. with_depth=True ise tek call'la RGB+ED (expected depth)
render eder — gsplat internal olarak RGB ve depth'i ayni geometric pass'te
hesaplar; backgrounds tensor'unu da otomatik genisletir (rendering.py:614-623).
Iki ayri call yapan eski yol depth ON'unu render maliyetini neredeyse iki
katina cikariyordu.
"""
from __future__ import annotations
import torch
import numpy as np


def render_view(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    with_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns:
        rgb:    (H, W, 3) ya da (H, W, 4) float [0, 1]  (with_depth=True ise last chan depth)
        alpha:  (H, W, 1)
        info:   gsplat'tan dönen metadata
    """
    try:
        from gsplat import rasterization
    except ImportError as e:
        raise ImportError(
            "gsplat kurulu değil. pip install gsplat"
        ) from e

    device = means.device
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    viewmat = w2c.unsqueeze(0)                              # (1, 4, 4) — assume already on device
    Ks = K.unsqueeze(0)                                     # (1, 3, 3) — assume already on device
    bg = torch.tensor(bg_color, device=device).unsqueeze(0)  # (1, 3)

    mode = "RGB+ED" if with_depth else "RGB"

    out, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree if colors.dim() == 3 else None,
        backgrounds=bg,                # gsplat extends with 0 for depth chan internally
        render_mode=mode,
        packed=False,
        rasterize_mode="antialiased",  # gsplat 1.5+ Mip-Splatting anti-alias
    )
    # out: (1, H, W, 3) for RGB, (1, H, W, 4) for RGB+ED. alpha: (1, H, W, 1)
    return out[0], alpha[0], info


def render_image_uint8(*args, **kwargs) -> np.ndarray:
    """Yardımcı: render_view → (H, W, 3) uint8."""
    rgb, _, _ = render_view(*args, **kwargs)
    # with_depth=True ise RGB only al
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sağlık testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠ CUDA yok")
        raise SystemExit(0)

    device = "cuda"
    N = 1000
    means = torch.randn(N, 3, device=device) * 0.5
    means[:, 2] += 5.0
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    scales = torch.full((N, 3), 0.05, device=device)
    opac = torch.full((N,), 0.8, device=device)
    colors = torch.rand(N, 3, device=device)

    K = torch.tensor([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]], device=device)
    w2c = torch.eye(4, device=device)

    rgb, alpha, info = render_view(means, quats, scales, opac, colors,
                                   K, w2c, width=640, height=480, sh_degree=0)
    print(f"RGB only: {rgb.shape}")

    rgbd, alpha, info = render_view(means, quats, scales, opac, colors,
                                    K, w2c, width=640, height=480, sh_degree=0,
                                    with_depth=True)
    print(f"RGB+D:    {rgbd.shape}  (last chan depth)")
