"""Faz 5 — gsplat rasterizer wrapper.

gsplat 1.x API'si kullanılır. Renderer çağrısı sırasında:
  - 3D Gaussian → 2D screen-space splat (CUDA)
  - Alpha compositing
  - SH renkleri view direction'a göre değerlendirilir
"""
from __future__ import annotations
import torch
import numpy as np


def render_view(
    means: torch.Tensor,        # (N, 3)
    quats: torch.Tensor,        # (N, 4)
    scales: torch.Tensor,       # (N, 3) > 0
    opacities: torch.Tensor,    # (N, 1) ya da (N,)
    colors: torch.Tensor,       # SH (N, K, 3) ya da RGB (N, 3)
    K: torch.Tensor,            # (3, 3)
    w2c: torch.Tensor,          # (4, 4)
    width: int,
    height: int,
    sh_degree: int = 3,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Bir kameradan render et.

    Returns:
        rgb:    (H, W, 3) float [0, 1]
        alpha:  (H, W, 1)
        info:   gsplat'tan döndürülen extra bilgiler (radii, screen_grad vb.)
    """
    try:
        from gsplat import rasterization
    except ImportError as e:
        raise ImportError(
            "gsplat kurulu değil. Kurulum: pip install gsplat\n"
            "Önce CUDA ile uyumlu PyTorch kur."
        ) from e

    N = means.shape[0]
    device = means.device
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    # gsplat (1, ...) batch boyutu bekler
    viewmat = w2c.unsqueeze(0).to(device)        # (1, 4, 4)
    Ks = K.unsqueeze(0).to(device)               # (1, 3, 3)

    bg = torch.tensor(bg_color, device=device).unsqueeze(0)  # (1, 3)

    # SH değerlendirmesi gsplat içinde otomatik (sh_degree verilirse)
    rgb, alpha, info = rasterization(
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
        backgrounds=bg,
        render_mode="RGB",
        packed=False,
    )
    # rgb: (1, H, W, 3), alpha: (1, H, W, 1)
    return rgb[0], alpha[0], info


def render_image_uint8(*args, **kwargs) -> np.ndarray:
    """Yardımcı: render_view → (H, W, 3) uint8."""
    rgb, _, _ = render_view(*args, **kwargs)
    return (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Hızlı (CUDA gerektiren) sağlık testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠ CUDA yok — bu test atlandı")
        raise SystemExit(0)

    device = "cuda"
    N = 1000
    means = torch.randn(N, 3, device=device) * 0.5
    means[:, 2] += 5.0
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    scales = torch.full((N, 3), 0.05, device=device)
    opac = torch.full((N,), 0.8, device=device)
    colors = torch.rand(N, 3, device=device)  # RGB modu

    K = torch.tensor([[400., 0, 320.], [0, 400., 240.], [0, 0, 1.]], device=device)
    w2c = torch.eye(4, device=device)

    rgb, alpha, info = render_view(means, quats, scales, opac, colors,
                                   K, w2c, width=640, height=480, sh_degree=0)
    print(f"RGB:   {rgb.shape}  [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"Alpha: {alpha.shape} [{alpha.min():.3f}, {alpha.max():.3f}]")
