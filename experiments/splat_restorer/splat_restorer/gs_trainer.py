"""Small, self-contained 3DGS trainer on top of gsplat 1.5.x.

It is intentionally plain: the pair generator only needs *plausible* splat
artefacts, and the distill-back test needs a fair, repeatable fine-tune.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


@dataclass
class View:
    image: torch.Tensor  # (H, W, 3) float in [0, 1], on device
    w2c: torch.Tensor  # (4, 4)
    K: torch.Tensor  # (3, 3)
    weight: float = 1.0


@dataclass
class TrainConfig:
    steps: int = 4000
    sh_degree: int = 3
    sh_increase_every: int = 1000
    ssim_lambda: float = 0.2
    densify: bool = True
    refine_start: int = 500
    refine_stop_frac: float = 0.75
    opacity_reset_every: int = 1_000_000  # disabled: short runs do not recover
    lr_means: float = 1.6e-4
    lr_means_final_frac: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    init_opacity: float = 0.1
    seed: int = 0
    log_every: int = 0
    extra: dict = field(default_factory=dict)


def _knn_mean_dist(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)
    return np.sqrt((d[:, 1:] ** 2).mean(axis=1))


def init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, init_opacity: float, device) -> torch.nn.ParameterDict:
    n = len(xyz)
    dist = np.clip(_knn_mean_dist(xyz), 1e-7, None)
    scales = np.log(np.repeat(dist[:, None], 3, axis=1))
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    opac = np.full(n, math.log(init_opacity / (1 - init_opacity)), np.float32)
    sh0 = ((rgb - 0.5) / SH_C0)[:, None, :]
    k = (sh_degree + 1) ** 2 - 1
    shN = np.zeros((n, k, 3), np.float32)
    t = lambda a: torch.nn.Parameter(torch.as_tensor(a, dtype=torch.float32, device=device))
    return torch.nn.ParameterDict(
        {
            "means": t(xyz),
            "scales": t(scales),
            "quats": t(quats),
            "opacities": t(opac),
            "sh0": t(sh0),
            "shN": t(shN),
        }
    )


def make_optimizers(params, cfg: TrainConfig, scene_scale: float) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": cfg.lr_means * scene_scale,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "sh0": cfg.lr_sh0,
        "shN": cfg.lr_shN,
    }
    return {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}


def render(params, w2c: torch.Tensor, K: torch.Tensor, width: int, height: int, sh_degree: int):
    """Render one view. Returns (image HxWx3 clamped to [0,1] in the graph, info)."""
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    img, _alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=w2c[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
    )
    return img[0].clamp(0.0, 1.0), info


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM of two (H, W, 3) images in [0, 1]."""
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    w = _gauss_window(device=a.device)
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x**2
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return float(10 * math.log10(1.0 / max(mse, 1e-12)))


def train_gaussians(
    views: list[View],
    width: int,
    height: int,
    cfg: TrainConfig,
    scene_scale: float = 1.1,
    params: torch.nn.ParameterDict | None = None,
    init_xyz: np.ndarray | None = None,
    init_rgb: np.ndarray | None = None,
    log=print,
) -> torch.nn.ParameterDict:
    """Optimise Gaussians on ``views``. Pass ``params`` to fine-tune existing ones."""
    from gsplat.strategy import DefaultStrategy

    device = views[0].image.device
    gen = torch.Generator().manual_seed(cfg.seed)
    if params is None:
        assert init_xyz is not None and init_rgb is not None
        params = init_params(init_xyz, init_rgb, cfg.sh_degree, cfg.init_opacity, device)
    optimizers = make_optimizers(params, cfg, scene_scale)
    means_gamma = cfg.lr_means_final_frac ** (1.0 / max(cfg.steps, 1))

    strategy = None
    if cfg.densify:
        strategy = DefaultStrategy(
            refine_start_iter=cfg.refine_start,
            refine_stop_iter=int(cfg.steps * cfg.refine_stop_frac),
            reset_every=cfg.opacity_reset_every,
            verbose=False,
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)

    weights = torch.tensor([v.weight for v in views], dtype=torch.float64)
    for step in range(cfg.steps):
        idx = int(torch.multinomial(weights, 1, generator=gen))
        v = views[idx]
        sh_deg = min(step // cfg.sh_increase_every, cfg.sh_degree) if cfg.densify else cfg.sh_degree
        img, info = render(params, v.w2c, v.K, width, height, sh_deg)
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (img - v.image).abs().mean()
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim(img, v.image))
        loss.backward()
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for g in optimizers["means"].param_groups:
            g["lr"] *= means_gamma
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            log(f"    step {step + 1}/{cfg.steps} loss={loss.item():.4f} n={len(params['means'])}")
    return params


def params_to_cpu_state(params: torch.nn.ParameterDict) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in params.items()}


def params_from_state(state: dict[str, torch.Tensor], device) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in state.items()})
