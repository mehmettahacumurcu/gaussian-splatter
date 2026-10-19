"""Where is a splat under-observed? Per-splat coverage from the training cameras.

For every training camera the splat's expected depth is rendered at low
resolution; each Gaussian centre is projected into the camera and counted as
seen when it is inside the image and not behind the rendered surface. Per
Gaussian this accumulates:

* ``count``: cameras that saw it,
* ``spread``: 1 - |mean viewing direction| (0 = always seen from one direction,
  the usual source of blur and floaters; ~0.3+ = seen from many sides),
* ``px_per_unit``: best resolution it was seen at (full-resolution pixels per
  scene unit), i.e. how much detail the photos could give it.

``score`` (0 bad .. 1 good) is the log-scaled view count: on ROOM2 it tracks the
held-out error best (Spearman -0.60 per test frame vs -0.45 for spread and
-0.47..-0.50 for count/spread blends), so spread and resolution stay separate
diagnostics. ``heatmap_params`` recolours the splat by score so the weak
regions show up in any splat viewer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .own_splat import _k_and_dist, read_sparse_model

GOOD_COUNT = 40  # views at which a surface counts as fully covered
SH_C0 = 0.28209479177387814


def load_cameras(sparse_dir: Path, prefix: str = "") -> dict:
    """World-to-camera matrices + full-resolution pinhole K of the images whose name starts with ``prefix``."""
    model = read_sparse_model(Path(sparse_dir))
    cam = next(iter(model.cameras.values()))
    K, _ = _k_and_dist(cam)
    images = sorted((im for im in model.images.values() if Path(im.name).name.startswith(prefix)),
                    key=lambda im: im.name)
    return {"names": [im.name for im in images],
            "w2c": np.stack([im.world_to_camera() for im in images]).astype(np.float32),
            "K": np.asarray(K, np.float32), "width": cam.width, "height": cam.height}


@torch.no_grad()
def _depth(params, w2c, K, w, h):
    from gsplat import rasterization

    out, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=params["sh0"], sh_degree=0,
        viewmats=w2c[None], Ks=K[None], width=w, height=h, packed=False, render_mode="ED")
    return out[0, ..., 0]


@torch.no_grad()
def visibility(params, w2c: torch.Tensor, K: torch.Tensor, w: int, h: int, tol: float = 0.05):
    """Mask of Gaussians seen by one camera (render-resolution K) and their camera-space depth."""
    depth = _depth(params, w2c, K, w, h)
    x = params["means"] @ w2c[:3, :3].T + w2c[:3, 3]
    z = x[:, 2]
    zs = z.clamp_min(1e-6)
    u = (K[0, 0] * x[:, 0] / zs + K[0, 2]).floor().long()
    v = (K[1, 1] * x[:, 1] / zs + K[1, 2]).floor().long()
    inside = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = depth[v.clamp(0, h - 1), u.clamp(0, w - 1)]
    # Seen = in front of (or on) the rendered surface; empty pixels count as seen.
    seen = inside & ((d <= 0) | (z <= d * (1 + tol)))
    return seen, z


@torch.no_grad()
def accumulate(params, cams: dict, width: int = 384, device="cuda", log=print) -> dict:
    """count / spread / px_per_unit for every Gaussian over the cameras in ``cams``."""
    n = params["means"].shape[0]
    s = width / cams["width"]
    h = round(cams["height"] * s)
    K = torch.tensor(cams["K"], device=device)
    Kr = K.clone()
    Kr[:2] *= s
    count = torch.zeros(n, device=device, dtype=torch.int32)
    dirsum = torch.zeros(n, 3, device=device)
    best = torch.zeros(n, device=device)
    for i, m in enumerate(cams["w2c"]):
        w2c = torch.tensor(m, device=device)
        seen, z = visibility(params, w2c, Kr, width, h)
        centre = -w2c[:3, :3].T @ w2c[:3, 3]
        d = params["means"][seen] - centre
        dirsum[seen] += d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        count[seen] += 1
        best[seen] = torch.maximum(best[seen], K[0, 0] / z[seen].clamp_min(1e-6))
        if (i + 1) % 200 == 0 or i + 1 == len(cams["w2c"]):
            log(f"[kapsama] {i + 1}/{len(cams['w2c'])} kamera")
    spread = 1 - dirsum.norm(dim=1) / count.clamp_min(1)
    spread[count == 0] = 0
    return {"count": count.cpu().numpy(), "spread": spread.cpu().numpy(), "px_per_unit": best.cpu().numpy()}


def score(cov: dict) -> np.ndarray:
    return np.clip(np.log1p(cov["count"]) / math.log1p(GOOD_COUNT), 0, 1)


def heat_colours(s: np.ndarray) -> np.ndarray:
    """0 red -> 0.5 yellow -> 1 green."""
    red, yellow, green = np.array([0.95, 0.12, 0.10]), np.array([1.0, 0.85, 0.10]), np.array([0.10, 0.80, 0.25])
    s = np.clip(s, 0, 1)[:, None]
    lo = red + (yellow - red) * np.clip(s * 2, 0, 1)
    return np.where(s < 0.5, lo, yellow + (green - yellow) * np.clip(s * 2 - 1, 0, 1))


def heatmap_params(params, s: np.ndarray, opacity_floor: float = 0.0):
    """Copy of the splat with view-independent colours from the score (shape and opacity unchanged)."""
    rgb = torch.tensor(heat_colours(s), dtype=torch.float32, device=params["means"].device)
    out = {k: v.detach().clone() for k, v in params.items()}
    out["sh0"] = ((rgb - 0.5) / SH_C0)[:, None, :]
    out["shN"] = torch.zeros_like(out["shN"])
    if opacity_floor:
        out["opacities"] = torch.maximum(out["opacities"], torch.full_like(out["opacities"], math.log(opacity_floor / (1 - opacity_floor))))
    return out


@torch.no_grad()
def view_scores(params, s: np.ndarray, cams: dict, width: int = 384, device="cuda") -> list[float]:
    """Opacity-weighted mean score of the Gaussians each camera sees (for checking against per-view error)."""
    st = torch.tensor(s, device=device, dtype=torch.float32)
    op = torch.sigmoid(params["opacities"]).reshape(-1)
    k = width / cams["width"]
    h = round(cams["height"] * k)
    Kr = torch.tensor(cams["K"], device=device)
    Kr[:2] *= k
    out = []
    for m in cams["w2c"]:
        seen, _ = visibility(params, torch.tensor(m, device=device), Kr, width, h)
        w = op[seen]
        out.append(float((st[seen] * w).sum() / w.sum().clamp_min(1e-9)))
    return out
