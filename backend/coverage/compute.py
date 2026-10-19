"""Per-splat coverage from the training cameras.

For every camera the splat's expected depth is rendered at low resolution;
each Gaussian centre is projected into the camera and counted as seen when it
is inside the image and not behind the rendered surface. Per Gaussian this
accumulates

* ``count``: cameras that saw it,
* ``spread``: 1 - |mean viewing direction| (0 = always seen from one side),
* ``px_per_unit``: best resolution it was seen at (pixels per scene unit).

``score`` (0 bad .. 1 good) is the log view count saturating at GOOD_COUNT
views. On the ROOM2 capture (1008 cameras, separate test walk) it tracked the
held-out LPIPS per test frame best (Spearman -0.60, vs -0.45 for spread and
lower for count/spread blends or a higher saturation point), so spread and
resolution are kept as diagnostics only.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..compose.plyio import GaussianCloud
from .colmap import CameraSet

GOOD_COUNT = 40     # views at which a surface counts as fully covered
WEAK_SCORE = 0.5    # below this a splat counts as weak (~6 views or fewer)
OPAQUE = 0.1        # splats more transparent than this are ignored in the statistics
SH_C0 = 0.28209479177387814
RENDER_WIDTH = 384  # depth test resolution; ~30 s for 3M splats x 1000 cameras on a 3060 Ti

ProgressFn = Callable[[float, str], None]


def splat_tensors(cloud: GaussianCloud, device: str = "cuda") -> dict:
    import torch

    return {
        "means": torch.tensor(cloud.means, device=device),
        "quats": torch.tensor(cloud.quats, device=device),
        "scales": torch.exp(torch.tensor(cloud.log_scales, device=device)),
        "opacities": torch.sigmoid(torch.tensor(cloud.opacities, device=device)).reshape(-1),
        "colors": torch.tensor(cloud.sh_dc, device=device)[:, None, :],
    }


def _render_K(cams: CameraSet, i: int, width: int):
    """Intrinsics scaled to the depth-test resolution, and that resolution."""
    w, h = (int(v) for v in cams.sizes[i])
    s = width / w
    K = cams.K[i].copy()
    K[:2] *= s
    return K, width, max(1, round(h * s))


def visibility(t: dict, w2c, K, width: int, height: int, tol: float = 0.05):
    """Mask of Gaussians seen by one camera and their camera-space depth (torch tensors)."""
    from gsplat import rasterization

    depth, _, _ = rasterization(
        means=t["means"], quats=t["quats"], scales=t["scales"], opacities=t["opacities"],
        colors=t["colors"], sh_degree=0, viewmats=w2c[None], Ks=K[None],
        width=width, height=height, packed=False, render_mode="ED")
    depth = depth[0, ..., 0]
    x = t["means"] @ w2c[:3, :3].T + w2c[:3, 3]
    z = x[:, 2]
    zs = z.clamp_min(1e-6)
    u = (K[0, 0] * x[:, 0] / zs + K[0, 2]).floor().long()
    v = (K[1, 1] * x[:, 1] / zs + K[1, 2]).floor().long()
    inside = (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    d = depth[v.clamp(0, height - 1), u.clamp(0, width - 1)]
    # Seen = in front of (or on) the rendered surface; empty pixels count as seen.
    return inside & ((d <= 0) | (z <= d * (1 + tol))), z


def accumulate(t: dict, cams: CameraSet, width: int = RENDER_WIDTH,
               on_progress: ProgressFn = lambda p, m: None,
               cancelled: Callable[[], bool] = lambda: False) -> dict[str, np.ndarray]:
    import torch

    dev = t["means"].device
    n = t["means"].shape[0]
    count = torch.zeros(n, device=dev, dtype=torch.int32)
    dirsum = torch.zeros(n, 3, device=dev)
    best = torch.zeros(n, device=dev)
    with torch.no_grad():
        for i in range(len(cams)):
            if cancelled():
                raise RuntimeError("Kapsama hesabı iptal edildi")
            K, w, h = _render_K(cams, i, width)
            w2c = torch.tensor(cams.w2c[i], device=dev)
            seen, z = visibility(t, w2c, torch.tensor(K, device=dev), w, h)
            centre = -w2c[:3, :3].T @ w2c[:3, 3]
            d = t["means"][seen] - centre
            dirsum[seen] += d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
            count[seen] += 1
            best[seen] = torch.maximum(best[seen], float(cams.K[i][0, 0]) / z[seen].clamp_min(1e-6))
            if (i + 1) % 25 == 0 or i + 1 == len(cams):
                on_progress((i + 1) / len(cams), f"{i + 1}/{len(cams)} kamera tarandı")
        spread = 1 - dirsum.norm(dim=1) / count.clamp_min(1)
        spread[count == 0] = 0
    return {"count": count.cpu().numpy(), "spread": spread.cpu().numpy(), "px_per_unit": best.cpu().numpy()}


def score(count: np.ndarray) -> np.ndarray:
    return np.clip(np.log1p(count) / math.log1p(GOOD_COUNT), 0, 1).astype(np.float32)


def view_stats(t: dict, s: np.ndarray, cams: CameraSet, width: int = RENDER_WIDTH,
               on_progress: ProgressFn = lambda p, m: None) -> list[dict]:
    """Per camera: opacity-weighted mean score and weak share of the splats it sees."""
    import torch

    dev = t["means"].device
    st = torch.tensor(s, device=dev)
    weak = (st < WEAK_SCORE).float()
    op = t["opacities"]
    out = []
    with torch.no_grad():
        for i in range(len(cams)):
            K, w, h = _render_K(cams, i, width)
            seen, _ = visibility(t, torch.tensor(cams.w2c[i], device=dev), torch.tensor(K, device=dev), w, h)
            wt = op[seen]
            total = wt.sum().clamp_min(1e-9)
            out.append({"score": float((st[seen] * wt).sum() / total), "weak": float((weak[seen] * wt).sum() / total)})
            if (i + 1) % 25 == 0 or i + 1 == len(cams):
                on_progress((i + 1) / len(cams), f"{i + 1}/{len(cams)} görüş puanlandı")
    return out


def heat_colours(s: np.ndarray) -> np.ndarray:
    """0 red -> 0.5 yellow -> 1 green, (N, 3) in [0, 1]."""
    red, yellow, green = np.array([0.95, 0.12, 0.10]), np.array([1.0, 0.85, 0.10]), np.array([0.10, 0.80, 0.25])
    s = np.clip(s, 0, 1)[:, None]
    lo = red + (yellow - red) * np.clip(s * 2, 0, 1)
    return np.where(s < 0.5, lo, yellow + (green - yellow) * np.clip(s * 2 - 1, 0, 1)).astype(np.float32)


def heat_cloud(cloud: GaussianCloud, s: np.ndarray) -> GaussianCloud:
    """The splat with view-independent score colours (geometry and opacity unchanged)."""
    return GaussianCloud(
        means=cloud.means, log_scales=cloud.log_scales, quats=cloud.quats, opacities=cloud.opacities,
        sh_dc=(heat_colours(s) - 0.5) / SH_C0, sh_rest=np.zeros((cloud.count, 0, 3), np.float32))


def summary(cov: dict[str, np.ndarray], s: np.ndarray, opacities_logit: np.ndarray) -> dict:
    opaque = 1 / (1 + np.exp(-opacities_logit.reshape(-1))) > OPAQUE
    n = max(int(opaque.sum()), 1)
    c = cov["count"][opaque]
    return {
        "splats": int(s.shape[0]),
        "opaque_splats": int(opaque.sum()),
        "unseen_share": float((c == 0).sum() / n),
        "weak_share": float((s[opaque] < WEAK_SCORE).sum() / n),
        "count_percentiles": {str(q): float(v) for q, v in zip((5, 25, 50, 75, 95), np.percentile(c, [5, 25, 50, 75, 95]))}
        if c.size else {},
        "good_count": GOOD_COUNT,
        "weak_score": WEAK_SCORE,
    }
