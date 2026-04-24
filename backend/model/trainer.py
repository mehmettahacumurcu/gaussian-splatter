"""Faz 5 — 4DGS training loop.

Her iterasyonda:
  1) Rastgele bir frame seç (idx → t = idx / (T-1))
  2) Deformation field uygula: (Δpos, Δquat, Δscale)
  3) gsplat ile render et (opsiyonel: RGB+D)
  4) L1 + (1-SSIM) + depth + motion regularizers → backward
  5) Density control (her N adımda)
  6) Checkpoint (her M adımda)

Foundation-driven losses (Stage 2):
  - Depth consistency (Metric3D .npy → scale-invariant L1)
  - Mask-weighted reconstruction (dinamik bölgelere daha fazla ağırlık)
"""
from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Callable, Sequence

from .gaussian_model import GaussianModel
from .deformation import DeformationField
from .renderer import render_view
from .density_control import DensityController


TrainProgressCallback = Callable[[int, int, float, float, int], None]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_ssim: float = 0.2,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    rendered, gt: (H, W, 3) float [0, 1]
    pixel_weight: opsiyonel (H, W) — L1 loss'u piksel bazlı ağırlıklandır
    """
    diff = (rendered - gt).abs()
    if pixel_weight is not None:
        # Broadcast (H, W) → (H, W, 3)
        l1 = (diff * pixel_weight.unsqueeze(-1)).mean()
    else:
        l1 = diff.mean()
    try:
        from pytorch_msssim import ssim
        ssim_val = ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            gt.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0, size_average=True,
        )
        return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)
    except ImportError:
        return l1


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((rendered - gt) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else -10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Frame / depth / mask yardımcıları
# ---------------------------------------------------------------------------
def load_frame_tensor(frame_path: Path, target_size: tuple[int, int] | None = None) -> torch.Tensor:
    """PNG → (H, W, 3) float [0, 1]."""
    import cv2
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).float() / 255.0


def _load_depth_for_frame(
    depth_dir: Path, frame_path: Path, target_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """Metric3D depth .npy → (H, W) float32, hedef boyuta resize."""
    import cv2
    candidate = Path(depth_dir) / f"{Path(frame_path).stem}_depth.npy"
    if not candidate.exists():
        return None
    arr = np.load(candidate).astype(np.float32)
    if target_size is not None:
        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(arr)


def _load_mask_for_frame(
    mask_dir: Path, frame_path: Path, target_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """Dynamic mask .png (1-based idx) → (H, W) float [0,1]."""
    import cv2
    import re
    stem = Path(frame_path).stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    frame_idx_0based = int(m.group(1))
    candidate = Path(mask_dir) / f"mask_{frame_idx_0based + 1:04d}.png"
    if not candidate.exists():
        return None
    img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(img.astype("float32") / 255.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer4DGS:
    def __init__(
        self,
        gs: GaussianModel,
        deform: DeformationField,
        device: str = "cuda",
        scene_extent: float = 1.0,
        # LR
        lr_means: float = 1.6e-4,
        lr_scales: float = 5e-3,
        lr_quats: float = 1e-3,
        lr_opacities: float = 5e-2,
        lr_sh_dc: float = 2.5e-3,
        lr_sh_rest: float = 2.5e-3 / 20,
        lr_deform: float = 1e-3,
        # Density
        density_start_iter: int = 500,
        density_end_iter: int = 15_000,
        density_interval: int = 100,
        densify_grad_threshold: float = 2e-4,
        prune_min_opacity: float = 0.005,
        prune_max_scale: float = 0.1,
        # Loss
        lambda_ssim: float = 0.2,
        # Motion regularizers (Stage 1)
        lambda_deform_reg: float = 1e-3,
        lambda_smoothness: float = 1e-2,
        lambda_rigidity: float = 1e-2,
        rigidity_sample_k: int = 512,
        smoothness_dt: float = 0.02,
        # Foundation losses (Stage 2)
        lambda_depth: float = 0.0,
        lambda_mask_motion: float = 1.0,
    ):
        self.gs = gs.to(device)
        self.deform = deform.to(device)
        self.device = device
        self.scene_extent = scene_extent
        self.lambda_ssim = lambda_ssim
        self.lambda_deform_reg = lambda_deform_reg
        self.lambda_smoothness = lambda_smoothness
        self.lambda_rigidity = lambda_rigidity
        self.rigidity_sample_k = rigidity_sample_k
        self.smoothness_dt = smoothness_dt
        self.lambda_depth = lambda_depth
        self.lambda_mask_motion = lambda_mask_motion

        self.density_start_iter = density_start_iter
        self.density_end_iter = density_end_iter
        self.density_interval = density_interval

        self.density = DensityController(
            grad_threshold=densify_grad_threshold,
            min_opacity=prune_min_opacity,
            max_scale=prune_max_scale,
        )

        self.lr_specs = {
            "means":     lr_means,
            "scales":    lr_scales,
            "quats":     lr_quats,
            "opacities": lr_opacities,
            "sh_dc":     lr_sh_dc,
            "sh_rest":   lr_sh_rest,
        }
        self.lr_deform = lr_deform
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        groups = [
            {"params": [self.gs.means],     "lr": self.lr_specs["means"]},
            {"params": [self.gs.scales],    "lr": self.lr_specs["scales"]},
            {"params": [self.gs.quats],     "lr": self.lr_specs["quats"]},
            {"params": [self.gs.opacities], "lr": self.lr_specs["opacities"]},
            {"params": [self.gs.sh_dc],     "lr": self.lr_specs["sh_dc"]},
            {"params": [self.gs.sh_rest],   "lr": self.lr_specs["sh_rest"]},
            {"params": list(self.deform.parameters()), "lr": self.lr_deform},
        ]
        self.optimizer = torch.optim.Adam(groups, eps=1e-15)

    def _apply_deformation(self, t: float):
        dpos, dquat, dscale = self.deform(self.gs.means, t, self.scene_extent)
        deformed_means  = self.gs.means + dpos
        deformed_quats  = F.normalize(self.gs.quats + dquat, dim=-1)
        deformed_scales = self.gs.get_scales * torch.exp(dscale)
        return deformed_means, deformed_quats, deformed_scales

    def train(
        self,
        frame_paths: Sequence[Path],
        cam_K: torch.Tensor,
        cam_w2c_per_frame: Sequence[torch.Tensor],
        n_iters: int = 30_000,
        image_size: tuple[int, int] = (640, 360),
        ckpt_dir: Path | None = None,
        ckpt_interval: int = 1000,
        log_interval: int = 50,
        progress_callback: TrainProgressCallback | None = None,
        depth_dir: Path | None = None,
        mask_dir: Path | None = None,
    ) -> dict:
        T = len(frame_paths)
        if len(cam_w2c_per_frame) != T:
            raise ValueError(
                f"frame_paths ve cam_w2c_per_frame uzunlukları farklı: {T} vs {len(cam_w2c_per_frame)}"
            )

        cam_K = cam_K.to(self.device)
        w2c_list = [w.to(self.device) for w in cam_w2c_per_frame]
        Ws, Hs = image_size
        sample = load_frame_tensor(Path(frame_paths[0]))
        H0, W0 = sample.shape[:2]
        sx, sy = Ws / W0, Hs / H0
        K_scaled = cam_K.clone()
        K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

        # Cache'ler
        frames_cached: list[torch.Tensor | None] = [None] * T
        depth_cached:  list[torch.Tensor | None] = [None] * T
        mask_cached:   list[torch.Tensor | None] = [None] * T

        use_depth = depth_dir is not None and self.lambda_depth > 0
        use_mask  = mask_dir is not None
        if use_depth:
            print(f"[trainer] Depth loss aktif (lambda={self.lambda_depth}), dir={depth_dir}")
        if use_mask:
            print(f"[trainer] Mask-weighted recon aktif (lambda={self.lambda_mask_motion}), dir={mask_dir}")

        history = {"loss": [], "psnr": [], "n_pts": []}
        t0 = time.time()

        for it in range(1, n_iters + 1):
            idx = int(torch.randint(0, T, (1,)).item())
            t_norm = idx / max(T - 1, 1)

            # Frame
            if frames_cached[idx] is None:
                frames_cached[idx] = load_frame_tensor(Path(frame_paths[idx]), (Ws, Hs))
            gt = frames_cached[idx].to(self.device)

            # Depth
            if use_depth and depth_cached[idx] is None:
                depth_cached[idx] = _load_depth_for_frame(depth_dir, frame_paths[idx], (Ws, Hs))
            gt_depth = depth_cached[idx].to(self.device) if (use_depth and depth_cached[idx] is not None) else None

            # Mask
            if use_mask and mask_cached[idx] is None:
                mask_cached[idx] = _load_mask_for_frame(mask_dir, frame_paths[idx], (Ws, Hs))
            frame_mask = mask_cached[idx].to(self.device) if (use_mask and mask_cached[idx] is not None) else None

            # Deformation
            d_means, d_quats, d_scales = self._apply_deformation(t_norm)

            # Render (RGB+D gerekirse)
            render_out, _alpha, _info = render_view(
                means=d_means,
                quats=d_quats,
                scales=d_scales,
                opacities=self.gs.get_opacities,
                colors=self.gs.get_colors,
                K=K_scaled,
                w2c=w2c_list[idx],
                width=Ws,
                height=Hs,
                sh_degree=self.gs.sh_degree,
                with_depth=use_depth,
            )
            if use_depth:
                rgb = render_out[..., :3]
                rendered_depth = render_out[..., 3]
            else:
                rgb = render_out
                rendered_depth = None

            # Mask-weighted reconstruction: dynamic bölgelere daha fazla ağırlık
            pixel_weight = None
            if frame_mask is not None and self.lambda_mask_motion > 0:
                # 1 + lambda*mask → static regions weight=1, dynamic regions weight=1+lambda
                pixel_weight = 1.0 + self.lambda_mask_motion * frame_mask

            loss_recon = compute_loss(rgb, gt, self.lambda_ssim, pixel_weight=pixel_weight)
            loss = loss_recon

            # Depth consistency (Stage 2)
            if use_depth and gt_depth is not None and rendered_depth is not None:
                valid = (gt_depth > 0.01) & (rendered_depth > 0.01)
                if valid.sum() > 100:
                    gt_v = gt_depth[valid]
                    r_v = rendered_depth[valid]
                    with torch.no_grad():
                        scale_ratio = (r_v.median() / gt_v.median()).clamp(min=1e-6)
                    gt_aligned = gt_v * scale_ratio
                    depth_l1 = (r_v - gt_aligned).abs().mean()
                    loss = loss + self.lambda_depth * depth_l1

            # Motion regularizers (Stage 1) — 100 iter warmup sonrası
            if it > 100 and (
                self.lambda_deform_reg > 0
                or self.lambda_smoothness > 0
                or self.lambda_rigidity > 0
            ):
                dpos, dquat, dscale = self.deform(
                    self.gs.means, t_norm, self.scene_extent,
                )
                if self.lambda_deform_reg > 0:
                    reg = dpos.pow(2).mean() + dquat.pow(2).mean() + dscale.pow(2).mean()
                    loss = loss + self.lambda_deform_reg * reg
                if self.lambda_smoothness > 0:
                    t2 = min(1.0, t_norm + self.smoothness_dt)
                    if t2 != t_norm:
                        dpos2, _, _ = self.deform(self.gs.means, t2, self.scene_extent)
                        smooth = (dpos - dpos2).pow(2).mean()
                        loss = loss + self.lambda_smoothness * smooth
                N = self.gs.num_points
                K = min(self.rigidity_sample_k, N)
                if self.lambda_rigidity > 0 and K >= 8:
                    sample_idx = torch.randint(0, N, (K,), device=self.device)
                    base_pts = self.gs.means[sample_idx]
                    deformed_pts = d_means[sample_idx]
                    dist_base = torch.cdist(base_pts, base_pts)
                    dist_def  = torch.cdist(deformed_pts, deformed_pts)
                    rigid = (dist_base - dist_def).abs().mean()
                    loss = loss + self.lambda_rigidity * rigid

            self.optimizer.zero_grad(set_to_none=False)
            loss.backward()
            self.density.accumulate(self.gs)
            self.optimizer.step()

            # Density control
            if (self.density_start_iter <= it < self.density_end_iter
                    and it % self.density_interval == 0):
                stats = self.density.step(self.gs)
                self._build_optimizer()
                if it % log_interval == 0:
                    print(f"  ↳ density: clone={stats['cloned']} split={stats['split']} "
                          f"prune={stats['pruned']} | {stats['before']} → {stats['after']}")
                torch.cuda.empty_cache()

            # Log
            if it % log_interval == 0:
                with torch.no_grad():
                    p = psnr(rgb, gt)
                history["loss"].append(loss.item())
                history["psnr"].append(p)
                history["n_pts"].append(self.gs.num_points)
                elapsed = time.time() - t0
                ips = it / max(elapsed, 1e-6)
                reg_part = loss.item() - loss_recon.item()
                reg_str = f" (+reg={reg_part:.4f})" if abs(reg_part) > 1e-5 else ""
                print(f"[{it:>6}/{n_iters}] loss={loss.item():.4f}{reg_str} "
                      f"psnr={p:.2f} N={self.gs.num_points:,} | {ips:.1f} it/s")
                if progress_callback is not None:
                    try:
                        progress_callback(it, n_iters, float(loss.item()),
                                          float(p), int(self.gs.num_points))
                    except Exception as _e:
                        print(f"  ⚠ progress_callback exception: {_e}")

            # Checkpoint
            if ckpt_dir is not None and it % ckpt_interval == 0:
                self._save_checkpoint(ckpt_dir, it)

        # Final checkpoint
        if ckpt_dir is not None:
            self._save_checkpoint(ckpt_dir, n_iters, final=True)

        return history

    def _save_checkpoint(self, ckpt_dir: Path, it: int, final: bool = False) -> None:
        ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        suffix = "final" if final else f"{it:06d}"
        p = ckpt_dir / f"ckpt_{suffix}.pt"
        torch.save({
            "iter": it,
            "gs":   self.gs.state_for_save(),
            "deform": self.deform.state_dict(),
            "scene_extent": self.scene_extent,
            "sh_degree": self.gs.sh_degree,
        }, p)
        print(f"  ✓ checkpoint → {p}")
