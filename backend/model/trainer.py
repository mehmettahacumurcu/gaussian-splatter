"""Faz 5 — 4DGS training loop.

Her iterasyonda:
  1) Rastgele bir frame seç (idx → t = idx / (T-1))
  2) Deformation field uygula: (Δpos, Δquat, Δscale)
  3) gsplat ile render et
  4) L1 + (1-SSIM) loss → backward
  5) Density control (her N adımda)
  6) Checkpoint (her M adımda)
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


# Trainer iç progress callback imzası:
#   (iter_idx, total_iters, loss, psnr, num_points) → None
TrainProgressCallback = Callable[[int, int, float, float, int], None]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(rendered: torch.Tensor, gt: torch.Tensor, lambda_ssim: float = 0.2) -> torch.Tensor:
    """
    rendered, gt: (H, W, 3) float [0, 1]
    """
    l1 = (rendered - gt).abs().mean()
    try:
        from pytorch_msssim import ssim
        ssim_val = ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            gt.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0, size_average=True,
        )
        return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)
    except ImportError:
        # SSIM kütüphanesi yoksa sadece L1
        return l1


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((rendered - gt) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else -10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Frame yardımcıları
# ---------------------------------------------------------------------------
def load_frame_tensor(frame_path: Path, target_size: tuple[int, int] | None = None) -> torch.Tensor:
    """PNG → (H, W, 3) float [0, 1]."""
    import cv2
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        # target_size = (W, H)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).float() / 255.0


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
        # LR'ler
        lr_means: float = 1.6e-4,
        lr_scales: float = 5e-3,
        lr_quats: float = 1e-3,
        lr_opacities: float = 5e-2,
        lr_sh_dc: float = 2.5e-3,
        lr_sh_rest: float = 2.5e-3 / 20,
        lr_deform: float = 1e-3,
        # Density control
        density_start_iter: int = 500,
        density_end_iter: int = 15_000,
        density_interval: int = 100,
        densify_grad_threshold: float = 2e-4,
        prune_min_opacity: float = 0.005,
        prune_max_scale: float = 0.1,
        # Loss
        lambda_ssim: float = 0.2,
        # --- Motion regularizers (Stage 1) ---
        # deformation field'in rastgele jitter üretmek yerine anlamlı
        # coherent motion öğrenmesine yardımcı olur.
        lambda_deform_reg: float = 1e-3,    # |Δpos|², |Δquat|², |Δscale|² — küçük tut
        lambda_smoothness: float = 1e-2,    # D(t) - D(t+dt) pürüzsüzlük
        lambda_rigidity: float = 1e-2,      # komşular arası mesafe korunsun
        rigidity_sample_k: int = 512,       # rigidity için kaç gaussian örnekle
        smoothness_dt: float = 0.02,        # t perturb adımı
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

    # ------------------------------------------------------------------
    def _apply_deformation(self, t: float):
        dpos, dquat, dscale = self.deform(self.gs.means, t, self.scene_extent)
        deformed_means  = self.gs.means + dpos
        deformed_quats  = F.normalize(self.gs.quats + dquat, dim=-1)
        deformed_scales = self.gs.get_scales * torch.exp(dscale)
        return deformed_means, deformed_quats, deformed_scales

    # ------------------------------------------------------------------
    def train(
        self,
        frame_paths: Sequence[Path],
        cam_K: torch.Tensor,                # (3, 3)
        cam_w2c_per_frame: Sequence[torch.Tensor],  # her frame için (4, 4)
        n_iters: int = 30_000,
        image_size: tuple[int, int] = (640, 360),
        ckpt_dir: Path | None = None,
        ckpt_interval: int = 1000,
        log_interval: int = 50,
        progress_callback: TrainProgressCallback | None = None,
    ) -> dict:
        """
        Args:
            frame_paths: PNG karelerin yolu (T tane)
            cam_K: tüm frame'ler için aynı intrinsics
            cam_w2c_per_frame: her frame için world-to-camera 4x4
            n_iters: toplam iterasyon
            image_size: (W, H)
        """
        T = len(frame_paths)
        if len(cam_w2c_per_frame) != T:
            raise ValueError(
                f"frame_paths ve cam_w2c_per_frame uzunlukları farklı: {T} vs {len(cam_w2c_per_frame)}"
            )

        cam_K = cam_K.to(self.device)
        w2c_list = [w.to(self.device) for w in cam_w2c_per_frame]
        # Intrinsics'i hedef boyuta ölçekle
        Ws, Hs = image_size
        # Orijinal frame boyutuna göre K skala faktörü
        sample = load_frame_tensor(Path(frame_paths[0]))
        H0, W0 = sample.shape[:2]
        sx, sy = Ws / W0, Hs / H0
        K_scaled = cam_K.clone()
        K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

        # Frame önbellek (RAM yeterse)
        frames_cached: list[torch.Tensor | None] = [None] * T

        history = {"loss": [], "psnr": [], "n_pts": []}
        t0 = time.time()
        for it in range(1, n_iters + 1):
            idx = int(torch.randint(0, T, (1,)).item())
            t_norm = idx / max(T - 1, 1)

            if frames_cached[idx] is None:
                frames_cached[idx] = load_frame_tensor(Path(frame_paths[idx]), (Ws, Hs))
            gt = frames_cached[idx].to(self.device)

            # Deformation
            d_means, d_quats, d_scales = self._apply_deformation(t_norm)

            # Render
            rgb, _alpha, _info = render_view(
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
            )

            # Reconstruction loss (L1 + SSIM)
            loss_recon = compute_loss(rgb, gt, self.lambda_ssim)
            loss = loss_recon

            # --- Motion regularizers (Stage 1) ---
            # Deformation yeterince ısındıktan sonra devreye gir (ilk 100 iter bekle)
            # ve sadece deformation field öğrenmeye başladığında etkili olsun.
            if it > 100 and (
                self.lambda_deform_reg > 0
                or self.lambda_smoothness > 0
                or self.lambda_rigidity > 0
            ):
                # Bu t anındaki ham deformation çıktıları (mesafe için tekrar çağırıyoruz
                # çünkü _apply_deformation sadece sonucu döner)
                dpos, dquat, dscale = self.deform(
                    self.gs.means, t_norm, self.scene_extent,
                )

                # (a) Magnitude regularizer — hareket olmayan yerde delta 0'a yaklasın
                if self.lambda_deform_reg > 0:
                    reg = (dpos.pow(2).mean()
                           + dquat.pow(2).mean()
                           + dscale.pow(2).mean())
                    loss = loss + self.lambda_deform_reg * reg

                # (b) Temporal smoothness — t ve t+dt'de yakın deformation
                if self.lambda_smoothness > 0:
                    t2 = min(1.0, t_norm + self.smoothness_dt)
                    if t2 != t_norm:
                        dpos2, _, _ = self.deform(
                            self.gs.means, t2, self.scene_extent,
                        )
                        smooth = (dpos - dpos2).pow(2).mean()
                        loss = loss + self.lambda_smoothness * smooth

                # (c) Isometric rigidity — rastgele K gaussian'ın komşuluk ilişkisi korunsun
                #     (tam cdist NxN belleği yakar; rastgele örnek alıp KxK matrisi çıkarıyoruz)
                N = self.gs.num_points
                K = min(self.rigidity_sample_k, N)
                if self.lambda_rigidity > 0 and K >= 8:
                    sample_idx = torch.randint(0, N, (K,), device=self.device)
                    base_pts = self.gs.means[sample_idx]
                    deformed_pts = d_means[sample_idx]
                    dist_base = torch.cdist(base_pts, base_pts)
                    dist_def = torch.cdist(deformed_pts, deformed_pts)
                    rigid = (dist_base - dist_def).abs().mean()
                    loss = loss + self.lambda_rigidity * rigid

            self.optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # Density gradyanlarini biriktir, sonra step
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
                reg_str = f" (+reg={reg_part:.4f})" if reg_part > 1e-5 else ""
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

    # ------------------------------------------------------------------
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
