"""Faz 5 — 4DGS training loop.

v3 — Stage 2 motion supervision tam:
  - L_recon: L1 + (1-SSIM), opsiyonel mask-weighted
  - L_depth: Metric3D/MiDaS scale-invariant L1
  - L_track: CoTracker 3D-anchored projection L1 (sparse motion supervision)
  - L_deform_reg / L_smoothness / L_rigidity (Stage 1 regularizers)
  - Regularizer'lar için linear warmup (0 → full over first warmup_iters iter)
  - Diagnostics: her log_interval'de Δpos mean/max, her loss bileşeni ayrı
"""
from __future__ import annotations
import math
import time
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, Callable, Sequence

from .gaussian_model import GaussianModel
from .deformation import DeformationField, decode_fourier_trajectory
from .renderer import render_view
from .density_control import DensityController


TrainProgressCallback = Callable[[int, int, float, float, int], None]


# ---------------------------------------------------------------------------
# Loss primitives
# ---------------------------------------------------------------------------
def compute_recon_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_ssim: float = 0.2,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 + (1-SSIM), opsiyonel pixel weighting."""
    diff = (rendered - gt).abs()
    if pixel_weight is not None:
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
# 4D Quality v6.1 — Madde 11: Adaptive SH degree schedule
# ---------------------------------------------------------------------------
def progressive_sh_degree(
    iter_idx: int, n_iters: int, max_degree: int = 3
) -> int:
    """Iter'a göre SH degree dön: %25/%50/%75/%100 bandlarinda 0/1/2/3.

    Erken iter'lerde DC + 1st order yeterken, late iter'lerde 3rd order detail
    kazandırır. Bu schedule ~0.3-0.5 dB PSNR artışı + daha temiz convergence.
    Standart 3DGS practice (INRIA reference impl).
    """
    if max_degree <= 0:
        return 0
    frac = iter_idx / max(n_iters, 1)
    if frac < 0.25:
        return 0
    elif frac < 0.50:
        return min(1, max_degree)
    elif frac < 0.75:
        return min(2, max_degree)
    return max_degree


# ---------------------------------------------------------------------------
# 4D Quality v6.1 — Madde 2-B: Mip-Splatting Python-side scale floor
# ---------------------------------------------------------------------------
def apply_mip_scale_floor(
    scales: torch.Tensor, means: torch.Tensor, cam_centers: torch.Tensor,
    floor_frac: float = 0.001,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """3D scale floor — her gauss'a min scale = floor_frac × distance_to_nearest_cam.

    Mip-Splatting paper'inin 3D filter approximation'i. Ekran-uzayinda
    Gaussian'a min pixel-size garantisi vermez ama yakin gauss'larin
    "noktalasmasini" engelliyor (ufak scale → tek piksel artifact'i).

    Memory-safe chunked impl: (N, M, 3) materialize etmek yerine N'i
    chunk_size'a bol → her chunk (chunk, M, 3) tensor (~ chunk×M×12 bytes).
    Default chunk=8192 + M=32 → ~12 MB peak, hizli + safe.

    Args:
      scales: (N, 3) gauss scale (post-activation, units of world)
      means: (N, 3) gauss world position
      cam_centers: (M, 3) sahnedeki cam centers (genelde 32'ye subsample edilmis)
      floor_frac: scale floor = floor_frac × min_dist (0.001-0.005 onerilen)
      chunk_size: gauss chunk boyutu (memory tradeoff)
    """
    if floor_frac <= 0 or cam_centers.numel() == 0:
        return scales
    N = means.shape[0]
    # Output buffer
    min_dist = torch.empty(N, device=means.device, dtype=means.dtype)
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        # (chunk, 1, 3) - (1, M, 3) -> (chunk, M, 3) -> (chunk, M) -> (chunk,)
        diff = means[i:j].unsqueeze(1) - cam_centers.unsqueeze(0)
        d = (diff * diff).sum(-1).sqrt()
        min_dist[i:j] = d.min(dim=1).values
    floor = (floor_frac * min_dist).unsqueeze(1).expand_as(scales)
    return torch.maximum(scales, floor)


# ---------------------------------------------------------------------------
# Frame / depth / mask / tracks yardımcıları
# ---------------------------------------------------------------------------
def load_frame_tensor(
    frame_path: Path,
    target_size: tuple[int, int] | None = None,
    dtype: str = "float32",
) -> torch.Tensor:
    """PNG → (H, W, 3) tensor.

    dtype:
      - "float32" (default, legacy): returns float32 tensor in [0, 1] (CPU).
      - "uint8": returns uint8 tensor (CPU). Caller is responsible for
        converting to float on GPU via `.float() / 255.0` after `.to(device)`.
        ~3.5x memory savings vs float32 — used by preload_to_ram cache.

    Corrupt-cache resilience: cv2.imread returns None for truncated/corrupt
    PNGs (from prior pipeline crashes). We raise FileNotFoundError so the
    upstream cache marker mismatch surfaces clearly. If you'd rather skip
    silently, swap the raise for `return None` + adjust the type hint.
    """
    import cv2
    img = cv2.imread(str(frame_path))
    if img is None:
        # Could be missing OR corrupt. Delete if exists so next run regenerates.
        if Path(frame_path).exists():
            print(f"[frame] WARN: corrupt {Path(frame_path).name}; deleting")
            try:
                Path(frame_path).unlink()
            except Exception:
                pass
        raise FileNotFoundError(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if dtype == "uint8":
        # cv2 returns uint8 by default; keep as-is for memory-efficient cache.
        return torch.from_numpy(np.ascontiguousarray(img))  # uint8 (H, W, 3)
    # Legacy default: float32 [0, 1]
    return torch.from_numpy(img).float() / 255.0


def _load_depth_for_frame(
    depth_dir: Path,
    frame_path: Path,
    target_size: tuple[int, int] | None = None,
    dtype: str = "float32",
) -> torch.Tensor | None:
    """Metric3D/MiDaS depth .npy → (H, W) tensor.

    dtype:
      - "float32" (default, legacy): float32 CPU tensor.
      - "float16": float16 CPU tensor (~2x memory savings). Caller should
        cast to float32 on GPU via `.to(device).float()`.

    Corrupt-cache resilience: if the .npy is truncated / unreadable
    (e.g., because a prior pipeline crash left a partial write),
    delete it + return None instead of crashing the run. Pipeline
    will see no depth for this frame and either regenerate it on
    the next foundation run, or skip depth supervision for it.
    """
    import cv2
    candidate = Path(depth_dir) / f"{Path(frame_path).stem}_depth.npy"
    if not candidate.exists():
        return None
    try:
        arr = np.load(candidate).astype(np.float32)
    except (EOFError, ValueError, OSError, Exception) as e:
        # Truncated / corrupt .npy from a prior crash — delete + skip.
        print(f"[depth] WARN: corrupt {candidate.name} ({type(e).__name__}: {e}); "
              f"deleting + skipping frame")
        try:
            candidate.unlink()
        except Exception:
            pass
        return None
    if target_size is not None:
        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(arr)
    if dtype == "float16":
        t = t.to(torch.float16)
    return t


def _load_mask_for_frame(
    mask_dir: Path,
    frame_path: Path,
    target_size: tuple[int, int] | None = None,
    dtype: str = "float32",
) -> torch.Tensor | None:
    """Dynamic mask .png → (H, W) tensor.

    dtype:
      - "float32" (default, legacy): float32 in [0, 1].
      - "uint8": raw uint8 [0, 255]. Caller converts via `.float() / 255.0`
        on GPU. Used for the preload_to_ram cache (~4x memory savings).
    """
    import cv2
    import re
    stem = Path(frame_path).stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    idx0 = int(m.group(1))
    candidate = Path(mask_dir) / f"mask_{idx0 + 1:04d}.png"
    if not candidate.exists():
        return None
    img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    if dtype == "uint8":
        return torch.from_numpy(np.ascontiguousarray(img))  # uint8 (H, W)
    return torch.from_numpy(img.astype("float32") / 255.0)


def _unproject_pixel_to_world(
    uv: torch.Tensor,         # (N, 2) pixel coords
    depth_at_uv: torch.Tensor,# (N,)   depth values
    K: torch.Tensor,          # (3, 3) intrinsics at UV resolution
    w2c: torch.Tensor,        # (4, 4) world-to-camera
) -> torch.Tensor:
    """Pixel + depth → world 3D. Returns (N, 3)."""
    # camera space: (K^-1 · [u, v, 1]) * depth
    K_inv = torch.linalg.inv(K)
    u, v = uv[:, 0], uv[:, 1]
    ones = torch.ones_like(u)
    pixels_h = torch.stack([u, v, ones], dim=0)  # (3, N)
    rays = K_inv @ pixels_h                       # (3, N), z=1 plane
    cam_pts = rays * depth_at_uv.unsqueeze(0)     # (3, N)
    # cam → world: (w2c)^-1 @ [X, Y, Z, 1]
    c2w = torch.linalg.inv(w2c)
    cam_pts_h = torch.cat([cam_pts, ones.unsqueeze(0)], dim=0)  # (4, N)
    world_h = c2w @ cam_pts_h                                     # (4, N)
    return world_h[:3].T                                          # (N, 3)


def _project_world_to_pixel(
    world_pts: torch.Tensor,  # (N, 3)
    K: torch.Tensor,          # (3, 3)
    w2c: torch.Tensor,        # (4, 4)
) -> tuple[torch.Tensor, torch.Tensor]:
    """World → pixel. Returns (uv (N, 2), valid_mask (N,) — behind-cam filter)."""
    ones = torch.ones(world_pts.shape[0], 1, device=world_pts.device)
    world_h = torch.cat([world_pts, ones], dim=-1)         # (N, 4)
    cam_pts = (w2c @ world_h.T).T                          # (N, 4)
    z = cam_pts[:, 2]
    valid = z > 1e-3
    cam_pts_safe = cam_pts[:, :3] / z.clamp(min=1e-3).unsqueeze(-1)  # (N, 3)
    img_h = (K @ cam_pts_safe.T).T                         # (N, 3)
    uv = img_h[:, :2]                                      # (N, 2)
    return uv, valid


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
        lambda_track: float = 0.0,
        track_sample_k: int = 256,
        # Scale regularizer (v3 — outlier blow-up önleme)
        lambda_scale: float = 1e-3,
        # v3.8: Anisotropy regularizer — streak/needle gaussian fix.
        # max_scale/min_scale ratio threshold üstünde quadratic ceza.
        # banana_demo'da scene'in etrafında uzun parlak çizgiler oluşturdu.
        lambda_aniso: float = 0.0,           # 0 = kapalı, 0.01-0.05 önerilen
        aniso_threshold: float = 5.0,        # ratio max/min < threshold serbest
        # v3.8: Total dpos clamp tightening (per-iter motion cap, fraction of scene)
        dpos_total_cap_frac: float = 0.2,    # v3.6.2 default, 0.05 daha sıkı
        # Opacity reset — INRIA 3DGS trick (her N iter low-opacity'yi topluca reset)
        opacity_reset_interval: int = 3000,
        # v3.7.2: Hard cap on N (0 = sınırsız). Banana ultra'da N=164k oldu, çöktü.
        max_gaussians: int = 0,
        # Warmup
        warmup_iters: int = 2000,
        # v3.6 / Yol C — Per-gaussian Fourier trajectory
        deform_pos_mode: str = "hybrid",     # "mlp" | "fourier" | "hybrid"
        lr_fourier: float = 5e-3,
        lambda_fourier_reg: float = 1e-4,
        # Phase 1.6 — LPIPS perceptual loss
        lambda_lpips: float = 0.0,
        lpips_net: str = "alex",
        lpips_warmup_iters: int = 1000,
        # Phase 1.7 — Multi-view cross-view consistency
        lambda_multiview_consistency: float = 0.0,
        # Phase 1.8 — RAFT optical flow loss
        lambda_flow: float = 0.0,
        flow_warmup_iters: int = 1000,
        # Phase 1.9 — Densify dynamics MV scale
        densify_mv_threshold_scale: float = 0.7,
        # Phase 2.2 — Multi-resolution training schedule [(iter, long_edge), ...]
        multires_schedule=None,
        # Phase 2.3 — Camera pose refinement
        lr_cam_K: float = 0.0,
        lr_cam_w2c: float = 0.0,
        cam_refine_start_iter: int = 5000,
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
        self.lambda_track = lambda_track
        self.track_sample_k = track_sample_k
        self.lambda_scale = lambda_scale
        # v3.8: Anisotropy + tightened clamp
        self.lambda_aniso = float(lambda_aniso)
        self.aniso_threshold = float(aniso_threshold)
        self.dpos_total_cap_frac = float(dpos_total_cap_frac)
        self.opacity_reset_interval = int(opacity_reset_interval)
        self.warmup_iters = max(1, warmup_iters)
        # v3.6 / Yol C
        self.deform_pos_mode = deform_pos_mode
        self.lr_fourier = lr_fourier
        self.lambda_fourier_reg = lambda_fourier_reg
        # Phase 1.6/1.7/1.8 — perceptual + cross-view + flow
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_net = str(lpips_net)
        self.lpips_warmup_iters = int(lpips_warmup_iters)
        self.lambda_multiview_consistency = float(lambda_multiview_consistency)
        self.lambda_flow = float(lambda_flow)
        self.flow_warmup_iters = int(flow_warmup_iters)
        self._lpips_module = None  # lazy init in train()
        # Phase 1.9
        self.densify_mv_threshold_scale = float(densify_mv_threshold_scale)
        # Phase 2.2
        self.multires_schedule = list(multires_schedule) if multires_schedule else []
        # Phase 2.3
        self.lr_cam_K = float(lr_cam_K)
        self.lr_cam_w2c = float(lr_cam_w2c)
        self.cam_refine_start_iter = int(cam_refine_start_iter)
        # Phase 2.1 — Static/Dynamic split (default off; promote_dynamic ile aktiflesir)
        self.use_static_dynamic_split = False
        # Static 3DGS Faz 1 — static mode flag (4D features bypass)
        self.static_mode = False
        # 4D Quality v6.1 — runtime knobs (default off; train() siginatursinde override)
        self.sh_progressive_schedule = False
        self.lambda_accel = 0.0
        self.cam_grad_clip_norm = 0.0
        self.mip_scale_floor_frac = 0.0
        self.dynamic_densify_scale = 1.0
        # Phase 2.3 — Camera pose refine optimizer placeholder (None = inactive)
        self._cam_refine_optimizer = None
        if self.lr_cam_K > 0 or self.lr_cam_w2c > 0:
            print(f"[trainer] Camera pose refinement requested "
                  f"(lr_K={self.lr_cam_K}, lr_w2c={self.lr_cam_w2c}). "
                  f"Aktivasyon train()'de scene cam'larina baglanir.")
        if deform_pos_mode not in ("mlp", "fourier", "hybrid"):
            raise ValueError(f"deform_pos_mode geçersiz: {deform_pos_mode}")
        if deform_pos_mode != "mlp" and (
            self.gs.fourier_pos_coeffs is None or self.gs.fourier_K == 0
        ):
            raise ValueError(
                f"deform_pos_mode='{deform_pos_mode}' ama GaussianModel fourier_K=0. "
                f"GaussianModel'i fourier_K>0 ile başlat."
            )
        print(f"[trainer] deform_pos_mode={deform_pos_mode} fourier_K={self.gs.fourier_K}")

        self.density_start_iter = density_start_iter
        self.density_end_iter = density_end_iter
        self.density_interval = density_interval

        # v3.2 FIX: prune_max_scale is now interpreted as FRACTION of scene_extent
        # (INRIA 3DGS original intent). Previous absolute-unit interpretation caused
        # mass prune on scenes with extent > 1 (cutlemon had extent=70, prune_max=0.1
        # → pruned %35 of initial points at first density step).
        effective_max_scale = prune_max_scale * max(scene_extent, 1e-6)
        print(f"[trainer] prune_max_scale effective: {prune_max_scale} × scene_extent({scene_extent:.2f}) = {effective_max_scale:.3f} units")
        self.density = DensityController(
            grad_threshold=densify_grad_threshold,
            min_opacity=prune_min_opacity,
            max_scale=effective_max_scale,
            max_gaussians=max_gaussians,
        )
        if max_gaussians > 0:
            print(f"[trainer] N hard cap: {max_gaussians:,} (cap dolunca split kapalı, prune devam)")

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
        # v3.6: Fourier trajectory param group (eğer açıksa)
        if self.gs.fourier_pos_coeffs is not None:
            groups.append(
                {"params": [self.gs.fourier_pos_coeffs], "lr": self.lr_fourier}
            )
        self.optimizer = torch.optim.Adam(groups, eps=1e-15)

    def _apply_deformation(self, t: float, return_raw: bool = False):
        # Static 3DGS Faz 1 — static_mode'da hiç deformation yok, identity döner
        if getattr(self, "static_mode", False) or self.deform is None:
            id_out = (
                self.gs.means,
                F.normalize(self.gs.quats, dim=-1),
                self.gs.get_scales,
            )
            if return_raw:
                return id_out, (None, None, None)
            return id_out
        # MLP her zaman çağrılır (dquat, dscale için gerekli, dpos opsiyonel)
        dpos_mlp, dquat, dscale = self.deform(self.gs.means, t, self.scene_extent)
        # v3.5: dscale clamp — MLP'nin scale delta'sı ±2 ile sınırlı.
        # T6 perf fix: clamp'i yeni isimde tut, RAW dscale'i regularizer icin koru.
        dscale_clamped = dscale.clamp(min=-2.0, max=2.0)

        # v3.6.2: dpos_mlp clamp — MLP serbest yazabildiği için 500+ unit
        # jump'lara yol açıyordu (chickchicken_fourier_v2'de Δpos max=513 görüldü).
        # Max motion per iter = scene_extent × 0.1 (yumuşak caydırıcı).
        # T6 perf fix: clamp'i yeni isimde tut, RAW dpos_mlp'i regularizer icin koru.
        mlp_cap = self.scene_extent * 0.1
        dpos_mlp_clamped = dpos_mlp.clamp(min=-mlp_cap, max=mlp_cap)

        # v3.6 / Yol C: Pozisyon deformasyonunu pos_mode'a göre seç
        if self.deform_pos_mode == "mlp":
            dpos = dpos_mlp_clamped
        elif self.deform_pos_mode == "fourier":
            dpos = decode_fourier_trajectory(self.gs.fourier_pos_coeffs, t)
        else:  # "hybrid"
            dpos_fourier = decode_fourier_trajectory(self.gs.fourier_pos_coeffs, t)
            dpos = dpos_mlp_clamped + dpos_fourier

        # v3.6.2 → v3.8: FINAL dpos clamp — overall motion cap
        # Default scene_extent × 0.2; Ultra Clean preset 0.05 (4× daha sıkı).
        # banana_demo Ultra'da Δpos max=13 (cap=21'de, ama görsel streak'leri için sıkıştırma faydalı).
        total_cap = self.scene_extent * self.dpos_total_cap_frac
        dpos = dpos.clamp(min=-total_cap, max=total_cap)

        deformed_means  = self.gs.means + dpos
        deformed_quats  = F.normalize(self.gs.quats + dquat, dim=-1)
        deformed_scales = self.gs.get_scales * torch.exp(dscale_clamped)
        if return_raw:
            # T6: RAW MLP output (pre-clamp) — regularizer block reg=dpos.pow(2)
            # tipi loss'larda kullanir; iki kez forward pass'i onlemek icin paylasiyoruz.
            return (deformed_means, deformed_quats, deformed_scales), (dpos_mlp, dquat, dscale)
        return deformed_means, deformed_quats, deformed_scales

    def _warmup_factor(self, it: int) -> float:
        """
        Linear warmup 0→1 over ilk warmup_iters iter.
        v3.2 fix: hardcoded 100 iter delay kaldırıldı — reg'ler iter 1'den
        itibaren yumuşak devreye giriyor. Önce 0-100 arası unregulated
        training yüzünden deformasyon MLP patlıyordu (Δpos 41, track 196).

        v5.0 multi-view fix: static warmup phase sirasinda warmup=0 (deformation
        donuk). Static phase bittikten sonra warmup_iters icinde 0->1.
        """
        static_phase = getattr(self, "_static_phase_iters", 0)
        if it <= static_phase:
            return 0.0
        return min(1.0, max(0.0, (it - static_phase) / max(1, self.warmup_iters)))

    @torch.no_grad()
    def _setup_static_dynamic_from_masks(
        self,
        masks_mv_dir: Path,
        train_cams: list,
        mv_cam_K: dict,
        mv_w2c: dict,
        threshold_frac: float = 0.10,
    ) -> None:
        """Phase 2.1 — Per-cam motion mask voting ile dynamic gauss seçimi.

        Algoritma:
          - Her cam icin masks_multiview/cam{N}/mask_*.png yukle (binary motion)
          - Her gauss center'i o cam'a project (pinhole)
          - In-frame ise: pixel mask'te ise +1 vote
          - Toplam vote / toplam view > threshold_frac -> DYNAMIC

        Args:
            masks_mv_dir: paths['masks_mv'] (cam{N}/mask_*.png yapisi)
            train_cams: train cam id list ['cam01', ..., 'cam20']
            mv_cam_K, mv_w2c: per-cam intrinsic + extrinsic dict (raw resolution)
            threshold_frac: gauss en az %X view'da motion -> dynamic
        """
        import cv2
        from pathlib import Path

        masks_mv_dir = Path(masks_mv_dir)
        if not masks_mv_dir.exists():
            print(f"[trainer.static_dyn] masks_mv yok: {masks_mv_dir}, skip")
            return
        if not hasattr(self.gs, "promote_dynamic"):
            print(f"[trainer.static_dyn] gs.promote_dynamic API yok, skip")
            return

        n_pts = self.gs.num_points
        device = self.gs.means.device
        dynamic_votes = torch.zeros(n_pts, dtype=torch.int32, device=device)
        n_total_views = 0

        means = self.gs.means.detach()  # [N, 3]
        for cam_id in train_cams:
            cam_mask_dir = masks_mv_dir / cam_id
            if not cam_mask_dir.exists():
                continue
            mask_files = sorted(cam_mask_dir.glob("mask_*.png"))
            if not mask_files:
                continue
            K = mv_cam_K[cam_id].to(device)
            w2c = mv_w2c[cam_id].to(device)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            for mask_path in mask_files:
                m_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m_np is None:
                    continue
                H_m, W_m = m_np.shape
                mask_t = torch.from_numpy(m_np > 127).to(device)

                # Project gauss centers to this cam
                pts_cam = (R @ means.T).T + t  # [N, 3]
                z = pts_cam[:, 2]
                valid_z = z > 0.1
                # Project (in raw resolution that K maps to)
                uv_h = (K @ pts_cam.T).T
                u = uv_h[:, 0] / uv_h[:, 2].clamp(min=0.1)
                v = uv_h[:, 1] / uv_h[:, 2].clamp(min=0.1)
                # Map to mask resolution if different (K and mask should match)
                in_frame = valid_z & (u >= 0) & (u < W_m) & (v >= 0) & (v < H_m)
                if not in_frame.any():
                    continue

                ui = u[in_frame].long().clamp(0, W_m - 1)
                vi = v[in_frame].long().clamp(0, H_m - 1)
                pix_motion = mask_t[vi, ui]  # [N_in_frame] bool
                # Update votes for in-frame gauss
                idx_in = torch.where(in_frame)[0]
                dynamic_votes[idx_in[pix_motion]] += 1
                n_total_views += 1

        if n_total_views == 0:
            print(f"[trainer.static_dyn] Hicbir mask okunamadi, skip")
            return

        threshold = max(1, int(n_total_views * threshold_frac))
        dynamic_mask = dynamic_votes > threshold
        n_promoted = self.gs.promote_dynamic(dynamic_mask)
        self.use_static_dynamic_split = True
        print(f"[trainer.static_dyn] {n_promoted:,} gauss DYNAMIC "
              f"({100 * n_promoted / max(n_pts, 1):.1f}%) "
              f"(threshold {threshold}/{n_total_views} views)")
        print(f"  Static: {self.gs.num_static:,}, Dynamic: {self.gs.num_dynamic:,}")

    @torch.no_grad()
    def _setup_static_dynamic_from_masks_sv(
        self,
        masks_dir: Path,
        frame_paths: list,
        cam_K: torch.Tensor,
        w2c_list: list,
        threshold_frac: float = 0.10,
    ) -> None:
        """Phase 2.1 — Single-view motion mask voting (tek cam, T frame).

        Multi-view varyantının single-view portu. Her gauss tek cam'a
        her frame'de project edilir, mask motion piksellerinde ise vote.
        """
        import cv2
        masks_dir = Path(masks_dir)
        if not masks_dir.exists():
            print(f"[trainer.static_dyn] masks yok: {masks_dir}, skip")
            return
        if not hasattr(self.gs, "promote_dynamic"):
            return

        n_pts = self.gs.num_points
        device = self.gs.means.device
        dynamic_votes = torch.zeros(n_pts, dtype=torch.int32, device=device)
        n_total = 0

        means = self.gs.means.detach()
        K = cam_K.to(device)

        # mask file naming: mask_0001.png (frame index 0-indexed → 1-indexed?)
        # dynamic_mask.py: mask_path = out / f"mask_{i+1:04d}.png" (1-indexed)
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        if not mask_files:
            print(f"[trainer.static_dyn] mask_*.png yok: {masks_dir}")
            return

        T_avail = min(len(mask_files), len(frame_paths), len(w2c_list))
        for t in range(T_avail):
            m_np = cv2.imread(str(mask_files[t]), cv2.IMREAD_GRAYSCALE)
            if m_np is None:
                continue
            H_m, W_m = m_np.shape
            mask_t = torch.from_numpy(m_np > 127).to(device)

            w2c = w2c_list[t].to(device)
            R = w2c[:3, :3]
            tvec = w2c[:3, 3]
            pts_cam = (R @ means.T).T + tvec
            z = pts_cam[:, 2]
            valid_z = z > 0.1
            uv_h = (K @ pts_cam.T).T
            u = uv_h[:, 0] / uv_h[:, 2].clamp(min=0.1)
            v = uv_h[:, 1] / uv_h[:, 2].clamp(min=0.1)
            in_frame = valid_z & (u >= 0) & (u < W_m) & (v >= 0) & (v < H_m)
            if not in_frame.any():
                continue
            ui = u[in_frame].long().clamp(0, W_m - 1)
            vi = v[in_frame].long().clamp(0, H_m - 1)
            pix_motion = mask_t[vi, ui]
            idx_in = torch.where(in_frame)[0]
            dynamic_votes[idx_in[pix_motion]] += 1
            n_total += 1

        if n_total == 0:
            print(f"[trainer.static_dyn] Hicbir mask okunmadi, skip")
            return

        threshold = max(1, int(n_total * threshold_frac))
        dynamic_mask = dynamic_votes > threshold
        n_promoted = self.gs.promote_dynamic(dynamic_mask)
        self.use_static_dynamic_split = True
        print(f"[trainer.static_dyn.sv] {n_promoted:,} gauss DYNAMIC "
              f"({100 * n_promoted / max(n_pts, 1):.1f}%) "
              f"(threshold {threshold}/{n_total} frames)")
        print(f"  Static: {self.gs.num_static:,}, Dynamic: {self.gs.num_dynamic:,}")

    def _preload_all_to_ram(
        self,
        T: int,
        target_size: tuple[int, int],
        is_multiview: bool,
        # SV
        frame_paths: Sequence[Path] | None = None,
        depth_dir: Path | None = None,
        mask_dir: Path | None = None,
        frames_cached: list | None = None,
        depth_cached: list | None = None,
        mask_cached: list | None = None,
        # MV
        train_cams: list | None = None,
        mv_frame_paths: dict | None = None,
        depth_mv_dir: Path | None = None,
        masks_mv_dir: Path | None = None,  # Reserved; MV does not preload masks (consumed at startup only)
        frames_cached_mv: dict | None = None,
        depth_cached_mv: dict | None = None,
        max_workers: int = 16,
    ) -> None:
        """Eagerly load all frames/depth/masks into RAM at master resolution.

        Uses ThreadPoolExecutor — most time is spent in cv2.imread/np.load (IO + decode),
        which releases the GIL. Each task writes to a pre-allocated cache slot.

        Cache dtypes (memory-efficient):
          - frames: uint8 (3.5x savings vs float32)
          - depth:  fp16  (2x savings)
          - mask:   uint8 (4x savings)
        Caller (per-iter load site) converts to float32 on GPU after .to(device).
        """
        # Build the task list. Each task = (cache_slot_setter, callable that returns tensor).
        tasks: list[tuple[Callable[[torch.Tensor | None], None], Callable[[], torch.Tensor | None]]] = []

        if is_multiview:
            assert train_cams is not None and mv_frame_paths is not None and frames_cached_mv is not None
            for cam_id in train_cams:
                paths_for_cam = mv_frame_paths[cam_id]
                cache_frames = frames_cached_mv[cam_id]
                cache_depth = depth_cached_mv[cam_id] if depth_cached_mv is not None else None
                for i in range(T):
                    fp = Path(paths_for_cam[i])
                    # RGB
                    def _set_rgb(t, cache=cache_frames, idx=i):
                        cache[idx] = t
                    def _load_rgb(p=fp, ts=target_size):
                        return load_frame_tensor(p, ts, dtype="uint8")
                    tasks.append((_set_rgb, _load_rgb))
                    # Depth
                    if cache_depth is not None and depth_mv_dir is not None:
                        depth_subdir = Path(depth_mv_dir) / cam_id
                        def _set_d(t, cache=cache_depth, idx=i):
                            cache[idx] = t
                        def _load_d(d=depth_subdir, p=fp, ts=target_size):
                            return _load_depth_for_frame(d, p, ts, dtype="float16")
                        tasks.append((_set_d, _load_d))
            # Note: MV path does NOT use per-frame mask cache at training time
            # (mask_cached_mv is not part of the existing trainer state). Masks
            # are consumed only by _setup_static_dynamic_from_masks at startup.
        else:
            assert frame_paths is not None and frames_cached is not None
            for i, fp in enumerate(frame_paths):
                fp = Path(fp)
                # RGB
                def _set_rgb(t, idx=i):
                    frames_cached[idx] = t
                def _load_rgb(p=fp, ts=target_size):
                    return load_frame_tensor(p, ts, dtype="uint8")
                tasks.append((_set_rgb, _load_rgb))
                # Depth
                if depth_cached is not None and depth_dir is not None:
                    def _set_d(t, idx=i):
                        depth_cached[idx] = t
                    def _load_d(d=depth_dir, p=fp, ts=target_size):
                        return _load_depth_for_frame(d, p, ts, dtype="float16")
                    tasks.append((_set_d, _load_d))
                # Mask
                if mask_cached is not None and mask_dir is not None:
                    def _set_m(t, idx=i):
                        mask_cached[idx] = t
                    def _load_m(d=mask_dir, p=fp, ts=target_size):
                        return _load_mask_for_frame(d, p, ts, dtype="uint8")
                    tasks.append((_set_m, _load_m))

        total = len(tasks)
        if total == 0:
            print("[preload] hicbir task yok, skip")
            return

        print(f"[preload] {total} task baslatiliyor "
              f"(workers={max_workers}, target_size={target_size}, "
              f"is_multiview={is_multiview}, T={T})")

        t_start = time.time()
        progress_step = max(1, total // 10)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for setter, loader in tasks:
                fut = ex.submit(loader)
                futures.append((fut, setter))
            for fut, setter in futures:
                try:
                    result = fut.result()
                    setter(result)
                except Exception as e:
                    # Don't kill preload on a single bad file; log + leave None.
                    print(f"[preload] task hatasi: {e}")
                done += 1
                if done % progress_step == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate = done / max(elapsed, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    print(f"[preload] {done}/{total} loaded "
                          f"({elapsed:.1f}s, {rate:.0f} task/s, eta {eta:.1f}s)")

        # Pin host memory so subsequent .to(device, non_blocking=True) is faster.
        # Optional best-effort — wrap in try; skip on failure (e.g. OOM).
        try:
            self._pin_cache_in_place(
                frames_cached_mv if is_multiview else frames_cached,
                depth_cached_mv if (is_multiview and depth_cached_mv is not None) else depth_cached,
                mask_cached if not is_multiview else None,
                is_multiview=is_multiview,
            )
        except Exception as _e:
            print(f"[preload] pin_memory skipped: {_e}")

        elapsed_total = time.time() - t_start
        print(f"[preload] DONE — {total} tensors in {elapsed_total:.1f}s")

    @staticmethod
    def _pin_cache_in_place(frames_obj, depth_obj, mask_obj, is_multiview: bool) -> None:
        """Pin host memory of every cached tensor in place. Best-effort."""
        def _pin_list(lst):
            if lst is None:
                return
            for i, t in enumerate(lst):
                if t is not None and not t.is_pinned():
                    try:
                        lst[i] = t.pin_memory()
                    except Exception:
                        pass
        if is_multiview:
            if frames_obj is not None:
                for c, lst in frames_obj.items():
                    _pin_list(lst)
            if depth_obj is not None:
                for c, lst in depth_obj.items():
                    _pin_list(lst)
        else:
            _pin_list(frames_obj)
            _pin_list(depth_obj)
            _pin_list(mask_obj)

    def _prepare_tracks(
        self,
        tracks_path: Path,
        frame_paths: Sequence[Path],
        cams_K: torch.Tensor,
        cams_w2c: Sequence[torch.Tensor],
        depth_dir: Path | None,
        frame_original_size: tuple[int, int],  # (W_frame, H_frame)
    ) -> dict | None:
        """
        CoTracker tracks'i yükle, frame 0'daki pixel koordinatlarını
        MiDaS depth'iyle 3D world anchor'a lift et.
        Return: {
            "anchors_3d": (N_valid, 3),
            "tracks_2d":  (T_tr, N_valid, 2) — frame resolution'a scale edilmiş,
            "visibility": (T_tr, N_valid),
            "anchor_indices_original": (N_valid,) CoTracker içi index,
        } ya da None.
        """
        if not tracks_path.exists():
            return None
        try:
            td = torch.load(tracks_path, map_location="cpu", weights_only=True)
        except Exception:
            td = torch.load(tracks_path, map_location="cpu")
        tracks_2d_v = td["tracks"][0]       # (T, N, 2) video resolution
        visibility = td["visibility"][0]    # (T, N) bool/float
        v_shape = td["video_shape"]         # (1, T, 3, H_v, W_v)
        H_v, W_v = int(v_shape[-2]), int(v_shape[-1])
        T_tr, N_tr = tracks_2d_v.shape[0], tracks_2d_v.shape[1]
        W_f, H_f = frame_original_size

        # video → frame resolution scaling
        sx = W_f / W_v
        sy = H_f / H_v
        tracks_2d_f = tracks_2d_v.clone().float()
        tracks_2d_f[..., 0] *= sx
        tracks_2d_f[..., 1] *= sy

        # Lift frame 0 tracks to 3D
        if depth_dir is None:
            print("[trainer.track] depth_dir yok — track loss kapalı")
            return None
        depth_0 = _load_depth_for_frame(depth_dir, frame_paths[0])
        if depth_0 is None:
            print("[trainer.track] frame 0 depth bulunamadı — track loss kapalı")
            return None

        tracks_f0 = tracks_2d_f[0]  # (N, 2)
        vis_f0 = visibility[0].bool() if visibility.dtype != torch.bool else visibility[0]

        # Pixel-safe: clamp to [0, W-1] x [0, H-1]
        u0 = tracks_f0[:, 0].clamp(0, W_f - 1).long()
        v0 = tracks_f0[:, 1].clamp(0, H_f - 1).long()
        # Depth lookup
        depth_at_tracks = depth_0[v0, u0]
        # Valid: visible + positive depth
        valid = vis_f0 & (depth_at_tracks > 1e-3)
        if valid.sum().item() < 32:
            print(f"[trainer.track] sadece {valid.sum().item()} geçerli anchor — track loss kapalı")
            return None

        tracks_f0_valid = tracks_f0[valid].to(self.device)      # (N', 2)
        depths_valid = depth_at_tracks[valid].to(self.device)   # (N',)

        # K and w2c for frame 0 at FRAME resolution (not training resolution)
        # cams_K / cams_w2c come in at frame resolution
        K_0 = cams_K.to(self.device)
        w2c_0 = cams_w2c[0].to(self.device)

        anchors_3d = _unproject_pixel_to_world(tracks_f0_valid, depths_valid, K_0, w2c_0)

        # Keep per-frame tracks 2D + visibility for valid subset
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        tracks_2d_valid = tracks_2d_f[:, valid_idx, :].to(self.device)    # (T, N', 2)
        vis_valid = visibility[:, valid_idx].to(self.device)               # (T, N')
        if vis_valid.dtype != torch.bool:
            vis_valid = vis_valid > 0.5

        print(f"[trainer.track] {anchors_3d.shape[0]} 3D anchor hazır "
              f"({T_tr} frame boyunca izleniyor)")
        return {
            "anchors_3d": anchors_3d,
            "tracks_2d":  tracks_2d_valid,
            "visibility": vis_valid,
            "T_tr": T_tr,
        }

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
        tracks_path: Path | None = None,
        # Phase 1.8 single-view — RAFT optical flow path (forward_*.pt)
        flow_dir: Path | None = None,
        run_logger: Any = None,   # RunLogger instance, opsiyonel
        mv_frame_paths=None,
        mv_cam_K=None,
        mv_w2c=None,
        mv_test_camera=None,
        # Phase 1.4 — Per-cam MV depth supervision
        depth_mv_dir: Path | None = None,
        # Phase 1.5 — Per-cam MV dynamic mask (Phase 2.1 promote için)
        masks_mv_dir: Path | None = None,
        # Phase 1.8 — Per-cam MV RAFT optical flow
        flow_mv_dir: Path | None = None,
        # Phase 2.1 — Static/Dynamic auto-promote
        auto_static_dynamic: bool = False,
        static_dynamic_threshold: float = 0.10,
        # Static 3DGS Faz 1 — 4D features bypass
        static_mode: bool = False,
        # 4D Quality v6.1 — runtime knobs
        sh_progressive_schedule: bool = False,
        lambda_accel: float = 0.0,
        cam_grad_clip_norm: float = 0.0,
        mip_scale_floor_frac: float = 0.0,
        dynamic_densify_scale: float = 1.0,
        # Perf — RAM preload + uint8/fp16 cache + multires fix
        preload_to_ram: bool = False,
        # NVS hold-out — frame indices to EXCLUDE from training (single-view path).
        # Eval can then measure true novel-view PSNR on these. None = train on all.
        # Multi-view path uses mv_test_camera for held-out; this is SV-only.
        holdout_indices: Sequence[int] | None = None,
        # Cooperative cancel: callable returning True when training should stop.
        # Polled every iter; if True, the loop exits cleanly and the function
        # returns the partial history. Caller transitions the job to a
        # cancelled/failed state. Default = no-op (training runs to completion).
        cancel_check: Callable[[], bool] | None = None,
        # Edit refit: per-frame inpaint mask. When provided, depth supervision
        # is skipped (zeroed) for masked pixels (no GT depth in inpainted
        # regions).  Shape: (T, H, W) bool, on CPU — fetched per-iter.
        edit_mask_stack: torch.Tensor | None = None,
    ) -> dict:
        # Static mode — 4D dynamic features bypass (deformation, Fourier, motion regs)
        self.static_mode = bool(static_mode)
        if self.static_mode:
            print(f"\n[trainer.STATIC] Static 3DGS modu — 4D dynamic features kapalı")
        # 4D Quality v6.1 — runtime knobs
        self.sh_progressive_schedule = bool(sh_progressive_schedule)
        self.lambda_accel = float(lambda_accel)
        self.cam_grad_clip_norm = float(cam_grad_clip_norm)
        self.mip_scale_floor_frac = float(mip_scale_floor_frac)
        self.dynamic_densify_scale = float(dynamic_densify_scale)
        if self.sh_progressive_schedule:
            print(f"[trainer.v6.1] SH progressive schedule aktif "
                  f"(0→{self.gs.sh_degree} degree over n_iters)")
        if self.lambda_accel > 0:
            print(f"[trainer.v6.1] 2nd-order accel reg aktif (λ={self.lambda_accel})")
        if self.mip_scale_floor_frac > 0:
            print(f"[trainer.v6.1] Mip-Splatting Python-side aktif "
                  f"(scale_floor_frac={self.mip_scale_floor_frac})")
        if self.cam_grad_clip_norm > 0 and (self.lr_cam_K > 0 or self.lr_cam_w2c > 0):
            print(f"[trainer.v6.1] Cam refine grad clip = {self.cam_grad_clip_norm}")
        # v5.0: Multi-view detect
        is_multiview = mv_frame_paths is not None
        # v5.0 ARCHITECTURAL FIX: Static warmup phase.
        # Multi-view'da deformation MLP, statik geometri oturmadan aktive
        # olursa per-cam noise'u "deformasyon" olarak ogrenip patliyor.
        # Standart 4DGaussians yaklasimi: ilk N iter sadece statik gaussian
        # + densify, sonra deformation devreye.
        #
        # v5.0.1 (uzun run optimizasyonu): static phase'i CAP'le. Geometry
        # genelde 5-8k iter'de oturur. 50k iter run icin 25k static phase
        # gereksiz uzun, dynamic phase'e zaman birak. min(n_iters//2, 8000).
        # Single-view'da bu sorun yok, iter 1'den deformation aktif kalabilir.
        # Static 3DGS modu: tüm training boyunca deformation kapalı, warmup gerekmez
        if self.static_mode:
            self._static_phase_iters = 0
        else:
            self._static_phase_iters = min(n_iters // 2, 8000) if is_multiview else 0
        if is_multiview and not self.static_mode:
            print(f"[trainer.mv] Static warmup phase: {self._static_phase_iters} iter "
                  f"(deformation tamamen donuk)")
            # Phase 1.9 — Multi-view densify tuning: threshold'u scale ile carp.
            # MV'de her cam ayri view; densify daha hassas olmali.
            mv_scale = float(getattr(self, "densify_mv_threshold_scale", 1.0))
            if 0.0 < mv_scale < 2.0 and abs(mv_scale - 1.0) > 1e-3:
                old_thr = self.density.grad_threshold
                self.density.grad_threshold = old_thr * mv_scale
                print(f"[trainer.mv] Densify grad_threshold {old_thr:.6f} → "
                      f"{self.density.grad_threshold:.6f} (scale={mv_scale}, MV daha hassas)")

        if is_multiview:
            train_cams = sorted([c for c in mv_frame_paths.keys() if c != mv_test_camera])
            if not train_cams:
                raise ValueError("mv_frame_paths bos veya test_camera disinda cam yok")
            cam_T_set = {len(mv_frame_paths[c]) for c in train_cams}
            if len(cam_T_set) != 1:
                raise ValueError(f"Cam'lar arasinda farkli frame sayisi: {cam_T_set}")
            T = cam_T_set.pop()
            print(f"[trainer.mv] Multi-view: {len(train_cams)} train cam x {T} frame "
                  f"(test_cam={mv_test_camera})")
            ref_cam = train_cams[0]
            cam_K_orig = mv_cam_K[ref_cam].to(self.device)
            w2c_list = None
        else:
            T = len(frame_paths)
            if len(cam_w2c_per_frame) != T:
                raise ValueError(
                    f"frame_paths ve cam_w2c_per_frame uzunluklari farkli: {T} vs {len(cam_w2c_per_frame)}"
                )
            cam_K_orig = cam_K.to(self.device)
            w2c_list = [w.to(self.device) for w in cam_w2c_per_frame]

        Ws, Hs = image_size
        if is_multiview:
            sample = load_frame_tensor(Path(mv_frame_paths[train_cams[0]][0]))
        else:
            sample = load_frame_tensor(Path(frame_paths[0]))
        H0, W0 = sample.shape[:2]             # original frame resolution

        # Phase 2.2 — Multi-resolution schedule resolver
        # multires_schedule: [(start_iter, long_edge), ...]; bos = single resolution.
        def _resolve_long_edge_for_iter(it_check: int) -> int | None:
            sched = self.multires_schedule
            if not sched:
                return None
            current = None
            for (it_start, le) in sorted(sched, key=lambda x: x[0]):
                if it_check >= it_start:
                    current = le
            return current

        def _resize_to_long_edge(le: int, base_w: int, base_h: int):
            ratio = le / max(base_w, base_h)
            return int(round(base_w * ratio)), int(round(base_h * ratio))

        # Initial resolution: schedule var ise schedule[0] kullan, yoksa image_size
        initial_le = _resolve_long_edge_for_iter(0)
        if initial_le is not None:
            Ws, Hs = _resize_to_long_edge(initial_le, W0, H0)
            print(f"[trainer.multires] Initial schedule: long_edge={initial_le} -> ({Ws}x{Hs})")

        # Perf — Master cache resolution.
        # OLD behavior: cache stored tensors at Ws,Hs. Multires resolution change
        # forced full disk re-read of all T frames per cam (5040x cam at 1080p MFS
        # = ~5min thrash per resolution step).
        # NEW: cache stores at LARGEST resolution that will appear in the schedule.
        # Per-iter, downsample on GPU via F.interpolate (bilinear+antialias for RGB,
        # bilinear for depth, nearest for mask). Resolution change is now free —
        # just toggle (Ws, Hs); cache untouched.
        if self.multires_schedule:
            max_le = max(le for (_, le) in self.multires_schedule)
            Wm, Hm = _resize_to_long_edge(max_le, W0, H0)
            # Master must be >= every scheduled resolution; if image_size > schedule
            # max (edge case), keep image_size as master.
            if Wm < image_size[0] or Hm < image_size[1]:
                Wm, Hm = image_size
        else:
            # No schedule — master == single training resolution.
            Wm, Hm = Ws, Hs
        print(f"[trainer.cache] Master cache resolution: ({Wm}x{Hm}) "
              f"(current train ({Ws}x{Hs}); preload_to_ram={preload_to_ram})")

        sx, sy = Ws / W0, Hs / H0
        K_scaled = cam_K_orig.clone()         # training resolution (single-view fallback)
        K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

        # v5.0: Per-cam K_scaled + w2c cache (multi-view)
        K_scaled_mv = {}
        w2c_mv = {}
        if is_multiview:
            for c in train_cams:
                K_c = mv_cam_K[c].to(self.device).clone()
                K_c[0, 0] *= sx; K_c[0, 2] *= sx
                K_c[1, 1] *= sy; K_c[1, 2] *= sy
                K_scaled_mv[c] = K_c
                w2c_mv[c] = mv_w2c[c].to(self.device)

        # Phase 2.3 — Camera pose refine: K_scaled_mv ve w2c_mv'yi nn.Parameter'a wrap et
        cam_refine_active = (
            is_multiview and (self.lr_cam_K > 0 or self.lr_cam_w2c > 0)
        )
        if cam_refine_active:
            cam_refine_params: list[dict] = []
            for c in train_cams:
                if self.lr_cam_K > 0:
                    K_p = torch.nn.Parameter(K_scaled_mv[c].clone().detach())
                    K_scaled_mv[c] = K_p
                    cam_refine_params.append({"params": [K_p], "lr": self.lr_cam_K})
                if self.lr_cam_w2c > 0:
                    w_p = torch.nn.Parameter(w2c_mv[c].clone().detach())
                    w2c_mv[c] = w_p
                    cam_refine_params.append({"params": [w_p], "lr": self.lr_cam_w2c})
            if cam_refine_params:
                self._cam_refine_optimizer = torch.optim.Adam(cam_refine_params)
                print(f"[trainer.cam_refine] Aktif: {len(train_cams)} cam, "
                      f"{len(cam_refine_params)} param group, "
                      f"start_iter={self.cam_refine_start_iter}")
            print(f"[trainer.mv] K scaled per-cam, {len(train_cams)} entries")

        # Phase 2.1 — Static/Dynamic auto-promote.
        # Mask voting ile ilk N iter'den ONCE statik/dinamik secimini yap,
        # boylece training boyunca statik bolgeler deformation BYPASS eder.
        # Static 3DGS Faz 1 — static_mode'da deformation komple kapali, dynamic
        # ayrimi anlamsiz; auto-promote'u atla.
        if self.static_mode:
            auto_static_dynamic = False
        if (is_multiview and auto_static_dynamic and masks_mv_dir is not None
                and hasattr(self.gs, "promote_dynamic")):
            print(f"\n[trainer.static_dyn] Phase 2.1 auto-promote (MV) — "
                  f"mask voting threshold={static_dynamic_threshold}")
            self._setup_static_dynamic_from_masks(
                masks_mv_dir=masks_mv_dir,
                train_cams=train_cams,
                mv_cam_K=mv_cam_K,
                mv_w2c=mv_w2c,
                threshold_frac=static_dynamic_threshold,
            )
        elif (not is_multiview and auto_static_dynamic and mask_dir is not None
              and hasattr(self.gs, "promote_dynamic")):
            print(f"\n[trainer.static_dyn] Phase 2.1 auto-promote (SV) — "
                  f"mask voting threshold={static_dynamic_threshold}")
            self._setup_static_dynamic_from_masks_sv(
                masks_dir=mask_dir,
                frame_paths=frame_paths,
                cam_K=cam_K_orig,
                w2c_list=w2c_list,
                threshold_frac=static_dynamic_threshold,
            )

        # Cache'ler — stored at master resolution (Wm, Hm); per-iter downsampled on GPU.
        # uint8 RGB / fp16 depth / uint8 mask (memory-efficient). Caller converts
        # to float on GPU after .to(device) (free with non_blocking=True).
        frames_cached: list = [None] * T
        depth_cached:  list = [None] * T
        mask_cached:   list = [None] * T
        # v5.0: Multi-view per-cam frame cache
        frames_cached_mv = {c: [None] * T for c in train_cams} if is_multiview else None
        # Phase 1.4 — Multi-view per-cam depth cache
        depth_cached_mv = (
            {c: [None] * T for c in train_cams}
            if (is_multiview and depth_mv_dir is not None and self.lambda_depth > 0)
            else None
        )
        # Phase 1.8 — Multi-view per-cam flow cache (sparse, only if file exists)
        flow_cached_mv = (
            {c: [None] * T for c in train_cams}
            if (is_multiview and flow_mv_dir is not None and self.lambda_flow > 0)
            else None
        )
        # Phase 1.8 single-view — flow cache (single timeline, T frame)
        flow_cached_sv = (
            [None] * T
            if (not is_multiview and flow_dir is not None and self.lambda_flow > 0)
            else None
        )

        use_depth = depth_dir is not None and self.lambda_depth > 0
        use_mask  = mask_dir is not None
        use_depth_mv = (
            is_multiview and depth_mv_dir is not None and self.lambda_depth > 0
        )
        use_flow_mv = (
            is_multiview and flow_mv_dir is not None and self.lambda_flow > 0
        )
        use_flow_sv = (
            not is_multiview and flow_dir is not None and self.lambda_flow > 0
        )
        if use_depth:
            print(f"[trainer] Depth loss aktif (lambda={self.lambda_depth})")
        if use_depth_mv:
            print(f"[trainer.mv] Phase 1.4 Depth-MV supervision: "
                  f"lambda={self.lambda_depth}, dir={depth_mv_dir}")
        if use_flow_mv:
            print(f"[trainer.mv] Phase 1.8 Flow-MV supervision: "
                  f"lambda={self.lambda_flow}, dir={flow_mv_dir}")
        if use_flow_sv:
            print(f"[trainer.sv] Phase 1.8 Flow-SV supervision: "
                  f"lambda={self.lambda_flow}, dir={flow_dir}")
        if use_mask:
            print(f"[trainer] Mask-weighted recon aktif (lambda={self.lambda_mask_motion})")

        # Perf — Eager preload all frames/depth/masks into RAM via ThreadPoolExecutor.
        # Eliminates per-iter MFS network filesystem latency. uint8 RGB (~3.5x mem
        # savings vs float32) + fp16 depth + uint8 mask. Cast to float on GPU after
        # .to(device, non_blocking=True) — free with pinned host memory.
        if preload_to_ram:
            self._preload_all_to_ram(
                T=T,
                target_size=(Wm, Hm),
                is_multiview=is_multiview,
                # SV
                frame_paths=(frame_paths if not is_multiview else None),
                depth_dir=(depth_dir if (not is_multiview and use_depth) else None),
                mask_dir=(mask_dir if (not is_multiview and use_mask) else None),
                frames_cached=(frames_cached if not is_multiview else None),
                depth_cached=(depth_cached if (not is_multiview and use_depth) else None),
                mask_cached=(mask_cached if (not is_multiview and use_mask) else None),
                # MV
                train_cams=(train_cams if is_multiview else None),
                mv_frame_paths=(mv_frame_paths if is_multiview else None),
                depth_mv_dir=(depth_mv_dir if (is_multiview and use_depth_mv) else None),
                masks_mv_dir=(masks_mv_dir if (is_multiview and use_mask) else None),
                frames_cached_mv=(frames_cached_mv if is_multiview else None),
                depth_cached_mv=(depth_cached_mv if (is_multiview and use_depth_mv) else None),
            )

        # Track data (CoTracker 3D anchors)
        track_data = None
        if tracks_path is not None and self.lambda_track > 0:
            try:
                track_data = self._prepare_tracks(
                    tracks_path, frame_paths, cam_K_orig, w2c_list,
                    depth_dir, (W0, H0),
                )
            except Exception as e:
                print(f"⚠ Track loss prep failed: {e}")
                track_data = None
        if track_data is not None:
            print(f"[trainer] Track loss aktif (lambda={self.lambda_track})")

        history = {"loss": [], "psnr": [], "n_pts": []}
        t0 = time.time()

        # v3.2: lr_deform için warmup — ilk warmup_iters iter'de 0 → target'a ramp up
        # Önce yüksek lr_deform MLP'yi patlatıp Δpos=41 spike'ları yaratıyordu.
        # v3.6: param group 6 = deform (MLP), group 7 (varsa) = fourier_pos_coeffs.
        # Her ikisini de warmup'la ramp up.
        DEFORM_GROUP_IDX = 6   # deform MLP
        FOURIER_GROUP_IDX = 7  # fourier (yoksa len < 8)
        # Static 3DGS modu: deformation/fourier optimizer'ları LR=0
        target_lr_deform = 0.0 if self.static_mode else self.lr_deform
        target_lr_fourier = 0.0 if self.static_mode else self.lr_fourier
        has_fourier_group = len(self.optimizer.param_groups) > FOURIER_GROUP_IDX

        # 4D Quality v6.1 — Madde 2-B: Mip-Splatting cam_centers (training boyunca sabit)
        # Performans icin max 32 cam kullaniyoruz — her gauss'in NEAREST cam'i
        # tahmini icin yeterli, distance hesabi sabit-zaman.
        mip_cam_centers = None
        MIP_MAX_CAMS = 32
        if self.mip_scale_floor_frac > 0:
            try:
                cam_centers_list = []
                if is_multiview:
                    for c in train_cams:
                        w2c = mv_w2c[c]
                        cam_center = -w2c[:3, :3].T @ w2c[:3, 3]
                        cam_centers_list.append(cam_center)
                else:
                    for w2c in w2c_list:
                        cam_center = -w2c[:3, :3].T @ w2c[:3, 3]
                        cam_centers_list.append(cam_center)
                if cam_centers_list:
                    full_centers = torch.stack(cam_centers_list).to(self.device)
                    M = full_centers.shape[0]
                    if M > MIP_MAX_CAMS:
                        # Uniform stride subsample (ilk + son + arası eşit aralıklı)
                        stride = max(1, M // MIP_MAX_CAMS)
                        idx = torch.arange(0, M, stride, device=full_centers.device)[:MIP_MAX_CAMS]
                        mip_cam_centers = full_centers[idx]
                        print(f"[trainer.v6.1] Mip-Splatting cam_centers: {M} → {mip_cam_centers.shape[0]} (subsample, performans)")
                    else:
                        mip_cam_centers = full_centers
                        print(f"[trainer.v6.1] Mip-Splatting cam_centers ({mip_cam_centers.shape[0]} cam) hazirlandi")
            except Exception as _e:
                print(f"[trainer.v6.1] Mip cam_centers hata, devre disi: {_e}")
                self.mip_scale_floor_frac = 0.0

        # SV training-index pool — excludes NVS hold-out frames so eval can
        # measure true novel views. MV uses mv_test_camera elsewhere; for MV
        # the pool is unused (train loop samples cam_id then idx from full T).
        if (not is_multiview) and holdout_indices:
            _holdout_set = {int(i) for i in holdout_indices if 0 <= int(i) < T}
            _train_idx_pool = torch.tensor(
                [i for i in range(T) if i not in _holdout_set],
                dtype=torch.long,
            )
            if _train_idx_pool.numel() == 0:
                raise ValueError(
                    f"holdout_indices excluded ALL {T} frames; nothing to train on"
                )
            print(f"[trainer.holdout] SV: {len(_holdout_set)}/{T} frames held out from training "
                  f"(pool size={_train_idx_pool.numel()})")
        else:
            _train_idx_pool = None  # sample uniformly from [0, T)

        for it in range(1, n_iters + 1):
            if cancel_check is not None and cancel_check():
                print(f"[trainer] cancel requested at iter {it}, stopping cleanly")
                break
            if _train_idx_pool is not None:
                idx = int(_train_idx_pool[
                    torch.randint(0, _train_idx_pool.numel(), (1,)).item()
                ].item())
            else:
                idx = int(torch.randint(0, T, (1,)).item())
            t_norm = idx / max(T - 1, 1)
            warmup = self._warmup_factor(it)

            # Update lr_deform + lr_fourier per iter (warmup schedule)
            self.optimizer.param_groups[DEFORM_GROUP_IDX]["lr"] = target_lr_deform * warmup
            if has_fourier_group:
                self.optimizer.param_groups[FOURIER_GROUP_IDX]["lr"] = target_lr_fourier * warmup

            # 4D Quality v6.1 — Madde 11: Adaptive SH degree
            if self.sh_progressive_schedule:
                active_sh_degree = progressive_sh_degree(it, n_iters, self.gs.sh_degree)
            else:
                active_sh_degree = self.gs.sh_degree

            # --- Phase 2.2: Multi-resolution transition ---
            # Perf fix: NO LONGER invalidate the cache on resolution change.
            # The cache holds master-resolution tensors (Wm, Hm); we just toggle
            # (Ws, Hs) and per-iter downsample on GPU below.
            new_le = _resolve_long_edge_for_iter(it)
            if new_le is not None:
                new_Ws, new_Hs = _resize_to_long_edge(new_le, W0, H0)
                if new_Ws != Ws or new_Hs != Hs:
                    print(f"[trainer.multires] Iter {it}: resolution {Ws}x{Hs} → {new_Ws}x{new_Hs} "
                          f"(long_edge={new_le}); cache PRESERVED at master ({Wm}x{Hm})")
                    Ws, Hs = new_Ws, new_Hs
                    sx, sy = Ws / W0, Hs / H0
                    K_scaled = cam_K_orig.clone()
                    K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
                    K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy
                    if is_multiview:
                        for c in train_cams:
                            K_c = mv_cam_K[c].to(self.device).clone()
                            K_c[0, 0] *= sx; K_c[0, 2] *= sx
                            K_c[1, 1] *= sy; K_c[1, 2] *= sy
                            K_scaled_mv[c] = K_c
                    # Flow cache: flow tensors are pre-resized per-iter via F.interpolate
                    # already (see flow blocks below), so safe to keep. We invalidate
                    # only flow caches that were stored at the OLD training resolution
                    # (defensive — they're loaded lazily, just clear stale entries).
                    # Note: existing flow code does its own per-iter F.interpolate, so
                    # we can leave them alone — but to stay consistent with prior
                    # behavior (flow magnitudes computed at training resolution), we
                    # keep the lazy load-cached-at-load-time approach: just drop them.
                    if is_multiview and flow_cached_mv is not None:
                        for c in train_cams:
                            flow_cached_mv[c] = [None] * T
                    if not is_multiview and flow_cached_sv is not None:
                        flow_cached_sv = [None] * T

            # GPU-side resize helpers — convert cached uint8/fp16 master tensor
            # to float32 at current training resolution (Ws, Hs).
            def _gt_rgb_from_cache(t_cpu: torch.Tensor) -> torch.Tensor:
                """Cached uint8 (H, W, 3) → device float32 (Hs, Ws, 3) in [0, 1]."""
                t = t_cpu.to(self.device, non_blocking=True)
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                else:
                    # Legacy float32 cache (preload_to_ram=False, dtype default)
                    t = t.float()
                # Resize if not at training resolution.
                # cache shape is (H_cache, W_cache, 3). F.interpolate wants NCHW.
                Hc, Wc = t.shape[:2]
                if Wc != Ws or Hc != Hs:
                    t_chw = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, Hc, Wc)
                    t_chw = F.interpolate(
                        t_chw, size=(Hs, Ws), mode="bilinear",
                        align_corners=False, antialias=True,
                    )
                    t = t_chw.squeeze(0).permute(1, 2, 0).contiguous()
                return t

            def _gt_depth_from_cache(t_cpu: torch.Tensor) -> torch.Tensor:
                """Cached fp16/fp32 (H, W) → device float32 (Hs, Ws)."""
                t = t_cpu.to(self.device, non_blocking=True).float()
                Hc, Wc = t.shape[:2]
                if Wc != Ws or Hc != Hs:
                    t_chw = t.unsqueeze(0).unsqueeze(0)  # (1, 1, Hc, Wc)
                    t_chw = F.interpolate(
                        t_chw, size=(Hs, Ws), mode="bilinear",
                        align_corners=False,
                    )
                    t = t_chw.squeeze(0).squeeze(0)
                return t

            def _gt_mask_from_cache(t_cpu: torch.Tensor) -> torch.Tensor:
                """Cached uint8/float (H, W) → device float32 (Hs, Ws) in [0, 1]."""
                t = t_cpu.to(self.device, non_blocking=True)
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                else:
                    t = t.float()
                Hc, Wc = t.shape[:2]
                if Wc != Ws or Hc != Hs:
                    t_chw = t.unsqueeze(0).unsqueeze(0)
                    t_chw = F.interpolate(
                        t_chw, size=(Hs, Ws), mode="nearest",
                    )
                    t = t_chw.squeeze(0).squeeze(0)
                return t

            # --- Ground truth load ---
            if is_multiview:
                # v5.0: random cam selection
                cam_id = train_cams[int(torch.randint(0, len(train_cams), (1,)).item())]
                if frames_cached_mv[cam_id][idx] is None:
                    # Lazy disk read at master resolution (cache survives multires).
                    frames_cached_mv[cam_id][idx] = load_frame_tensor(
                        Path(mv_frame_paths[cam_id][idx]), (Wm, Hm),
                        dtype=("uint8" if preload_to_ram else "float32"),
                    )
                gt = _gt_rgb_from_cache(frames_cached_mv[cam_id][idx])
                # Phase 1.4 — Per-cam depth load (MV path)
                gt_depth = None
                if use_depth_mv:
                    if depth_cached_mv[cam_id][idx] is None:
                        # frame_path'in stem'inden depth dosyası adı
                        frame_stem = Path(mv_frame_paths[cam_id][idx]).stem
                        depth_path = depth_mv_dir / cam_id / f"{frame_stem}_depth.npy"
                        if depth_path.exists():
                            depth_cached_mv[cam_id][idx] = _load_depth_for_frame(
                                depth_mv_dir / cam_id,
                                Path(mv_frame_paths[cam_id][idx]),
                                (Wm, Hm),
                                dtype=("float16" if preload_to_ram else "float32"),
                            )
                    if depth_cached_mv[cam_id][idx] is not None:
                        gt_depth = _gt_depth_from_cache(depth_cached_mv[cam_id][idx])
                frame_mask = None
                K_active = K_scaled_mv[cam_id]
                w2c_active = w2c_mv[cam_id]
            else:
                if frames_cached[idx] is None:
                    frames_cached[idx] = load_frame_tensor(
                        Path(frame_paths[idx]), (Wm, Hm),
                        dtype=("uint8" if preload_to_ram else "float32"),
                    )
                gt = _gt_rgb_from_cache(frames_cached[idx])

                if use_depth and depth_cached[idx] is None:
                    depth_cached[idx] = _load_depth_for_frame(
                        depth_dir, frame_paths[idx], (Wm, Hm),
                        dtype=("float16" if preload_to_ram else "float32"),
                    )
                gt_depth = (
                    _gt_depth_from_cache(depth_cached[idx])
                    if (use_depth and depth_cached[idx] is not None) else None
                )

                if use_mask and mask_cached[idx] is None:
                    mask_cached[idx] = _load_mask_for_frame(
                        mask_dir, frame_paths[idx], (Wm, Hm),
                        dtype=("uint8" if preload_to_ram else "float32"),
                    )
                frame_mask = (
                    _gt_mask_from_cache(mask_cached[idx])
                    if (use_mask and mask_cached[idx] is not None) else None
                )

                K_active = K_scaled
                w2c_active = w2c_list[idx]

            # --- Deformation + render ---
            # v5.0: Static phase'de deformation tamamen bypass — MLP init
            # weights nonzero olabilir, sadece lr=0 yapmak yetersiz, forward
            # pass'i da kapatmak gerek. Statik gaussian'lar direkt render edilir.
            # T6 perf fix: raw_dpos/raw_dquat/raw_dscale = MLP'nin pre-clamp output'u,
            # regularizer block'da yeniden hesaplanmasin diye saklayalim.
            raw_dpos = raw_dquat = raw_dscale = None
            if it <= getattr(self, "_static_phase_iters", 0):
                d_means = self.gs.means
                d_quats = F.normalize(self.gs.quats, dim=-1)
                d_scales = self.gs.get_scales
            else:
                (d_means, d_quats, d_scales), (raw_dpos, raw_dquat, raw_dscale) = \
                    self._apply_deformation(t_norm, return_raw=True)
                # Phase 2.1 + 2.4 — Static + Background gaussian'lari deformation'dan bypass et.
                # is_static=True veya is_background=True olanlar undeformed kalir.
                bypass_mask = None
                if (self.use_static_dynamic_split
                        and hasattr(self.gs, "is_static")):
                    bypass_mask = self.gs.is_static.to(d_means.device)
                if hasattr(self.gs, "is_background") and self.gs.is_background.any():
                    bg = self.gs.is_background.to(d_means.device)
                    bypass_mask = bg if bypass_mask is None else (bypass_mask | bg)
                if bypass_mask is not None and bypass_mask.any():
                    means_undef = self.gs.means
                    quats_undef = F.normalize(self.gs.quats, dim=-1)
                    scales_undef = self.gs.get_scales
                    d_means = torch.where(bypass_mask.unsqueeze(-1), means_undef, d_means)
                    d_quats = torch.where(bypass_mask.unsqueeze(-1), quats_undef, d_quats)
                    d_scales = torch.where(bypass_mask.unsqueeze(-1), scales_undef, d_scales)

            # 4D Quality v6.1 — Madde 2-B: Mip-Splatting 3D scale floor
            # Anti-aliasing approximation — yakin gauss'larin tek-piksel'e
            # cokmesini engeller, yumusak edge'ler verir.
            if self.mip_scale_floor_frac > 0 and mip_cam_centers is not None:
                d_scales = apply_mip_scale_floor(
                    d_scales, d_means, mip_cam_centers,
                    floor_frac=self.mip_scale_floor_frac,
                )

            render_out, _alpha, _info = render_view(
                means=d_means, quats=d_quats, scales=d_scales,
                opacities=self.gs.get_opacities, colors=self.gs.get_colors,
                K=K_active, w2c=w2c_active,
                width=Ws, height=Hs,
                sh_degree=active_sh_degree,
                with_depth=((use_depth and not is_multiview) or use_depth_mv),
            )
            # Phase 1.4 fix: with_depth MV path'te de True olabilir, RGB extract et
            if use_depth or use_depth_mv:
                rgb = render_out[..., :3]
                rendered_depth = render_out[..., 3]
            else:
                rgb = render_out
                rendered_depth = None

            # --- Reconstruction loss ---
            pixel_weight = None
            if frame_mask is not None and self.lambda_mask_motion > 0:
                pixel_weight = 1.0 + self.lambda_mask_motion * frame_mask
            loss_recon = compute_recon_loss(rgb, gt, self.lambda_ssim, pixel_weight=pixel_weight)
            loss = loss_recon

            # Perf v2: defer .item() — store detached tensors during accumulation, sync once at log time.
            # Eskisi her iter ~12 cuda sync (her .item() bir host-device sync). Yeni: sadece log_interval'de bir kere.
            will_log = (it % log_interval == 0)
            comp_t: dict = {"recon": loss_recon.detach()}

            # --- Phase 1.6: LPIPS perceptual loss ---
            # rgb / gt: [H, W, 3] in [0, 1] -> [3, H, W] beklenir
            if self.lambda_lpips > 0:
                if self._lpips_module is None:
                    from .losses_perceptual import LPIPSLoss
                    self._lpips_module = LPIPSLoss(net=self.lpips_net)
                lpips_warmup = min(1.0, it / max(1, self.lpips_warmup_iters))
                pred_chw = rgb.permute(2, 0, 1).contiguous()
                gt_chw = gt.permute(2, 0, 1).contiguous()
                lpips_val = self._lpips_module(pred_chw, gt_chw)
                if torch.isfinite(lpips_val):
                    loss = loss + self.lambda_lpips * lpips_warmup * lpips_val
                    comp_t["lpips"] = lpips_val.detach()

            # --- Phase 1.7: Multi-view consistency loss ---
            # Aynı t'de farkli bir cam'da second render + recon. 2 cam supervision
            # ile single-cam overfit baski. ~%20 iter time art.
            # Maliyet: ekstra render. Sadece is_multiview ve lambda > 0.
            if (is_multiview and self.lambda_multiview_consistency > 0
                    and len(train_cams) >= 2):
                try:
                    # Farkli bir cam sec
                    other_cams = [c for c in train_cams if c != cam_id]
                    cam_id2 = other_cams[int(torch.randint(0, len(other_cams), (1,)).item())]
                    if frames_cached_mv[cam_id2][idx] is None:
                        frames_cached_mv[cam_id2][idx] = load_frame_tensor(
                            Path(mv_frame_paths[cam_id2][idx]), (Wm, Hm),
                            dtype=("uint8" if preload_to_ram else "float32"),
                        )
                    gt2 = _gt_rgb_from_cache(frames_cached_mv[cam_id2][idx])
                    K2 = K_scaled_mv[cam_id2]
                    w2c2 = w2c_mv[cam_id2]
                    render_out2, _, _ = render_view(
                        means=d_means, quats=d_quats, scales=d_scales,
                        opacities=self.gs.get_opacities, colors=self.gs.get_colors,
                        K=K2, w2c=w2c2,
                        width=Ws, height=Hs,
                        sh_degree=active_sh_degree,
                        with_depth=False,
                    )
                    rgb2 = render_out2 if render_out2.shape[-1] == 3 else render_out2[..., :3]
                    mv_l1 = (rgb2 - gt2).abs().mean()
                    if torch.isfinite(mv_l1):
                        loss = loss + self.lambda_multiview_consistency * mv_l1
                        comp_t["mv_consist"] = mv_l1.detach()
                except Exception as _e:
                    if it < 50:
                        print(f"  ⚠ mv_consistency failed iter {it}: {_e}")

            # --- Phase 1.8: RAFT optical flow supervision (multi-view) ---
            # Yaklasim A (proxy): t ve t+1 frame'lerini ayni cam'da render et,
            # RGB diff magnitude'i RAFT flow magnitude ile L1 ile esitle.
            # Gercek 2D motion vector degil — proxy ama motion supervision sinyali iyi.
            if (use_flow_mv and self.lambda_flow > 0
                    and idx < T - 1
                    and it > getattr(self, "_static_phase_iters", 0)):
                try:
                    flow_path = flow_mv_dir / cam_id / f"forward_{idx:04d}.pt"
                    if flow_path.exists():
                        if flow_cached_mv[cam_id][idx] is None:
                            flow_cached_mv[cam_id][idx] = torch.load(
                                flow_path, map_location=self.device
                            ).float()
                        gt_flow = flow_cached_mv[cam_id][idx]  # [2, H_raw, W_raw]
                        # Resize flow to training resolution
                        gt_flow_resized = F.interpolate(
                            gt_flow.unsqueeze(0), size=(Hs, Ws),
                            mode='bilinear', align_corners=False,
                        ).squeeze(0)
                        gt_flow_mag = gt_flow_resized.norm(dim=0)  # [Hs, Ws]
                        # Render frame t+1 same cam (extra render)
                        t_next_norm = (idx + 1) / max(T - 1, 1)
                        d_means_n, d_quats_n, d_scales_n = self._apply_deformation(t_next_norm)
                        if (self.use_static_dynamic_split
                                and hasattr(self.gs, "is_static")):
                            bypass = self.gs.is_static.to(d_means_n.device)
                            if hasattr(self.gs, "is_background"):
                                bypass = bypass | self.gs.is_background.to(d_means_n.device)
                            if bypass.any():
                                m_und = self.gs.means
                                q_und = F.normalize(self.gs.quats, dim=-1)
                                s_und = self.gs.get_scales
                                d_means_n = torch.where(bypass.unsqueeze(-1), m_und, d_means_n)
                                d_quats_n = torch.where(bypass.unsqueeze(-1), q_und, d_quats_n)
                                d_scales_n = torch.where(bypass.unsqueeze(-1), s_und, d_scales_n)
                        ro_n, _, _ = render_view(
                            means=d_means_n, quats=d_quats_n, scales=d_scales_n,
                            opacities=self.gs.get_opacities, colors=self.gs.get_colors,
                            K=K_active, w2c=w2c_active,
                            width=Ws, height=Hs,
                            sh_degree=active_sh_degree,
                            with_depth=False,
                        )
                        rgb_n = ro_n if ro_n.shape[-1] == 3 else ro_n[..., :3]
                        rendered_diff_mag = (rgb_n - rgb).abs().mean(dim=-1)  # [Hs, Ws]
                        # Normalize gt_flow magnitude by max + threshold high-motion regions
                        flow_warmup = min(1.0, it / max(1, self.flow_warmup_iters))
                        # Loss: rendered diff magnitude high motion'da yuksek olmali
                        gt_norm = gt_flow_mag / (gt_flow_mag.max() + 1e-6)
                        diff_norm = rendered_diff_mag / (rendered_diff_mag.max() + 1e-6)
                        flow_l1 = (diff_norm - gt_norm).abs().mean().clamp(max=1.0)
                        if torch.isfinite(flow_l1):
                            loss = loss + self.lambda_flow * flow_warmup * flow_l1
                            comp_t["flow"] = flow_l1.detach()
                except Exception as _e:
                    if it < 50:
                        print(f"  ⚠ flow loss failed iter {it}: {_e}")

            # --- Phase 1.8 SV: RAFT optical flow supervision (single-view) ---
            # MV varyantın single-view portu. flow_dir/forward_*.pt cache'inden
            # gt flow yukle, t+1 frame'i ayni cam'da render et, RGB diff vs gt L1.
            if (use_flow_sv and self.lambda_flow > 0
                    and idx < T - 1
                    and it > getattr(self, "_static_phase_iters", 0)
                    and not is_multiview):
                try:
                    flow_path = flow_dir / f"forward_{idx:04d}.pt"
                    if flow_path.exists():
                        if flow_cached_sv[idx] is None:
                            flow_cached_sv[idx] = torch.load(
                                flow_path, map_location=self.device
                            ).float()
                        gt_flow = flow_cached_sv[idx]
                        gt_flow_resized = F.interpolate(
                            gt_flow.unsqueeze(0), size=(Hs, Ws),
                            mode='bilinear', align_corners=False,
                        ).squeeze(0)
                        gt_flow_mag = gt_flow_resized.norm(dim=0)
                        # Render frame t+1 same cam pose (single-view next frame'in pose'u list'te)
                        t_next_norm = (idx + 1) / max(T - 1, 1)
                        d_means_n, d_quats_n, d_scales_n = self._apply_deformation(t_next_norm)
                        if (self.use_static_dynamic_split
                                and hasattr(self.gs, "is_static")):
                            bypass = self.gs.is_static.to(d_means_n.device)
                            if hasattr(self.gs, "is_background"):
                                bypass = bypass | self.gs.is_background.to(d_means_n.device)
                            if bypass.any():
                                m_und = self.gs.means
                                q_und = F.normalize(self.gs.quats, dim=-1)
                                s_und = self.gs.get_scales
                                d_means_n = torch.where(bypass.unsqueeze(-1), m_und, d_means_n)
                                d_quats_n = torch.where(bypass.unsqueeze(-1), q_und, d_quats_n)
                                d_scales_n = torch.where(bypass.unsqueeze(-1), s_und, d_scales_n)
                        # Single-view next frame için K + w2c (cam_K_orig + w2c_list[idx+1])
                        ro_n, _, _ = render_view(
                            means=d_means_n, quats=d_quats_n, scales=d_scales_n,
                            opacities=self.gs.get_opacities, colors=self.gs.get_colors,
                            K=K_scaled, w2c=w2c_list[idx + 1],
                            width=Ws, height=Hs,
                            sh_degree=active_sh_degree,
                            with_depth=False,
                        )
                        rgb_n = ro_n if ro_n.shape[-1] == 3 else ro_n[..., :3]
                        rendered_diff_mag = (rgb_n - rgb).abs().mean(dim=-1)
                        flow_warmup = min(1.0, it / max(1, self.flow_warmup_iters))
                        gt_norm = gt_flow_mag / (gt_flow_mag.max() + 1e-6)
                        diff_norm = rendered_diff_mag / (rendered_diff_mag.max() + 1e-6)
                        flow_l1 = (diff_norm - gt_norm).abs().mean().clamp(max=1.0)
                        if torch.isfinite(flow_l1):
                            loss = loss + self.lambda_flow * flow_warmup * flow_l1
                            comp_t["flow"] = flow_l1.detach()
                except Exception as _e:
                    if it < 50:
                        print(f"  ⚠ flow_sv loss failed iter {it}: {_e}")

            # --- Depth consistency (log-space, warmup-gated, clamped) ---
            # Log-space L1 scale-invariant ve outlier'a karşı dayanıklı.
            # Warmup gate: ilk 100 iter statik GS otursun, sonra depth devreye.
            # Clamp(2.0): tek frame outlier'ı tüm run'ı batırmasın.
            if (use_depth or use_depth_mv) and gt_depth is not None and rendered_depth is not None:
                valid = (gt_depth > 0.01) & (rendered_depth > 0.01)
                # Edit refit: inpainted pixels have no reliable GT depth —
                # exclude them from depth supervision.
                if edit_mask_stack is not None:
                    inpaint_m = edit_mask_stack[idx].to(self.device).bool()
                    # Resize mask to match depth map spatial dims if needed
                    if inpaint_m.shape != valid.shape:
                        inpaint_m = torch.nn.functional.interpolate(
                            inpaint_m.float().unsqueeze(0).unsqueeze(0),
                            size=valid.shape[-2:],
                            mode="nearest",
                        ).squeeze(0).squeeze(0).bool()
                    valid = valid & (~inpaint_m)
                if valid.sum() > 100:
                    gt_v = gt_depth[valid].clamp(min=0.01, max=100.0)
                    r_v  = rendered_depth[valid].clamp(min=0.01, max=100.0)
                    log_gt = torch.log(gt_v)
                    log_r  = torch.log(r_v)
                    with torch.no_grad():
                        shift = log_r.median() - log_gt.median()
                    log_gt_aligned = log_gt + shift
                    depth_l1 = (log_r - log_gt_aligned).abs().mean().clamp(max=2.0)
                    loss = loss + self.lambda_depth * warmup * depth_l1
                    comp_t["depth"] = depth_l1.detach()

            # --- Track loss (CoTracker 3D-anchored projection) ---
            if track_data is not None and idx < track_data["T_tr"]:
                anchors = track_data["anchors_3d"]                  # (N, 3)
                gt_tracks = track_data["tracks_2d"][idx]            # (N, 2) frame-res pixel
                vis = track_data["visibility"][idx]                 # (N,) bool

                K_scaled_frame = K_scaled  # training res ile kıyaslayacağız; tracks'i scale edelim
                gt_tracks_scaled = gt_tracks.clone()
                gt_tracks_scaled[..., 0] *= sx
                gt_tracks_scaled[..., 1] *= sy

                # Sample K tracks for speed
                # Perf v2: use sum without .item() — bool() in if catches it but only when track loss enabled.
                # Default config has lambda_track > 0, so this still costs one sync; acceptable.
                N_visible = int(vis.sum().item())
                if N_visible >= 16:
                    sample_size = min(self.track_sample_k, N_visible)
                    vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(-1)
                    sel = vis_idx[torch.randperm(vis_idx.shape[0], device=self.device)[:sample_size]]
                    anchors_sel = anchors[sel]
                    gt_uv = gt_tracks_scaled[sel]

                    # Apply deformation at t to anchors (static base + delta)
                    # v3.6.1: HEM MLP hem nearest-gaussian fourier trajectory.
                    # Bu sayede track loss gradient'i hem MLP'ye hem Fourier'a akar.
                    # Önceden sadece MLP'ye akıyordu → Fourier motion rastgele öğreniliyordu.
                    d_anchors_mlp, _, _ = self.deform(anchors_sel, t_norm, self.scene_extent)
                    if (self.deform_pos_mode != "mlp" and
                            self.gs.fourier_pos_coeffs is not None):
                        # KNN: her anchor için en yakın gaussian
                        with torch.no_grad():
                            dists = torch.cdist(anchors_sel, self.gs.means)  # (A, N)
                            nearest_idx = dists.argmin(dim=-1)                # (A,)
                        nearest_coeffs = self.gs.fourier_pos_coeffs[nearest_idx]  # (A, K, 2, 3)
                        d_anchors_fourier = decode_fourier_trajectory(nearest_coeffs, t_norm)
                        d_anchors_pos = d_anchors_mlp + d_anchors_fourier
                    else:
                        d_anchors_pos = d_anchors_mlp
                    deformed_anchors = anchors_sel + d_anchors_pos

                    # Project to training image space
                    uv_pred, valid_proj = _project_world_to_pixel(
                        deformed_anchors, K_scaled_frame, w2c_list[idx],
                    )
                    if valid_proj.any():
                        diff = (uv_pred[valid_proj] - gt_uv[valid_proj]).abs().mean()
                        # Normalize by image diagonal so lambda ~ O(1)
                        diag = (Ws ** 2 + Hs ** 2) ** 0.5
                        track_l = diff / diag
                        loss = loss + self.lambda_track * warmup * track_l
                        comp_t["track"] = track_l.detach()

            # --- Motion regularizers (warmup'la scale) ---
            if warmup > 0 and raw_dpos is not None and (
                self.lambda_deform_reg > 0
                or self.lambda_smoothness > 0
                or self.lambda_rigidity > 0
                or self.lambda_accel > 0
            ):
                # T6 perf fix: _apply_deformation'dan gelen RAW MLP output'u kullan,
                # ayni t_norm icin ikinci kez self.deform(...) cagrisi yapma.
                # Onceden bu blok her iter'de duplicate forward yapiyordu.
                dpos, dquat, dscale = raw_dpos, raw_dquat, raw_dscale
                if self.lambda_deform_reg > 0:
                    reg = dpos.pow(2).mean() + dquat.pow(2).mean() + dscale.pow(2).mean()
                    loss = loss + self.lambda_deform_reg * warmup * reg
                    comp_t["deform_reg"] = reg.detach()
                if self.lambda_smoothness > 0:
                    t2 = min(1.0, t_norm + self.smoothness_dt)
                    if t2 != t_norm:
                        dpos2, _, _ = self.deform(self.gs.means, t2, self.scene_extent)
                        smooth = (dpos - dpos2).pow(2).mean()
                        loss = loss + self.lambda_smoothness * warmup * smooth
                        comp_t["smooth"] = smooth.detach()
                # 4D Quality v6.1 — Madde 6: 2nd-order acceleration regularizer
                # D(t-1) - 2D(t) + D(t+1) magnitude. Slow-motion render'da titremeyi azalt.
                # Sample t-dt and t+dt; eger sinirlarda ise yumusakca skip.
                if self.lambda_accel > 0:
                    t_minus = max(0.0, t_norm - self.smoothness_dt)
                    t_plus  = min(1.0, t_norm + self.smoothness_dt)
                    if t_minus < t_norm < t_plus:
                        dpos_m, _, _ = self.deform(self.gs.means, t_minus, self.scene_extent)
                        dpos_p, _, _ = self.deform(self.gs.means, t_plus,  self.scene_extent)
                        accel = (dpos_m - 2.0 * dpos + dpos_p).pow(2).mean()
                        loss = loss + self.lambda_accel * warmup * accel
                        comp_t["accel"] = accel.detach()
                N = self.gs.num_points
                K = min(self.rigidity_sample_k, N)
                if self.lambda_rigidity > 0 and K >= 8:
                    sample_idx = torch.randint(0, N, (K,), device=self.device)
                    base_pts = self.gs.means[sample_idx]
                    deformed_pts = d_means[sample_idx]
                    dist_base = torch.cdist(base_pts, base_pts)
                    dist_def  = torch.cdist(deformed_pts, deformed_pts)
                    rigid = (dist_base - dist_def).abs().mean()
                    loss = loss + self.lambda_rigidity * warmup * rigid
                    comp_t["rigid"] = rigid.detach()

            # --- Scale regularizer v3.1 — ASIMETRIK HINGE ---
            # v3'te symmetric log_scale² formülü tüm scale'leri 1'e itip
            # homogenization yaratmıştı (arka plan gaussian'ları küçüldü, detaylar büyüdü).
            # v3.1: sadece scene_extent'in %5'inden büyük scale'leri cezalandır.
            # Küçük/orta gaussian'lar serbest, sadece outlier bloat'a müdahale.
            if self.lambda_scale > 0:
                # Threshold: log(scene_extent × 0.05). Örn scene_extent=48 → log(2.4)=0.875
                # log_scale > threshold olan kısım cezalandırılır, altı = 0 gradient.
                log_threshold = math.log(max(self.scene_extent * 0.05, 1e-3))
                excess = (self.gs.scales - log_threshold).clamp(min=0)
                scale_reg = excess.pow(2).mean()
                loss = loss + self.lambda_scale * scale_reg
                comp_t["scale"] = scale_reg.detach()

            # --- Anisotropy regularizer v3.8 — STREAK / NEEDLE GAUSSIAN FIX ---
            # banana_demo Ultra'da scene'in etrafında uzun parlak çizgiler oluştu.
            # Sebep: gaussian (0.05, 2, 0.05) gibi extreme aspect ratio (max/min=40).
            # Magnitude reg (yukarıdaki scale_reg) sadece büyük scale'leri cezalandırır,
            # küçük-iğne'ye dokunmaz. Aspect ratio reg eksikti.
            #
            # Quadratic hinge: ratio > threshold (default 5) ise (ratio - threshold)² ceza.
            # threshold = 5 normal 3DGS aspect ratio; 10+ kuyruklu yıldız.
            if self.lambda_aniso > 0 and self.gs.num_points > 0:
                # gs.scales is log-space → linear scales pozitif
                lin_scales = torch.exp(self.gs.scales)  # (N, 3)
                sc_max = lin_scales.max(dim=-1).values  # (N,)
                sc_min = lin_scales.min(dim=-1).values.clamp(min=1e-6)  # (N,)
                aniso_ratio = sc_max / sc_min  # (N,) — ≥ 1
                aniso_excess = (aniso_ratio - self.aniso_threshold).clamp(min=0)
                aniso_reg = aniso_excess.pow(2).mean()
                loss = loss + self.lambda_aniso * warmup * aniso_reg
                comp_t["aniso"] = aniso_reg.detach()

            # --- Fourier trajectory regularizer (v3.6 / Yol C) ---
            # v5.0 (Adim 2): L2 → L1 sparsity. Lasso effect — kucuk coefficient'lari
            # TAM sifira ceker, sadece gercekten motion isteyen gaussian'lar
            # nonzero kalir. Bu implicit static-dynamic separation:
            #   - fourier_pos_coeffs ≈ 0 → gaussian static (motion yok)
            #   - fourier_pos_coeffs > 0 → gaussian dynamic (motion var)
            # Multi-view'da kritik: 100k+ gaussian × 8 K = 800k+ DOF, hepsi
            # birden hareket etmesin diye L1 baskisi gerekli.
            # Frekans agirligi k^2 → k (L1 zaten daha guclu, k^2 fazla agir).
            if (self.lambda_fourier_reg > 0 and
                    self.gs.fourier_pos_coeffs is not None and
                    self.deform_pos_mode != "mlp"):
                K = self.gs.fourier_K
                # Frekans ağırlığı: k (lineer). Yüksek k'lar biraz daha cezalı.
                freq_weights = torch.arange(1, K + 1, device=self.gs.fourier_pos_coeffs.device,
                                            dtype=self.gs.fourier_pos_coeffs.dtype)
                # L1 magnitude: |coeff|
                mag = self.gs.fourier_pos_coeffs.abs()  # (N, K, 2, 3)
                # Sum over sin/cos + xyz, mean over N, weighted sum over K
                per_freq_l1 = mag.sum(dim=(0, 2, 3)) / max(self.gs.num_points, 1)  # (K,)
                fourier_reg = (per_freq_l1 * freq_weights).sum()
                # L1 magnitude fark farkli L2'den. Initial (Adim 2) 5× idi —
                # cok agresif, dynamic motion'i da bastirdi. v5.0.1: 2× multiplier
                # ile recon ve sparsity arasinda denge. Static gaussian'lar
                # cogunlukla sifir, dynamic gaussian'lar yeterli motion freedom.
                loss = loss + self.lambda_fourier_reg * 2.0 * fourier_reg
                comp_t["fourier"] = fourier_reg.detach()

            # --- Fourier SPATIAL smoothness (v3.6.2) ---
            # Komşu gaussian'lar benzer trajectory'ye sahip olsun. Motion diffuse et.
            # Aksi halde sadece track anchor'larına yakın gaussian hareket ediyor,
            # geri kalan statik — chickchicken'da %92 gaussian durgundu.
            # Her iter random K=128 gaussian sample + 1-NN üzerinden smoothness.
            #
            # v5.0 (Adim 2): Multi-view'da BU ISTENMEZ. L1 sparsity ile statik
            # gaussian'lar tam sifir kalsin; spatial smoothness motion bulasimi
            # yaratir, statik bolgeler dynamic'e dogru kayar. Multi-view tespiti
            # icin: train()'e is_multiview parametresi tasimak yerine, daha
            # genel bir flag kullanilsa daha iyi olur. Su an kapali tutuyoruz.
            if (False and  # v5.0: spatial smoothness disabled for multi-view static preservation
                    self.lambda_fourier_reg > 0 and
                    self.gs.fourier_pos_coeffs is not None and
                    self.deform_pos_mode != "mlp" and
                    self.gs.num_points > 16):
                Ns = min(128, self.gs.num_points)
                sample_idx = torch.randint(0, self.gs.num_points, (Ns,), device=self.device)
                sampled_means = self.gs.means[sample_idx].detach()  # (Ns, 3)
                # Her sample için en yakın komşu (kendisi hariç)
                with torch.no_grad():
                    dists = torch.cdist(sampled_means, self.gs.means)  # (Ns, N)
                    # En yakın ilk 2'yi al (0 = kendisi eğer sample gaussian ise)
                    _, nearest_k = torch.topk(dists, k=2, largest=False, dim=-1)
                    # Ikinci en yakını (ilki kendisi olabilir)
                    neighbor_idx = nearest_k[:, 1]  # (Ns,)
                sampled_coeffs = self.gs.fourier_pos_coeffs[sample_idx]      # (Ns, K, 2, 3)
                neighbor_coeffs = self.gs.fourier_pos_coeffs[neighbor_idx]   # (Ns, K, 2, 3)
                spatial_smooth = (sampled_coeffs - neighbor_coeffs).pow(2).mean()
                # Weight: reg'in %50'si kadar — motion'u öldürmeden smooth
                loss = loss + self.lambda_fourier_reg * 0.5 * spatial_smooth
                # comp'ta ayrı tutmayalım, fourier ile toplu logla

            # --- NaN/Inf guard (pre-backward) ---
            # Loss patlıyorsa backward yapma, bu iter'i atla. 30k iter'de
            # tek patlayan iter bile means/scales/quats'u NaN'layıp modeli
            # geri dönülmez kılar — cutlemon_full'de oldu.
            # v3.4: consecutive skip counter — 10 ardışık skip olursa abort
            #
            # Perf v2: bu check her iter SHARED-stream sync gerektiriyor (bool() on
            # cuda scalar). Kritik bir guard oldugu icin tutuluyor — bir iter atlamak
            # bile divergent eder, sonraki iter'lerde model patlar. Sync maliyetinin
            # asagidaki grad/param NaN check'lerini log_interval'a kaydirip net iter
            # baslarinda 1 sync hedefliyoruz.
            if not hasattr(self, "_consecutive_skip"):
                self._consecutive_skip = 0
            if not torch.isfinite(loss):
                self._consecutive_skip += 1
                print(f"  ⚠ iter {it}: loss={loss.item()} non-finite → skip (#{self._consecutive_skip})")
                if run_logger is not None:
                    try:
                        run_logger.warn(
                            f"iter {it} skipped: loss non-finite",
                            consecutive=self._consecutive_skip,
                        )
                    except Exception:
                        pass
                if self._consecutive_skip >= 10:
                    if run_logger is not None:
                        try:
                            run_logger.log_event("training:aborted", iter=it, reason="10+ consecutive non-finite loss")
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Training aborted at iter {it}: 10+ consecutive iters with non-finite loss. "
                        f"Model state diverged (check scales/params). "
                        f"Try: lower lambda_track, raise lambda_smoothness, or reduce lr_deform."
                    )
                self.optimizer.zero_grad(set_to_none=False)
                continue
            self._consecutive_skip = 0

            self.optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # --- Gradient clipping (stability) ---
            # Tüm parametre grup'larından global norm ile clip — divergence'ı engeller.
            # 1.0 conservative, GS için 10.0 daha uygun (büyük LR + densification).
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                max_norm=10.0,
            )

            # --- Post-backward NaN guard on params ---
            # Perf v2: per-iter sync (16+ params x .all()) ciddi maliyet. Divergence
            # nadir ve onceki pre-backward isfinite(loss) guard'ı yakaliyor; bu yuzden
            # log_interval cadence yeterli (her ~50 iter bir kontrol). Ilk 200 iter'de
            # daha sik (init asamasi divergence olasiligi yuksek).
            run_grad_check = will_log or it < 200
            grad_nan = False
            if run_grad_check:
                grad_nan = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for g in self.optimizer.param_groups for p in g["params"]
                )
            if grad_nan:
                print(f"  ⚠ iter {it}: non-finite gradient → skip step")
                self.optimizer.zero_grad(set_to_none=False)
                continue

            self.density.accumulate(self.gs)
            # Edit refit: zero gradients on out-of-zone (frozen) Gaussians so
            # only Gaussians inside the affected zone are updated this step.
            _fm = getattr(self.gs, "_freeze_mask", None)
            if _fm is not None:
                _fm_dev = _fm.to(self.device)
                for _attr in ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest"):
                    _p = getattr(self.gs, _attr, None)
                    if isinstance(_p, torch.nn.Parameter) and _p.grad is not None:
                        _p.grad[_fm_dev] = 0.0
                if self.gs.fourier_pos_coeffs is not None:
                    _p = self.gs.fourier_pos_coeffs
                    if isinstance(_p, torch.nn.Parameter) and _p.grad is not None:
                        _p.grad[_fm_dev] = 0.0
            self.optimizer.step()
            # Phase 2.3 — Camera pose refinement step (warmup sonrasi aktif)
            if (self._cam_refine_optimizer is not None
                    and it >= self.cam_refine_start_iter):
                # 4D Quality v6.1 — Madde 12: cam params icin ayri grad norm clip.
                # Cam parametreleri gauss'larla aynı global clip'e tabi tutmak yerine
                # kendi norm budget'ini kullanir. Large-scale scene'de drift'i azalt.
                if self.cam_grad_clip_norm > 0:
                    cam_params = [p for g in self._cam_refine_optimizer.param_groups
                                  for p in g["params"] if p.grad is not None]
                    if cam_params:
                        torch.nn.utils.clip_grad_norm_(
                            cam_params, max_norm=self.cam_grad_clip_norm,
                        )
                self._cam_refine_optimizer.step()
                self._cam_refine_optimizer.zero_grad(set_to_none=True)

            # --- Post-step param NaN/Inf check — v3.4 expanded ---
            # means + scales + quats hepsi kontrol ediliyor çünkü scales inf'e
            # gitmesi training'i sessizce bozuyordu (chickchicken_v3_3'te log_scale=70+,
            # exp()=inf, her sonraki iter loss non-finite → skipped → silent garbage).
            #
            # Perf v2: 3 ayri .all() reduction + bool() per iter ~3 cuda sync. Asagidaki
            # log_interval gating sadece her ~50 iter bir kontrol; arada NaN olusursa
            # bir sonraki iter'in pre-backward isfinite(loss) guard'ı yakalar ve skip
            # eder, training diverge etmez (clamp'ler yardim eder).
            if run_grad_check:
                means_bad = not torch.isfinite(self.gs.means).all()
                scales_bad = not torch.isfinite(self.gs.scales).all()
                quats_bad = not torch.isfinite(self.gs.quats).all()
                if means_bad or scales_bad or quats_bad:
                    reason = []
                    if means_bad: reason.append("means")
                    if scales_bad: reason.append("scales")
                    if quats_bad: reason.append("quats")
                    print(f"  ⚠⚠ iter {it}: {'+'.join(reason)} NaN/Inf AFTER step — training aborted")
                    if run_logger is not None:
                        try:
                            run_logger.log_event("training:aborted", iter=it, reason=f"{'+'.join(reason)} non-finite post-step")
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Training diverged at iter {it}: {'+'.join(reason)} non-finite. "
                        f"Last good checkpoint: run scripts/export_from_ckpt.py --scene <scene>"
                    )

            # v3.7.4: Hard clamp on log_scale — TIGHTER bound.
            # Önce scene_extent idi (gaussian sahnenin tamamı kadar olabiliyordu),
            # banana_high'ta max_scale=108=scene_extent → streak/overlap.
            # Şimdi: scene_extent × 0.05 = sahnenin %5'i max.
            with torch.no_grad():
                max_log_scale = math.log(max(self.scene_extent * 0.05, 1e-3))
                self.gs.scales.data.clamp_(max=max_log_scale)

                # v3.6.1: Hard clamp on Fourier coefficients.
                # Her katsayı büyüklüğü scene_extent × 0.05 = max 5% motion contribution.
                # Bir gaussian maksimum motion = K × max_coeff = 8 × 0.05 × scene = 40% scene.
                # Reg çalışsa bile katsayılar kontrolsüz büyüyordu (0.12 → 6.38 in 500 iter)
                # — bu clamp hard limit.
                if self.gs.fourier_pos_coeffs is not None:
                    max_coeff = self.scene_extent * 0.03  # 3% of scene per coefficient
                    self.gs.fourier_pos_coeffs.data.clamp_(min=-max_coeff, max=max_coeff)

            # --- Density control ---
            # T4 perf fix: optimizer'i density.step'e geciriyoruz; densitiy ops
            # (clone/split/prune) Adam state'i (exp_avg, exp_avg_sq) parametre
            # mutation'ı ile lockstep gunceller. Eskiden self._build_optimizer()
            # ile her density step optimizer'i sifirdan kuruyorduk → 200+ kez
            # Adam momentum siliniyordu → gerek convergence yavasligi, gerekse
            # kalite kaybi. torch.cuda.empty_cache() de kaldirildi (allocator
            # thrash + sync stall sebebiyle stuck'lara yol aciyordu).
            if (self.density_start_iter <= it < self.density_end_iter
                    and it % self.density_interval == 0):
                stats = self.density.step(
                    self.gs,
                    optimizer=self.optimizer,
                    dynamic_densify_scale=self.dynamic_densify_scale,
                )
                if it % log_interval == 0:
                    print(f"  ↳ density: clone={stats['cloned']} split={stats['split']} "
                          f"prune={stats['pruned']} | {stats['before']} → {stats['after']}")
                if run_logger is not None:
                    try:
                        run_logger.log_event(
                            "density",
                            iter=it,
                            cloned=stats["cloned"],
                            split=stats["split"],
                            pruned=stats["pruned"],
                            before=stats["before"],
                            after=stats["after"],
                        )
                    except Exception:
                        pass

            # --- Opacity reset (INRIA 3DGS trick, v3) ---
            # Her N iter'de tüm gaussian'ların opacity'sini low bir değere reset et
            # → recon loss bu opacity'leri tekrar >0.005 eşiğine yükseltemeyenleri
            # bir sonraki density step'te pruneyecek (outlier cleanup mekanizması).
            # Sadece density window içinde yap.
            if (self.opacity_reset_interval > 0
                    and self.density_start_iter <= it < self.density_end_iter
                    and it % self.opacity_reset_interval == 0
                    and it > self.density_start_iter):
                with torch.no_grad():
                    # Mevcut opacity'lerin min(current, 0.01 logit ≈ sigmoid^-1(0.01))
                    # Logit(0.01) ≈ -4.595. Daha yüksek olanları buna indir.
                    reset_logit = float(np.log(0.01 / (1 - 0.01)))  # ≈ -4.595
                    new_op = torch.minimum(
                        self.gs.opacities,
                        torch.full_like(self.gs.opacities, reset_logit),
                    )
                    self.gs.opacities.data.copy_(new_op)
                # Optimizer state'te opacity momentum sıfırla (yeni değer başlıyor)
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        if p is self.gs.opacities:
                            state = self.optimizer.state.get(p, None)
                            if state:
                                if "exp_avg" in state: state["exp_avg"].zero_()
                                if "exp_avg_sq" in state: state["exp_avg_sq"].zero_()
                print(f"  ↳ opacity reset @ iter {it}")
                if run_logger is not None:
                    try:
                        run_logger.log_event("opacity_reset", iter=it)
                    except Exception:
                        pass

            # --- Log ---
            if it % log_interval == 0:
                with torch.no_grad():
                    p = psnr(rgb, gt)
                    # v3.7.1 METRIC FIX: Önceden Δpos = raw MLP dpos (pre-clamp, Fourier'sız).
                    # Artık _apply_deformation'un gerçek dpos çıktısını ölçüyor
                    # (MLP clamp + Fourier + total clamp dahil). Bu, PLY export'taki
                    # gerçek motion'la tutarlı. Önceden metrics yanıltıcıydı.
                    d_means_dbg, _, _ = self._apply_deformation(t_norm)
                    dpos_applied = d_means_dbg - self.gs.means
                    dpos_mean = dpos_applied.abs().mean().item()
                    dpos_max = dpos_applied.abs().max().item()
                # Perf v2: comp_t'deki tensor'leri tek seferde sync edip float dict'e cevir.
                # Eskiden her iter ~12 sync; simdi sadece log_interval'de bir.
                comp = {k: float(v.item()) if torch.is_tensor(v) else float(v)
                        for k, v in comp_t.items()}
                # Default'lar (downstream get(...,0.0) calismasi icin):
                for _k in ("recon", "depth", "track", "deform_reg", "smooth", "rigid",
                           "scale", "fourier", "accel", "lpips", "mv_consist", "flow",
                           "aniso"):
                    comp.setdefault(_k, 0.0)
                history["loss"].append(loss.item())
                history["psnr"].append(p)
                history["n_pts"].append(self.gs.num_points)
                elapsed = time.time() - t0
                ips = it / max(elapsed, 1e-6)

                comp_str = " ".join(
                    f"{k}={v:.4f}" for k, v in comp.items()
                    if (k == "recon") or (v > 1e-5)
                )
                print(f"[{it:>6}/{n_iters}] loss={loss.item():.4f} [{comp_str}] "
                      f"psnr={p:.2f} N={self.gs.num_points:,} "
                      f"Δpos={dpos_mean:.4f}/{dpos_max:.4f} wu={warmup:.2f} | {ips:.1f} it/s")

                if progress_callback is not None:
                    try:
                        progress_callback(it, n_iters, float(loss.item()),
                                          float(p), int(self.gs.num_points))
                    except Exception as _e:
                        print(f"  ⚠ progress_callback exception: {_e}")

                # Run logger — metrics.jsonl'a yaz
                if run_logger is not None:
                    try:
                        run_logger.log_metric(
                            iter=it, n_iters=n_iters,
                            loss=float(loss.item()),
                            psnr=float(p),
                            n_points=int(self.gs.num_points),
                            dpos_mean=float(dpos_mean),
                            dpos_max=float(dpos_max),
                            warmup=float(warmup),
                            it_per_sec=float(ips),
                            recon=float(comp.get("recon", 0.0)),
                            depth=float(comp.get("depth", 0.0)),
                            track=float(comp.get("track", 0.0)),
                            deform_reg=float(comp.get("deform_reg", 0.0)),
                            smooth=float(comp.get("smooth", 0.0)),
                            rigid=float(comp.get("rigid", 0.0)),
                            scale=float(comp.get("scale", 0.0)),
                            fourier=float(comp.get("fourier", 0.0)),
                        )
                    except Exception as _e:
                        print(f"  ⚠ run_logger exception: {_e}")

            if ckpt_dir is not None and it % ckpt_interval == 0:
                self._save_checkpoint(ckpt_dir, it)

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
