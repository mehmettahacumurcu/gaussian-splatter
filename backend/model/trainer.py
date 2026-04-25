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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Callable, Sequence

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
# Frame / depth / mask / tracks yardımcıları
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
    """Metric3D/MiDaS depth .npy → (H, W) float32."""
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
    """Dynamic mask .png → (H, W) float [0,1]."""
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
        self.opacity_reset_interval = int(opacity_reset_interval)
        self.warmup_iters = max(1, warmup_iters)
        # v3.6 / Yol C
        self.deform_pos_mode = deform_pos_mode
        self.lr_fourier = lr_fourier
        self.lambda_fourier_reg = lambda_fourier_reg
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

    def _apply_deformation(self, t: float):
        # MLP her zaman çağrılır (dquat, dscale için gerekli, dpos opsiyonel)
        dpos_mlp, dquat, dscale = self.deform(self.gs.means, t, self.scene_extent)
        # v3.5: dscale clamp — MLP'nin scale delta'sı ±2 ile sınırlı.
        dscale = dscale.clamp(min=-2.0, max=2.0)

        # v3.6.2: dpos_mlp clamp — MLP serbest yazabildiği için 500+ unit
        # jump'lara yol açıyordu (chickchicken_fourier_v2'de Δpos max=513 görüldü).
        # Max motion per iter = scene_extent × 0.1 (yumuşak caydırıcı).
        mlp_cap = self.scene_extent * 0.1
        dpos_mlp = dpos_mlp.clamp(min=-mlp_cap, max=mlp_cap)

        # v3.6 / Yol C: Pozisyon deformasyonunu pos_mode'a göre seç
        if self.deform_pos_mode == "mlp":
            dpos = dpos_mlp
        elif self.deform_pos_mode == "fourier":
            dpos = decode_fourier_trajectory(self.gs.fourier_pos_coeffs, t)
        else:  # "hybrid"
            dpos_fourier = decode_fourier_trajectory(self.gs.fourier_pos_coeffs, t)
            dpos = dpos_mlp + dpos_fourier

        # v3.6.2: FINAL dpos clamp — overall motion cap
        # Herhangi bir gaussian maks sahne'nin %20'si kadar oynayabilir.
        total_cap = self.scene_extent * 0.2
        dpos = dpos.clamp(min=-total_cap, max=total_cap)

        deformed_means  = self.gs.means + dpos
        deformed_quats  = F.normalize(self.gs.quats + dquat, dim=-1)
        deformed_scales = self.gs.get_scales * torch.exp(dscale)
        return deformed_means, deformed_quats, deformed_scales

    def _warmup_factor(self, it: int) -> float:
        """
        Linear warmup 0→1 over ilk warmup_iters iter.
        v3.2 fix: hardcoded 100 iter delay kaldırıldı — reg'ler iter 1'den
        itibaren yumuşak devreye giriyor. Önce 0-100 arası unregulated
        training yüzünden deformasyon MLP patlıyordu (Δpos 41, track 196).
        """
        return min(1.0, max(0.0, it / max(1, self.warmup_iters)))

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
        run_logger: Any = None,   # RunLogger instance, opsiyonel
    ) -> dict:
        T = len(frame_paths)
        if len(cam_w2c_per_frame) != T:
            raise ValueError(
                f"frame_paths ve cam_w2c_per_frame uzunlukları farklı: {T} vs {len(cam_w2c_per_frame)}"
            )

        cam_K_orig = cam_K.to(self.device)   # frame resolution
        w2c_list = [w.to(self.device) for w in cam_w2c_per_frame]
        Ws, Hs = image_size
        sample = load_frame_tensor(Path(frame_paths[0]))
        H0, W0 = sample.shape[:2]             # original frame resolution
        sx, sy = Ws / W0, Hs / H0
        K_scaled = cam_K_orig.clone()         # training resolution
        K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

        # Cache'ler
        frames_cached: list[torch.Tensor | None] = [None] * T
        depth_cached:  list[torch.Tensor | None] = [None] * T
        mask_cached:   list[torch.Tensor | None] = [None] * T

        use_depth = depth_dir is not None and self.lambda_depth > 0
        use_mask  = mask_dir is not None
        if use_depth:
            print(f"[trainer] Depth loss aktif (lambda={self.lambda_depth})")
        if use_mask:
            print(f"[trainer] Mask-weighted recon aktif (lambda={self.lambda_mask_motion})")

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
        target_lr_deform = self.lr_deform
        target_lr_fourier = self.lr_fourier
        has_fourier_group = len(self.optimizer.param_groups) > FOURIER_GROUP_IDX

        for it in range(1, n_iters + 1):
            idx = int(torch.randint(0, T, (1,)).item())
            t_norm = idx / max(T - 1, 1)
            warmup = self._warmup_factor(it)

            # Update lr_deform + lr_fourier per iter (warmup schedule)
            self.optimizer.param_groups[DEFORM_GROUP_IDX]["lr"] = target_lr_deform * warmup
            if has_fourier_group:
                self.optimizer.param_groups[FOURIER_GROUP_IDX]["lr"] = target_lr_fourier * warmup

            # --- Ground truth load ---
            if frames_cached[idx] is None:
                frames_cached[idx] = load_frame_tensor(Path(frame_paths[idx]), (Ws, Hs))
            gt = frames_cached[idx].to(self.device)

            if use_depth and depth_cached[idx] is None:
                depth_cached[idx] = _load_depth_for_frame(depth_dir, frame_paths[idx], (Ws, Hs))
            gt_depth = depth_cached[idx].to(self.device) if (use_depth and depth_cached[idx] is not None) else None

            if use_mask and mask_cached[idx] is None:
                mask_cached[idx] = _load_mask_for_frame(mask_dir, frame_paths[idx], (Ws, Hs))
            frame_mask = mask_cached[idx].to(self.device) if (use_mask and mask_cached[idx] is not None) else None

            # --- Deformation + render ---
            d_means, d_quats, d_scales = self._apply_deformation(t_norm)

            render_out, _alpha, _info = render_view(
                means=d_means, quats=d_quats, scales=d_scales,
                opacities=self.gs.get_opacities, colors=self.gs.get_colors,
                K=K_scaled, w2c=w2c_list[idx],
                width=Ws, height=Hs,
                sh_degree=self.gs.sh_degree,
                with_depth=use_depth,
            )
            if use_depth:
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

            # Loss bileşenlerini ayrı tut (diagnostics için)
            comp = {"recon": loss_recon.item(), "depth": 0.0, "track": 0.0,
                    "deform_reg": 0.0, "smooth": 0.0, "rigid": 0.0,
                    "scale": 0.0, "fourier": 0.0}

            # --- Depth consistency (log-space, warmup-gated, clamped) ---
            # Log-space L1 scale-invariant ve outlier'a karşı dayanıklı.
            # Warmup gate: ilk 100 iter statik GS otursun, sonra depth devreye.
            # Clamp(2.0): tek frame outlier'ı tüm run'ı batırmasın.
            if use_depth and gt_depth is not None and rendered_depth is not None:
                valid = (gt_depth > 0.01) & (rendered_depth > 0.01)
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
                    comp["depth"] = depth_l1.item()

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
                        comp["track"] = track_l.item()

            # --- Motion regularizers (warmup'la scale) ---
            if warmup > 0 and (
                self.lambda_deform_reg > 0
                or self.lambda_smoothness > 0
                or self.lambda_rigidity > 0
            ):
                dpos, dquat, dscale = self.deform(
                    self.gs.means, t_norm, self.scene_extent,
                )
                if self.lambda_deform_reg > 0:
                    reg = dpos.pow(2).mean() + dquat.pow(2).mean() + dscale.pow(2).mean()
                    loss = loss + self.lambda_deform_reg * warmup * reg
                    comp["deform_reg"] = reg.item()
                if self.lambda_smoothness > 0:
                    t2 = min(1.0, t_norm + self.smoothness_dt)
                    if t2 != t_norm:
                        dpos2, _, _ = self.deform(self.gs.means, t2, self.scene_extent)
                        smooth = (dpos - dpos2).pow(2).mean()
                        loss = loss + self.lambda_smoothness * warmup * smooth
                        comp["smooth"] = smooth.item()
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
                    comp["rigid"] = rigid.item()

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
                comp["scale"] = scale_reg.item()

            # --- Fourier trajectory regularizer (v3.6 / Yol C) ---
            # High-freq katsayıları bastır — noise/overfitting önle.
            # Frekansa göre ağırlıklı L2: yüksek k → büyük ceza (low-pass prior).
            if (self.lambda_fourier_reg > 0 and
                    self.gs.fourier_pos_coeffs is not None and
                    self.deform_pos_mode != "mlp"):
                K = self.gs.fourier_K
                # Frekans ağırlığı: k² (k=1,2,...,K). Yüksek k'lar daha çok cezalandırılır.
                freq_weights = torch.arange(1, K + 1, device=self.gs.fourier_pos_coeffs.device,
                                            dtype=self.gs.fourier_pos_coeffs.dtype)
                freq_weights = freq_weights ** 2  # (K,)
                # coeffs: (N, K, 2, 3) → per-coeff squared magnitude × freq weight
                sq = self.gs.fourier_pos_coeffs.pow(2)  # (N, K, 2, 3)
                # Sum over sin/cos + xyz, mean over N, weighted sum over K
                per_freq_energy = sq.sum(dim=(0, 2, 3)) / max(self.gs.num_points, 1)  # (K,)
                fourier_reg = (per_freq_energy * freq_weights).sum()
                loss = loss + self.lambda_fourier_reg * fourier_reg
                comp["fourier"] = fourier_reg.item()

            # --- Fourier SPATIAL smoothness (v3.6.2) ---
            # Komşu gaussian'lar benzer trajectory'ye sahip olsun. Motion diffuse et.
            # Aksi halde sadece track anchor'larına yakın gaussian hareket ediyor,
            # geri kalan statik — chickchicken'da %92 gaussian durgundu.
            # Her iter random K=128 gaussian sample + 1-NN üzerinden smoothness.
            if (self.lambda_fourier_reg > 0 and
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
            # Gradient clip'e rağmen bazı durumlarda (örn. gradient NaN yayılması)
            # param'ların kendisi NaN olabilir. Güvenlik için adım öncesi sanity check.
            grad_nan = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for g in self.optimizer.param_groups for p in g["params"]
            )
            if grad_nan:
                if it % log_interval == 0 or it < 50:
                    print(f"  ⚠ iter {it}: non-finite gradient → skip step")
                self.optimizer.zero_grad(set_to_none=False)
                continue

            self.density.accumulate(self.gs)
            self.optimizer.step()

            # --- Post-step param NaN/Inf check — v3.4 expanded ---
            # means + scales + quats hepsi kontrol ediliyor çünkü scales inf'e
            # gitmesi training'i sessizce bozuyordu (chickchicken_v3_3'te log_scale=70+,
            # exp()=inf, her sonraki iter loss non-finite → skipped → silent garbage).
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
                    f"Last good checkpoint: run scripts/recover_scene.py <scene>"
                )

            # v3.4: Hard clamp on log_scale — safety net against runaway growth.
            # Upper bound: scene_extent (absolute size). log(scene_extent) is max reasonable.
            # Without this, Δpos huge spike can push scales via gradient to log_scale=70+.
            with torch.no_grad():
                max_log_scale = math.log(max(self.scene_extent, 1.0))
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
            if (self.density_start_iter <= it < self.density_end_iter
                    and it % self.density_interval == 0):
                stats = self.density.step(self.gs)
                self._build_optimizer()
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
                torch.cuda.empty_cache()

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
