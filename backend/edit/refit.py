"""Local refit after Gaussian deletion + inpaint.

Strategy:
  1. Load source ckpt, build GaussianModel.
  2. Drop Gaussians flagged by classifier.
  3. Freeze gradients on all surviving Gaussians outside the affected zone
     (zone defined as a sphere of radius 1.5 * max_deleted_extent around the
     centroid of deleted Gaussians).
  4. Run training loop with inpainted views as new GT for n_iters.
     - RGB + SSIM + LPIPS-512 normal
     - Depth supervision masked off inside the SAM mask
     - Densify only allowed for Gaussians inside the affected zone
  5. Save new ckpt + PLY.

NOTE: Full body wired in Task C2 (which also adds the trainer-side hooks
this needs: filter_in_place, set_freeze_mask, edit_mask_stack support).
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import torch


def compute_affected_zone(
    means: torch.Tensor,            # (N, 3) all gaussians (incl. flagged)
    delete_flags: torch.Tensor,     # (N,) bool
    multiplier: float = 1.5,
) -> tuple[torch.Tensor, float]:
    """Return (centroid: (3,), radius: float) defining the affected zone.

    The radius is multiplier × max distance from the deletion centroid to any
    deleted Gaussian. This bounds the region where surviving Gaussians may be
    re-trained / densified after deletion + inpaint.
    """
    if delete_flags.sum() == 0:
        return torch.zeros(3, device=means.device), 0.0
    deleted = means[delete_flags]
    centroid = deleted.mean(dim=0)
    max_extent = (deleted - centroid).norm(dim=1).max().item()
    radius = multiplier * max_extent
    return centroid, radius


def is_in_zone(
    means: torch.Tensor,            # (N, 3)
    centroid: torch.Tensor,         # (3,)
    radius: float,
) -> torch.Tensor:
    """Bool[N] — True for Gaussians within the affected zone."""
    if radius == 0.0:
        return torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    return (means - centroid).norm(dim=1) <= radius


def run_refit(
    source_ckpt_path: Path,
    inpainted_frames_dir: Path,
    inpainted_mask_stack: torch.Tensor,  # (T, H, W) bool — for depth-loss masking
    delete_flags: torch.Tensor,          # (N,) bool from classifier
    output_ckpt_dir: Path,
    n_iters: int = 5000,
    cancel_check: Callable[[], bool] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Path:
    """Run local refit, save new ckpt to output_ckpt_dir/ckpt_final.pt. Returns path.

    Wired in Task C2 — depends on trainer-side changes (filter_in_place,
    set_freeze_mask, edit_mask_stack parameter on Trainer4DGS.train()).
    """
    from ..config import default_config
    from ..model.gaussian_model import GaussianModel
    from ..model.deformation import DeformationField
    from ..model.trainer import Trainer4DGS
    from ..preprocess.parse_colmap import parse_cameras

    # ------------------------------------------------------------------
    # 1. Load source checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(source_ckpt_path, map_location="cuda", weights_only=False)
    gs = GaussianModel.from_checkpoint(ckpt["gs"]).to("cuda")
    scene_extent = float(ckpt.get("scene_extent", 1.0))

    # ------------------------------------------------------------------
    # 2. Compute affected zone BEFORE deletion (uses original means)
    # ------------------------------------------------------------------
    centroid, radius = compute_affected_zone(gs.means.detach(), delete_flags.to("cuda"))

    # ------------------------------------------------------------------
    # 3. Drop deleted Gaussians
    # ------------------------------------------------------------------
    keep = ~delete_flags.to("cuda")
    gs.filter_in_place(keep)

    # ------------------------------------------------------------------
    # 4. Mark frozen vs free (re-compute zone membership for survivors)
    # ------------------------------------------------------------------
    in_zone = is_in_zone(gs.means.detach(), centroid, radius)
    gs.set_freeze_mask(~in_zone)  # frozen = OUT of zone

    # ------------------------------------------------------------------
    # 5. Resolve camera poses from the source scene's COLMAP
    # ------------------------------------------------------------------
    # The runner (C3) passes inpainted_frames_dir that lives at:
    #   data/<scene>/output/edits/edit_NNN/inpainted_frames/
    # Two parents up gives data/<scene>/output/edits/edit_NNN/,
    # three parents up gives data/<scene>/output/edits/,
    # four parents up gives data/<scene>/output/,
    # but parse_cameras needs data/<scene>/colmap/ — which is a sibling of
    # data/<scene>/output/.  Use the source_ckpt_path to find the scene root
    # instead of relying on inpainted_frames_dir hierarchy (more robust).
    #
    # ckpt is typically at:  data/<scene>/output/ckpt/ckpt_final.pt
    # → scene root = ckpt.parent.parent.parent
    scene_dir = source_ckpt_path.parent.parent.parent
    colmap_dir = scene_dir / "colmap"

    cams = parse_cameras(colmap_dir)
    cam_names_sorted = sorted(cams.keys())
    K = torch.from_numpy(cams[cam_names_sorted[0]]["K"]).float()
    w2cs = [torch.from_numpy(cams[n]["w2c"]).float() for n in cam_names_sorted]

    # ------------------------------------------------------------------
    # 6. Build frame-path list from the inpainted directory
    # ------------------------------------------------------------------
    # Inpainted frames are named frame_000000.png, frame_000001.png ...
    # matching the sorted COLMAP camera order produced by the inpainter (C3).
    frame_paths = [
        inpainted_frames_dir / f"frame_{i:06d}.png"
        for i in range(len(cam_names_sorted))
    ]

    # ------------------------------------------------------------------
    # 7. Build a minimal static-refit config
    # ------------------------------------------------------------------
    cfg = default_config()
    cfg.train.static_mode = True
    cfg.train.n_iters = n_iters
    cfg.train.density_start_iter = max(100, n_iters // 10)
    cfg.train.density_end_iter = int(n_iters * 0.7)
    # Softer depth weight — inpainted regions already masked out in trainer,
    # but surrounding valid pixels shouldn't dominate the gradient.
    cfg.train.lambda_depth = 0.05

    output_ckpt_dir = Path(output_ckpt_dir)
    output_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 8. Minimal placeholder DeformationField (static mode — no dynamics)
    #    Same pattern as pipeline.py:839 for static_mode=True.
    #    Trainer needs a non-None deform for _build_optimizer / _save_checkpoint.
    # ------------------------------------------------------------------
    deform = DeformationField(
        resolution=8, feat_dim=4, mlp_width=32, mlp_depth=1, num_time_freqs=2,
    )
    for p in deform.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 9. Construct Trainer4DGS
    #    Mirrors pipeline.py:911 constructor call, omitting multi-view /
    #    flow / perceptual args that are irrelevant for a quick local refit.
    # ------------------------------------------------------------------
    trainer = Trainer4DGS(
        gs=gs,
        deform=deform,
        device="cuda",
        scene_extent=scene_extent,
        lr_means=cfg.train.lr_means,
        lr_scales=cfg.train.lr_scales,
        lr_quats=cfg.train.lr_quats,
        lr_opacities=cfg.train.lr_opacities,
        lr_sh_dc=cfg.train.lr_sh_dc,
        lr_sh_rest=cfg.train.lr_sh_rest,
        lr_deform=cfg.train.lr_deform,
        density_start_iter=cfg.train.density_start_iter,
        density_end_iter=cfg.train.density_end_iter,
        density_interval=cfg.train.density_interval,
        densify_grad_threshold=cfg.train.densify_grad_threshold,
        prune_min_opacity=cfg.train.prune_min_opacity,
        prune_max_scale=cfg.train.prune_max_scale,
        lambda_ssim=cfg.train.lambda_ssim,
        lambda_depth=cfg.train.lambda_depth,
        lambda_deform_reg=cfg.train.lambda_deform_reg,
        lambda_smoothness=cfg.train.lambda_smoothness,
        lambda_rigidity=cfg.train.lambda_rigidity,
        lambda_scale=cfg.train.lambda_scale,
        lambda_aniso=cfg.train.lambda_aniso,
        aniso_threshold=cfg.train.aniso_threshold,
        dpos_total_cap_frac=cfg.train.dpos_total_cap_frac,
        opacity_reset_interval=cfg.train.opacity_reset_interval,
        warmup_iters=cfg.train.warmup_iters,
        max_gaussians=cfg.train.max_gaussians,
        # Fourier disabled — loaded ckpt likely has fourier_K=0; mlp mode safe.
        deform_pos_mode="mlp",
        lr_fourier=cfg.train.lr_fourier,
        lambda_fourier_reg=cfg.train.lambda_fourier_reg,
        # No motion-mask, track, LPIPS, multi-view, flow, cam-refine for refit.
        lambda_mask_motion=0.0,
        lambda_track=0.0,
        lambda_lpips=0.0,
        lambda_flow=0.0,
    )

    # ------------------------------------------------------------------
    # 10. Trainer progress callback adapter
    # ------------------------------------------------------------------
    def _refit_progress(iter_idx: int, total: int, loss: float,
                        psnr_val: float, n_pts: int) -> None:
        if progress_cb is not None:
            progress_cb(iter_idx / max(total, 1), f"refit iter {iter_idx}/{total} — loss={loss:.4f}")

    # ------------------------------------------------------------------
    # 11. Run training
    # ------------------------------------------------------------------
    trainer.train(
        frame_paths=frame_paths,
        cam_K=K,
        cam_w2c_per_frame=w2cs,
        n_iters=n_iters,
        image_size=cfg.train.image_resolution,
        ckpt_dir=output_ckpt_dir,
        ckpt_interval=n_iters,          # only save at end
        log_interval=200,
        progress_callback=_refit_progress,
        static_mode=True,
        cancel_check=cancel_check,
        edit_mask_stack=(
            inpainted_mask_stack.cpu()
            if isinstance(inpainted_mask_stack, torch.Tensor)
            else inpainted_mask_stack
        ),
    )

    # Trainer._save_checkpoint(final=True) writes ckpt_final.pt
    return output_ckpt_dir / "ckpt_final.pt"
