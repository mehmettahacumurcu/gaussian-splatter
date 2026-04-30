"""End-to-end integration test for Phase 1.3-1.9 + Phase 2.1-2.5.

Real multi-view sahne uzerinde calisip implement ettigimiz tum ozellikleri
tetikler:

ROUND 1 — Heavy preprocess (standard profile)
    - Phase 1.3: cache marker + settings_hash yazilir
    - Phase 1.4: per-cam multi-view depth (depth_multiview/cam*/*.npy)
    - Phase 1.5: per-cam multi-view dynamic mask (masks_multiview/cam*/mask_*.png)

ROUND 2 — Aynı preprocess tekrar (cache hit dogrulama)
    - Hash match -> tum step'ler 'CACHE HIT' loglari basar
    - 30-60 sn'de biter

ROUND 3 — Settings degisikligi (premium profile)
    - depth model standard'in vit_small'ından premium'un vit_large'ına geçer
    - colmap_mv_timestamps 5 -> 25 (mv_dense_mvs True)
    - depth_mv + colmap_mv hash mismatch -> CACHE MISS, re-run

ROUND 4 — Mini training run (300 iter), TUM Phase 1+2 ozellikleri AKTIF
    - Phase 1.6: LPIPS perceptual loss (lambda=0.02)
    - Phase 1.7: Multi-view consistency loss (lambda=0.05)
    - Phase 1.9: densify_mv_threshold_scale=0.7 (default MV)
    - Phase 2.2: multires_schedule [(0, 240), (150, 320)] resolution transition
    - Phase 2.3: lr_cam_K=1e-7, lr_cam_w2c=1e-7 (camera pose refine)
    - Phase 2.5: multires_resolutions=[12, 24] (HexPlane stack)

NOT (skipped — runtime activation gerektirir):
    - Phase 1.8 RAFT flow: lambda_flow=0 (preprocess ~3 saat, mini run icin yapma)
    - Phase 2.1 static/dyn split: buffer mevcut, ama promote_dynamic() runtime
      hook olmadigi icin tetiklenmez (smoke_phase_1_2 testinde dogrulandi)
    - Phase 2.4 background flag: ayni - flag_background_by_distance() programatik

Calistir:
    python scripts/integration_test_phase_1_2.py --scene flame_steak

Mini scenario:
    python scripts/integration_test_phase_1_2.py --scene flame_steak --iters 300

Sadece preprocess (no training):
    python scripts/integration_test_phase_1_2.py --scene flame_steak --no-training
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}\n")


def _resolve_video_path(scene_name: str):
    from backend.config import scene_paths, is_multiview_scene
    paths = scene_paths(scene_name)
    if not paths["base"].exists():
        raise FileNotFoundError(f"Scene not found: {paths['base']}")
    if is_multiview_scene(scene_name):
        videos = sorted(paths["videos_mv"].glob("cam*.mp4"))
        if not videos:
            videos = sorted(paths["videos_mv"].glob("cam*.MP4"))
        if not videos:
            raise FileNotFoundError(f"No cam*.mp4 in {paths['videos_mv']}")
        return videos[0], paths, True
    else:
        if not paths["video"].exists():
            raise FileNotFoundError(f"video.mp4 not found: {paths['video']}")
        return paths["video"], paths, False


# ---------------------------------------------------------------------------
# ROUND helpers
# ---------------------------------------------------------------------------

PROFILE_OVERRIDES = {
    "standard": dict(fps=10, resize=960, mv_ts=5, dense=False,
                     m3d="metric3d_vit_small", grid=30),
    "premium":  dict(fps=30, resize=1280, mv_ts=25, dense=True,
                     m3d="metric3d_vit_large", grid=75),
}


def _build_preprocess_cfg(profile_name: str):
    from backend.config import default_config
    p = PROFILE_OVERRIDES[profile_name]
    cfg = default_config()
    cfg.preprocess.fps = p["fps"]
    cfg.preprocess.resize_long_edge = p["resize"]
    cfg.preprocess.colmap_matching = "exhaustive" if profile_name != "standard" else "sequential"
    cfg.preprocess.colmap_mv_timestamps = p["mv_ts"]
    cfg.preprocess.colmap_mv_dense_mvs = p["dense"]
    cfg.foundation.metric3d_model = p["m3d"]
    cfg.foundation.cotracker_grid_size = p["grid"]
    return cfg


def round_preprocess(scene_name: str, profile: str, force: bool = False, label: str = "") -> dict:
    from backend.pipeline import run_pipeline
    video_path, paths, is_mv = _resolve_video_path(scene_name)
    cfg = _build_preprocess_cfg(profile)
    print(f"  Profile: {profile}  Multi-view: {is_mv}  Force: {force}")
    print(f"  Video reference: {video_path.name}")
    t0 = time.time()
    status = run_pipeline(
        video_path=str(video_path), scene_name=scene_name, cfg=cfg,
        skip_training=True, skip_export=True, skip_foundation=False,
        force_preprocess=force,
    )
    dt = time.time() - t0
    print(f"\n  {label} elapsed: {dt:.1f} sec ({dt/60:.1f} dk)")
    for k, v in status.items():
        print(f"    status[{k}] = {v}")
    return status


def round_training(scene_name: str, n_iters: int, quality: str = "mini",
                   export: bool = False) -> dict:
    """Training run — Phase 1.6 + 1.7 + 1.9 + 2.2 + 2.3 + 2.5 actively engaged.

    quality:
        "mini" — 240p, 80k cap, smoke testi (default)
        "high" — 720p, 150k cap, premium-equivalent (Secenek A: ~2-3 saat 5000 iter)
    export: True ise pipeline.export koşar (PLY'ler frontend için).
    """
    from backend.config import default_config
    from backend.pipeline import run_pipeline

    video_path, paths, is_mv = _resolve_video_path(scene_name)
    cfg = default_config()

    cfg.train.n_iters = n_iters
    cfg.train.warmup_iters = max(20, n_iters // 6)
    cfg.train.density_start_iter = max(50, n_iters // 6)
    cfg.train.density_end_iter = max(100, n_iters - max(20, n_iters // 10))
    cfg.train.density_interval = 50 if quality == "mini" else 100
    cfg.train.opacity_reset_interval = 0 if quality == "mini" else max(2000, n_iters // 4)
    cfg.train.ckpt_interval = max(n_iters * 2, 99999) if quality == "mini" else max(1000, n_iters // 5)
    cfg.train.log_interval = max(20, n_iters // 15) if quality == "mini" else max(100, n_iters // 20)

    if quality == "mini":
        cfg.train.image_resolution = (320, 180)
        cfg.train.max_gaussians = 80_000
    elif quality == "high":
        # 640p — RTX 3060 Ti 8GB ile guvenli (onceki 720p test'te VRAM 7.8/8 GB,
        # OOM riski yuksekti). 640p ile ~6 GB hedef.
        cfg.train.image_resolution = (640, 480)
        cfg.train.max_gaussians = 120_000  # 150k -> 120k, ek 200 MB tasarruf
    else:
        raise ValueError(f"Bilinmeyen quality: {quality}")

    # ---- Phase 1.6 — LPIPS perceptual loss ----
    if quality == "mini":
        cfg.train.lambda_lpips = 0.02
    else:  # high
        cfg.train.lambda_lpips = 0.05  # premium tipik
    cfg.train.lpips_net = "alex"
    cfg.train.lpips_warmup_iters = max(50, n_iters // 10)

    # ---- Phase 1.7 — Multi-view cross-cam consistency ----
    cfg.train.lambda_multiview_consistency = 0.05 if quality == "mini" else 0.10

    # ---- Phase 1.8 — Flow loss disabled (preprocess yok, ~3 saat) ----
    cfg.train.lambda_flow = 0.0

    # ---- Phase 1.9 — Densify MV threshold scale ----
    cfg.train.densify_mv_threshold_scale = 0.7

    # ---- Phase 2.2 — Multi-resolution training schedule ----
    if quality == "mini":
        cfg.train.multires_schedule = [(0, 240), (n_iters // 2, 320)]
    else:  # high — 480p → 640p geçişi (720p'den hafifletildi)
        cfg.train.multires_schedule = [(0, 480), (n_iters // 2, 640)]

    # ---- Phase 2.3 — Camera pose refinement ----
    cfg.train.lr_cam_K = 1e-7
    cfg.train.lr_cam_w2c = 1e-7
    cfg.train.cam_refine_start_iter = max(30, n_iters // 5)

    # ---- Phase 2.5 — Multi-resolution HexPlane ----
    if quality == "mini":
        cfg.model.multires_resolutions = [12, 24]
        cfg.model.multires_feat_dim = 12
    else:  # high — 2-scale (3-scale 432 dim VRAM heavy idi)
        cfg.model.multires_resolutions = [32, 64]
        cfg.model.multires_feat_dim = 24  # total = 24*6*2 = 288 dim (was 432)

    # Deform MLP capacity
    if quality == "mini":
        cfg.model.mlp_width = 256
        cfg.model.mlp_depth = 2
    else:  # high — daha fazla kapasite premium icin
        cfg.model.mlp_width = 512
        cfg.model.mlp_depth = 4

    # Export ayarlari (high quality + export'ta PLY uretilir frontend icin)
    if export:
        cfg.export.num_timestamps = 30 if quality == "high" else 20
    # Foundation lambda'lar düşük (depth_mv consume kodu tam wire değil)
    cfg.train.lambda_depth = 0.0
    cfg.train.lambda_track = 0.0
    cfg.train.lambda_mask_motion = 1.0

    # Stability — fourier coef patlamasi onlemek icin reg 5x (1e-3 -> 5e-3)
    # Onceki 720p run'da iter 3500'de fourier=2.89'a cikti, lambda yetersizdi.
    if quality == "high":
        cfg.train.lambda_fourier_reg = 5e-3
        cfg.train.dpos_total_cap_frac = 0.1  # 0.2'den siki, max Delta_pos=10% scene extent

    print(f"  cfg snapshot ({quality}):")
    print(f"    n_iters={cfg.train.n_iters}, res={cfg.train.image_resolution}, "
          f"max_gauss={cfg.train.max_gaussians}")
    print(f"    [Phase 1.6] lambda_lpips={cfg.train.lambda_lpips} (warmup={cfg.train.lpips_warmup_iters})")
    print(f"    [Phase 1.7] lambda_multiview_consistency={cfg.train.lambda_multiview_consistency}")
    print(f"    [Phase 1.9] densify_mv_threshold_scale={cfg.train.densify_mv_threshold_scale}")
    print(f"    [Phase 2.2] multires_schedule={cfg.train.multires_schedule}")
    print(f"    [Phase 2.3] lr_cam_K={cfg.train.lr_cam_K} lr_cam_w2c={cfg.train.lr_cam_w2c} "
          f"start={cfg.train.cam_refine_start_iter}")
    print(f"    [Phase 2.5] multires_resolutions={cfg.model.multires_resolutions} "
          f"feat_dim={cfg.model.multires_feat_dim}")
    print(f"    Export: {'ENABLED (' + str(cfg.export.num_timestamps) + ' frame PLY)' if export else 'skipped'}")

    t0 = time.time()
    status = run_pipeline(
        video_path=str(video_path), scene_name=scene_name, cfg=cfg,
        skip_training=False, skip_export=(not export), skip_foundation=False,
        force_preprocess=False,  # önceki round'lardan cache kullan
    )
    dt = time.time() - t0
    print(f"\n  Training elapsed: {dt:.1f} sec ({dt/60:.1f} dk)")
    for k, v in status.items():
        print(f"    status[{k}] = {v}")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1+2 integration test")
    ap.add_argument("--scene", default="flame_steak", help="Multi-view scene name")
    ap.add_argument("--iters", type=int, default=300, help="Training iter count")
    ap.add_argument("--quality", default="mini", choices=["mini", "high"],
                    help="mini=240p smoke (mevcut), high=720p premium-equivalent (Secenek A)")
    ap.add_argument("--export", action="store_true",
                    help="Export PLY frame'leri (frontend gorselelmesi icin)")
    ap.add_argument("--no-training", action="store_true",
                    help="Sadece preprocess round'lari (training atlanir)")
    ap.add_argument("--skip-cache-tests", action="store_true",
                    help="Cache hit + invalidation round'larini atla")
    ap.add_argument("--standard-only", action="store_true",
                    help="Premium round'i atla (sadece standard preprocess test)")
    args = ap.parse_args()

    banner(f"PHASE 1+2 INTEGRATION TEST  scene='{args.scene}'  iters={args.iters}")

    from backend.config import scene_paths
    paths = scene_paths(args.scene)
    if not paths["base"].exists():
        print(f"✗ Scene not found: {paths['base']}")
        sys.exit(1)

    overall_t0 = time.time()
    rounds_done = []

    if not args.skip_cache_tests:
        banner("ROUND 1 — Heavy preprocess (standard profile, cache build)")
        round_preprocess(args.scene, profile="standard", force=False, label="Round 1")
        rounds_done.append("R1: preprocess(standard) cache build")

        banner("ROUND 2 — Heavy preprocess (standard tekrar, cache HIT bekleniyor)")
        round_preprocess(args.scene, profile="standard", force=False, label="Round 2")
        rounds_done.append("R2: preprocess(standard) cache HIT verification")

        if not args.standard_only:
            banner("ROUND 3 — Heavy preprocess (premium, settings degisikligi -> CACHE MISS)")
            round_preprocess(args.scene, profile="premium", force=False, label="Round 3")
            rounds_done.append("R3: preprocess(premium) settings invalidation")

    if not args.no_training:
        label = "Mini" if args.quality == "mini" else "HIGH (premium-equivalent)"
        banner(f"ROUND 4 — {label} training ({args.iters} iter, ALL Phase 1+2 features ON)")
        round_training(args.scene, n_iters=args.iters,
                       quality=args.quality, export=args.export)
        rounds_done.append(
            f"R4: {label} training {args.iters} iter "
            f"(Phase 1.6-1.9 + 2.2-2.5)" + (" + EXPORT" if args.export else "")
        )

    total = time.time() - overall_t0
    banner("INTEGRATION TEST COMPLETE")
    print(f"  Total elapsed: {total:.1f} sec ({total/60:.1f} dk)\n")
    print("  Rounds run:")
    for r in rounds_done:
        print(f"    ✓ {r}")
    print()
    print("  Cikti'da grep ile dogrula:")
    print("    Phase 1.3 — 'CACHE HIT' / 'settings degisti' / 'cache marker yazildi'")
    print("    Phase 1.4 — '[depth_mv]' veya 'depth_mv:cache_hit'")
    print("    Phase 1.5 — '[masks_mv]' veya 'masks_mv:cache_hit'")
    print("    Phase 1.6 — '[trainer] LPIPS yuklendi' + comp tablosunda 'lpips':")
    print("    Phase 1.7 — comp tablosunda 'mv_consist': nonzero")
    print("    Phase 1.9 — '[trainer.mv] Densify grad_threshold ... scale=0.7'")
    print("    Phase 2.2 — '[trainer.multires] Iter X: resolution YxZ -> AxB'")
    print("    Phase 2.3 — '[trainer.cam_refine] Aktif: N cam'")
    print("    Phase 2.5 — '[deformation] MultiResHexPlane: resolutions=[12, 24]'")
