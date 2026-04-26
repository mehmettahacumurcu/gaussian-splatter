"""End-to-end pipeline orchestrator.

Bir video → 4DGS sahnesi:
  1) Frame çıkarma (ffmpeg)
  2) COLMAP SfM
  3) Foundation modeller (depth + tracking + masking)  [opsiyonel: --skip-foundation]
  4) Gaussian model + DeformationField init
  5) Training loop
  6) Export (.ply)

Kullanım:
  python -m backend.pipeline data/test_scene/video.mp4 --scene test_scene
"""
from __future__ import annotations
import argparse
import sys
import time
import torch
from pathlib import Path
from typing import Any, Callable

from .config import default_config, cloud_config, scene_paths, Config
from .preprocess.extract_frames import extract_frames
from .preprocess.run_colmap     import run_colmap
from .preprocess.parse_colmap   import parse_cameras, load_points3d, scene_extent as compute_scene_extent
from .model.gaussian_model      import GaussianModel
from .model.deformation         import DeformationField
from .model.trainer             import Trainer4DGS
from .export.to_splat           import export_to_ply
from .run_logger                import RunLogger


# progress_callback imzası:
#   (phase_name: str, progress_0_1: float, message: str, details: dict) -> None
ProgressCallback = Callable[[str, float, str, dict[str, Any]], None]


def _noop_cb(phase: str, progress: float, message: str = "",
             details: dict[str, Any] | None = None) -> None:
    """Callback verilmediğinde kullanılan no-op."""
    pass


def run_pipeline(
    video_path: str | Path,
    scene_name: str = "test_scene",
    cfg: Config | None = None,
    skip_foundation: bool = True,
    skip_training: bool = False,
    skip_export: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """
    Returns: { phase: durum }

    Args:
        progress_callback: (phase, progress_0_1, message, details) → None
            Her faz başlangıcında ve bitişinde, ayrıca training sırasında
            periyodik olarak çağrılır. None ise hiçbir şey yapmaz.
    """
    if cfg is None:
        cfg = default_config()
    paths = scene_paths(scene_name)
    paths["base"].mkdir(parents=True, exist_ok=True)
    (paths["output"] / "logs").mkdir(parents=True, exist_ok=True)

    cb: ProgressCallback = progress_callback or _noop_cb
    status = {}

    # Run logger — tüm pipeline'ı takip eder, metrics/events/summary yazar
    run_logger = RunLogger(paths["output"] / "logs", scene=scene_name)
    run_logger.log_event(
        "config",
        skip_foundation=skip_foundation,
        skip_training=skip_training,
        skip_export=skip_export,
        video_path=str(video_path),
    )
    try:
        # Config'i de event olarak yaz (dataclass → dict)
        from dataclasses import asdict as _asdict
        run_logger.log_event("config_snapshot", **{
            "cfg": _asdict(cfg),
        })
    except Exception:
        pass

    # -------- Faz 2a: Frame çıkarma --------
    print("\n[Faz 2a] Frame çıkarma")
    cb("frames", 0.0, "Video karelere ayrılıyor", {})
    extract_frames(
        video_path, paths["frames"],
        fps=cfg.preprocess.fps,
        resize_long_edge=cfg.preprocess.resize_long_edge,
    )
    status["frames"] = "ok"
    cb("frames", 1.0, "Kareler hazır", {})

    # -------- Faz 2b-c: COLMAP (cache'li + progress hook) --------
    print("\n[Faz 2b] COLMAP SfM")
    cb("colmap", 0.0, "COLMAP başlıyor", {})
    # Cache check — eğer sparse zaten hazırsa tekrar koşma
    try:
        cams = parse_cameras(paths["colmap"])
        xyz, rgb = load_points3d(paths["colmap"])
        print(f"✓ COLMAP cache hit: {len(cams)} kamera, {len(xyz)} nokta (rerun atlandı)")
        cb("colmap", 1.0, f"cache hit: {len(cams)} kamera", {"cache": True})
    except (FileNotFoundError, RuntimeError):
        # COLMAP stream progress → pipeline callback'e relay
        def _colmap_on_progress(frac: float, msg: str) -> None:
            cb("colmap", frac, msg, {"colmap_fraction": frac})

        run_colmap(
            paths["frames"], paths["colmap"],
            camera_model=cfg.preprocess.colmap_camera_model,
            use_gpu=cfg.preprocess.colmap_use_gpu,
            sequential=True,
            colmap_exe=cfg.preprocess.colmap_exe,
            on_progress=_colmap_on_progress,
        )
        cams = parse_cameras(paths["colmap"])
        xyz, rgb = load_points3d(paths["colmap"])
    status["colmap"] = f"{len(cams)} kamera, {len(xyz)} nokta"
    cb("colmap", 1.0, f"{len(cams)} kamera, {len(xyz)} 3B nokta",
       {"cameras": len(cams), "points": len(xyz)})

    # -------- Faz 3: Foundation modeller (opsiyonel, resilient) --------
    # Her fazın hatası diğerlerini etkilemesin — biri patlasa bile pipeline devam eder.
    if not skip_foundation:
        foundation_status = {"depth": "pending", "tracks": "pending", "masks": "pending"}

        # 3a — Depth
        print("\n[Faz 3a] MiDaS derinlik tahmini")
        cb("foundation", 0.0, "Derinlik tahmini", {})
        try:
            from .preprocess.depth_estimate import estimate_depth
            estimate_depth(paths["frames"], paths["depth"],
                           model_name=cfg.foundation.metric3d_model)
            foundation_status["depth"] = "ok"
        except Exception as e:
            print(f"⚠ Derinlik tahmini başarısız, atlanıyor: {e}")
            foundation_status["depth"] = f"failed: {e}"

        # 3a.5 — MiDaS → COLMAP scale alignment (v3.7 / Option B)
        # MiDaS relative depth üretiyor, COLMAP world scale ile uyumsuz.
        # Anchor unprojection doğru 3D koordinat üretebilsin diye align ediyoruz.
        # Track loss'un 0.15'te takılmasının ana sebebi bu uyumsuzluktu.
        run_logger.log_event("phase:start", phase="depth_align")
        if foundation_status["depth"] == "ok":
            try:
                print("\n[Faz 3a.5] Depth → COLMAP scale alignment")
                from .preprocess.align_depth import align_depth_to_colmap
                # cams = Dict[image_name, {K, w2c, width, height}]
                # Sıralı isim listesi al, frame_paths ile uyumlu sırada w2c topla
                frame_paths_all = sorted(paths["frames"].glob("frame_*.png"))
                cam_names_sorted = sorted(cams.keys())
                first_cam = cams[cam_names_sorted[0]]
                w2c_list = [
                    torch.from_numpy(cams[n]["w2c"]).float()
                    for n in cam_names_sorted
                ]
                K_first = torch.from_numpy(first_cam["K"]).float()
                W_frame = int(first_cam["width"])
                H_frame = int(first_cam["height"])
                xyz_t = torch.from_numpy(xyz).float()

                # Frame sayısı eşleşmesini garantile (bazen COLMAP az register edebilir)
                n = min(len(frame_paths_all), len(w2c_list))
                align_stats = align_depth_to_colmap(
                    depth_dir=paths["depth"],
                    frame_paths=frame_paths_all[:n],
                    cam_K=K_first,
                    cam_w2c_per_frame=w2c_list[:n],
                    colmap_xyz=xyz_t,
                    frame_size=(W_frame, H_frame),
                    overwrite=True,
                )
                foundation_status["depth"] = f"ok (COLMAP-aligned, scale={align_stats['global_scale_median']:.3f})"
                run_logger.log_event(
                    "depth_align:ok",
                    n_aligned=align_stats["n_frames_aligned"],
                    n_skipped=align_stats["n_frames_skipped"],
                    global_scale=align_stats["global_scale_median"],
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"⚠ Depth align başarısız (kritik değil, training devam): {e}")
                print(tb)
                run_logger.log_event("depth_align:failed", error=str(e)[:200])
                # Alignment opsiyonel, pipeline devam eder
        else:
            run_logger.log_event("depth_align:skipped",
                                 reason=f"depth status={foundation_status['depth']}")

        cb("foundation", 0.33, "Depth done, CoTracker başlıyor", {})

        # 3b — Tracks
        # v3.7.5: MiDaS modelini eksplisit release et — module-level cache'te
        # 700MB-1.4GB tutuyordu. empty_cache() bunu temizlemiyor.
        # Sonra CoTracker yer bulamadığı için OOM oluyordu.
        print("\n[Faz 3b] CoTracker piksel takibi")
        try:
            from .preprocess.depth_estimate import release_models as release_depth_models
            release_depth_models()
        except Exception as _e:
            print(f"  ⚠ depth model release: {_e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            from .preprocess.point_tracking import track_points
            track_points(paths["frames"], paths["tracks"] / "tracks.pt",
                         grid_size=cfg.foundation.cotracker_grid_size)
            foundation_status["tracks"] = "ok"
        except Exception as e:
            print(f"⚠ Tracking başarısız, atlanıyor: {e}")
            foundation_status["tracks"] = f"failed: {e}"
        # Cleanup after CoTracker too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cb("foundation", 0.66, "Tracks done, dinamik maske başlıyor", {})

        # 3c — Dynamic mask
        print("\n[Faz 3c] Dinamik maske")
        try:
            from .preprocess.dynamic_mask import compute_dynamic_masks
            compute_dynamic_masks(paths["frames"], paths["masks"])
            foundation_status["masks"] = "ok"
        except Exception as e:
            print(f"⚠ Maske başarısız, atlanıyor: {e}")
            foundation_status["masks"] = f"failed: {e}"

        status["foundation"] = ", ".join(f"{k}={v}" for k, v in foundation_status.items())
        cb("foundation", 1.0, f"Foundation bitti: {status['foundation']}", foundation_status)
    else:
        print("\n[Faz 3] Foundation modeller atlandı (--skip-foundation)")
        status["foundation"] = "skipped"
        cb("foundation", 1.0, "Atlandı", {"skipped": True})

    # -------- Faz 4-5: Model + Training --------
    if skip_training:
        print("\n[Faz 4-5] Training atlandı (--skip-training)")
        status["training"] = "skipped"
        return status

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠ CUDA bulunamadı — training çok yavaş olacak (test amaçlı)")

    print("\n[Faz 4] Gaussian model + DeformationField")
    cb("init", 0.0, "Model başlatılıyor", {})
    init_pts = torch.from_numpy(xyz)
    init_rgb = torch.from_numpy(rgb).float() / 255.0

    # v3.7.3: Initial point subsample. COLMAP bazen banana gibi sahnelerde 100k+
    # sparse point döndürüyor. max_gaussians cap sadece split/clone'u durduruyor,
    # initial N'i shrink etmiyor → render aşırı yavaşlar (banana_ultra_v2'de
    # 112k init → 0.19 it/s = 117 saat ETA).
    # Bu fix: Eğer N_init > max_gaussians × 0.7, random downsample yap (cap'in
    # %70'i, density'nin büyümeye yer bırakması için).
    if cfg.train.max_gaussians > 0:
        target_init = int(cfg.train.max_gaussians * 0.7)
        if init_pts.shape[0] > target_init:
            print(f"⚠ Initial COLMAP points ({init_pts.shape[0]:,}) > target "
                  f"({target_init:,}) — random subsample (max_gaussians × 0.7)")
            perm = torch.randperm(init_pts.shape[0])[:target_init]
            init_pts = init_pts[perm]
            init_rgb = init_rgb[perm]
            run_logger.log_event("init_subsample",
                                 before=int(xyz.shape[0]),
                                 after=int(init_pts.shape[0]),
                                 max_gaussians=cfg.train.max_gaussians)

    gs = GaussianModel(init_pts, init_colors=init_rgb,
                       sh_degree=cfg.model.sh_degree,
                       fourier_K=cfg.model.fourier_K)
    deform = DeformationField(
        resolution=cfg.model.hexplane_resolution,
        feat_dim=cfg.model.hexplane_feat_dim,
        mlp_width=cfg.model.mlp_width,
        mlp_depth=cfg.model.mlp_depth,
        num_time_freqs=cfg.model.num_time_freqs,
    )
    extent = compute_scene_extent(xyz)
    print(f"  Sahne kapsamı: {extent:.3f}, başlangıç Gaussian: {gs.num_points:,}")
    cb("init", 1.0, f"Başlangıç Gaussian: {gs.num_points:,}",
       {"num_points": gs.num_points, "scene_extent": float(extent)})

    # Frame yolları + kamera pozları (COLMAP'in kullandığı isim sırasına göre)
    frame_paths, w2c_list, K_first = [], [], None
    for name in sorted(cams.keys()):
        cam = cams[name]
        fp = Path(paths["frames"]) / name
        if not fp.exists():
            continue
        frame_paths.append(fp)
        w2c_list.append(torch.from_numpy(cam["w2c"]).float())
        if K_first is None:
            K_first = torch.from_numpy(cam["K"]).float()

    if not frame_paths:
        raise RuntimeError("COLMAP kameraları ile frame dosyaları eşleşmedi")
    print(f"  Eğitim için {len(frame_paths)} frame eşleşti")

    print("\n[Faz 5] Training loop")
    trainer = Trainer4DGS(
        gs, deform,
        device=device,
        scene_extent=extent,
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
        lambda_deform_reg=cfg.train.lambda_deform_reg,
        lambda_smoothness=cfg.train.lambda_smoothness,
        lambda_rigidity=cfg.train.lambda_rigidity,
        lambda_depth=cfg.train.lambda_depth,
        lambda_mask_motion=cfg.train.lambda_mask_motion,
        lambda_track=cfg.train.lambda_track,
        track_sample_k=cfg.train.track_sample_k,
        lambda_scale=cfg.train.lambda_scale,
        # v3.8 — anisotropy + tightened dpos clamp
        lambda_aniso=cfg.train.lambda_aniso,
        aniso_threshold=cfg.train.aniso_threshold,
        dpos_total_cap_frac=cfg.train.dpos_total_cap_frac,
        opacity_reset_interval=cfg.train.opacity_reset_interval,
        warmup_iters=cfg.train.warmup_iters,
        # v3.6 / Yol C — Per-gaussian Fourier trajectory
        deform_pos_mode=cfg.model.deform_pos_mode,
        lr_fourier=cfg.train.lr_fourier,
        lambda_fourier_reg=cfg.train.lambda_fourier_reg,
        # v3.7.2 — N hard cap
        max_gaussians=cfg.train.max_gaussians,
    )
    # Foundation çıktıları varsa trainer'a ver (stage 2 loss'lar için)
    depth_dir_arg = paths["depth"] if (not skip_foundation and paths["depth"].exists()) else None
    mask_dir_arg  = paths["masks"] if (not skip_foundation and paths["masks"].exists()) else None
    tracks_file = paths["tracks"] / "tracks.pt"
    tracks_path_arg = tracks_file if (not skip_foundation and tracks_file.exists()) else None
    # Trainer ilerlemesini pipeline callback'ine relay eden köprü:
    # trainer iç ilerlemesini (0-1 arası) "training" fazına map ederiz.
    def _train_progress(iter_idx: int, total: int, loss: float,
                        psnr_val: float, n_pts: int) -> None:
        cb(
            "training",
            iter_idx / max(total, 1),
            f"iter {iter_idx}/{total} — loss={loss:.4f} psnr={psnr_val:.2f} N={n_pts:,}",
            {
                "iter": iter_idx, "total": total,
                "loss": loss, "psnr": psnr_val, "num_points": n_pts,
            },
        )

    cb("training", 0.0, "Training başlıyor", {"total_iters": cfg.train.n_iters})
    run_logger.phase_start("training")
    history = trainer.train(
        frame_paths, K_first, w2c_list,
        n_iters=cfg.train.n_iters,
        image_size=cfg.train.image_resolution,
        ckpt_dir=paths["output"] / "ckpt",
        ckpt_interval=cfg.train.ckpt_interval,
        log_interval=cfg.train.log_interval,
        progress_callback=_train_progress,
        depth_dir=depth_dir_arg,
        mask_dir=mask_dir_arg,
        tracks_path=tracks_path_arg,
        run_logger=run_logger,
    )
    run_logger.phase_end(
        "training",
        final_loss=history["loss"][-1] if history["loss"] else None,
        final_psnr=history["psnr"][-1] if history["psnr"] else None,
        final_n=history["n_pts"][-1] if history["n_pts"] else None,
    )
    # Gerçek log entry sayısından effective iter hesapla (cfg.n_iters değil — aborted olabilir)
    actual_logs = len(history.get("loss", []))
    effective_iters = actual_logs * cfg.train.log_interval
    if effective_iters < cfg.train.n_iters:
        status["training"] = f"⚠ {effective_iters} iter (konfigdeki {cfg.train.n_iters} tamamlanamadı), son loss={history['loss'][-1]:.4f}"
        run_logger.warn("training_incomplete",
                        effective_iters=effective_iters,
                        config_iters=cfg.train.n_iters)
    else:
        status["training"] = f"{effective_iters} iter, son loss={history['loss'][-1]:.4f}"
    cb("training", 1.0, f"Training bitti, son loss={history['loss'][-1]:.4f}",
       {"final_loss": history["loss"][-1] if history["loss"] else None,
        "final_psnr": history["psnr"][-1] if history["psnr"] else None,
        "final_num_points": history["n_pts"][-1] if history["n_pts"] else None})

    # -------- Faz 6: Export --------
    if skip_export:
        print("\n[Faz 6] Export atlandı (--skip-export)")
        status["export"] = "skipped"
        return status

    print("\n[Faz 6] PLY export")
    cb("export", 0.0, f"{cfg.export.num_timestamps} timestamp export ediliyor", {})
    export_to_ply(
        gs=trainer.gs,
        deform=trainer.deform,
        output_dir=paths["output"] / "ply",
        num_timestamps=cfg.export.num_timestamps,
        scene_extent=extent,
        device=device,
    )
    status["export"] = f"{cfg.export.num_timestamps} timestamp"
    cb("export", 1.0, f"{cfg.export.num_timestamps} .ply yazıldı",
       {"num_timestamps": cfg.export.num_timestamps,
        "ply_dir": str(paths["output"] / "ply")})

    # Run summary — final metrics, config, PLY stats, phase durations
    try:
        ply_dir = paths["output"] / "ply"
        ply_files = sorted(ply_dir.glob("*.ply")) if ply_dir.exists() else []
        total_ply_bytes = sum(p.stat().st_size for p in ply_files)
        run_logger.save_summary({
            "status": status,
            "config": cfg,
            "final_loss": history["loss"][-1] if history.get("loss") else None,
            "final_psnr": history["psnr"][-1] if history.get("psnr") else None,
            "final_n_points": history["n_pts"][-1] if history.get("n_pts") else None,
            "ply_count": len(ply_files),
            "ply_total_bytes": total_ply_bytes,
            "scene_extent": float(extent),
        })
    except Exception as _e:
        print(f"  ⚠ summary save failed: {_e}")
    run_logger.close()

    return status


def main():
    p = argparse.ArgumentParser(description="4DGS Studio - end-to-end pipeline")
    p.add_argument("video", type=str, help="Giriş video dosyası")
    p.add_argument("--scene", default="test_scene", help="Sahne adı (data/<scene>/ altında)")
    p.add_argument("--cloud", action="store_true", help="Cloud config kullan (RTX 4090)")
    p.add_argument("--skip-foundation", action="store_true",
                   help="Foundation model çıkarımını atla (varsayılan)")
    p.add_argument("--with-foundation", action="store_true",
                   help="Foundation modelleri çalıştır")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--colmap-exe", default=None,
                   help="COLMAP exe tam yolu (PATH'te yoksa). "
                        "Alternatif: COLMAP_EXE ortam değişkeni.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Hızlı uçtan uca test: 500 iter, 480x270, 10 timestamp (~1-2 dk)")
    args = p.parse_args()

    cfg = cloud_config() if args.cloud else default_config()
    if args.colmap_exe:
        cfg.preprocess.colmap_exe = args.colmap_exe

    if args.smoke_test:
        cfg.train.n_iters = 500
        cfg.train.image_resolution = (480, 270)
        cfg.train.ckpt_interval = 500
        cfg.train.log_interval = 25
        cfg.train.density_start_iter = 100
        cfg.train.density_end_iter = 400
        cfg.train.density_interval = 50
        cfg.export.num_timestamps = 10
        print("→ SMOKE TEST modu: 500 iter, 480×270, 10 timestamp")

    skip_found = not args.with_foundation if args.with_foundation else True
    if args.skip_foundation:
        skip_found = True

    t0 = time.time()
    status = run_pipeline(
        args.video, args.scene, cfg,
        skip_foundation=skip_found,
        skip_training=args.skip_training,
        skip_export=args.skip_export,
    )
    print(f"\n=== Pipeline tamamlandi ({time.time()-t0:.1f}s) ===")
    for k, v in status.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
