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

    cb: ProgressCallback = progress_callback or _noop_cb
    status = {}

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

    # -------- Faz 2b-c: COLMAP --------
    print("\n[Faz 2b] COLMAP SfM")
    cb("colmap", 0.0, "Kamera pozları tahmin ediliyor (feature + match + mapper)", {})
    run_colmap(
        paths["frames"], paths["colmap"],
        camera_model=cfg.preprocess.colmap_camera_model,
        use_gpu=cfg.preprocess.colmap_use_gpu,
        sequential=True,
        colmap_exe=cfg.preprocess.colmap_exe,
    )
    cams = parse_cameras(paths["colmap"])
    xyz, rgb = load_points3d(paths["colmap"])
    status["colmap"] = f"{len(cams)} kamera, {len(xyz)} nokta"
    cb("colmap", 1.0, f"{len(cams)} kamera, {len(xyz)} 3B nokta",
       {"cameras": len(cams), "points": len(xyz)})

    # -------- Faz 3: Foundation modeller (opsiyonel) --------
    if not skip_foundation:
        print("\n[Faz 3a] Metric3D-v2 derinlik tahmini")
        cb("foundation", 0.0, "Derinlik tahmini (Metric3D)", {})
        from .preprocess.depth_estimate import estimate_depth
        estimate_depth(paths["frames"], paths["depth"],
                       model_name=cfg.foundation.metric3d_model)
        cb("foundation", 0.33, "CoTracker çalışıyor", {})
        print("\n[Faz 3b] CoTracker piksel takibi")
        from .preprocess.point_tracking import track_points
        track_points(paths["frames"], paths["tracks"] / "tracks.pt",
                     grid_size=cfg.foundation.cotracker_grid_size)
        cb("foundation", 0.66, "Dinamik maske hesaplanıyor", {})
        print("\n[Faz 3c] Dinamik maske")
        from .preprocess.dynamic_mask import compute_dynamic_masks
        compute_dynamic_masks(paths["frames"], paths["masks"])
        status["foundation"] = "ok"
        cb("foundation", 1.0, "Foundation modeller tamamlandı", {})
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
    gs = GaussianModel(init_pts, init_colors=init_rgb,
                       sh_degree=cfg.model.sh_degree)
    deform = DeformationField(
        resolution=cfg.model.hexplane_resolution,
        feat_dim=cfg.model.hexplane_feat_dim,
        mlp_width=cfg.model.mlp_width,
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
    )
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
    history = trainer.train(
        frame_paths, K_first, w2c_list,
        n_iters=cfg.train.n_iters,
        image_size=cfg.train.image_resolution,
        ckpt_dir=paths["output"] / "ckpt",
        ckpt_interval=cfg.train.ckpt_interval,
        log_interval=cfg.train.log_interval,
        progress_callback=_train_progress,
    )
    status["training"] = f"{cfg.train.n_iters} iter, son loss={history['loss'][-1]:.4f}"
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
        # Uçtan uca her şeyin çalıştığını hızlıca doğrulayan preset
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
    print(f"\n=== Pipeline tamamlandı ({time.time()-t0:.1f}s) ===")
    for k, v in status.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
