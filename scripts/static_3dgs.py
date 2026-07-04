"""Static 3D Gaussian Splatting — Faz 2 preset runner.

Mevcut 4D pipeline'i `cfg.train.static_mode=True` ile statik 3DGS olarak kosar.
4D dynamic features (deformation MLP + Fourier trajectories + motion losses)
tamamen kapali; sadece RGB + SSIM + (opsiyonel) LPIPS supervision.

Workflow:
    # Photo set veya tek-frame multi-view sahne icin:
    python scripts/static_3dgs.py --scene truck --preset balanced

    # Hizli smoke (5-8 dk):
    python scripts/static_3dgs.py --scene truck --preset fast

    # Premium overnight (6-10 saat, 4dv.ai-tier):
    python scripts/static_3dgs.py --scene truck --preset premium

    # SOTA benchmark paritesi (Mip-NeRF360 protokolu — vanilla-3DGS loss/densify;
    # --native-res ILE ve --foundation OLMADAN kos, yoksa parite bozulur):
    python scripts/static_3dgs.py --scene garden --preset sota --nvs-eval --native-res

Presets (3060 Ti, 8GB VRAM hedefli; sota A100/24GB+ ister):
    fast      —  7,000 iter,  720p,  100k cap,  5-8 dk     | preview kalite
    balanced  — 30,000 iter, 1080p,  250k cap,  30-45 dk   | sosyal medya
    high      — 50,000 iter, 1080p,  500k cap,  2-3 saat   | profesyonel
    premium   —100,000 iter, native, 1M cap,    6-10 saat  | 4dv.ai-tier
    sota      — 30,000 iter, native, 6M cap,    ~1 saat/A100| PSNR paritesi

Tek argumanlar:
    --preset      fast | balanced | high | premium | sota
    --scene       data/<scene>/ altinda (default: test_scene)
    --video       Photo set yoksa video.mp4 path. Default scene_paths uzerinden bulur.
    --no-export   PLY export'u atla (cache test icin)
    --force       Tum cache'leri yoksay, baştan calistir
    --dry-run     Sadece config tablosunu yazdir, calistirma
    --colmap-cpu  COLMAP CPU SIFT (headless/CUDA'siz build'ler, orn. Colab apt colmap)
    --native-res  Train/eval cozunurlugu kaynak frame boyutundan (protokol paritesi)

Bagimliliklar: backend.pipeline.run_pipeline (single-view path).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Windows consoles default to cp1254/cp1252 and choke on the → glyphs below
# (same guard as scripts/sota_compare.py).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (  # noqa: E402
    default_config, scene_paths, is_static_scene,
)
# NOT: backend.pipeline torch'a bagimli — deferred import (main icinde)
# boylece --list-presets / --dry-run torch yokken bile calisir.


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "fast": {
        "description": "5-8 dk preview",
        "n_iters": 7_000,
        "image_resolution": (1280, 720),
        "max_gaussians": 100_000,
        "ckpt_interval": 7_000,
        "log_interval": 50,
        "density_start_iter": 500,
        "density_end_iter": 5_500,
        "density_interval": 100,
        "densify_grad_threshold": 4e-4,
        "opacity_reset_interval": 0,
        "lambda_ssim": 0.2,
        "lambda_lpips": 0.0,
        # Static depth supervision — Metric3D-aligned mono depth
        "lambda_depth": 0.05,
        "metric3d_model": "metric3d_vit_small",
        "lambda_aniso": 5e-3,
        "aniso_threshold": 5.0,
        "fps": 10,
        "resize_long_edge": 960,
        "colmap_matching": "exhaustive",
        "init_subsample_mode": "confidence",
        "multires_schedule": [],
        "bg_distance_ratio": 2.0,
    },
    "balanced": {
        "description": "30-45 dk production",
        "n_iters": 30_000,
        "image_resolution": (1920, 1080),
        "max_gaussians": 250_000,
        "ckpt_interval": 5_000,
        "log_interval": 100,
        "density_start_iter": 500,
        "density_end_iter": 22_000,
        "density_interval": 100,
        "densify_grad_threshold": 2e-4,
        "opacity_reset_interval": 3_000,
        "lambda_ssim": 0.2,
        "lambda_lpips": 0.05,
        "lpips_net": "vgg",
        "lpips_warmup_iters": 1_000,
        "lambda_depth": 0.10,
        "metric3d_model": "metric3d_vit_small",
        "lambda_aniso": 5e-3,
        "aniso_threshold": 5.0,
        "fps": 15,
        "resize_long_edge": 1280,
        "colmap_matching": "exhaustive",
        "init_subsample_mode": "confidence",
        "multires_schedule": [],
        "bg_distance_ratio": 2.0,
    },
    "high": {
        "description": "2-3 saat profesyonel",
        "n_iters": 50_000,
        "image_resolution": (1920, 1080),
        "max_gaussians": 500_000,
        "ckpt_interval": 10_000,
        "log_interval": 100,
        "density_start_iter": 500,
        "density_end_iter": 35_000,
        "density_interval": 100,
        "densify_grad_threshold": 1.5e-4,
        "opacity_reset_interval": 3_000,
        "lambda_ssim": 0.2,
        "lambda_lpips": 0.10,
        "lpips_net": "vgg",
        "lpips_warmup_iters": 2_000,
        "lambda_depth": 0.10,
        "metric3d_model": "metric3d_vit_large",
        "lambda_aniso": 1e-2,
        "aniso_threshold": 4.0,
        "fps": 20,
        "resize_long_edge": 1600,
        "colmap_matching": "exhaustive",
        "init_subsample_mode": "confidence",
        "multires_schedule": [(0, 720), (15_000, 1_080)],
        "bg_distance_ratio": 2.0,
    },
    "premium": {
        "description": "6-10 saat 4dv.ai-tier",
        "n_iters": 100_000,
        "image_resolution": (2560, 1440),
        "max_gaussians": 1_000_000,
        "ckpt_interval": 20_000,
        "log_interval": 200,
        "density_start_iter": 500,
        "density_end_iter": 70_000,
        "density_interval": 100,
        "densify_grad_threshold": 1e-4,
        "opacity_reset_interval": 3_000,
        "lambda_ssim": 0.2,
        "lambda_lpips": 0.15,
        "lpips_net": "vgg",
        "lpips_warmup_iters": 3_000,
        "lambda_depth": 0.15,
        "metric3d_model": "metric3d_vit_large",
        "lambda_aniso": 1e-2,
        "aniso_threshold": 3.5,
        "fps": 30,
        "resize_long_edge": 2_560,
        "colmap_matching": "exhaustive",
        "init_subsample_mode": "confidence",
        "multires_schedule": [(0, 720), (25_000, 1_080), (60_000, 1_440)],
        "bg_distance_ratio": 1.5,
    },
    # PSNR-parity run against published Mip-NeRF360 numbers. The product presets
    # above trade PSNR for perceptual quality (LPIPS/depth/aniso losses, low N cap);
    # this one mirrors vanilla 3DGS so the sota_compare verdict measures the
    # pipeline, not the preset's objective: pure L1+0.2*D-SSIM, densify until
    # 15k @ grad 2e-4, opacity reset 3k, 6M cap, no multires warmup.
    # Run WITH --native-res and WITHOUT --foundation (2026-07-04 garden premium run:
    # 24.94 dB / LPIPS-VGG 0.0776 vs 3DGS 27.41 dB / 0.103 — the LPIPS win is the
    # perceptual-loss trade-off, which this preset removes).
    "sota": {
        "description": "Mip-NeRF360 PSNR paritesi (--native-res ile, --foundation'siz)",
        "n_iters": 30_000,
        "image_resolution": (1920, 1080),   # --native-res kaynaktan turetir; bunu kullanma
        "max_gaussians": 6_000_000,
        "ckpt_interval": 10_000,
        "log_interval": 100,
        "density_start_iter": 500,
        "density_end_iter": 15_000,
        "density_interval": 100,
        "densify_grad_threshold": 2e-4,
        "opacity_reset_interval": 3_000,
        "lambda_ssim": 0.2,
        "lambda_lpips": 0.0,
        "lambda_depth": 0.0,
        "lambda_aniso": 0.0,
        "aniso_threshold": 5.0,
        "fps": 30,
        "resize_long_edge": 1600,
        "colmap_matching": "exhaustive",
        "init_subsample_mode": "confidence",
        "multires_schedule": [],
        "bg_distance_ratio": 2.0,
    },
}


def _apply_preset(cfg, preset_name: str) -> None:
    """Preset ayarlarini cfg'ye uygula. Static mode otomatik aktif."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}"
        )
    p = PRESETS[preset_name]
    cfg.train.static_mode = True

    cfg.train.n_iters = p["n_iters"]
    cfg.train.image_resolution = tuple(p["image_resolution"])
    cfg.train.max_gaussians = p["max_gaussians"]
    cfg.train.ckpt_interval = p["ckpt_interval"]
    cfg.train.log_interval = p["log_interval"]

    cfg.train.density_start_iter = p["density_start_iter"]
    cfg.train.density_end_iter = p["density_end_iter"]
    cfg.train.density_interval = p["density_interval"]
    cfg.train.densify_grad_threshold = p["densify_grad_threshold"]
    cfg.train.opacity_reset_interval = p["opacity_reset_interval"]

    cfg.train.lambda_ssim = p["lambda_ssim"]
    cfg.train.lambda_lpips = p.get("lambda_lpips", 0.0)
    if "lpips_net" in p:
        cfg.train.lpips_net = p["lpips_net"]
    if "lpips_warmup_iters" in p:
        cfg.train.lpips_warmup_iters = p["lpips_warmup_iters"]
    cfg.train.lambda_aniso = p["lambda_aniso"]
    cfg.train.aniso_threshold = p["aniso_threshold"]

    # Static depth supervision (Metric3D + COLMAP scale alignment)
    cfg.train.lambda_depth = p.get("lambda_depth", 0.0)
    if "metric3d_model" in p:
        cfg.foundation.metric3d_model = p["metric3d_model"]

    # 4D-only loss'lar 0'lanmali (motion regs / track / mask / flow)
    cfg.train.lambda_mask_motion = 0.0
    cfg.train.lambda_track = 0.0
    cfg.train.lambda_flow = 0.0
    cfg.train.lambda_smoothness = 0.0
    cfg.train.lambda_rigidity = 0.0
    cfg.train.lambda_deform_reg = 0.0
    cfg.train.lambda_fourier_reg = 0.0
    cfg.train.lambda_multiview_consistency = 0.0
    cfg.train.auto_static_dynamic = False

    cfg.train.multires_schedule = list(p.get("multires_schedule", []))
    cfg.train.bg_distance_ratio = p["bg_distance_ratio"]

    cfg.preprocess.fps = p["fps"]
    cfg.preprocess.resize_long_edge = p["resize_long_edge"]
    cfg.preprocess.colmap_matching = p["colmap_matching"]
    cfg.preprocess.init_subsample_mode = p["init_subsample_mode"]

    cfg.export.num_timestamps = 1


def _print_preset_table() -> None:
    print("\n" + "=" * 78)
    print("  Static 3DGS Presets")
    print("=" * 78)
    print(f"  {'preset':<10} {'iters':>8} {'res':>11} {'cap':>8} {'lpips':>6}  desc")
    print("  " + "-" * 76)
    for name, p in PRESETS.items():
        res = f"{p['image_resolution'][0]}x{p['image_resolution'][1]}"
        if p["max_gaussians"] >= 1_000_000:
            cap = f"{p['max_gaussians'] // 1_000_000}M"
        else:
            cap = f"{p['max_gaussians'] // 1000}k"
        lpips = p.get("lambda_lpips", 0.0)
        print(f"  {name:<10} {p['n_iters']:>8} {res:>11} {cap:>8} {lpips:>6.2f}  {p['description']}")
    print()


def _resolve_input(scene_name: str, video_arg):
    """Static 3DGS input cozumu.

    Oncelik:
      1. data/<scene>/images/ klasoru (photo set)
      2. data/<scene>/video.mp4
      3. CLI --video flag
    Returns: (video_path, scene_paths_dict, is_photo_set)
    """
    paths = scene_paths(scene_name)
    if not paths["base"].exists():
        raise FileNotFoundError(f"Scene not found: {paths['base']}")

    images_dir = paths["base"] / "images"
    has_imgs = False
    if images_dir.exists():
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
            if any(images_dir.glob(ext)):
                has_imgs = True
                break

    if has_imgs:
        n_imgs = sum(1 for _ in images_dir.glob("*.*"))
        print(f"[static_3dgs] Photo set: {n_imgs} images in {images_dir}")
        frames_dst = paths["frames"]
        if not frames_dst.exists() or not any(frames_dst.glob("frame_*.png")):
            print(f"[static_3dgs] images/ -> frames/ kopyalaniyor...")
            frames_dst.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            all_imgs = []
            for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
                all_imgs.extend(images_dir.glob(ext))
            for i, img_path in enumerate(sorted(all_imgs)):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.save(frames_dst / f"frame_{i:06d}.png")
                except Exception as e:
                    print(f"  ! {img_path.name} skipped: {e}")
            n_done = len(list(frames_dst.glob("frame_*.png")))
            print(f"[static_3dgs]   ok: {n_done} frame")
        return paths["video"], paths, True

    if video_arg:
        return Path(video_arg), paths, False
    if paths["video"].exists():
        return paths["video"], paths, False

    raise FileNotFoundError(
        f"Static input bulunamadi: ne {images_dir} ne de {paths['video']}.\n"
        f"Coz: data/{scene_name}/images/IMG_*.jpg KOY veya --video <path> ver."
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Static 3DGS preset runner")
    p.add_argument("--scene", default="test_scene")
    p.add_argument("--preset", default="balanced", choices=list(PRESETS.keys()))
    p.add_argument("--video", default=None)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--nvs-eval", action="store_true",
                   help="Enable held-out NVS eval (static: every-8 interleaved) → "
                        "writes output/eval/nvs_eval.json for scripts/sota_compare.py")
    p.add_argument("--foundation", action="store_true",
                   help="Run foundation depth (Metric3D) supervision instead of the "
                        "default skip_foundation=True (needed for max-quality SOTA runs)")
    p.add_argument("--colmap-cpu", action="store_true",
                   help="Force COLMAP CPU SIFT (use_gpu=0). Needed on headless/CUDA-less "
                        "COLMAP builds (e.g. Colab's apt colmap) where GPU SIFT crashes")
    p.add_argument("--native-res", action="store_true",
                   help="Derive train/eval resolution from the source frames instead of "
                        "the preset's fixed image_resolution (photo-set protocol parity, "
                        "e.g. Mip-NeRF360 images_4 — required for comparable SOTA metrics)")
    args = p.parse_args()

    if args.list_presets:
        _print_preset_table()
        return 0

    print("=" * 78)
    print(f"  Static 3DGS — preset={args.preset}, scene={args.scene}")
    print("=" * 78)

    if is_static_scene(args.scene):
        print(f"[static_3dgs] is_static_scene({args.scene}) = True")
    else:
        print(f"[static_3dgs] WARN: is_static_scene({args.scene}) = False")

    cfg = default_config()
    _apply_preset(cfg, args.preset)
    print(f"[static_3dgs] Preset '{args.preset}' uygulandi")
    print(f"  static_mode={cfg.train.static_mode}, iters={cfg.train.n_iters}, "
          f"res={cfg.train.image_resolution}, max_gauss={cfg.train.max_gaussians}")
    print(f"  lambda_lpips={cfg.train.lambda_lpips}, lambda_aniso={cfg.train.lambda_aniso}, "
          f"densify_thr={cfg.train.densify_grad_threshold}")

    if args.nvs_eval:
        cfg.train.nvs_eval_enabled = True
        print("[static_3dgs] NVS eval ON → held-out PSNR/SSIM/LPIPS → output/eval/nvs_eval.json")
    if args.foundation:
        print("[static_3dgs] Foundation ON → Metric3D depth supervision active "
              f"(lambda_depth={cfg.train.lambda_depth})")
    if args.colmap_cpu:
        cfg.preprocess.colmap_use_gpu = False
        print("[static_3dgs] COLMAP CPU SIFT ON (use_gpu=0 — headless/CUDA'siz build)")
    if args.native_res:
        cfg.train.native_resolution = True
        print("[static_3dgs] Native-res ON → image_resolution kaynak frame boyutundan turetilecek")

    if args.dry_run:
        print("\n[dry-run] Calistirilmadi.")
        return 0

    from backend.pipeline import run_pipeline  # deferred (torch dep)

    video_path, paths, is_photo = _resolve_input(args.scene, args.video)
    src_kind = "photo-set" if is_photo else "video"
    print(f"  input: {src_kind} -> {video_path}")

    t0 = time.time()
    status = run_pipeline(
        video_path=video_path,
        scene_name=args.scene,
        cfg=cfg,
        skip_foundation=not args.foundation,
        skip_training=False,
        skip_export=args.no_export,
        force_preprocess=args.force,
    )
    elapsed = time.time() - t0
    print(f"\n=== Static 3DGS done ({elapsed/60:.1f} dk) ===")
    for k, v in status.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
