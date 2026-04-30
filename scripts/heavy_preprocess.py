"""Phase 1.2 — Heavy preprocessing premium runner.

Bir sahne icin TUM preprocessing adimlarini bir kerede, profile'a gore
maksimum kalitede calistirir. Sonuc disk'e cache'lenir, training run'lari
bu cache'i kullanir (her parametre degisikliginde COLMAP'a donmek yerine).

Workflow:
    # Bir kere (uzun, agir):
    python scripts/heavy_preprocess.py --scene flame_steak --profile premium

    # Sonra defalarca training (her biri 10-15 dk, COLMAP cache'den hizli):
    curl /process -F "scene=flame_steak" -F "iters=10000" ...
    curl /process -F "scene=flame_steak" -F "iters=20000" -F "lambda_rigid=0.5" ...
    curl /process -F "scene=flame_steak" -F "max_gaussians=500000" ...

Profiles:
    standard — varsayilan ayarlar (5-10 dk MV, 30 dk SV)
    high     — orta yogunluk (30 dk MV, 1.5 saat SV)
    premium  — maksimum kalite (2-3 saat MV, 4-5 saat SV)

Usage:
    python scripts/heavy_preprocess.py --scene <name> [--profile premium] [--force]

Coklu sahne:
    for scene in flame_steak banana_demo chickchicken_v3_5_full; do
        python scripts/heavy_preprocess.py --scene $scene --profile premium
    done
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path so backend imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import default_config, scene_paths, is_multiview_scene  # noqa: E402
from backend.pipeline import run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Profile definitions — preprocessing intensity presets
# ---------------------------------------------------------------------------

PROFILES = {
    "standard": {
        "description": "Default settings — fast smoke (~5-10 dk MV, ~30 dk SV)",
        # Frame extract
        "fps": 10,
        "resize_long_edge": 960,
        # COLMAP single-view
        "colmap_matching": "sequential",
        # COLMAP multi-view
        "colmap_mv_timestamps": 5,
        "colmap_mv_dense_mvs": False,
        # Foundation
        "metric3d_model": "metric3d_vit_small",
        "cotracker_grid_size": 30,
    },
    "high": {
        "description": "Orta yogunluk — production-ish (~30 dk MV, ~1.5h SV)",
        "fps": 20,
        "resize_long_edge": 1280,
        "colmap_matching": "exhaustive",
        "colmap_mv_timestamps": 10,
        "colmap_mv_dense_mvs": True,
        "metric3d_model": "metric3d_vit_large",  # Phase 1.4: DPT_Large (~1.4 GB)
        "cotracker_grid_size": 50,
    },
    "premium": {
        "description": "Maksimum kalite — chickchicken-grade (~2-3h MV, ~4-5h SV)",
        "fps": 30,
        "resize_long_edge": 1280,
        "colmap_matching": "exhaustive",
        "colmap_mv_timestamps": 25,
        "colmap_mv_dense_mvs": True,
        "metric3d_model": "metric3d_vit_large",  # Phase 1.4: DPT_Large premium
        "cotracker_grid_size": 75,
    },
}


def _apply_profile(cfg, profile_name: str) -> None:
    """Profile ayarlarini cfg'ye uygula."""
    p = PROFILES[profile_name]
    cfg.preprocess.fps = p["fps"]
    cfg.preprocess.resize_long_edge = p["resize_long_edge"]
    cfg.preprocess.colmap_matching = p["colmap_matching"]
    cfg.preprocess.colmap_mv_timestamps = p["colmap_mv_timestamps"]
    cfg.preprocess.colmap_mv_dense_mvs = p["colmap_mv_dense_mvs"]
    cfg.foundation.metric3d_model = p["metric3d_model"]
    cfg.foundation.cotracker_grid_size = p["cotracker_grid_size"]


def _print_profile_table() -> None:
    """Tum profile'lari ve ayarlarini tablo halinde yazdir."""
    print("\n" + "=" * 78)
    print("  Available Profiles:")
    print("=" * 78)
    for name, p in PROFILES.items():
        print(f"\n  {name.upper()}: {p['description']}")
        print(f"    fps={p['fps']}, resize_long_edge={p['resize_long_edge']}")
        print(f"    colmap_matching={p['colmap_matching']}")
        print(f"    multi-view: {p['colmap_mv_timestamps']} timesteps, "
              f"dense_mvs={p['colmap_mv_dense_mvs']}")
        print(f"    foundation: metric3d={p['metric3d_model']}, "
              f"cotracker_grid={p['cotracker_grid_size']}")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heavy preprocessing runner — bir sahne icin tum preprocessing'i bir kerede yapar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0] if __doc__ else "",
    )
    parser.add_argument("--scene", default=None,
                        help="Sahne adi (data/<scene>/ altinda olmali). "
                             "--list-profiles disindaki tum komutlarda zorunlu.")
    parser.add_argument("--profile", default="high",
                        choices=list(PROFILES.keys()),
                        help="Preprocessing yogunlugu (default: high)")
    parser.add_argument("--force", action="store_true",
                        help="Cache'leri yoksay, hepsini bastan kos")
    parser.add_argument("--video", default=None,
                        help="Single-view sahne icin video dosyasi yolu "
                             "(multi-view sahnelerde gerekmez, otomatik bulunur)")
    parser.add_argument("--list-profiles", action="store_true",
                        help="Tum profile'lari listele ve cik")
    parser.add_argument("--no-foundation", action="store_true",
                        help="Foundation modellerini atla (depth/tracks/masks). "
                             "Sadece frame extract + COLMAP (+ MVS) yapar.")
    args = parser.parse_args()

    if args.list_profiles:
        _print_profile_table()
        return 0

    if not args.scene:
        parser.error("--scene zorunludur (--list-profiles disinda)")

    # 1) Scene path validation
    paths = scene_paths(args.scene)
    if not paths["base"].exists():
        print(f"✗ Scene not found: {paths['base']}")
        print(f"  Expected layout: data/{args.scene}/")
        print(f"  - Single-view: data/{args.scene}/video.mp4")
        print(f"  - Multi-view:  data/{args.scene}/videos/cam*.mp4")
        return 1

    is_mv = is_multiview_scene(args.scene)
    print(f"\n{'=' * 78}")
    print(f"  Heavy Preprocess: scene='{args.scene}' profile='{args.profile}' "
          f"({'MULTI-VIEW' if is_mv else 'SINGLE-VIEW'})")
    print(f"{'=' * 78}\n")
    print(f"  {PROFILES[args.profile]['description']}")
    print(f"  Force re-run: {args.force}")
    print(f"  Skip foundation: {args.no_foundation}\n")

    # 2) Video path resolution
    video_path = None
    if is_mv:
        # Multi-view: dummy, pipeline kullanmaz (videos/ folder'i okur)
        videos_dir = paths.get("videos_mv", paths["base"] / "videos")
        if videos_dir.exists():
            mp4_files = sorted(videos_dir.glob("cam*.mp4"))
            if mp4_files:
                video_path = mp4_files[0]
                print(f"  Multi-view detected: {len(mp4_files)} cams, "
                      f"using {video_path.name} as nominal video reference")
        if video_path is None:
            print(f"✗ Multi-view but no cam*.mp4 in {videos_dir}")
            return 1
    else:
        # Single-view: --video arg veya default video.mp4
        if args.video:
            video_path = Path(args.video)
        else:
            video_path = paths["video"]
        if not video_path.exists():
            print(f"✗ Single-view video not found: {video_path}")
            print(f"  Provide --video <path> or place at {paths['video']}")
            return 1
        print(f"  Single-view video: {video_path}")

    # 3) Build config with profile overrides
    cfg = default_config()
    _apply_profile(cfg, args.profile)
    print(f"\n  Profile applied:")
    print(f"    fps={cfg.preprocess.fps}, resize={cfg.preprocess.resize_long_edge}")
    print(f"    colmap_matching={cfg.preprocess.colmap_matching}")
    print(f"    mv_timestamps={cfg.preprocess.colmap_mv_timestamps}, "
          f"mv_dense_mvs={cfg.preprocess.colmap_mv_dense_mvs}")
    print(f"    metric3d={cfg.foundation.metric3d_model}, "
          f"cotracker_grid={cfg.foundation.cotracker_grid_size}")

    # 4) Run pipeline with training/export skipped (preprocess only)
    print(f"\n{'-' * 78}")
    print(f"  Running preprocessing pipeline (training/export skipped)...")
    print(f"{'-' * 78}\n")

    t0 = time.time()
    try:
        status = run_pipeline(
            video_path=str(video_path),
            scene_name=args.scene,
            cfg=cfg,
            skip_foundation=args.no_foundation,
            skip_training=True,   # Phase 1.2: SADECE preprocessing
            skip_export=True,
            progress_callback=None,  # CLI mode, console log yeterli
            force_preprocess=args.force,
        )
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0

    # 5) Summary
    print(f"\n{'=' * 78}")
    print(f"  ✓ Heavy Preprocessing Done — {elapsed:.0f} sec ({elapsed / 60:.1f} dk)")
    print(f"{'=' * 78}\n")
    for k, v in status.items():
        print(f"  {k:12s}: {v}")

    # Cache summary
    print(f"\n  Cache markers ({paths['base'] / '.cache_markers'}):")
    marker_dir = paths["base"] / ".cache_markers"
    if marker_dir.exists():
        for marker in sorted(marker_dir.glob("*.json")):
            print(f"    ✓ {marker.stem}")
    else:
        print(f"    (no markers — multi-view path may not write all)")

    # Phase 1.3: manifest dosyasi yaz (toplu cache rapor)
    try:
        from backend.preprocess.cache_utils import write_manifest, build_manifest
        manifest_path = write_manifest(paths["base"], cfg=cfg)
        manifest = build_manifest(paths["base"], cfg=cfg)
        print(f"\n  Manifest: {manifest_path}")
        for step, entry in manifest["steps"].items():
            if entry["cached"]:
                m = "✓" if entry.get("match", True) else "✗"
                cur = entry.get("current_hash", "?")
                cached = (entry["marker"] or {}).get("settings_hash", "?")
                print(f"    {m} {step:12s} cached={cached} current={cur}")
    except Exception as e:
        print(f"  ⚠ Manifest yazilirken hata: {e}")

    print(f"\n  Cache hazir. Training run'lari su cache'i kullanacak (force_preprocess=False default).")
    print(f"  Tum sahne icin yeniden preprocessing isteniyorsa:")
    print(f"    python scripts/heavy_preprocess.py --scene {args.scene} --profile {args.profile} --force\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
