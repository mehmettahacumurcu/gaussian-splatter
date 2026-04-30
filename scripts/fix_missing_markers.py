"""Patch script — Round 1 sırasında UnboundLocalError yüzünden yazılmayan
depth_mv ve masks_mv marker'larini manuel yaz. Pipeline crash'ten sonra
bir kerelik kullanim. Standard profile cfg'ye gore hash hesaplar.

Usage:
    python scripts/fix_missing_markers.py --scene flame_steak
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import default_config, scene_paths  # noqa
from backend.preprocess.cache_utils import (  # noqa
    write_cache_marker, read_cache_marker, compute_settings_hash,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--profile", default="standard",
                    choices=["standard", "premium"])
    args = ap.parse_args()

    paths = scene_paths(args.scene)
    if not paths["base"].exists():
        print(f"✗ Scene not found: {paths['base']}")
        sys.exit(1)

    # Profile'a gore cfg kur (integration_test_phase_1_2.py ile ayni override'lar)
    cfg = default_config()
    if args.profile == "standard":
        cfg.preprocess.fps = 10
        cfg.preprocess.resize_long_edge = 960
        cfg.preprocess.colmap_matching = "sequential"
        cfg.preprocess.colmap_mv_timestamps = 5
        cfg.preprocess.colmap_mv_dense_mvs = False
        cfg.foundation.metric3d_model = "metric3d_vit_small"
        cfg.foundation.cotracker_grid_size = 30
    elif args.profile == "premium":
        cfg.preprocess.fps = 30
        cfg.preprocess.resize_long_edge = 1280
        cfg.preprocess.colmap_matching = "exhaustive"
        cfg.preprocess.colmap_mv_timestamps = 25
        cfg.preprocess.colmap_mv_dense_mvs = True
        cfg.foundation.metric3d_model = "metric3d_vit_large"
        cfg.foundation.cotracker_grid_size = 75

    # depth_mv marker
    depth_mv_dir = paths.get("depth_mv")
    if depth_mv_dir and depth_mv_dir.exists():
        n_total = len(list(depth_mv_dir.glob("cam*/*_depth.npy")))
        cam_dirs = sorted(d for d in depth_mv_dir.iterdir()
                          if d.is_dir() and d.name.startswith("cam"))
        n_cams = len(cam_dirs)
        n_per_cam = (
            len(list(cam_dirs[0].glob("*_depth.npy"))) if cam_dirs else 0
        )
        if n_total > 0 and read_cache_marker(paths["base"], "depth_mv") is None:
            write_cache_marker(paths["base"], "depth_mv", {
                "model": cfg.foundation.metric3d_model,
                "n_cams": n_cams,
                "n_frames_per_cam": n_per_cam,
                "n_depth_total": n_total,
                "patched_post_hoc": True,
            }, cfg=cfg)
            h = compute_settings_hash(cfg, "depth_mv")
            print(f"✓ depth_mv marker yazildi (n={n_total}, hash={h})")
        else:
            print(f"  depth_mv: marker zaten var veya dosya yok")
    else:
        print(f"  depth_mv klasoru yok: {depth_mv_dir}")

    # masks_mv marker
    masks_mv_dir = paths.get("masks_mv")
    if masks_mv_dir and masks_mv_dir.exists():
        n_total = len(list(masks_mv_dir.glob("cam*/mask_*.png")))
        cam_dirs = sorted(d for d in masks_mv_dir.iterdir()
                          if d.is_dir() and d.name.startswith("cam"))
        n_cams = len(cam_dirs)
        n_per_cam = (
            len(list(cam_dirs[0].glob("mask_*.png"))) if cam_dirs else 0
        )
        if n_total > 0 and read_cache_marker(paths["base"], "masks_mv") is None:
            write_cache_marker(paths["base"], "masks_mv", {
                "n_cams": n_cams,
                "n_frames_per_cam": n_per_cam,
                "n_masks_total": n_total,
                "patched_post_hoc": True,
            }, cfg=cfg)
            h = compute_settings_hash(cfg, "masks_mv")
            print(f"✓ masks_mv marker yazildi (n={n_total}, hash={h})")
        else:
            print(f"  masks_mv: marker zaten var veya dosya yok")
    else:
        print(f"  masks_mv klasoru yok: {masks_mv_dir}")

    # frames_mv marker (extract_frames_multiview marker yazimi pipeline icinde
    # crash'ten once tamamlandi muhtemelen, ama yine de kontrol)
    frames_mv_dir = paths.get("frames_mv")
    if frames_mv_dir and frames_mv_dir.exists():
        if read_cache_marker(paths["base"], "frames_mv") is None:
            cam_dirs = sorted(d for d in frames_mv_dir.iterdir()
                              if d.is_dir() and d.name.startswith("cam"))
            n_per = len(list(cam_dirs[0].glob("frame_*.png"))) if cam_dirs else 0
            write_cache_marker(paths["base"], "frames_mv", {
                "n_cams": len(cam_dirs),
                "n_frames_per_cam": n_per,
                "fps": cfg.preprocess.fps,
                "resize_long_edge": cfg.preprocess.resize_long_edge,
                "patched_post_hoc": True,
            }, cfg=cfg)
            h = compute_settings_hash(cfg, "frames_mv")
            print(f"✓ frames_mv marker yazildi (n_cams={len(cam_dirs)}, hash={h})")
        else:
            print(f"  frames_mv: marker zaten var")

    print("\nManifest yaziliyor...")
    from backend.preprocess.cache_utils import write_manifest
    p = write_manifest(paths["base"], cfg=cfg)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
