"""Multi-view preprocessing utilities (v5.0).

Bizim mevcut single-view pipeline'i ile uyumlu shekilde multi-camera
data'yi handle eden helper'lar. Single-view path'leri DEGISTIRMIYOR.

Anahtar fonksiyonlar:
  - extract_frames_multiview(): paralel multi-cam frame extraction
  - parse_n3v_calibration(): N3V calibration.json → bizim cams Dict format
  - poses_to_colmap_compat(): N3V poses → COLMAP-compatible (parse_cameras output)

Pipeline orchestration (pipeline.py'de):
  1. is_multiview_scene(scene) ile detect
  2. Yes → extract_frames_multiview, parse_n3v_calibration, COLMAP atla
  3. No → eski single-view path

Note: Multi-view'da COLMAP genelde gerek değil — N3V poses_bounds.npy zaten
calibrated. Eğer custom multi-cam (kullanıcı kendi rig'i) ise COLMAP
multi-camera registration calistirilir (Sprint 2.2 ileri).
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# Multi-camera frame extraction
# =============================================================================

def extract_frames_multiview(
    videos_dir: Path,
    output_dir: Path,
    fps: int = 10,
    resize_long_edge: int | None = 960,
    overwrite: bool = False,
) -> Dict[str, List[Path]]:
    """Multi-camera batch frame extraction.

    Her cam.mp4 için ayrı klasöre frame çıkar. Cache check per-camera.

    Args:
        videos_dir: cam*.mp4 dosyaları içeren klasör
        output_dir: frames_multiview/ → cam00/, cam01/, ...
        fps: frame extraction rate
        resize_long_edge: video uzun kenar (None = orijinal)
        overwrite: True → mevcut frame'leri sil, yeniden çıkar

    Returns:
        {"cam00": [Path...], "cam01": [Path...], ...}
    """
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_files = sorted(videos_dir.glob("cam*.mp4"))
    if not cam_files:
        raise FileNotFoundError(f"{videos_dir} altında cam*.mp4 yok")

    print(f"[multiview] {len(cam_files)} kamera frame extraction başlıyor "
          f"(fps={fps}, edge={resize_long_edge})")

    results: Dict[str, List[Path]] = {}
    for cam_file in cam_files:
        cam_name = cam_file.stem  # cam00, cam01, ...
        cam_out = output_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        existing = sorted(cam_out.glob("frame_*.png"))
        if existing and not overwrite:
            print(f"  ⚠ {cam_name}: {len(existing)} mevcut frame, skip "
                  f"(overwrite=True ile yeniden çıkar)")
            results[cam_name] = existing
            continue

        # ffmpeg invocation
        vf_parts = [f"fps={fps}"]
        if resize_long_edge:
            vf_parts.append(
                f"scale='if(gt(iw,ih),{resize_long_edge},-2)':'if(gt(iw,ih),-2,{resize_long_edge})'"
            )
        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(cam_file),
            "-vf", vf,
            "-q:v", "1",
            str(cam_out / "frame_%04d.png"),
        ]
        print(f"  → {cam_name}: ffmpeg starting...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {cam_name} ffmpeg fail: {e}")
            continue

        extracted = sorted(cam_out.glob("frame_*.png"))
        print(f"  ✓ {cam_name}: {len(extracted)} frame extracted")
        results[cam_name] = extracted

    if not results:
        raise RuntimeError("Hiçbir kameradan frame extract edilemedi")

    # Sanity check: tüm cam'lar aynı sayıda frame'e sahip mi?
    counts = {name: len(frames) for name, frames in results.items()}
    if len(set(counts.values())) > 1:
        print(f"  ⚠ Frame counts farklı kamera arasında: {counts}")
        print(f"    Bu sync problemi olduğunu gösterir, train zorlanabilir")
    else:
        n = list(counts.values())[0]
        print(f"  ✓ Tüm {len(results)} kamera {n} frame ile sync")

    return results


# =============================================================================
# Calibration parsing
# =============================================================================

def parse_n3v_calibration(calibration_path: Path) -> Dict[str, Dict]:
    """N3V calibration.json → bizim parse_cameras() format'ı.

    parse_cameras() döndürür:
        { image_name: {"K": (3,3) np, "R": (3,3) np, "t": (3,) np,
                       "w2c": (4,4) np, "width": int, "height": int} }

    Multi-view'de "image_name" = "cam00", "cam01", ... (per-camera, tek pose)

    Args:
        calibration_path: load_n3v.py'nin ürettiği calibration.json

    Returns:
        Dict[cam_name, {K, R, t, w2c, width, height}]
    """
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cams_out: Dict[str, Dict] = {}
    for cam in data["cameras"]:
        K = np.array(cam["K"], dtype=np.float64)
        w2c = np.array(cam["w2c"], dtype=np.float64)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cams_out[cam["name"]] = {
            "K": K,
            "R": R,
            "t": t,
            "w2c": w2c,
            "width": cam["width"],
            "height": cam["height"],
            "near": cam.get("near"),
            "far": cam.get("far"),
        }
    return cams_out


def estimate_scene_extent_from_n3v(calibration: Dict[str, Dict]) -> Tuple[np.ndarray, float]:
    """N3V scene center + extent — near/far bounds'tan look-point projection.

    LLFF/N3V'de scene KAMERALARIN MERKEZINDE DEGIL — kameralar 'in onunde
    near/far depth aralginda. Eski kod camera centroid'ini scene merkezi
    saniyordu → init points 30 unit yanlis yerde → ogrenme yok.

    Yeni yaklasim:
      1) Her kameranin near/far ortalamasi mid_depth
      2) look_point[i] = pos[i] + forward[i] * mid_depth[i]
         (kameranin forward direction'da scene'e bakti tahmini nokta)
      3) scene_center = mean(look_points)
      4) scene_extent = max_dist(look_points - scene_center) * 1.5 + buffer
         (look point'ler arasindaki yayilim = scene size estimation)

    Returns:
        scene_center: (3,) world coord — gercek scene merkezi
        scene_extent: float — scene yariçapi (init bbox + prune_max_scale icin)
    """
    centers, fwds, nears, fars = [], [], [], []
    for cam in calibration.values():
        c2w = np.linalg.inv(cam["w2c"])
        pos = c2w[:3, 3]
        fwd = c2w[:3, 2]
        near = float(cam.get("near") or 1.0)
        far = float(cam.get("far") or 50.0)
        centers.append(pos)
        fwds.append(fwd)
        nears.append(near)
        fars.append(far)
    centers = np.array(centers)
    fwds = np.array(fwds)
    nears = np.array(nears)
    fars = np.array(fars)

    # NEAR depth'e gore look-point hesabi (foreground-focused).
    # Eski yaklasim: mid = (near+far)/2 — N3V'de far=56 (background, sky)
    # foreground'i overshoot eder, init bbox sahne disinda kalir.
    # Yeni: focal_depth = near * 1.3 (yakin foreground bolgesini hedefle)
    focal_depth = nears * 1.3
    look_points = centers + fwds * focal_depth[:, None]  # (N, 3)
    scene_center = look_points.mean(axis=0)

    # Scene extent: TIGHT — look point spread + small buffer.
    # max(spread, near_min * 0.5) — minimum 50% near depth wide
    look_spread = float(np.linalg.norm(look_points - scene_center, axis=1).max())
    near_min = float(nears.min())
    scene_extent = float(max(look_spread + near_min * 0.3, near_min * 0.5))
    cam_to_look = float(np.linalg.norm(centers - look_points, axis=1).mean())

    print(f"  [scene] camera centroid: {centers.mean(0)}")
    print(f"  [scene] look-point center: {scene_center} (near*1.3 projection)")
    print(f"  [scene] near range: [{nears.min():.2f}, {nears.max():.2f}]")
    print(f"  [scene] far  range: [{fars.min():.2f}, {fars.max():.2f}]  (used: near only)")
    print(f"  [scene] mean cam→focal depth: {cam_to_look:.2f}")
    print(f"  [scene] look-point spread: {look_spread:.2f}")
    print(f"  [scene] scene_extent (init bbox half-size): {scene_extent:.2f}")
    return scene_center, scene_extent


# =============================================================================
# Sparse 3D point initialization (N3V'de COLMAP yok, dummy init gerekir)
# =============================================================================

def init_random_points_in_bbox(
    centroid: np.ndarray,
    extent: float,
    n_points: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """COLMAP sparse cloud yokken random 3D points üret.

    N3V (poses_bounds.npy verili) durumunda COLMAP atlanır, sparse cloud
    yoktur. Trainer için init points gerek → bbox içinde uniform random.

    Args:
        centroid: (3,) sahne merkezi
        extent: float — yarıçap
        n_points: kaç random point

    Returns:
        xyz: (N, 3) float32
        rgb: (N, 3) uint8 — gri (128, 128, 128)
    """
    rng = np.random.default_rng(42)
    # Uniform in bbox.
    # NOT: centroid float64 ise (calibration w2c'den geliyor) numpy upcast'leyip
    # xyz'yi float64 yapar → torch.from_numpy(...) float64 → gsplat patlar.
    # astype(float32) ile garantiye al.
    #
    # NOT: estimate_scene_extent_from_n3v artik look-point projection ile
    # GERCEK scene merkezini ve extent'i hesapliyor (camera centroid degil).
    # Bu yuzden bbox = extent (full half-size) olabilir, scene'i tam kapsar.
    bbox_half = np.float32(extent)
    centroid_f32 = np.asarray(centroid, dtype=np.float32)
    xyz = ((rng.random((n_points, 3), dtype=np.float32) * 2 - 1)
           * bbox_half + centroid_f32).astype(np.float32)
    rgb = np.full((n_points, 3), 128, dtype=np.uint8)
    return xyz, rgb


# =============================================================================
# Multi-view scene info (pipeline'a özet)
# =============================================================================

def multiview_scene_info(scene_paths: dict) -> Dict:
    """Multi-view sahnenin durumunu özetle (pipeline log için).

    Returns:
        {
            "n_cameras": int,
            "cameras": list[str],
            "frames_per_camera": dict,
            "has_poses": bool,
            "has_calibration": bool,
        }
    """
    info = {
        "n_cameras": 0,
        "cameras": [],
        "frames_per_camera": {},
        "has_poses": False,
        "has_calibration": False,
    }

    videos_mv = scene_paths.get("videos_mv")
    if videos_mv and videos_mv.exists():
        cams = sorted([p.stem for p in videos_mv.glob("cam*.mp4")])
        info["n_cameras"] = len(cams)
        info["cameras"] = cams

    frames_mv = scene_paths.get("frames_mv")
    if frames_mv and frames_mv.exists():
        for cam_dir in sorted(frames_mv.iterdir()):
            if cam_dir.is_dir():
                count = len(list(cam_dir.glob("frame_*.png")))
                info["frames_per_camera"][cam_dir.name] = count

    poses_path = scene_paths.get("poses_bounds")
    if poses_path and poses_path.exists():
        info["has_poses"] = True

    calib_path = scene_paths.get("calibration")
    if calib_path and calib_path.exists():
        info["has_calibration"] = True

    return info


# =============================================================================
# CLI test
# =============================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-view preprocessing test")
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--extract", action="store_true", help="Frame extraction çalıştır")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--edge", type=int, default=960)
    args = p.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    videos = scene_dir / "videos"

    if args.extract:
        frames_dir = scene_dir / "frames_multiview"
        results = extract_frames_multiview(
            videos, frames_dir,
            fps=args.fps,
            resize_long_edge=args.edge,
        )
        print(f"\n{len(results)} kamera frame extracted")

    calib = scene_dir / "calibration.json"
    if calib.exists():
        cams = parse_n3v_calibration(calib)
        print(f"\nCalibration parsed: {len(cams)} cameras")
        first = list(cams.values())[0]
        print(f"  cam00: {first['width']}x{first['height']}, K[0][0]={first['K'][0][0]:.1f}")
        centroid, extent = estimate_scene_extent_from_n3v(cams)
        print(f"  Scene centroid: {centroid}")
        print(f"  Scene extent: {extent:.2f}")
