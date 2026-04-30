"""Multi-view standalone test runner (Sprint 2.3 — pipeline integration öncesi).

Mevcut backend/pipeline.py single-view için yazılmış. Multi-view pipeline'a
tam entegrasyon (Sprint 3 — trainer dataloader) tamamlanana kadar bu script
multi-view sahneyi single-view'e "düşürerek" trainer'ı çalıştırır:

    Strateji A (varsayılan):
        Sadece cam00 (test camera) hariç, kalan kameralardan birini SEÇ
        (örn cam05) — pipeline'a single-view gibi besle. Bu MVP.

    Strateji B (gerçek multi-view, Sprint 3):
        Tüm kameraları training data olarak kullan, dataloader (cam, t)
        random sampling. Henüz implement edilmedi.

Kullanım:
    # Önce N3V import:
    python scripts/load_n3v.py --src ~/Downloads/flame_steak --dst data/flame_steak

    # Multi-view'i bizim format'da hazırla:
    python scripts/run_multiview.py data/flame_steak --extract

    # Sonra single-view runner (mevcut pipeline) ile cam05 ile train et:
    python scripts/run_multiview.py data/flame_steak --train --select-cam cam05
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.config import scene_paths, is_multiview_scene, list_multiview_cameras  # noqa
from backend.preprocess.multiview import (  # noqa
    extract_frames_multiview,
    parse_n3v_calibration,
    estimate_scene_extent_from_n3v,
    multiview_scene_info,
)


def cmd_info(scene_dir: Path):
    """Sahne durumunu göster."""
    scene_name = scene_dir.name
    paths = scene_paths(scene_name)
    print(f"\n{'=' * 70}")
    print(f"  Multi-view Scene Info: {scene_name}")
    print(f"{'=' * 70}")

    is_mv = is_multiview_scene(scene_name)
    print(f"  Multi-view detected: {is_mv}")

    cams = list_multiview_cameras(scene_name)
    print(f"  Cameras: {len(cams)} → {cams[:5]}{'...' if len(cams) > 5 else ''}")

    info = multiview_scene_info(paths)
    print(f"  Has poses_bounds.npy: {info['has_poses']}")
    print(f"  Has calibration.json: {info['has_calibration']}")
    if info["frames_per_camera"]:
        counts = list(info["frames_per_camera"].values())
        print(f"  Frames per camera: {counts[0]} (range {min(counts)}-{max(counts)})")

    if info["has_calibration"]:
        cams = parse_n3v_calibration(paths["calibration"])
        centroid, extent = estimate_scene_extent_from_n3v(cams)
        print(f"  Scene centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"  Scene extent:   {extent:.2f}")


def cmd_extract(scene_dir: Path, fps: int, edge: int):
    """Multi-camera frame extraction."""
    scene_name = scene_dir.name
    paths = scene_paths(scene_name)

    if not is_multiview_scene(scene_name):
        print(f"✗ {scene_name} multi-view değil. videos/cam*.mp4 bekleniyor.")
        sys.exit(1)

    print(f"\n→ Multi-view frame extraction: {scene_name}")
    print(f"  fps={fps}, max_long_edge={edge}")
    results = extract_frames_multiview(
        paths["videos_mv"],
        paths["frames_mv"],
        fps=fps,
        resize_long_edge=edge,
    )
    print(f"\n✓ {len(results)} kameradan frame extract edildi")


def cmd_setup_single_view_proxy(scene_dir: Path, select_cam: str):
    """MVP: Multi-view sahneden tek bir kamerayı seç, single-view layout'a kopyala.

    Bu mevcut pipeline.py'i değiştirmeden multi-view sahneyi train edebilmek için
    pratik workaround. Tek kamera × dynamic = monocular 4DGS.

    Kopyalama:
        videos/<select_cam>.mp4  →  video.mp4
        frames_multiview/<select_cam>/  →  frames/

    Sonra mevcut /process endpoint'i çalışır.
    """
    scene_name = scene_dir.name
    paths = scene_paths(scene_name)

    src_video = paths["videos_mv"] / f"{select_cam}.mp4"
    if not src_video.exists():
        print(f"✗ {src_video} bulunamadı")
        sys.exit(1)

    src_frames = paths["frames_mv"] / select_cam
    if not src_frames.exists() or not list(src_frames.glob("frame_*.png")):
        print(f"⚠ {src_frames} altında frame yok — önce 'extract' çalıştır")

    # Single-view layout
    dst_video = paths["video"]
    if dst_video.exists():
        print(f"⚠ {dst_video} zaten var — siliniyor")
        dst_video.unlink()
    print(f"→ {select_cam}.mp4 → video.mp4 (symlink/copy)")
    try:
        # Önce symlink dene, fail ise copy
        try:
            dst_video.symlink_to(src_video.resolve())
            print(f"  ✓ symlink: {dst_video} → {src_video}")
        except (OSError, NotImplementedError):
            shutil.copy2(src_video, dst_video)
            print(f"  ✓ copy: {src_video} → {dst_video}")
    except Exception as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    # Frames
    dst_frames = paths["frames"]
    if dst_frames.exists():
        print(f"⚠ {dst_frames} zaten var — temizleniyor")
        shutil.rmtree(dst_frames)
    if src_frames.exists() and list(src_frames.glob("frame_*.png")):
        try:
            dst_frames.symlink_to(src_frames.resolve(), target_is_directory=True)
            print(f"  ✓ frames symlink: {dst_frames} → {src_frames}")
        except (OSError, NotImplementedError):
            shutil.copytree(src_frames, dst_frames)
            print(f"  ✓ frames copy")
    else:
        print(f"  (frames yok — pipeline kendisi ffmpeg ile çıkarır)")

    print(f"\n✓ {scene_name} single-view-proxy hazır ({select_cam})")
    print(f"  Sıradaki: pipeline ile train et (single-view path):")
    print(f"    curl -X POST http://127.0.0.1:8000/process \\")
    print(f"      -F 'video=@{dst_video}' \\")
    print(f"      -F 'scene={scene_name}' \\")
    print(f"      -F 'static_max=true' \\")
    print(f"      -F 'iters=15000'")


def main():
    parser = argparse.ArgumentParser(description="Multi-view pipeline runner")
    parser.add_argument("scene_dir", type=Path, help="data/<scene> klasörü")
    parser.add_argument("--info", action="store_true", help="Sahne durumunu göster")
    parser.add_argument("--extract", action="store_true",
                        help="Multi-cam frame extraction")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--edge", type=int, default=960)
    parser.add_argument("--select-cam", type=str, default=None,
                        help="MVP: tek kamera seç (cam05 vs), single-view-proxy oluştur")
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.exists():
        print(f"✗ {scene_dir} yok")
        sys.exit(1)

    # Default: info
    if not (args.info or args.extract or args.select_cam):
        args.info = True

    if args.info:
        cmd_info(scene_dir)
    if args.extract:
        cmd_extract(scene_dir, args.fps, args.edge)
    if args.select_cam:
        cmd_setup_single_view_proxy(scene_dir, args.select_cam)


if __name__ == "__main__":
    main()
