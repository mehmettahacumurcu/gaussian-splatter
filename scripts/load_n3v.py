"""Neural 3D Video (N3V) dataset loader → bizim multi-view layout.

N3V kaynak format (Meta CVPR 2022):
    <src>/
        cam00.mp4
        cam01.mp4
        ...
        cam17.mp4
        poses_bounds.npy        # (N_cam, 17) — pose + intrinsics + near/far

poses_bounds.npy format (LLFF/N3V convention):
    Row per camera. Each row 17 floats:
        [0:15]  → 3x5 matrix flattened: [R|t|hwf]
                  Last column = (height, width, focal_length)
        [15:17] → near, far depth bounds

Bizim hedef layout:
    data/<scene>/
        videos/
            cam00.mp4
            cam01.mp4
            ...
        poses_bounds.npy        # kopya
        calibration.json        # parsed insanca okunabilir versiyon

Kullanım:
    python scripts/load_n3v.py \
        --src ~/Downloads/flame_steak \
        --dst data/flame_steak

Validation:
    Sonuç:
      ✓ 18 cam mp4 kopyalandı
      ✓ poses_bounds.npy parse edildi (18 rows)
      ✓ calibration.json yazıldı
      ✓ data/flame_steak hazır → backend pipeline çalışabilir
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_poses_bounds(path: Path) -> dict:
    """N3V poses_bounds.npy parse et → human-readable dict.

    Returns:
        {
            "n_cameras": int,
            "cameras": [
                {
                    "cam_id": 0,
                    "name": "cam00",
                    "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
                    "w2c": 4x4 list,  # world-to-camera
                    "c2w": 4x4 list,  # camera-to-world (LLFF original)
                    "width": int,
                    "height": int,
                    "near": float,
                    "far": float,
                }, ...
            ]
        }
    """
    pb = np.load(str(path))  # (N, 17)
    n = pb.shape[0]
    if pb.shape[1] != 17:
        raise ValueError(f"poses_bounds.npy shape ({pb.shape}) beklenen (N, 17) degil")

    cameras = []
    for i in range(n):
        row = pb[i]
        # Reshape [0:15] → 3x5 matrix
        mat3x5 = row[:15].reshape(3, 5)
        # First 4 columns = c2w, 5th = (h, w, f)
        c2w_llff_3x4 = mat3x5[:, :4].astype(np.float64)  # (3, 4) raw LLFF
        h, w, f = mat3x5[:, 4]
        near, far = row[15], row[16]

        # ─────────────────────────────────────────────────────────────────────
        # CAMERA CONVENTION FIX — LLFF → COLMAP/OpenCV (gsplat icin zorunlu).
        #
        # LLFF/N3V poses_bounds.npy raw 3x5 matrix'inde c2w'nin columns'i
        # [right, up, backward] OpenGL camera basisidir AMA world frame'i
        # NeRF/standart degil — world axes farkli labellanmis.
        #
        # Standart LLFF processing (Mildenhall LLFF, NeRF, 4D-GS, hepsi ayni):
        #
        #   poses = np.concatenate([poses[1:2,:], -poses[0:1,:], poses[2:,:]], axis=0)
        #
        # Bu bir ROW operation (axis=0 / row swap+negate). Etkisi: world frame
        # eksen labellarini permute eder (LLFF-world → NeRF-world, y-up).
        # Camera basis (cols) DEGISMEZ — hala [right, up, backward] OpenGL.
        #
        # Sonra OpenGL → OpenCV: col 1 (up→down) ve col 2 (back→forward) flip.
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Row permutation (world frame relabel) — LLFF → NeRF/OpenGL world
        c2w_opengl_3x4 = np.stack([
            c2w_llff_3x4[1, :],
            -c2w_llff_3x4[0, :],
            c2w_llff_3x4[2, :],
        ], axis=0)  # (3, 4)
        # Step 2: Column flip — OpenGL camera [right, up, back] → OpenCV [right, down, fwd]
        R_opencv = c2w_opengl_3x4[:, :3].copy()
        R_opencv[:, 1] *= -1
        R_opencv[:, 2] *= -1
        t = c2w_opengl_3x4[:, 3:4]

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_opencv
        c2w[:3, 3:4] = t

        # w2c = inverse(c2w)
        w2c = np.linalg.inv(c2w)

        # K matrix (focal length f shared, principal point at image center)
        K = np.array([
            [float(f), 0.0, float(w) / 2.0],
            [0.0, float(f), float(h) / 2.0],
            [0.0, 0.0, 1.0],
        ])

        cameras.append({
            "cam_id": i,
            "name": f"cam{i:02d}",
            "K": K.tolist(),
            "w2c": w2c.tolist(),
            "c2w": c2w.tolist(),
            "width": int(w),
            "height": int(h),
            "near": float(near),
            "far": float(far),
        })

    return {
        "format": "neural_3d_video_llff_to_opencv",
        "convention": "OpenCV/COLMAP (right-down-forward); converted from LLFF",
        "n_cameras": n,
        "cameras": cameras,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neural 3D Video dataset → bizim multi-view layout'a aktar",
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="N3V dataset klasörü (cam*.mp4 + poses_bounds.npy içerir)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Hedef sahne klasörü (örn data/flame_steak)")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="MP4'leri kopyala (default: True). False ise symlink.")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    print(f"\n{'=' * 70}")
    print(f"  Neural 3D Video → 4DGS Studio multi-view")
    print(f"{'=' * 70}")
    print(f"  Source:  {src}")
    print(f"  Target:  {dst}")
    print()

    # 1) Validate source
    if not src.exists() or not src.is_dir():
        print(f"✗ Kaynak klasör bulunamadı: {src}")
        sys.exit(1)

    cam_files = sorted(src.glob("cam*.mp4"))
    if not cam_files:
        print(f"✗ {src} altında cam*.mp4 yok")
        print(f"  Dizinde olanlar: {[p.name for p in src.iterdir()]}")
        sys.exit(1)
    print(f"  ✓ {len(cam_files)} kamera dosyası bulundu: "
          f"{cam_files[0].name} ... {cam_files[-1].name}")

    poses_src = src / "poses_bounds.npy"
    if not poses_src.exists():
        print(f"⚠ poses_bounds.npy yok — calibration JSON üretilemeyecek")
        print(f"  Pipeline yine de çalışır ama COLMAP rerun gerekir")
        poses_src = None
    else:
        print(f"  ✓ poses_bounds.npy bulundu ({poses_src.stat().st_size / 1024:.1f} KB)")

    # 2) Create dst layout
    videos_dir = dst / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ Hedef yapı oluşturuluyor: {dst}/")

    # 3) Copy mp4 files
    print(f"\n→ {len(cam_files)} mp4 kopyalanıyor...")
    total_size = 0
    for cam_file in cam_files:
        dst_file = videos_dir / cam_file.name
        if args.copy:
            if dst_file.exists() and dst_file.stat().st_size == cam_file.stat().st_size:
                print(f"  ⚠ {cam_file.name} zaten var (skip)")
            else:
                shutil.copy2(cam_file, dst_file)
                print(f"  ✓ {cam_file.name} → videos/{cam_file.name} "
                      f"({dst_file.stat().st_size / 1024 / 1024:.1f} MB)")
        total_size += dst_file.stat().st_size

    print(f"  Total videos boyutu: {total_size / 1024 / 1024:.1f} MB")

    # 4) Copy + parse poses_bounds.npy
    if poses_src is not None:
        poses_dst = dst / "poses_bounds.npy"
        shutil.copy2(poses_src, poses_dst)
        print(f"\n→ poses_bounds.npy kopyalandı")

        try:
            calib = parse_poses_bounds(poses_dst)
            calib_dst = dst / "calibration.json"
            calib_dst.write_text(json.dumps(calib, indent=2))
            print(f"  ✓ calibration.json üretildi ({len(calib['cameras'])} kamera)")
            # First cam intrinsics print
            cam0 = calib["cameras"][0]
            K = cam0["K"]
            print(f"  cam00 example: {cam0['width']}x{cam0['height']}, "
                  f"fx={K[0][0]:.1f}, near={cam0['near']:.3f}, far={cam0['far']:.3f}")
        except Exception as e:
            print(f"  ⚠ poses_bounds parse hatası (non-fatal): {e}")

    # 5) Validation
    print(f"\n{'=' * 70}")
    print(f"  ✓ TAMAM")
    print(f"{'=' * 70}")
    print(f"  Sahne hazır: {dst}")
    print(f"  Kameralar: {len(cam_files)} ({cam_files[0].stem} - {cam_files[-1].stem})")
    print(f"  Multi-view detection bekleniyor: data/{dst.name}/videos/ var → multi-view")
    print(f"\n  Sıradaki adım:")
    print(f"    Backend pipeline submit (multi-view auto-detect):")
    print(f"      curl -X POST http://127.0.0.1:8000/process \\")
    print(f"        -F 'video=@{dst}/videos/cam00.mp4' \\")
    print(f"        -F 'scene={dst.name}' \\")
    print(f"        -F 'static_max=true'")
    print(f"    veya frontend'den scene='{dst.name}' ile yükle")
    print()


if __name__ == "__main__":
    main()
