"""Faz 2c — COLMAP text çıktısını parse et: cameras.txt, images.txt, points3D.txt."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) → 3x3 rotasyon matrisi."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _find_sparse_dir(colmap_dir: Path) -> Path:
    """
    cameras.txt'yi içeren sparse sub-model'i bul.

    COLMAP mapper birden fazla sub-model üretebilir (sparse/0, sparse/1, ...).
    run_colmap en büyüğünü seçip SADECE ona model_converter (BIN→TXT) uyguluyor.
    Yani .txt dosyaları her zaman 'sparse/0'da değil — en büyük parçada.

    Strateji:
      1) sparse/* alt klasörlerinde cameras.txt ara, bulduklarını dosya boyutuna göre
         sırala (run_colmap ile tutarlı), en büyüğü döndür.
      2) Düz yerleşim: colmap_dir/cameras.txt.
      3) Yoksa hata.
    """
    sparse_root = colmap_dir / "sparse"
    if sparse_root.exists() and sparse_root.is_dir():
        candidates = [
            d for d in sparse_root.iterdir()
            if d.is_dir() and (d / "cameras.txt").exists()
        ]
        if candidates:
            def _size(d: Path) -> int:
                total = 0
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    f = d / name
                    if f.exists():
                        total += f.stat().st_size
                return total
            candidates.sort(key=_size, reverse=True)
            return candidates[0]

    if (colmap_dir / "cameras.txt").exists():
        return colmap_dir

    raise FileNotFoundError(
        f"cameras.txt bulunamadı: {colmap_dir}\n"
        f"  Denenen: sparse/*/cameras.txt ve {colmap_dir}/cameras.txt\n"
        f"  (run_colmap çalıştı mı? model_converter BIN→TXT adımı başarılı mı?)"
    )


def parse_cameras(colmap_dir: str | Path) -> Dict[str, Dict]:
    """
    COLMAP sparse/0/ klasöründen kamera intrinsic + extrinsic'leri oku.

    Returns:
        { image_name: { "K": (3,3), "R": (3,3), "t": (3,), "w2c": (4,4),
                        "width": int, "height": int } }
    """
    sparse = _find_sparse_dir(Path(colmap_dir))

    # cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    cameras: Dict[int, Dict] = {}
    with open(sparse / "cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                f_, cx, cy = params[:3]
                fx = fy = f_
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            else:
                raise NotImplementedError(f"Camera model desteklenmiyor: {model}")

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            cameras[cam_id] = {"K": K, "width": width, "height": height, "model": model}

    # images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #             POINTS2D[] (ikinci satırda)
    images: Dict[str, Dict] = {}
    with open(sparse / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]

        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)

        # World-to-camera 4x4
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        cam = cameras[cam_id]
        images[name] = {
            "K": cam["K"], "R": R, "t": t, "w2c": w2c,
            "width": cam["width"], "height": cam["height"],
        }

    if not images:
        raise RuntimeError("images.txt parse edildi ama hiç kamera çıkmadı")
    print(f"✓ {len(images)} kamera pozu yüklendi")
    return images


def load_points3d(colmap_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse 3D nokta bulutunu yükle.

    Returns:
        xyz: (N, 3) float32
        rgb: (N, 3) uint8
    """
    sparse = _find_sparse_dir(Path(colmap_dir))
    pts_file = sparse / "points3D.txt"
    if not pts_file.exists():
        raise FileNotFoundError(f"points3D.txt bulunamadı: {pts_file}")

    xyz, rgb = [], []
    with open(pts_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]),   int(parts[5]),   int(parts[6])])

    if not xyz:
        raise RuntimeError("points3D.txt boş — COLMAP rekonstrüksiyonu başarısız")
    xyz = np.array(xyz, dtype=np.float32)
    rgb = np.array(rgb, dtype=np.uint8)
    print(f"✓ {len(xyz)} sparse 3D nokta yüklendi")
    return xyz, rgb


def scene_extent(xyz: np.ndarray) -> float:
    """Sahnenin radyal kapsamı — density control için kullanılır."""
    centroid = xyz.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(xyz - centroid, axis=1).max())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="COLMAP çıktısını parse et")
    p.add_argument("colmap_dir", type=str)
    args = p.parse_args()

    cams = parse_cameras(args.colmap_dir)
    xyz, rgb = load_points3d(args.colmap_dir)
    first = next(iter(cams.values()))
    print(f"\nÖrnek kamera intrinsics:\n{first['K']}")
    print(f"Sahne kapsamı (radyal): {scene_extent(xyz):.3f}")
