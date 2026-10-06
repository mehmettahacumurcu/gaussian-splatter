"""Minimal COLMAP sparse-model reader (binary and text).

Only what the pair generator needs: cameras, registered images (pose + name)
and the 3D points used to initialise the Gaussians.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_NAME_TO_COUNT = {name: count for name, count in CAMERA_MODELS.values()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self) -> tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy); distortion terms are ignored."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    def has_distortion(self) -> bool:
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return False
        extra = self.params[3:] if self.model.startswith("SIMPLE") or self.model in ("RADIAL", "FOV") else self.params[4:]
        return bool(np.any(np.abs(extra) > 1e-9))


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # w, x, y, z (world -> camera)
    tvec: np.ndarray
    camera_id: int
    name: str

    def world_to_camera(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = qvec_to_rotmat(self.qvec)
        m[:3, 3] = self.tvec
        return m


@dataclass
class SparseModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points_xyz: np.ndarray  # (N, 3) float64
    points_rgb: np.ndarray  # (N, 3) uint8


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read(f, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def _read_cameras_bin(path: Path) -> dict[int, Camera]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = _read(f, "iiQQ")
            name, count = CAMERA_MODELS[model_id]
            params = np.array(_read(f, "d" * count))
            cams[cam_id] = Camera(cam_id, name, int(width), int(height), params)
    return cams


def _read_images_bin(path: Path) -> dict[int, Image]:
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            props = _read(f, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                if not c:
                    raise ValueError("Truncated image name")
                name += c
            (n2d,) = _read(f, "Q")
            f.seek(24 * n2d, 1)  # x, y (double) + point3D id (int64)
            imgs[image_id] = Image(image_id, qvec, tvec, camera_id, name.decode("utf-8"))
    return imgs


def _read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        xyz = np.empty((n, 3), np.float64)
        rgb = np.empty((n, 3), np.uint8)
        for i in range(n):
            props = _read(f, "QdddBBBd")
            xyz[i] = props[1:4]
            rgb[i] = props[4:7]
            (track_len,) = _read(f, "Q")
            f.seek(8 * track_len, 1)
    return xyz, rgb


def _read_cameras_txt(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = np.array([float(v) for v in parts[4 : 4 + _NAME_TO_COUNT[model]]])
        cams[cam_id] = Camera(cam_id, model, w, h, params)
    return cams


def _read_images_txt(path: Path) -> dict[int, Image]:
    imgs = {}
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    # Pose lines alternate with 2D-point lines (which may be empty).
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.array([float(v) for v in parts[1:5]])
        tvec = np.array([float(v) for v in parts[5:8]])
        imgs[image_id] = Image(image_id, qvec, tvec, int(parts[8]), " ".join(parts[9:]))
    return imgs


def _read_points_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append([float(v) for v in parts[1:4]])
        rgb.append([int(v) for v in parts[4:7]])
    return np.array(xyz, np.float64).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def read_sparse_model(sparse_dir: Path) -> SparseModel:
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.bin").exists():
        cams = _read_cameras_bin(sparse_dir / "cameras.bin")
        imgs = _read_images_bin(sparse_dir / "images.bin")
        xyz, rgb = _read_points_bin(sparse_dir / "points3D.bin")
    elif (sparse_dir / "cameras.txt").exists():
        cams = _read_cameras_txt(sparse_dir / "cameras.txt")
        imgs = _read_images_txt(sparse_dir / "images.txt")
        xyz, rgb = _read_points_txt(sparse_dir / "points3D.txt")
    else:
        raise FileNotFoundError(f"No COLMAP model in {sparse_dir}")
    return SparseModel(cams, imgs, xyz, rgb)
