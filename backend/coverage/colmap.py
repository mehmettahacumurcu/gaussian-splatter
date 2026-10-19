"""Camera poses and pinhole intrinsics from a COLMAP sparse model (binary or text).

Only cameras and registered images are read; points3D and the 2D observations
are skipped. Distortion is ignored: coverage only needs where each camera looked.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_PARAM_COUNT = {name: count for name, count in CAMERA_MODELS.values()}
_SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


class ColmapError(ValueError):
    """The folder holds no readable COLMAP model."""


@dataclass
class CameraSet:
    names: list[str]       # image names as stored in the model, sorted
    w2c: np.ndarray        # (N, 4, 4) float32 world -> camera (OpenCV: x right, y down, z forward)
    K: np.ndarray          # (N, 3, 3) float32 full-resolution intrinsics
    sizes: np.ndarray      # (N, 2) int (width, height)

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, idx) -> "CameraSet":
        idx = list(idx)
        return CameraSet([self.names[i] for i in idx], self.w2c[idx], self.K[idx], self.sizes[idx])


def has_model(folder: Path) -> bool:
    return any((folder / f"cameras.{ext}").is_file() and (folder / f"images.{ext}").is_file()
               for ext in ("bin", "txt"))


def qvec_to_rotmat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _intrinsics(model: str, params) -> np.ndarray:
    if model not in _PARAM_COUNT:
        raise ColmapError(f"Bilinmeyen kamera modeli: {model}")
    if model in _SINGLE_FOCAL:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _read(f, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ColmapError("COLMAP dosyası yarım kalmış")
    return struct.unpack("<" + fmt, data)


def _cameras_bin(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = _read(f, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ColmapError(f"Bilinmeyen kamera model id: {model_id}")
            name, count = CAMERA_MODELS[model_id]
            cams[cam_id] = (name, int(width), int(height), list(_read(f, "d" * count)))
    return cams


def _images_bin(path: Path) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    out = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            props = _read(f, "idddddddi")
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                if not c:
                    raise ColmapError("Görüntü adı yarım kalmış")
                name += c
            (n2d,) = _read(f, "Q")
            f.seek(24 * n2d, 1)  # x, y (double) + point3D id (int64)
            out.append((name.decode("utf-8"), np.array(props[1:5]), np.array(props[5:8]), props[8]))
    return out


def _cameras_txt(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    cams = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        model = p[1]
        if model not in _PARAM_COUNT:
            raise ColmapError(f"Bilinmeyen kamera modeli: {model}")
        cams[int(p[0])] = (model, int(p[2]), int(p[3]), [float(v) for v in p[4:4 + _PARAM_COUNT[model]]])
    return cams


def _images_txt(path: Path) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    out = []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    # Pose lines alternate with 2D-point lines (which may be empty).
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        out.append((" ".join(p[9:]), np.array([float(v) for v in p[1:5]]),
                    np.array([float(v) for v in p[5:8]]), int(p[8])))
    return out


def read_cameras(folder: str | Path) -> CameraSet:
    folder = Path(folder)
    if (folder / "cameras.bin").is_file():
        cams, images = _cameras_bin(folder / "cameras.bin"), _images_bin(folder / "images.bin")
    elif (folder / "cameras.txt").is_file():
        cams, images = _cameras_txt(folder / "cameras.txt"), _images_txt(folder / "images.txt")
    else:
        raise ColmapError(f"COLMAP modeli bulunamadı: {folder}")
    if not images:
        raise ColmapError(f"COLMAP modelinde kayıtlı görüntü yok: {folder}")
    images.sort(key=lambda im: im[0])
    w2c, K, sizes = [], [], []
    for _, q, t, cam_id in images:
        if cam_id not in cams:
            raise ColmapError(f"Görüntü bilinmeyen kameraya bağlı: {cam_id}")
        model, w, h, params = cams[cam_id]
        m = np.eye(4)
        m[:3, :3] = qvec_to_rotmat(q)
        m[:3, 3] = t
        w2c.append(m)
        K.append(_intrinsics(model, params))
        sizes.append((w, h))
    return CameraSet([im[0] for im in images], np.asarray(w2c, np.float32),
                     np.asarray(K, np.float32), np.asarray(sizes, np.int64))
