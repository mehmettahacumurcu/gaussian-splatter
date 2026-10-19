"""Synthetic COLMAP models and splats for the coverage tests."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from backend.compose.plyio import GaussianCloud, write_ply


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def look_at_w2c(eye, target, down=(0, 1, 0)) -> np.ndarray:
    """OpenCV world->camera looking from ``eye`` at ``target`` (image y along ``down``)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(down, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m


def write_model(folder: Path, names: list[str], w2cs: list[np.ndarray], fmt: str = "bin",
                width: int = 640, height: int = 480, focal: float = 500.0) -> Path:
    """A one-camera PINHOLE model with the given poses (no 3D points)."""
    folder.mkdir(parents=True, exist_ok=True)
    params = (focal, focal, width / 2, height / 2)
    poses = [(rotmat_to_qvec(m[:3, :3]), m[:3, 3]) for m in w2cs]
    if fmt == "bin":
        with open(folder / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<iiQQ", 1, 1, width, height))
            f.write(struct.pack("<4d", *params))
        with open(folder / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", len(names)))
            for i, (name, (q, t)) in enumerate(zip(names, poses)):
                f.write(struct.pack("<i7di", i + 1, *q, *t, 1))
                f.write(name.encode() + b"\x00")
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<ddq", 1.0, 2.0, -1))
        (folder / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    else:
        (folder / "cameras.txt").write_text(
            "# Camera list\n1 PINHOLE %d %d %s\n" % (width, height, " ".join(map(str, params))), encoding="utf-8")
        lines = ["# Image list"]
        for i, (name, (q, t)) in enumerate(zip(names, poses)):
            lines.append(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {name}")
            lines.append("1.0 2.0 -1")
        (folder / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / "points3D.txt").write_text("", encoding="utf-8")
    return folder


def plane_cloud(n_side: int = 40, size: float = 2.0) -> GaussianCloud:
    """Opaque grey splats on the z=0 plane, x and y in [-size/2, size/2]."""
    g = np.linspace(-size / 2, size / 2, n_side, dtype=np.float32)
    xx, yy = np.meshgrid(g, g)
    n = xx.size
    means = np.stack([xx.ravel(), yy.ravel(), np.zeros(n, np.float32)], 1)
    step = size / (n_side - 1)
    return GaussianCloud(
        means=means,
        log_scales=np.tile(np.log([step * 0.6, step * 0.6, 1e-3]).astype(np.float32), (n, 1)),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, 4.0, np.float32),
        sh_dc=np.zeros((n, 3), np.float32),
        sh_rest=np.zeros((n, 0, 3), np.float32),
    )


def write_splat(path: Path, cloud: GaussianCloud | None = None) -> Path:
    return write_ply(cloud or plane_cloud(), path)
