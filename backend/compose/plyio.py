"""Standard INRIA 3DGS .ply read/write for the scene composer.

Same on-disk layout as ``backend/export/to_splat.write_ply`` but without the
torch import, so composer code and its tests stay light.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
# Number of f_rest_* fields for SH degree 0..3.
_VALID_REST_FIELD_COUNTS = (0, 9, 24, 45)


class PlyFormatError(ValueError):
    """The file is not a usable Gaussian-splat .ply."""


@dataclass
class GaussianCloud:
    means: np.ndarray       # (N, 3) float32
    log_scales: np.ndarray  # (N, 3) float32, log-space
    quats: np.ndarray       # (N, 4) float32, (w, x, y, z), not necessarily unit
    opacities: np.ndarray   # (N,)   float32, logit-space
    sh_dc: np.ndarray       # (N, 3) float32
    sh_rest: np.ndarray     # (N, K, 3) float32, K = (degree + 1)^2 - 1

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return sh_degree_for_coeffs(self.sh_rest.shape[1])


def sh_degree_for_coeffs(k: int) -> int:
    for degree in range(4):
        if (degree + 1) ** 2 - 1 == k:
            return degree
    raise PlyFormatError(f"Unsupported SH coefficient count: {k}")


def read_ply_header(path: str | Path) -> tuple[int, list[str]]:
    """Parse only the header: (vertex count, vertex property names)."""
    count: int | None = None
    props: list[str] = []
    in_vertex = False
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise PlyFormatError("Not a PLY file (missing 'ply' magic)")
        for _ in range(10_000):
            raw = fh.readline()
            if not raw:
                raise PlyFormatError("PLY header has no end_header")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    try:
                        count = int(parts[2])
                    except ValueError as exc:
                        raise PlyFormatError(f"Bad vertex count: {parts[2]}") from exc
            elif parts[0] == "property" and in_vertex:
                props.append(parts[-1])
        else:
            raise PlyFormatError("PLY header too long")
    if count is None:
        raise PlyFormatError("PLY has no vertex element")
    return count, props


def validate_ply(path: str | Path) -> int:
    """Check that ``path`` is a Gaussian-splat PLY. Returns the Gaussian count."""
    count, props = read_ply_header(path)
    missing = [f for f in REQUIRED_FIELDS if f not in props]
    if missing:
        raise PlyFormatError(f"Not a Gaussian-splat PLY, missing fields: {', '.join(missing)}")
    rest_suffixes = [p[len("f_rest_"):] for p in props if p.startswith("f_rest_")]
    n_rest = len(rest_suffixes)
    if n_rest not in _VALID_REST_FIELD_COUNTS:
        raise PlyFormatError(f"Unsupported number of f_rest fields: {n_rest}")
    if set(rest_suffixes) != {str(i) for i in range(n_rest)}:
        raise PlyFormatError("f_rest fields must be numbered contiguously from f_rest_0")
    if count <= 0:
        raise PlyFormatError("PLY contains no Gaussians")
    return count


def read_ply(path: str | Path) -> GaussianCloud:
    from plyfile import PlyData

    validate_ply(path)
    # Default memmap read (mmap=False is pure-Python and ~300x slower). Everything
    # stored on the cloud must be a copy, or it would keep the file locked on Windows.
    vertex = PlyData.read(str(path))["vertex"].data
    n = len(vertex)

    def col(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float32)

    rest_names = sorted(
        (f for f in vertex.dtype.names if f.startswith("f_rest_")),
        key=lambda f: int(f[len("f_rest_"):]),
    )
    k = len(rest_names) // 3
    if rest_names:
        # Channel-major on disk: f_rest_{c*K + k}.
        flat = np.stack([col(f) for f in rest_names], axis=1)
        sh_rest = flat.reshape(n, 3, k).transpose(0, 2, 1).copy()
    else:
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    return GaussianCloud(
        means=np.stack([col("x"), col("y"), col("z")], axis=1),
        log_scales=np.stack([col(f"scale_{i}") for i in range(3)], axis=1),
        quats=np.stack([col(f"rot_{i}") for i in range(4)], axis=1),
        opacities=col("opacity").copy(),  # single column: would otherwise alias the memmap
        sh_dc=np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1),
        sh_rest=sh_rest,
    )


def write_ply(cloud: GaussianCloud, path: str | Path) -> Path:
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.count
    k = cloud.sh_rest.shape[1]
    dtype = (
        [(p, "f4") for p in ("x", "y", "z", "nx", "ny", "nz")]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [(f"f_rest_{i}", "f4") for i in range(3 * k)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = cloud.means[:, 0], cloud.means[:, 1], cloud.means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(3):
        arr[f"f_dc_{i}"] = cloud.sh_dc[:, i]
    rest_flat = cloud.sh_rest.transpose(0, 2, 1).reshape(n, 3 * k)
    for i in range(3 * k):
        arr[f"f_rest_{i}"] = rest_flat[:, i]
    arr["opacity"] = cloud.opacities
    for i in range(3):
        arr[f"scale_{i}"] = cloud.log_scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = cloud.quats[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path
