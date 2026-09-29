"""Rotate real spherical-harmonic colour coefficients (INRIA 3DGS convention).

Instead of closed-form Wigner-D matrices, each band's rotation matrix is solved
numerically from the *same* basis functions the renderer uses, so sign and
ordering conventions cannot disagree with gsplat / 3DGS.
"""
from __future__ import annotations

import numpy as np

from .plyio import sh_degree_for_coeffs

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)

# Column ranges of each band inside f_rest (band 0 is the DC term).
BAND_SLICES = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}


def sh_basis(dirs: np.ndarray, degree: int) -> np.ndarray:
    """Basis values for bands 1..degree at unit ``dirs`` (M, 3) → (M, (degree+1)^2 - 1)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols: list[np.ndarray] = []
    if degree >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        cols += [
            C2[0] * x * y,
            C2[1] * y * z,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * x * z,
            C2[4] * (xx - yy),
        ]
    if degree >= 3:
        cols += [
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * x * y * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ]
    if not cols:
        return np.zeros((dirs.shape[0], 0))
    return np.stack(cols, axis=1)


def eval_sh_color(sh_dc: np.ndarray, sh_rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Rendered colour (N, 3) for per-Gaussian unit view ``dirs`` (N, 3), unclamped."""
    degree = sh_degree_for_coeffs(sh_rest.shape[1])
    basis = sh_basis(dirs, degree)
    return 0.5 + C0 * sh_dc + np.einsum("nk,nkc->nc", basis, sh_rest)


def _fibonacci_sphere(n: int = 256) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_SAMPLE_DIRS = _fibonacci_sphere()


def band_rotation_matrices(rotation: np.ndarray, degree: int) -> dict[int, np.ndarray]:
    """Per band ``l``: ``D_l`` with ``c'_l = D_l @ c_l`` so that ``SH'(R d) = SH(d)``."""
    if degree == 0:
        return {}
    r = np.asarray(rotation, dtype=np.float64)
    a = sh_basis(_SAMPLE_DIRS, degree)
    b = sh_basis(_SAMPLE_DIRS @ r, degree)  # rows are R^T d
    out: dict[int, np.ndarray] = {}
    for band in range(1, degree + 1):
        s = BAND_SLICES[band]
        # Solve Y(d) D = Y(R^T d) in the least-squares sense (exact for SH).
        out[band] = np.linalg.lstsq(a[:, s], b[:, s], rcond=None)[0]
    return out


def rotate_sh_rest(sh_rest: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate (N, K, 3) higher-order SH coefficients by ``rotation`` (3, 3)."""
    degree = sh_degree_for_coeffs(sh_rest.shape[1])
    if degree == 0:
        return sh_rest.copy()
    out = np.empty_like(sh_rest)
    for band, d in band_rotation_matrices(rotation, degree).items():
        s = BAND_SLICES[band]
        out[:, s, :] = np.einsum("ij,njc->nic", d, sh_rest[:, s, :])
    return out
