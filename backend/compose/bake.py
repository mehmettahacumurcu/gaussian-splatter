"""Bake scene-composer objects into one Gaussian cloud (pure numpy).

Order per object: crop (object-local) → uniform scale → rotation → translation,
then SH rotation and colour adjustment. Must match the Spark preview in
``frontend/src/compose`` (checked by the shared golden fixture).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plyio import GaussianCloud
from .sh_rotation import C0, rotate_sh_rest

Vec3 = tuple[float, float, float]
QuatXYZW = tuple[float, float, float, float]

LUMA = np.array([0.2126, 0.7152, 0.0722])  # Rec.709


@dataclass(frozen=True)
class Crop:
    center: Vec3
    half_size: Vec3
    quaternion_xyzw: QuatXYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ColorAdjust:
    exposure: float = 0.0
    tint: Vec3 = (1.0, 1.0, 1.0)
    saturation: float = 1.0


@dataclass(frozen=True)
class Placement:
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion_xyzw: QuatXYZW = (0.0, 0.0, 0.0, 1.0)
    scale: float = 1.0
    crop: Crop | None = None
    color: ColorAdjust | None = None


def quat_to_matrix(q_xyzw: QuatXYZW) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64) / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for one quaternion ``a`` (4,) and many ``b`` (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def crop_mask(means: np.ndarray, crop: Crop) -> np.ndarray:
    """True for Gaussians whose centre lies inside the (oriented) crop box."""
    r = quat_to_matrix(crop.quaternion_xyzw)
    local = (means.astype(np.float64) - np.asarray(crop.center)) @ r  # rows: R^T (p - c)
    return np.all(np.abs(local) <= np.asarray(crop.half_size), axis=1)


def color_matrix(adj: ColorAdjust) -> np.ndarray:
    """``M = S · diag(gain)``: exposure/tint first, then saturation around luma."""
    gain = (2.0 ** adj.exposure) * np.asarray(adj.tint, dtype=np.float64)
    luma = np.outer(np.ones(3), LUMA)
    sat = luma + adj.saturation * (np.eye(3) - luma)
    return sat @ np.diag(gain)


def _apply_color(sh_dc: np.ndarray, sh_rest: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Rendered colour is 0.5 + C0*dc + Σ rest·Y, so applying M to (0.5 + C0*dc)
    # and to every rest coefficient equals M · colour for every view direction.
    rgb = 0.5 + C0 * sh_dc.astype(np.float64)
    dc = (rgb @ m.T - 0.5) / C0
    rest = np.einsum("ij,nkj->nki", m, sh_rest)
    return dc.astype(np.float32), rest.astype(np.float32)


def _select(cloud: GaussianCloud, keep: np.ndarray) -> GaussianCloud:
    return GaussianCloud(
        means=cloud.means[keep], log_scales=cloud.log_scales[keep], quats=cloud.quats[keep],
        opacities=cloud.opacities[keep], sh_dc=cloud.sh_dc[keep], sh_rest=cloud.sh_rest[keep],
    )


def transform_cloud(cloud: GaussianCloud, placement: Placement) -> GaussianCloud:
    src = cloud if placement.crop is None else _select(cloud, crop_mask(cloud.means, placement.crop))
    q = np.asarray(placement.quaternion_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q)
    r = quat_to_matrix(tuple(q))
    s = float(placement.scale)

    means = (s * (src.means.astype(np.float64) @ r.T) + np.asarray(placement.position)).astype(np.float32)
    # Unit q_o keeps the norm of the stored (possibly unnormalised) quaternion.
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])
    quats = quat_multiply_wxyz(q_wxyz, src.quats.astype(np.float64)).astype(np.float32)
    log_scales = (src.log_scales.astype(np.float64) + math.log(s)).astype(np.float32)
    sh_rest = rotate_sh_rest(src.sh_rest, r).astype(np.float32)
    sh_dc = src.sh_dc.copy()
    if placement.color is not None:
        sh_dc, sh_rest = _apply_color(sh_dc, sh_rest, color_matrix(placement.color))
    return GaussianCloud(means, log_scales, quats, src.opacities.copy(), sh_dc, sh_rest)


def merge_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    if not clouds:
        raise ValueError("Nothing to merge: no visible splat objects")
    k = max(c.sh_rest.shape[1] for c in clouds)
    rest = [np.pad(c.sh_rest, ((0, 0), (0, k - c.sh_rest.shape[1]), (0, 0))) for c in clouds]
    return GaussianCloud(
        means=np.concatenate([c.means for c in clouds]),
        log_scales=np.concatenate([c.log_scales for c in clouds]),
        quats=np.concatenate([c.quats for c in clouds]),
        opacities=np.concatenate([c.opacities for c in clouds]),
        sh_dc=np.concatenate([c.sh_dc for c in clouds]),
        sh_rest=np.concatenate(rest),
    )
