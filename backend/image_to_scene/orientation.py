"""World up-vector estimation — gravity alignment for world bundles.

Reconstruction frames are arbitrary: COLMAP fixes its gauge from the first
camera / bundle adjustment, single-image generators use their own camera
convention. Measured on real bundles (2026-07-04): the garden Colab run came
out inverted and pitched 31 deg off the naive 180-flip, myroom-v2 lies on its
side, fixture-a is pitched 90 deg toward +Z. Nothing downstream may assume
raw +Y is up.

World bundles are therefore minted in VIEWER space (three.js, +Y up): the
wrap step measures 'up' here, derives the collider in the rotated frame, and
stores the quaternion in the collider JSON (`worldRotation`) for the viewer
to apply to the splat object. The .ply itself stays in the raw frame —
rebaking SH coefficients under a rotation is not worth it when the viewer can
rotate the object instead.

'Up' is defined by the dominant *walkable* plane: RANSAC over candidate
planes, scored by inlier mass x content asymmetry. The asymmetry term is what
rejects walls — a floor has (nearly) nothing below it, a wall has content on
both sides. Sign = the side the content mass lives on. A far bright shell
(sky), when present, is reported as a consistency check only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

# Below this tilt the raw frame is already fine — don't churn good worlds
# with sub-degree rotations that would dirty every collider re-wrap.
IDENTITY_SNAP_DEG = 5.0

# Floors reach above/below ratios of 100-1000x; walls sit near 1. The cap
# keeps plane choice driven by (mass x is-it-a-floor), not by ratio noise.
ASYM_CAP = 50.0


@dataclass
class WorldOrientation:
    quaternion: np.ndarray    # (4,) x,y,z,w — rotates raw frame -> viewer frame
    up_raw: np.ndarray        # (3,) measured up direction in the raw frame
    tilt_deg: float           # angle between up_raw and +Y
    plane_inlier_frac: float  # fraction of core sample on the chosen plane
    above_below_ratio: float  # content-mass asymmetry across the plane
    sky_agrees: bool | None   # far-bright-shell check; None if no usable shell


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotate_points(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Apply the (x, y, z, w) quaternion to an (N, 3) point array."""
    return np.asarray(points, dtype=np.float64) @ _rotation_matrix(quaternion).T


def _quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) rotating unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return IDENTITY_QUAT.copy()
        # Antiparallel: 180 deg about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = v / s
    half = np.arccos(np.clip(c, -1.0, 1.0)) / 2.0
    return np.array([*(axis * np.sin(half)), np.cos(half)])


def estimate_world_orientation(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    *,
    seed: int = 0,
    n_sample: int = 60_000,
    n_iters: int = 600,
    top_k: int = 8,
) -> WorldOrientation:
    """Estimate the raw-frame up direction and the rotation to viewer +Y.

    Args:
        points:  (N, 3) gaussian means.
        weights: (N,) importance (e.g. opacities). Defaults to uniform.
        colors:  (N, 3) linear RGB in [0, 1] — only used for the sky check.
        seed / n_sample / n_iters / top_k: RANSAC controls.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    n = pts.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    if n < 500:
        # Too sparse to trust a plane fit; leave the frame alone.
        return WorldOrientation(IDENTITY_QUAT.copy(), np.array([0.0, 1.0, 0.0]),
                                0.0, 0.0, 1.0, None)

    rng = np.random.default_rng(seed)
    center = np.median(pts, axis=0)
    r = np.linalg.norm(pts - center, axis=1)
    r97 = float(np.quantile(r, 0.97))
    core_idx = np.flatnonzero(r < r97)
    sample = rng.choice(core_idx, size=min(n_sample, core_idx.size), replace=False)
    sp, sw = pts[sample], w[sample]
    thresh = 0.02 * r97

    # -- RANSAC: collect candidate planes ranked by inlier mass ------------
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []  # (mass, n, p0)
    for _ in range(n_iters):
        i = rng.choice(sp.shape[0], size=3, replace=False)
        p0, p1, p2 = sp[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = (sp - p0) @ nrm
        candidates.append((float(sw[np.abs(d) < thresh].sum()), nrm, p0))
    candidates.sort(key=lambda c: -c[0])

    # Dedupe by normal direction (parallel planes share an up axis anyway).
    distinct: list[tuple[float, np.ndarray, np.ndarray]] = []
    for cand in candidates:
        if len(distinct) >= top_k:
            break
        if all(abs(cand[1] @ d[1]) < np.cos(np.radians(15.0)) for d in distinct):
            distinct.append(cand)
    if not distinct:
        return WorldOrientation(IDENTITY_QUAT.copy(), np.array([0.0, 1.0, 0.0]),
                                0.0, 0.0, 1.0, None)

    # -- Score candidates: inlier mass x content asymmetry ------------------
    best = None  # (score, up, inlier_frac, asym)
    eps = 1e-6
    for _, nrm, p0 in distinct:
        d = (sp - p0) @ nrm
        inl = np.abs(d) < thresh
        if inl.sum() < 50:
            continue
        # Least-squares refine on inliers.
        c = sp[inl].mean(axis=0)
        _, _, vt = np.linalg.svd(sp[inl] - c, full_matrices=False)
        nrm = vt[2] / np.linalg.norm(vt[2])
        d = (sp - c) @ nrm
        inl_mass = float(sw[np.abs(d) < thresh].sum())
        band = np.abs(d) < 0.5 * r97
        pos = float(sw[band & (d > thresh)].sum())
        neg = float(sw[band & (d < -thresh)].sum())
        sign = 1.0 if pos >= neg else -1.0
        asym = (max(pos, neg) + eps) / (min(pos, neg) + eps)
        score = inl_mass * min(asym, ASYM_CAP)
        if best is None or score > best[0]:
            best = (score, nrm * sign, inl_mass / max(sw.sum(), eps), asym)

    if best is None:
        return WorldOrientation(IDENTITY_QUAT.copy(), np.array([0.0, 1.0, 0.0]),
                                0.0, 0.0, 1.0, None)
    _, up, inlier_frac, asym = best

    # -- Sky consistency check (report only; content mass is authoritative) -
    sky_agrees: bool | None = None
    if colors is not None:
        far = r > 1.3 * r97
        if int(far.sum()) >= 300:
            bright = np.asarray(colors, dtype=np.float64)[far].mean(axis=1)
            sky_dir = ((pts[far] - center) * (w[far] * bright)[:, None]).sum(axis=0)
            nlen = float(np.linalg.norm(sky_dir))
            if nlen > eps:
                sky_agrees = bool((sky_dir / nlen) @ up > 0)

    tilt_deg = float(np.degrees(np.arccos(np.clip(up @ [0.0, 1.0, 0.0], -1, 1))))
    if tilt_deg < IDENTITY_SNAP_DEG:
        quat = IDENTITY_QUAT.copy()
    else:
        quat = _quat_between(up, np.array([0.0, 1.0, 0.0]))

    return WorldOrientation(quat, up, tilt_deg, float(inlier_frac), float(asym),
                            sky_agrees)
