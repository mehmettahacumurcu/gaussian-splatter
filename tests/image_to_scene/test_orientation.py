"""Tests for world up-vector estimation (gravity alignment at wrap time).

Reconstruction frames are arbitrary (COLMAP gauge freedom / generator
convention): the 2026-07-04 garden Colab run came out inverted + pitched 31
degrees, myroom-v2 lies on its side, fixture-a is pitched 90 degrees. The wrap
step must measure 'up' from the dominant walkable plane and the side content
lives on, and bundles must carry the correcting rotation.
"""
import json

import numpy as np
import pytest

from backend.image_to_scene.orientation import (
    estimate_world_orientation, rotate_points,
)
from backend.image_to_scene.collider import MinimalCollider, write_collider_json


# ---------------------------------------------------------------------------
# Synthetic scene builders
# ---------------------------------------------------------------------------

def _basis_perp(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(up, a)
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    return u, v


def _disc(up, n, radius, thickness, rng, center=None):
    """Points on a disc perpendicular to `up`."""
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    u, v = _basis_perp(up)
    r = np.sqrt(rng.uniform(0, 1, n)) * radius
    th = rng.uniform(0, 2 * np.pi, n)
    pts = (r * np.cos(th))[:, None] * u + (r * np.sin(th))[:, None] * v \
        + rng.normal(0, thickness, (n, 1)) * up
    if center is not None:
        pts = pts + np.asarray(center, dtype=float)
    return pts


def _garden_like(up, seed=0, n_floor=8000, n_content=3000, n_shell=600):
    """Dense floor disc + content strictly above it + bright far sky shell."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    u, v = _basis_perp(up)

    floor = _disc(up, n_floor, radius=6.0, thickness=0.02, rng=rng)

    h = rng.uniform(0.2, 2.0, n_content)
    r = np.sqrt(rng.uniform(0, 1, n_content)) * 5.0
    th = rng.uniform(0, 2 * np.pi, n_content)
    content = (r * np.cos(th))[:, None] * u + (r * np.sin(th))[:, None] * v \
        + h[:, None] * up

    # Far shell biased to the up hemisphere (sky).
    dirs = rng.normal(size=(n_shell * 4, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs[dirs @ up > 0.2][:n_shell]
    shell = dirs * 30.0

    pts = np.concatenate([floor, content, shell])
    weights = np.ones(len(pts))
    colors = np.full((len(pts), 3), 0.4)
    colors[-len(shell):] = 0.85  # sky is bright
    return pts, weights, colors


def _angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# estimate_world_orientation
# ---------------------------------------------------------------------------

def test_recovers_tilted_inverted_up():
    # The measured garden frame: inverted and pitched ~31 deg off the naive flip.
    true_up = np.array([0.0, -0.859, -0.512])
    true_up /= np.linalg.norm(true_up)
    pts, w, colors = _garden_like(true_up)

    est = estimate_world_orientation(pts, weights=w, colors=colors)

    assert _angle_deg(est.up_raw, true_up) < 3.0
    # Rotating the raw frame by the quaternion must map measured-up onto +Y.
    mapped = rotate_points(true_up[None, :], est.quaternion)[0]
    assert _angle_deg(mapped, [0.0, 1.0, 0.0]) < 3.0
    # And the floor must become horizontal: tiny Y spread on the floor slab.
    rot = rotate_points(pts[:8000], est.quaternion)
    assert np.std(rot[:, 1]) < 0.15
    # Content sits ABOVE the rotated floor.
    rot_content = rotate_points(pts[8000:11000], est.quaternion)
    assert np.median(rot_content[:, 1]) > np.median(rot[:, 1]) + 0.1


def test_upright_scene_snaps_to_identity():
    pts, w, colors = _garden_like(np.array([0.0, 1.0, 0.0]), seed=1)
    est = estimate_world_orientation(pts, weights=w, colors=colors)
    assert est.tilt_deg < 5.0
    np.testing.assert_allclose(est.quaternion, [0.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_floor_beats_larger_wall():
    # myroom regression: a perimeter wall can out-inlier the floor. Model the
    # physical case — the wall STANDS ON the floor near its edge, room content
    # is interior, a few reconstruction floaters sit outside. The wall then has
    # moderate one-sided asymmetry, the floor has (capped) extreme asymmetry;
    # the floor must still win despite 3x fewer points.
    true_up = np.array([0.0, -1.0, 0.0])
    rng = np.random.default_rng(7)
    u, v = _basis_perp(true_up)

    floor = _disc(true_up, 5000, radius=5.0, thickness=0.02, rng=rng)
    h = rng.uniform(0.2, 2.2, 2500)
    r = np.sqrt(rng.uniform(0, 1, 2500)) * 4.0
    th = rng.uniform(0, 2 * np.pi, 2500)
    content = (r * np.cos(th))[:, None] * u + (r * np.sin(th))[:, None] * v \
        + h[:, None] * true_up

    # Perimeter wall at x=+4.5, rising from the floor (0 to 4m along up).
    wall_n = np.array([1.0, 0.0, 0.0])
    n_wall = 15000
    wall_span = rng.uniform(-5.0, 5.0, n_wall)          # along the wall
    wall_rise = rng.uniform(0.0, 4.0, n_wall)           # up from the floor
    wall_axis = np.cross(wall_n, true_up)
    wall = (np.array([4.5, 0.0, 0.0])
            + wall_span[:, None] * wall_axis
            + wall_rise[:, None] * true_up
            + rng.normal(0, 0.02, (n_wall, 1)) * wall_n)

    # Sparse floaters outside the wall.
    floaters = (np.array([5.5, 0.0, 0.0])
                + rng.uniform(-4, 4, (500, 1)) * wall_axis
                + rng.uniform(0, 3, (500, 1)) * true_up
                + rng.uniform(0, 2, (500, 1)) * wall_n)

    pts = np.concatenate([floor, content, wall, floaters])
    est = estimate_world_orientation(pts)

    assert _angle_deg(est.up_raw, true_up) < 5.0, (
        f"picked up={est.up_raw} — likely locked onto the wall")


def test_rotate_points_is_rigid():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 3))
    q = np.array([0.9640, 0.0, -0.0011, 0.2658])
    q = q / np.linalg.norm(q)
    rot = rotate_points(pts, q)
    # Rigid rotation: pairwise distances preserved.
    d0 = np.linalg.norm(pts[:100] - pts[100:200], axis=1)
    d1 = np.linalg.norm(rot[:100] - rot[100:200], axis=1)
    np.testing.assert_allclose(d0, d1, atol=1e-6)


# ---------------------------------------------------------------------------
# collider JSON carries the rotation
# ---------------------------------------------------------------------------

def _dummy_collider():
    return MinimalCollider(
        ground_y=0.0,
        bbox_min=np.array([-3.0, 0.0, -3.0], dtype=np.float32),
        bbox_max=np.array([3.0, 2.5, 3.0], dtype=np.float32),
        spawn_position=np.array([0.0, 1.7, 0.0], dtype=np.float32),
        spawn_look_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_collider_json_includes_world_rotation(tmp_path):
    q = [0.9640, 0.0, -0.0011, 0.2658]
    p = tmp_path / "collider.json"
    write_collider_json(p, _dummy_collider(), world_rotation=q)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1  # additive, not a breaking change
    got = data["worldRotation"]["quaternion"]
    assert got == pytest.approx(q, abs=1e-6)


def test_collider_json_omits_world_rotation_when_absent(tmp_path):
    p = tmp_path / "collider.json"
    write_collider_json(p, _dummy_collider())
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "worldRotation" not in data
