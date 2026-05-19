import json
import numpy as np
import pytest
from pathlib import Path
from backend.image_to_scene.collider import (
    derive_minimal_collider, MinimalCollider, write_collider_json,
)


def test_derive_finds_floor_y():
    # Cluster of points: most at y=0 (floor), some at y=2.5 (ceiling).
    rng = np.random.default_rng(42)
    pts = np.concatenate([
        rng.uniform(-1, 1, (1000, 3)) * np.array([2, 0.01, 2]) + np.array([0, 0, 0]),
        rng.uniform(-1, 1, (200, 3)) * np.array([2, 0.01, 2]) + np.array([0, 2.5, 0]),
    ])
    collider = derive_minimal_collider(pts)
    # 10th percentile of Y values for this mix should be very close to 0.
    assert collider.ground_y == pytest.approx(0.0, abs=0.1)


def test_derive_finds_bounding_box():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-3, 3, (5000, 3))
    collider = derive_minimal_collider(pts)
    bx_min, bx_max = collider.bbox_min, collider.bbox_max
    # 2.5th-97.5th percentile should be inside (-3, 3).
    assert bx_min[0] >= -3.1 and bx_max[0] <= 3.1
    assert bx_min[2] >= -3.1 and bx_max[2] <= 3.1


def test_spawn_at_ground_plus_eye_height():
    pts = np.zeros((100, 3))
    pts[:, 1] = np.linspace(0.0, 2.5, 100)  # Y from 0 to 2.5
    collider = derive_minimal_collider(pts, eye_height_m=1.7)
    # ground_y ≈ 0 (10th percentile), spawn.y ≈ 1.7
    assert collider.spawn_position[1] == pytest.approx(collider.ground_y + 1.7, abs=1e-3)
    np.testing.assert_allclose(collider.spawn_look_direction, [0.0, 0.0, 1.0], atol=1e-6)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        derive_minimal_collider(np.zeros((5, 3)))


def test_write_collider_json(tmp_path):
    collider = MinimalCollider(
        ground_y=0.0,
        bbox_min=np.array([-3.0, 0.0, -3.0], dtype=np.float32),
        bbox_max=np.array([3.0, 2.5, 3.0], dtype=np.float32),
        spawn_position=np.array([0.0, 1.7, 0.0], dtype=np.float32),
        spawn_look_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    p = tmp_path / "collider.json"
    write_collider_json(p, collider)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["groundPlane"]["y"] == 0.0
    assert "boundingWalls" in data
    bw = data["boundingWalls"]
    assert bw["xMin"] == -3.0 and bw["xMax"] == 3.0
    assert bw["zMin"] == -3.0 and bw["zMax"] == 3.0
    assert bw["yMax"] == 2.5
    spawn = data["spawn"]
    assert spawn["position"] == [0.0, 1.7, 0.0]
    assert spawn["lookDirection"] == [0.0, 0.0, 1.0]
