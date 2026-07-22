from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.learned_quality.depth import SupportedDepthCloud
from experiments.learned_quality.floor_recovery import (
    FloorRecoveryPolicy,
    build_floor_hole_map,
    estimate_floor_plane,
    generate_floor_seed_artifact,
)


def _digest(character: str) -> str:
    return character * 64


def _room() -> tuple[np.ndarray, SupportedDepthCloud]:
    floor_x = np.linspace(-2.0, 2.0, 41)
    floor_z = np.linspace(0.0, 4.0, 41)
    xx, zz = np.meshgrid(floor_x, floor_z, indexing="xy")
    floor = np.stack((xx.ravel(), np.zeros(xx.size), zz.ravel()), axis=-1)
    missing = (
        (floor[:, 0] >= -0.65)
        & (floor[:, 0] <= 0.65)
        & (floor[:, 2] >= 1.15)
        & (floor[:, 2] <= 2.85)
    )
    sparse_floor = floor[~missing]

    wall_y = np.linspace(0.15, 2.5, 20)
    wall_z = np.linspace(0.0, 4.0, 20)
    yy, wz = np.meshgrid(wall_y, wall_z, indexing="xy")
    wall = np.stack(
        (np.full(yy.size, -2.0), yy.ravel(), wz.ravel()), axis=-1
    )
    furniture_rng = np.random.default_rng(4)
    furniture = furniture_rng.uniform(
        (-1.5, 0.2, 0.5), (1.5, 1.8, 3.5), size=(300, 3)
    )
    sparse = np.concatenate((sparse_floor, wall, furniture))

    candidates = floor.copy()
    frame_index = np.arange(len(candidates), dtype=np.int32) % 3
    source_xy = np.stack(
        (
            np.arange(len(candidates), dtype=np.int32) % 640,
            np.arange(len(candidates), dtype=np.int32) % 480,
        ),
        axis=-1,
    )
    cloud = SupportedDepthCloud(
        xyz=candidates.astype(np.float64),
        rgb=np.tile(np.array([[120, 105, 90]], dtype=np.uint8), (len(candidates), 1)),
        confidence=np.full(len(candidates), 0.9, dtype=np.float64),
        view_support=np.full(len(candidates), 3, dtype=np.uint16),
        source_frame_index=frame_index,
        source_xy=source_xy,
        camera_centers=np.array(
            ((-1.0, 1.7, -0.5), (0.0, 1.7, -0.5), (1.0, 1.7, -0.5)),
            dtype=np.float64,
        ),
        source_model_digest=_digest("a"),
        source_mask_digest=_digest("b"),
        source_depth_digest=_digest("c"),
        source_frame_digest=_digest("d"),
    )
    return sparse.astype(np.float64), cloud


def _policy(**changes: object) -> FloorRecoveryPolicy:
    base = FloorRecoveryPolicy(
        plane_inlier_fraction=0.02,
        minimum_plane_inliers=100,
        minimum_plane_support_fraction=0.005,
        maximum_up_angle_degrees=15.0,
        minimum_above_below_ratio=4.0,
        grid_fraction=0.025,
        footprint_erosion_cells=1,
        sparse_support_radius_cells=1.5,
        minimum_component_cells=4,
        minimum_seed_count=10,
        maximum_seed_count=1_000,
    )
    return dataclasses.replace(base, **changes)


def test_floor_plane_orients_toward_cameras_and_fits_floor() -> None:
    sparse, cloud = _room()

    plane = estimate_floor_plane(
        sparse,
        cloud.camera_centers,
        policy=_policy(),
        seed=8,
    )

    assert np.dot(plane.normal, np.array((0.0, 1.0, 0.0))) > 0.98
    assert abs(plane.offset) < plane.inlier_tolerance
    assert plane.inlier_count >= 100
    assert plane.median_camera_height > 1.0
    assert plane.above_below_ratio >= 4.0


def test_floor_plane_never_flips_a_ceiling_toward_the_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(19)
    floor = np.column_stack(
        (rng.uniform(-2.0, 2.0, 450), np.zeros(450), rng.uniform(0.0, 4.0, 450))
    )
    ceiling = np.column_stack(
        (
            rng.uniform(-2.0, 2.0, 1_100),
            np.full(1_100, 3.0),
            rng.uniform(0.0, 4.0, 1_100),
        )
    )
    interior = rng.uniform((-1.8, 0.2, 0.2), (1.8, 2.8, 3.8), size=(500, 3))
    points = np.concatenate((floor, ceiling, interior))
    cameras = np.array(((-1.0, 1.6, 0.5), (0.0, 1.7, 2.0), (1.0, 1.6, 3.5)))
    monkeypatch.setattr(
        "experiments.learned_quality.floor_recovery.estimate_world_orientation",
        lambda *_args, **_kwargs: SimpleNamespace(
            up_raw=np.array((0.0, 1.0, 0.0))
        ),
    )

    plane = estimate_floor_plane(points, cameras, policy=_policy(), seed=31)

    assert np.dot(plane.normal, np.array((0.0, 1.0, 0.0))) > 0.98
    assert abs(plane.offset) < plane.inlier_tolerance
    assert plane.median_camera_height > 1.0


def test_floor_hole_map_and_seeds_stay_inside_missing_patch(tmp_path: Path) -> None:
    sparse, cloud = _room()
    policy = _policy()
    plane = estimate_floor_plane(
        sparse, cloud.camera_centers, policy=policy, seed=8
    )
    holes = build_floor_hole_map(sparse, cloud, plane, policy=policy)

    artifact = generate_floor_seed_artifact(
        sparse,
        cloud,
        plane,
        holes,
        (tmp_path / "floor-artifact").resolve(),
        policy=policy,
    )

    assert holes.initial_hole_cells >= policy.minimum_component_cells
    assert artifact.point_count >= policy.minimum_seed_count
    with np.load(artifact.npz_path, allow_pickle=False) as values:
        assert set(values.files) == {
            "xyz",
            "rgb",
            "confidence",
            "view_support",
            "source_frame_index",
            "hole_cell_id",
        }
        xyz = values["xyz"]
        assert np.all(np.abs(xyz[:, 1]) <= 1e-6)
        assert np.all((xyz[:, 0] >= -0.8) & (xyz[:, 0] <= 0.8))
        assert np.all((xyz[:, 2] >= 0.9) & (xyz[:, 2] <= 3.1))


def test_floor_seed_cap_and_bytes_are_deterministic(tmp_path: Path) -> None:
    sparse, cloud = _room()
    policy = _policy(maximum_seed_count=20)
    plane = estimate_floor_plane(
        sparse, cloud.camera_centers, policy=policy, seed=8
    )
    holes = build_floor_hole_map(sparse, cloud, plane, policy=policy)

    first = generate_floor_seed_artifact(
        sparse, cloud, plane, holes, (tmp_path / "first").resolve(), policy=policy
    )
    second = generate_floor_seed_artifact(
        sparse, cloud, plane, holes, (tmp_path / "second").resolve(), policy=policy
    )

    assert first.point_count == 20
    assert first.content_fingerprint == second.content_fingerprint
    assert first.npz_path.read_bytes() == second.npz_path.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()


def test_floor_plane_and_hole_detection_fail_closed() -> None:
    cameras = np.array(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)), dtype=np.float64)
    with pytest.raises(ValueError, match="reliable floor plane"):
        estimate_floor_plane(
            np.stack((np.arange(200), np.zeros(200), np.zeros(200)), axis=-1),
            cameras,
            policy=_policy(),
            seed=1,
        )

    sparse, cloud = _room()
    policy = _policy(minimum_seed_count=10_000, maximum_seed_count=10_000)
    plane = estimate_floor_plane(
        sparse, cloud.camera_centers, policy=policy, seed=8
    )
    holes = build_floor_hole_map(sparse, cloud, plane, policy=policy)
    with pytest.raises(ValueError, match="no_recoverable_floor_hole"):
        generate_floor_seed_artifact(
            sparse,
            cloud,
            plane,
            holes,
            Path.cwd() / "insufficient-floor-seeds",
            policy=policy,
        )


def test_floor_policy_rejects_unsafe_limits() -> None:
    with pytest.raises(ValueError):
        _policy(maximum_seed_count=150_001)
    with pytest.raises(ValueError):
        _policy(minimum_seed_count=0)
    with pytest.raises(ValueError):
        _policy(maximum_up_angle_degrees=90.0)
