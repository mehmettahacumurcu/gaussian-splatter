import math
import numpy as np
import pytest
from backend.image_to_scene.trajectory import (
    generate_bounded_room_trajectory, CameraPose,
)


def test_first_pose_is_origin_facing_forward():
    poses = generate_bounded_room_trajectory(n_views=10, bubble_radius_m=3.0)
    assert len(poses) == 10
    first = poses[0]
    np.testing.assert_allclose(first.position, [0.0, 0.0, 0.0], atol=1e-6)
    forward = first.rotation[:, 2]
    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-6)


def test_poses_stay_within_bubble():
    poses = generate_bounded_room_trajectory(n_views=30, bubble_radius_m=3.0)
    for pose in poses:
        r = float(np.linalg.norm(pose.position))
        assert r <= 3.0 + 1e-4, f"pose at r={r} exceeds bubble"


def test_yaw_covers_360_when_requested():
    poses = generate_bounded_room_trajectory(
        n_views=12, bubble_radius_m=2.0, yaw_full_360=True,
    )
    yaws = []
    for pose in poses[1:]:
        f = pose.rotation[:, 2]
        yaws.append(math.atan2(f[0], f[2]))
    yaw_range = max(yaws) - min(yaws)
    assert yaw_range >= math.radians(300), f"yaw range too small: {math.degrees(yaw_range):.1f}°"


def test_camera_pose_extrinsic_matrix_shape():
    poses = generate_bounded_room_trajectory(n_views=3, bubble_radius_m=2.0)
    M = poses[0].as_world_from_camera()
    assert M.shape == (4, 4)
    assert M[3, 3] == 1.0


def test_single_view_returns_just_origin():
    poses = generate_bounded_room_trajectory(n_views=1, bubble_radius_m=3.0)
    assert len(poses) == 1
    np.testing.assert_allclose(poses[0].position, [0.0, 0.0, 0.0], atol=1e-6)


def test_zero_or_negative_views_raises():
    with pytest.raises(ValueError):
        generate_bounded_room_trajectory(n_views=0)
