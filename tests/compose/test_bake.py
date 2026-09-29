from __future__ import annotations

import math

import numpy as np
import pytest

from backend.compose.bake import (
    ColorAdjust, Crop, Placement, color_matrix, crop_mask, merge_clouds, transform_cloud,
)
from backend.compose.plyio import GaussianCloud
from backend.compose.sh_rotation import eval_sh_color
from tests.compose.helpers import random_cloud

S45 = math.sin(math.pi / 4)


def _dirs(n, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_identity_placement_is_a_noop():
    cloud = random_cloud(30)
    out = transform_cloud(cloud, Placement())
    for field in ("means", "log_scales", "quats", "opacities", "sh_dc"):
        np.testing.assert_array_equal(getattr(out, field), getattr(cloud, field))
    np.testing.assert_allclose(out.sh_rest, cloud.sh_rest, atol=1e-6)


def test_translation_moves_only_means():
    cloud = random_cloud(10)
    out = transform_cloud(cloud, Placement(position=(1.0, -2.0, 3.0)))
    np.testing.assert_allclose(out.means, cloud.means + np.array([1, -2, 3]), atol=1e-6)
    np.testing.assert_array_equal(out.quats, cloud.quats)


def test_rotation_rotates_means_and_gaussian_orientation():
    cloud = GaussianCloud(
        means=np.array([[1.0, 0.0, 0.0]], np.float32),
        log_scales=np.zeros((1, 3), np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.zeros(1, np.float32),
        sh_dc=np.zeros((1, 3), np.float32),
        sh_rest=np.zeros((1, 0, 3), np.float32),
    )
    rz90 = Placement(quaternion_xyzw=(0.0, 0.0, S45, S45))
    out = transform_cloud(cloud, rz90)
    np.testing.assert_allclose(out.means, [[0.0, 1.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(out.quats, [[S45, 0.0, 0.0, S45]], atol=1e-6)


def test_uniform_scale_scales_means_and_log_scales():
    cloud = random_cloud(10)
    out = transform_cloud(cloud, Placement(scale=2.0))
    np.testing.assert_allclose(out.means, cloud.means * 2, atol=1e-6)
    np.testing.assert_allclose(out.log_scales, cloud.log_scales + math.log(2.0), atol=1e-6)


def test_crop_mask_axis_aligned_and_rotated():
    means = np.array([[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0], [0, 0, 0.99]], np.float32)
    box = Crop(center=(0, 0, 0), half_size=(1, 1, 1), quaternion_xyzw=(0, 0, 0, 1))
    assert crop_mask(means, box).tolist() == [True, True, False, True]

    rotated = Crop(center=(0, 0, 0), half_size=(1, 0.1, 1),
                   quaternion_xyzw=(0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
    pts = np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0]], np.float32)
    assert crop_mask(pts, rotated).tolist() == [True, False]


def test_crop_happens_in_local_frame_before_transform():
    cloud = random_cloud(3)
    cloud.means[:] = [[0, 0, 0], [5, 0, 0], [0.2, 0, 0]]
    placement = Placement(
        position=(10.0, 0.0, 0.0),
        crop=Crop(center=(0, 0, 0), half_size=(1, 1, 1), quaternion_xyzw=(0, 0, 0, 1)),
    )
    out = transform_cloud(cloud, placement)
    np.testing.assert_allclose(out.means, [[10, 0, 0], [10.2, 0, 0]], atol=1e-6)


def test_exposure_plus_one_doubles_rendered_colour():
    cloud = random_cloud(40, seed=2)
    dirs = _dirs(40)
    out = transform_cloud(cloud, Placement(color=ColorAdjust(exposure=1.0)))
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs),
        2.0 * eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs),
        atol=1e-5,
    )


def test_saturation_zero_gives_grey():
    cloud = random_cloud(40, seed=3)
    out = transform_cloud(cloud, Placement(color=ColorAdjust(saturation=0.0)))
    rgb = eval_sh_color(out.sh_dc, out.sh_rest, _dirs(40))
    np.testing.assert_allclose(rgb[:, 0], rgb[:, 1], atol=1e-5)
    np.testing.assert_allclose(rgb[:, 1], rgb[:, 2], atol=1e-5)


def test_colour_matrix_applies_to_every_view_direction():
    adj = ColorAdjust(exposure=0.3, tint=(1.2, 0.9, 0.7), saturation=1.4)
    cloud = random_cloud(40, seed=4)
    dirs = _dirs(40, seed=1)
    out = transform_cloud(cloud, Placement(color=adj))
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs),
        eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs) @ color_matrix(adj).T,
        atol=1e-5,
    )


def test_rotation_keeps_view_dependent_colour_consistent():
    cloud = random_cloud(40, seed=5)
    q = np.array([0.3, -0.2, 0.5, 0.78])
    q /= np.linalg.norm(q)
    placement = Placement(quaternion_xyzw=tuple(q))
    from backend.compose.bake import quat_to_matrix

    r = quat_to_matrix(tuple(q))
    dirs = _dirs(40, seed=2)
    out = transform_cloud(cloud, placement)
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs @ r.T),
        eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs),
        atol=1e-5,
    )


def test_merge_pads_lower_sh_degree():
    a = random_cloud(5, degree=1, seed=1)
    b = random_cloud(3, degree=3, seed=2)
    merged = merge_clouds([a, b])
    assert merged.count == 8
    assert merged.sh_rest.shape == (8, 15, 3)
    np.testing.assert_array_equal(merged.sh_rest[:5, :3], a.sh_rest)
    assert not merged.sh_rest[:5, 3:].any()
    np.testing.assert_array_equal(merged.means[5:], b.means)


def test_merge_rejects_all_cropped_away():
    empty = random_cloud(0, degree=3)
    with pytest.raises(ValueError, match="All Gaussians were cropped away"):
        merge_clouds([empty, random_cloud(0, degree=1)])


def test_quaternion_composition_order():
    from backend.compose.bake import quat_to_matrix

    cloud = random_cloud(25, seed=9)
    q = np.array([0.3, -0.6, 0.2, 0.7])
    q /= np.linalg.norm(q)
    out = transform_cloud(cloud, Placement(quaternion_xyzw=tuple(q)))
    r_o = quat_to_matrix(tuple(q))
    for qi, qo in zip(cloud.quats.astype(np.float64), out.quats.astype(np.float64)):
        expected = r_o @ quat_to_matrix((qi[1], qi[2], qi[3], qi[0]))
        got = quat_to_matrix((qo[1], qo[2], qo[3], qo[0]))
        np.testing.assert_allclose(got, expected, atol=1e-5)


def test_merge_requires_at_least_one_cloud():
    with pytest.raises(ValueError, match="Nothing to merge"):
        merge_clouds([])
