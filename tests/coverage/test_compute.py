"""CPU-side coverage maths (scoring, colours, statistics, view records)."""
from __future__ import annotations

import numpy as np

from backend.coverage import compute
from backend.coverage.colmap import CameraSet
from backend.coverage.runner import camera_to_world, is_test_view, view_records, world_up
from tests.coverage.helpers import look_at_w2c, plane_cloud


def test_score_is_log_count_saturating_at_good_count():
    s = compute.score(np.array([0, 5, compute.GOOD_COUNT, 500]))
    assert s[0] == 0 and s[2] == 1 and s[3] == 1
    assert 0.45 < s[1] < compute.WEAK_SCORE  # ~5 views is still weak


def test_heat_colours_red_yellow_green():
    c = compute.heat_colours(np.array([0.0, 0.5, 1.0]))
    assert c[0, 0] > 0.9 and c[0, 1] < 0.2   # red
    assert c[1, 0] > 0.9 and c[1, 1] > 0.8   # yellow
    assert c[2, 1] > 0.7 and c[2, 0] < 0.2   # green


def test_heat_cloud_keeps_geometry_and_encodes_colour_in_dc():
    cloud = plane_cloud(4)
    s = np.linspace(0, 1, cloud.count).astype(np.float32)
    heat = compute.heat_cloud(cloud, s)
    np.testing.assert_array_equal(heat.means, cloud.means)
    np.testing.assert_array_equal(heat.opacities, cloud.opacities)
    assert heat.sh_rest.shape == (cloud.count, 0, 3)
    np.testing.assert_allclose(0.5 + compute.SH_C0 * heat.sh_dc, compute.heat_colours(s), atol=1e-6)


def test_summary_ignores_transparent_splats():
    count = np.array([0, 0, 3, 100, 100])
    logits = np.array([5.0, -9.0, 5.0, 5.0, -9.0])  # #1 and #4 nearly transparent
    out = compute.summary({"count": count}, compute.score(count), logits)
    assert out["opaque_splats"] == 3
    assert out["unseen_share"] == 1 / 3
    assert out["weak_share"] == 2 / 3


def test_camera_to_world_inverts_and_up_points_away_from_image_down():
    w2c = look_at_w2c([1, 2, 3], [0, 0, 0], down=(0, 0, -1))
    np.testing.assert_allclose(camera_to_world(w2c) @ w2c, np.eye(4), atol=1e-9)
    cams = CameraSet(["a"], w2c[None].astype(np.float32), np.eye(3, dtype=np.float32)[None], np.array([[4, 3]]))
    up = world_up(cams)
    assert np.dot(up, [0, 0, 1]) > 0.5


def test_view_records_fov_and_test_flags():
    K = np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]], np.float32)
    cams = CameraSet(["eval_1.png"], np.eye(4, dtype=np.float32)[None], K, np.array([[640, 480]]))
    rec = view_records(cams, [{"score": 0.5, "weak": 0.25}], [True])[0]
    assert rec["test"] and rec["weak"] == 0.25
    assert abs(rec["fov_y"] - np.degrees(2 * np.arctan(240 / 500))) < 1e-6
    assert len(rec["c2w"]) == 16


def test_test_views_are_recognised_by_prefix():
    assert is_test_view("images/eval_00012.png") and is_test_view("TEST_1.jpg")
    assert not is_test_view("train_00001.png") and not is_test_view("frame_eval.png")
