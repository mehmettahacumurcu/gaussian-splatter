from __future__ import annotations

import numpy as np
import pytest

from backend.coverage.colmap import ColmapError, has_model, read_cameras
from tests.coverage.helpers import look_at_w2c, write_model

POSES = {
    "b_0002.png": look_at_w2c([0, 0, 3], [0, 0, 0]),
    "a_0001.png": look_at_w2c([2, 1, 3], [0, 0, 0]),
    "eval_0003.png": look_at_w2c([-1, 0.5, 2], [0.2, 0, 0]),
}


@pytest.mark.parametrize("fmt", ["bin", "txt"])
def test_reads_poses_intrinsics_and_sorts_by_name(tmp_path, fmt):
    folder = write_model(tmp_path / "sparse" / "0", list(POSES), list(POSES.values()), fmt=fmt,
                         width=640, height=480, focal=500)
    assert has_model(folder)
    cams = read_cameras(folder)
    assert cams.names == sorted(POSES)
    for i, name in enumerate(cams.names):
        np.testing.assert_allclose(cams.w2c[i], POSES[name], atol=1e-5)
    np.testing.assert_allclose(cams.K[0], [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    assert cams.sizes.tolist() == [[640, 480]] * 3


def test_single_focal_models_use_one_focal(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "cameras.txt").write_text("1 SIMPLE_RADIAL 100 50 80 50 25 0.01\n", encoding="utf-8")
    (folder / "images.txt").write_text("1 1 0 0 0 0 0 0 1 x.png\n\n", encoding="utf-8")
    cams = read_cameras(folder)
    np.testing.assert_allclose(cams.K[0], [[80, 0, 50], [0, 80, 25], [0, 0, 1]])


def test_subset_keeps_order(tmp_path):
    cams = read_cameras(write_model(tmp_path, list(POSES), list(POSES.values())))
    sub = cams.subset([2, 0])
    assert sub.names == [cams.names[2], cams.names[0]]
    np.testing.assert_array_equal(sub.w2c[0], cams.w2c[2])


def test_missing_or_empty_model_raises(tmp_path):
    assert not has_model(tmp_path)
    with pytest.raises(ColmapError):
        read_cameras(tmp_path)
    write_model(tmp_path, [], [], fmt="txt")
    with pytest.raises(ColmapError):
        read_cameras(tmp_path)
