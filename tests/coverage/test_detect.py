from __future__ import annotations

import numpy as np
import pytest

from backend.coverage.detect import DetectError, detect, find_images
from tests.coverage.helpers import look_at_w2c, plane_cloud, write_model, write_splat


def _spirula_run(root):
    """Layout of a Spirula Colab run synced from Drive."""
    splat = write_splat(root / "outputs" / "training" / "run" / "splat.ply", plane_cloud(30))
    write_splat(root / "outputs" / "small.ply", plane_cloud(3))
    write_splat(root / "coverage_heat.ply", plane_cloud(50))  # bigger, but a coverage output
    ds = root / "diagnostic_dataset"
    write_model(ds / "sparse" / "0", ["train_0001.png"], [look_at_w2c([0, 0, 2], [0, 0, 0])])
    write_model(ds / "sparse_components" / "1", ["x.png"], [np.eye(4)])
    (ds / "images").mkdir()
    (ds / "images" / "train_0001.png").write_bytes(b"png")
    return splat, ds


def test_finds_splat_model_and_images_in_a_run_folder(tmp_path):
    splat, ds = _spirula_run(tmp_path)
    ply, sparse = detect(str(tmp_path))
    assert ply == splat
    assert sparse == ds / "sparse" / "0"
    assert find_images(sparse, ["train_0001.png"]) == ds / "images"


def test_a_ply_path_climbs_to_the_run_folder_model(tmp_path):
    splat, ds = _spirula_run(tmp_path)
    ply, sparse = detect(str(splat))
    assert ply == splat and sparse == ds / "sparse" / "0"


def test_trainer_folder_finds_the_shared_model_one_level_up(tmp_path):
    """Final-pipeline layout: ROOM2/sfm/sparse_0 next to ROOM2/<trainer>/splat.ply."""
    room = tmp_path / "ROOM2"
    sparse = write_model(room / "sfm" / "sparse_0", ["train_0001.png"], [np.eye(4)])
    splat = write_splat(room / "spirula_high" / "splat.ply", plane_cloud(10))
    write_splat(room / "spirula_ultra" / "splat.ply", plane_cloud(30))
    assert detect(str(room / "spirula_high")) == (splat, sparse)


def test_explicit_paths_win(tmp_path):
    splat, ds = _spirula_run(tmp_path)
    other = write_model(tmp_path / "other", ["a.png"], [np.eye(4)], fmt="txt")
    ply, sparse = detect("", str(splat), str(other))
    assert (ply, sparse) == (splat, other)


def test_missing_folder(tmp_path):
    with pytest.raises(DetectError, match="bulunamadı"):
        detect(str(tmp_path / "nope"))


def test_folder_without_model(tmp_path):
    write_splat(tmp_path / "a" / "b" / "c" / "d" / "s.ply")
    with pytest.raises(DetectError, match="COLMAP"):
        detect(str(tmp_path / "a" / "b" / "c" / "d"))


def test_folder_without_splat(tmp_path):
    write_model(tmp_path / "sparse" / "0", ["a.png"], [np.eye(4)])
    with pytest.raises(DetectError, match="splat .ply"):
        detect(str(tmp_path))


def test_images_missing_gives_none(tmp_path):
    sparse = write_model(tmp_path / "sparse" / "0", ["a.png"], [np.eye(4)])
    assert find_images(sparse, ["a.png"]) is None
