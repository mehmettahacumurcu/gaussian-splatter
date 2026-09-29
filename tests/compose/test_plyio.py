from __future__ import annotations

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from backend.compose.plyio import PlyFormatError, read_ply, read_ply_header, validate_ply, write_ply
from tests.compose.helpers import random_cloud


def test_round_trip_preserves_every_field(tmp_path):
    cloud = random_cloud(50, degree=3, seed=1)
    path = write_ply(cloud, tmp_path / "a.ply")
    back = read_ply(path)
    for field in ("means", "log_scales", "quats", "opacities", "sh_dc", "sh_rest"):
        np.testing.assert_array_equal(getattr(back, field), getattr(cloud, field))
    assert back.sh_degree == 3


def test_header_reports_count_and_fields(tmp_path):
    path = write_ply(random_cloud(7, degree=3), tmp_path / "a.ply")
    count, props = read_ply_header(path)
    assert count == 7
    assert "f_rest_44" in props and "rot_3" in props


def test_reads_spirula_style_file_with_extra_fields_and_no_rest(tmp_path):
    names = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "extra_thing")
    arr = np.zeros(4, dtype=[(n, "f4") for n in names])
    arr["x"] = np.arange(4)
    arr["rot_0"] = 1.0
    path = tmp_path / "splat.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    cloud = read_ply(path)

    assert cloud.sh_degree == 0
    assert cloud.sh_rest.shape == (4, 0, 3)
    np.testing.assert_array_equal(cloud.means[:, 0], np.arange(4))


def test_rejects_plain_mesh_ply(tmp_path):
    arr = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    path = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    with pytest.raises(PlyFormatError, match="missing fields"):
        validate_ply(path)


def test_rejects_non_ply_bytes(tmp_path):
    path = tmp_path / "x.ply"
    path.write_bytes(b"hello world")
    with pytest.raises(PlyFormatError):
        validate_ply(path)


def test_rejects_unsupported_rest_count(tmp_path):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] + [f"f_rest_{i}" for i in range(5)]
    arr = np.zeros(2, dtype=[(n, "f4") for n in names])
    path = tmp_path / "odd.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    with pytest.raises(PlyFormatError, match="f_rest"):
        validate_ply(path)


def test_read_ply_does_not_keep_the_file_locked(tmp_path):
    path = write_ply(random_cloud(10, degree=3, seed=2), tmp_path / "a.ply")
    cloud = read_ply(path)
    path.unlink()
    assert not path.exists()
    assert cloud.count == 10

    path = write_ply(random_cloud(10, degree=3, seed=3), tmp_path / "b.ply")
    cloud = read_ply(path)
    write_ply(random_cloud(4, degree=3, seed=4), path)  # overwrite while cloud is alive
    assert cloud.count == 10 and read_ply(path).count == 4


def test_write_ply_handles_empty_cloud(tmp_path):
    empty = random_cloud(0, degree=3)
    path = write_ply(empty, tmp_path / "empty.ply")
    count, props = read_ply_header(path)
    assert count == 0 and "f_rest_44" in props


def _rest_ply(tmp_path, rest_names):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] + rest_names
    arr = np.zeros(2, dtype=[(n, "f4") for n in names])
    path = tmp_path / "rest.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path


def test_rejects_non_numeric_f_rest_suffix(tmp_path):
    path = _rest_ply(tmp_path, [f"f_rest_{i}" for i in range(8)] + ["f_rest_x"])
    with pytest.raises(PlyFormatError, match="f_rest"):
        validate_ply(path)


def test_rejects_gap_in_f_rest_suffixes(tmp_path):
    path = _rest_ply(tmp_path, [f"f_rest_{i}" for i in range(8)] + ["f_rest_10"])
    with pytest.raises(PlyFormatError, match="f_rest"):
        validate_ply(path)
