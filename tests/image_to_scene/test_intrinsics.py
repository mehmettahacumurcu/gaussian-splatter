import math
import numpy as np
import pytest
from backend.image_to_scene.intrinsics import (
    intrinsics_from_fov, CameraIntrinsics,
)


def test_intrinsics_square_image_60deg():
    K = intrinsics_from_fov(width=512, height=512, fov_horizontal_deg=60.0)
    assert K.cx == pytest.approx(256.0)
    assert K.cy == pytest.approx(256.0)
    expected_fx = 256.0 / math.tan(math.radians(30.0))
    assert K.fx == pytest.approx(expected_fx, rel=1e-6)
    assert K.fy == pytest.approx(expected_fx, rel=1e-6)


def test_intrinsics_landscape_image():
    K = intrinsics_from_fov(width=800, height=600, fov_horizontal_deg=60.0)
    assert K.cx == pytest.approx(400.0)
    assert K.cy == pytest.approx(300.0)
    expected_fx = 400.0 / math.tan(math.radians(30.0))
    assert K.fx == pytest.approx(expected_fx, rel=1e-6)
    assert K.fy == pytest.approx(expected_fx, rel=1e-6)


def test_intrinsics_as_matrix():
    K = intrinsics_from_fov(width=512, height=512, fov_horizontal_deg=60.0)
    M = K.as_matrix()
    assert M.shape == (3, 3)
    assert M[0, 0] == pytest.approx(K.fx)
    assert M[1, 1] == pytest.approx(K.fy)
    assert M[0, 2] == pytest.approx(K.cx)
    assert M[1, 2] == pytest.approx(K.cy)
    assert M[2, 2] == 1.0


from pathlib import Path
from backend.image_to_scene.intrinsics import (
    intrinsics_for_image, fov_from_exif,
)


def test_fov_from_exif_returns_none_for_no_exif(tmp_path):
    from PIL import Image
    p = tmp_path / "no_exif.png"
    Image.new("RGB", (32, 32)).save(p)
    assert fov_from_exif(p) is None


def test_intrinsics_for_image_falls_back_to_default_fov(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (640, 480)).save(p)
    K = intrinsics_for_image(p, fallback_fov_deg=60.0)
    assert K.width == 640
    assert K.height == 480
    import math
    expected_fx = 320.0 / math.tan(math.radians(30.0))
    assert K.fx == pytest.approx(expected_fx, rel=1e-6)
