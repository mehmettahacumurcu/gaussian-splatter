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
