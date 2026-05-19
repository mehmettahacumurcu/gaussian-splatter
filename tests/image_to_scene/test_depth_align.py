import numpy as np
import pytest
from backend.image_to_scene.depth_align import (
    solve_scale_shift, apply_scale_shift,
)


def test_solve_scale_shift_identity():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    source = target.copy()
    s, b, _ = solve_scale_shift(source, target)
    assert s == pytest.approx(1.0, abs=1e-6)
    assert b == pytest.approx(0.0, abs=1e-6)


def test_solve_scale_shift_recovers_known_transform():
    source = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.5 * source + 0.7
    s, b, _ = solve_scale_shift(source, target)
    assert s == pytest.approx(2.5, abs=1e-6)
    assert b == pytest.approx(0.7, abs=1e-6)


def test_apply_scale_shift_clamps_negative_to_zero():
    src = np.array([-1.0, 0.0, 1.0])
    out = apply_scale_shift(src, scale=1.0, shift=0.0)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 1.0


def test_residual_reported():
    source = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.1, 3.0])
    _, _, residual = solve_scale_shift(source, target)
    assert residual > 0
    assert residual < 0.1


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        solve_scale_shift(np.array([1.0]), np.array([2.0]))
