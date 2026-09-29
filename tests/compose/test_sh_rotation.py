from __future__ import annotations

import numpy as np
import pytest

from backend.compose.sh_rotation import band_rotation_matrices, eval_sh_color, rotate_sh_rest
from tests.compose.helpers import random_rotation


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_rotated_sh_in_rotated_direction_equals_original(degree):
    rng = np.random.default_rng(degree)
    k = (degree + 1) ** 2 - 1
    for _ in range(5):
        r = random_rotation(rng)
        dc = rng.normal(size=(20, 3))
        rest = rng.normal(size=(20, k, 3))
        dirs = rng.normal(size=(20, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        rotated = rotate_sh_rest(rest, r)

        np.testing.assert_allclose(
            eval_sh_color(dc, rotated, dirs @ r.T), eval_sh_color(dc, rest, dirs), atol=1e-6,
        )


def test_identity_rotation_gives_identity_matrices():
    for l, d in band_rotation_matrices(np.eye(3), 3).items():
        np.testing.assert_allclose(d, np.eye(2 * l + 1), atol=1e-9)


def test_band_matrices_compose_and_are_orthogonal():
    rng = np.random.default_rng(7)
    r1, r2 = random_rotation(rng), random_rotation(rng)
    d1 = band_rotation_matrices(r1, 3)
    d2 = band_rotation_matrices(r2, 3)
    d12 = band_rotation_matrices(r1 @ r2, 3)
    for l in (1, 2, 3):
        np.testing.assert_allclose(d12[l], d1[l] @ d2[l], atol=1e-9)
        np.testing.assert_allclose(d1[l] @ d1[l].T, np.eye(2 * l + 1), atol=1e-9)


def test_degree_zero_is_a_noop():
    rest = np.zeros((3, 0, 3), dtype=np.float32)
    assert rotate_sh_rest(rest, np.eye(3)).shape == (3, 0, 3)
