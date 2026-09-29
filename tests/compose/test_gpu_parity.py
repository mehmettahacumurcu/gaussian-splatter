from __future__ import annotations

import subprocess

import numpy as np
import pytest

from backend.compose.bake import Placement, quat_to_matrix, transform_cloud
from tests.compose.helpers import random_cloud

pytestmark = pytest.mark.gpu


def _render(torch, rasterization, cloud, viewmat):
    dev = "cuda"
    means = torch.tensor(cloud.means, device=dev)
    quats = torch.tensor(cloud.quats, device=dev)
    scales = torch.exp(torch.tensor(cloud.log_scales, device=dev))
    opac = torch.sigmoid(torch.tensor(cloud.opacities, device=dev))
    sh = torch.cat([torch.tensor(cloud.sh_dc, device=dev)[:, None], torch.tensor(cloud.sh_rest, device=dev)], 1)
    k = torch.tensor([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]], device=dev)[None]
    img, _, _ = rasterization(means, quats, scales, opac, sh,
                              torch.tensor(viewmat, dtype=torch.float32, device=dev)[None], k, 128, 128,
                              sh_degree=3)
    return img[0].cpu().numpy()


@pytest.fixture(scope="module")
def gpu():
    """(torch, rasterization) if a tiny CUDA render works here, otherwise skip with the reason."""
    try:
        import torch
        import gsplat

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        probe_view = np.eye(4)
        probe_view[2, 3] = 4.0
        _render(torch, gsplat.rasterization, random_cloud(1, degree=3, seed=0), probe_view)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ImportError) as exc:
        # gsplat's CUDA JIT needs MSVC (cl) on Windows; a missing toolchain is not a test failure.
        pytest.skip(f"gsplat CUDA rendering unavailable: {type(exc).__name__}: {exc}")
    return torch, gsplat.rasterization


def test_rotated_scene_matches_rotated_camera(gpu):
    torch, rasterization = gpu
    cloud = random_cloud(3000, degree=3, seed=11)
    cloud.means *= 0.5
    cloud.opacities[:] = 2.0
    q = np.array([0.2, -0.4, 0.3, 0.84])
    q /= np.linalg.norm(q)
    r = quat_to_matrix(tuple(q))
    view = np.eye(4)
    view[2, 3] = 4.0  # camera 4 units in front of the cloud
    view_rot = view @ np.block([[r.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    rotated_cloud = transform_cloud(cloud, Placement(quaternion_xyzw=tuple(q)))

    original = _render(torch, rasterization, cloud, view)
    rotated = _render(torch, rasterization, rotated_cloud, view_rot)
    unrotated_view = _render(torch, rasterization, rotated_cloud, view)

    assert original.max() > 0.1  # the render is not trivially empty
    assert np.abs(original - unrotated_view).max() > 0.05  # the rotation actually matters
    assert np.abs(original - rotated).max() < 2e-3
