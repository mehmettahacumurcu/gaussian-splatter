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


def test_rotated_scene_matches_rotated_camera():
    torch = pytest.importorskip("torch")
    gsplat = pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cloud = random_cloud(3000, degree=3, seed=11)
    cloud.means *= 0.5
    cloud.opacities[:] = 2.0
    q = np.array([0.2, -0.4, 0.3, 0.84])
    q /= np.linalg.norm(q)
    r = quat_to_matrix(tuple(q))
    view = np.eye(4)
    view[2, 3] = 4.0  # camera 4 units in front of the cloud
    view_rot = view @ np.block([[r.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])

    try:
        original = _render(torch, gsplat.rasterization, cloud, view)
    except subprocess.CalledProcessError as exc:  # gsplat's CUDA JIT needs MSVC (cl) on Windows
        pytest.skip(f"gsplat CUDA extension could not be compiled: {exc}")
    rotated = _render(torch, gsplat.rasterization, transform_cloud(cloud, Placement(quaternion_xyzw=tuple(q))), view_rot)

    assert np.abs(original - rotated).max() < 2e-3
