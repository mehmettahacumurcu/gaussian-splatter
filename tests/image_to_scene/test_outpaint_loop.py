"""Tests for the iterative outpaint loop (Phase 6)."""
import math
import numpy as np
import pytest
import torch
from backend.image_to_scene.outpaint_loop import render_pose
from backend.image_to_scene.trajectory import generate_bounded_room_trajectory
from backend.image_to_scene.intrinsics import intrinsics_from_fov
from backend.image_to_scene.seed import init_gaussian_model_from_seed


@pytest.mark.gpu
def test_render_pose_returns_rgb_alpha_depth():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Seed a tiny model in front of camera (z > 0).
    points = torch.randn(1000, 3) * 0.5 + torch.tensor([0.0, 0.0, 2.0])
    colors = torch.rand(1000, 3)
    model = init_gaussian_model_from_seed(points, colors, sh_degree=0).cuda()

    pose = generate_bounded_room_trajectory(n_views=1)[0]
    K = intrinsics_from_fov(64, 64, 60.0)

    rgb, alpha, depth = render_pose(model, pose, K)
    assert rgb.shape == (64, 64, 3)
    assert alpha.shape == (64, 64)
    assert depth.shape == (64, 64)
    assert rgb.dtype == np.float32
    assert alpha.max() > 0.0
