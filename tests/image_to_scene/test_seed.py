import math
import numpy as np
import pytest
import torch
from backend.image_to_scene.seed import deproject_depth, image_to_pointcloud
from backend.image_to_scene.intrinsics import intrinsics_from_fov


def test_deproject_center_pixel_at_unit_depth():
    K = intrinsics_from_fov(512, 512, 60.0)
    depth = np.ones((512, 512), dtype=np.float32)
    points, valid = deproject_depth(depth, K)
    assert points.shape == (512 * 512, 3)
    # Center pixel maps to (0, 0, 1) — we use +Y up convention
    center_idx = 256 * 512 + 256
    np.testing.assert_allclose(points[center_idx], [0.0, 0.0, 1.0], atol=1e-3)
    assert valid.sum() == 512 * 512


def test_deproject_skips_invalid_depth():
    K = intrinsics_from_fov(8, 8, 60.0)
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[7, 7] = -1.0
    points, valid = deproject_depth(depth, K)
    assert valid.sum() == 8 * 8 - 2


def test_image_to_pointcloud_returns_torch_tensors(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(p)

    K = intrinsics_from_fov(32, 32, 60.0)
    depth = np.ones((32, 32), dtype=np.float32) * 2.0

    points, colors = image_to_pointcloud(p, depth, K)
    assert isinstance(points, torch.Tensor)
    assert isinstance(colors, torch.Tensor)
    assert points.shape == (32 * 32, 3)
    assert colors.shape == (32 * 32, 3)
    assert colors.max() <= 1.0 and colors.min() >= 0.0
    np.testing.assert_allclose(colors[0].numpy(), [200/255, 100/255, 50/255], atol=1e-3)
