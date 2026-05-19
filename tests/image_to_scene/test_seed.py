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


from backend.image_to_scene.seed import init_gaussian_model_from_seed


def test_init_gaussian_model_has_correct_count():
    points = torch.randn(1000, 3)
    colors = torch.rand(1000, 3)
    model = init_gaussian_model_from_seed(points, colors, sh_degree=0)
    assert model.means.shape == (1000, 3)


def test_init_gaussian_model_handles_empty_input():
    with pytest.raises(ValueError):
        init_gaussian_model_from_seed(torch.empty((0, 3)), torch.empty((0, 3)))


def test_init_gaussian_model_subsamples_if_too_many():
    points = torch.randn(2_000_000, 3)
    colors = torch.rand(2_000_000, 3)
    model = init_gaussian_model_from_seed(points, colors, max_points=500_000)
    assert model.means.shape == (500_000, 3)


@pytest.mark.integration
def test_seed_end_to_end_tiny_image(tmp_path):
    """Full seed pipeline: synthetic image → fake depth → point cloud → GaussianModel."""
    from PIL import Image
    img_path = tmp_path / "tiny.png"
    Image.new("RGB", (64, 48), color=(128, 128, 128)).save(img_path)

    K = intrinsics_from_fov(64, 48, 60.0)
    depth = np.ones((48, 64), dtype=np.float32) * 2.5

    points, colors = image_to_pointcloud(img_path, depth, K)
    model = init_gaussian_model_from_seed(points, colors)

    assert model.means.shape[0] == 64 * 48
    z_values = model.means[:, 2]
    assert torch.allclose(z_values, torch.tensor(2.5), atol=1e-3)
