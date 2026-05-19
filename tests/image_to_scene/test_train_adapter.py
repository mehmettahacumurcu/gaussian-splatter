# tests/image_to_scene/test_train_adapter.py
import numpy as np
import torch
import pytest
from pathlib import Path
from backend.image_to_scene.runner import train_on_generated_views
from backend.image_to_scene.intrinsics import intrinsics_from_fov
from backend.image_to_scene.trajectory import generate_bounded_room_trajectory
from backend.image_to_scene.seed import init_gaussian_model_from_seed


def test_train_on_generated_views_writes_frames_and_calls_trainer(tmp_path, mocker):
    """Verify the adapter writes images to disk, builds per-frame w2c, and calls
    Trainer4DGS.train with the correct kwargs. Does NOT actually train (mocked)."""
    # Mock the Trainer4DGS class entirely.
    mock_trainer_class = mocker.patch("backend.image_to_scene.runner.Trainer4DGS")
    mock_trainer_instance = mock_trainer_class.return_value
    mock_trainer_instance.train.return_value = {"final_iter": 2}

    # Build tiny inputs.
    K = intrinsics_from_fov(64, 64, 60.0)
    poses = generate_bounded_room_trajectory(n_views=4)
    images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in poses]

    points = torch.randn(100, 3) * 0.5 + torch.tensor([0.0, 0.0, 2.0])
    colors = torch.rand(100, 3)
    model = init_gaussian_model_from_seed(points, colors, sh_degree=0)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    stats = train_on_generated_views(
        model=model,
        poses=poses,
        images=images,
        K=K,
        n_iterations=2,
        output_dir=output_dir,
    )

    # Trainer4DGS class was constructed with gs=model + a deform field + static_mode in train()
    assert mock_trainer_class.called
    construct_kwargs = mock_trainer_class.call_args.kwargs
    assert construct_kwargs.get("gs") is model
    assert construct_kwargs.get("deform") is not None  # DeformationField instance

    # train() was called with len(poses) frame_paths, K tensor, w2c per frame
    assert mock_trainer_instance.train.called
    train_kwargs = mock_trainer_instance.train.call_args.kwargs
    assert len(train_kwargs["frame_paths"]) == 4
    assert all(Path(p).exists() for p in train_kwargs["frame_paths"])
    assert isinstance(train_kwargs["cam_K"], torch.Tensor)
    assert train_kwargs["cam_K"].shape == (3, 3)
    assert len(train_kwargs["cam_w2c_per_frame"]) == 4
    for w2c in train_kwargs["cam_w2c_per_frame"]:
        assert w2c.shape == (4, 4)
    assert train_kwargs["n_iters"] == 2
    assert train_kwargs["static_mode"] is True
    assert train_kwargs["image_size"] == (64, 64)

    # Returns a stats dict
    assert "final_iteration" in stats or "final_iter" in stats


def test_train_on_generated_views_handles_zero_iterations(tmp_path, mocker):
    """Even with n_iterations=0 the adapter should still set up + call train."""
    mocker.patch("backend.image_to_scene.runner.Trainer4DGS")
    K = intrinsics_from_fov(32, 32, 60.0)
    poses = generate_bounded_room_trajectory(n_views=2)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in poses]
    points = torch.randn(50, 3)
    colors = torch.rand(50, 3)
    model = init_gaussian_model_from_seed(points, colors, sh_degree=0)

    train_on_generated_views(
        model=model, poses=poses, images=images, K=K,
        n_iterations=0, output_dir=tmp_path,
    )
    # No assertion error means it ran.
