from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.learned_quality.da3 import DA3Frame, run_anchor_inference


def _required_local_assets() -> tuple[Path, Path, tuple[Path, ...]]:
    if os.environ.get("LEARNED_DA3_GPU_TESTS") != "1":
        pytest.skip("set LEARNED_DA3_GPU_TESTS=1 with explicit local assets")
    source = Path(os.environ.get("LEARNED_DA3_SOURCE_CHECKOUT", ""))
    checkpoint = Path(os.environ.get("LEARNED_DA3_BASE_CHECKPOINT", ""))
    raw_images = os.environ.get("LEARNED_DA3_GPU_IMAGES", "")
    images = tuple(Path(value) for value in raw_images.split(os.pathsep) if value)
    if (
        not source.is_absolute()
        or not (source / "src" / "depth_anything_3").is_dir()
        or not checkpoint.is_absolute()
        or not checkpoint.is_dir()
        or not (checkpoint / "config.json").is_file()
        or not (checkpoint / "model.safetensors").is_file()
        or not 3 <= len(images) <= 6
        or any(not path.is_absolute() or not path.is_file() for path in images)
    ):
        pytest.skip("explicit pinned DA3 source/checkpoint and 3-6 local images required")
    return source, checkpoint, images


@pytest.mark.gpu
@pytest.mark.integration
def test_pinned_da3_anchor_gpu_shapes_are_finite_without_downloads(
    tmp_path: Path,
) -> None:
    source, checkpoint, image_paths = _required_local_assets()
    sys.path.insert(0, str(source / "src"))
    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = DepthAnything3.from_pretrained(
        str(checkpoint),
        local_files_only=True,
    ).to("cuda")
    try:
        frames = tuple(
            DA3Frame(
                image_name=path.name,
                frame_id=f"gpu-{index:03d}",
                path=path,
            )
            for index, path in enumerate(image_paths)
        )
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result = run_anchor_inference(
            model,
            frames,
            tmp_path / "anchor",
            vram_gb=total_vram_gb,
            torch_module=torch,
        )

        assert len(result.artifacts) == len(frames)
        shapes = []
        for artifact, camera in zip(result.artifacts, result.cameras):
            depth = np.load(artifact.depth_path, allow_pickle=False)
            assert depth.ndim == 2
            assert np.isfinite(depth).all()
            assert np.all(depth > 0.0)
            shapes.append(depth.shape)
            if artifact.confidence_path is not None:
                confidence = np.load(artifact.confidence_path, allow_pickle=False)
                assert confidence.shape == depth.shape
                assert np.isfinite(confidence).all()
                assert np.all(confidence >= 0.0)
            assert np.asarray(camera.w2c).shape == (4, 4)
            assert np.isfinite(np.asarray(camera.w2c)).all()
        assert len(set(shapes)) == 1
        assert (result.shared_camera.height, result.shared_camera.width) == shapes[0]
    finally:
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
