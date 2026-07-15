from __future__ import annotations

import gc
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.learned_quality.da3 import DA3Frame, run_anchor_inference


_PINNED_DA3_SOURCE_COMMIT = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
_PINNED_DA3_BASE_REVISION = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"


def _source_checkout_is_pinned(
    source: Path,
    revision: str,
    *,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> bool:
    try:
        head = runner(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = runner(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return head.stdout.strip() == revision and not status.stdout.strip()


def _checkpoint_is_pinned(
    checkpoint: Path,
    revision: str,
    *,
    reader: Callable[[Path, str], object | None] | None = None,
) -> bool:
    if reader is None:
        try:
            from huggingface_hub._local_folder import read_download_metadata
        except ImportError:
            return False
        reader = read_download_metadata
    for filename in ("config.json", "model.safetensors"):
        try:
            metadata = reader(checkpoint, filename)
        except (OSError, ValueError):
            return False
        if metadata is None or getattr(metadata, "commit_hash", None) != revision:
            return False
    return True


def test_gpu_source_authentication_requires_exact_clean_pinned_commit(
    tmp_path: Path,
) -> None:
    pinned = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
    calls: list[list[str]] = []

    def clean_runner(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        stdout = f"{pinned}\n" if argv[-2:] == ["rev-parse", "HEAD"] else ""
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    assert _source_checkout_is_pinned(tmp_path, pinned, runner=clean_runner)
    assert calls == [
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        ["git", "-C", str(tmp_path), "status", "--porcelain"],
    ]

    def wrong_runner(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        stdout = f"{'0' * 40}\n" if argv[-2:] == ["rev-parse", "HEAD"] else ""
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    assert not _source_checkout_is_pinned(tmp_path, pinned, runner=wrong_runner)

    def dirty_runner(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        stdout = f"{pinned}\n" if argv[-2:] == ["rev-parse", "HEAD"] else " M api.py\n"
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    assert not _source_checkout_is_pinned(tmp_path, pinned, runner=dirty_runner)


def test_gpu_checkpoint_authentication_requires_both_local_metadata_revisions(
    tmp_path: Path,
) -> None:
    pinned = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
    calls: list[str] = []

    def pinned_reader(root: Path, filename: str) -> object:
        assert root == tmp_path
        calls.append(filename)
        return SimpleNamespace(commit_hash=pinned)

    assert _checkpoint_is_pinned(tmp_path, pinned, reader=pinned_reader)
    assert calls == ["config.json", "model.safetensors"]

    def mismatched_reader(root: Path, filename: str) -> object:
        revision = pinned if filename == "config.json" else "0" * 40
        return SimpleNamespace(commit_hash=revision)

    assert not _checkpoint_is_pinned(tmp_path, pinned, reader=mismatched_reader)
    assert not _checkpoint_is_pinned(
        tmp_path,
        pinned,
        reader=lambda root, filename: None,
    )


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
    if not _source_checkout_is_pinned(source, _PINNED_DA3_SOURCE_COMMIT):
        pytest.skip("DA3 source must be the exact clean pinned commit")
    if not _checkpoint_is_pinned(checkpoint, _PINNED_DA3_BASE_REVISION):
        pytest.skip("DA3-BASE local metadata must authenticate the pinned revision")
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
