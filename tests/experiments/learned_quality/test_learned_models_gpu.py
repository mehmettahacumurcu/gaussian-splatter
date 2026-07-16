from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.flow import FlowPairRequest
from experiments.learned_quality.lifecycle import release_cuda_model
from experiments.learned_quality.model_adapters import (
    Sam2ImageAdapter,
    SeaRaftTorchAdapter,
    TransformersGroundingDinoAdapter,
)
from experiments.learned_quality.segmentation import TRANSIENT_PROMPT, SamBoxPrompt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assets():
    if os.environ.get("LEARNED_MODELS_GPU_TESTS") != "1":
        pytest.skip("set LEARNED_MODELS_GPU_TESTS=1 with pinned local assets")
    source_root = Path(os.environ["LEARNED_SOURCE_ROOT"])
    checkpoint_root = Path(os.environ["LEARNED_CHECKPOINT_ROOT"])
    images = tuple(
        Path(value)
        for value in os.environ.get("LEARNED_GPU_IMAGES", "").split(os.pathsep)
        if value
    )
    if not 3 <= len(images) <= 6 or any(not image.is_file() for image in images):
        pytest.skip("three to six same-size local images are required")
    frames = tuple(
        FrameArtifact(path.name, f"gpu-{index:03d}", path, _sha256(path))
        for index, path in enumerate(images)
    )
    return source_root, checkpoint_root, frames


@pytest.mark.gpu
@pytest.mark.integration
def test_pinned_semantic_and_flow_adapters_join_finite_outputs() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source_root, checkpoint_root, frames = _assets()
    dino = TransformersGroundingDinoAdapter(
        checkpoint_root / "IDEA-Research--grounding-dino-tiny"
    )
    detections = dino.detect_batch(
        frames,
        prompt=TRANSIENT_PROMPT,
        box_threshold=0.30,
        text_threshold=0.25,
    )
    assert tuple(detections) == tuple(frame.frame_id for frame in frames)
    release_cuda_model(dino)
    del dino

    sam = Sam2ImageAdapter(
        source_root / "sam2",
        checkpoint_root / "facebook--sam2.1-hiera-large",
    )
    prompts = tuple(
        SamBoxPrompt(frame, box)
        for frame in frames
        for box in detections[frame.frame_id][:1]
    )
    masks = sam.refine_batch(prompts)
    assert tuple(masks) == tuple(
        (prompt.frame.frame_id, prompt.box.box_id) for prompt in prompts
    )
    assert all(np.asarray(mask).dtype == np.bool_ for mask in masks.values())
    release_cuda_model(sam)
    del sam

    raft = SeaRaftTorchAdapter(
        source_root / "sea-raft",
        checkpoint_root / "MemorySlices--Tartan-C-T-TSKH-spring540x960-M",
    )
    pairs = tuple(
        FlowPairRequest(
            index,
            source.frame_id,
            target.frame_id,
            source.path,
            target.path,
        )
        for index, (source, target) in enumerate(zip(frames, frames[1:]))
    )
    predictions = raft.infer_bidirectional(pairs, batch_size=2)
    assert tuple(
        (prediction.source_frame_id, prediction.target_frame_id)
        for prediction in predictions
    ) == tuple((pair.source_frame_id, pair.target_frame_id) for pair in pairs)
    for prediction in predictions:
        assert prediction.forward_flow.shape[-1] == 2
        assert prediction.backward_flow.shape == prediction.forward_flow.shape
        assert np.isfinite(prediction.forward_flow).all()
        assert np.isfinite(prediction.forward_uncertainty).all()
    release_cuda_model(raft)
