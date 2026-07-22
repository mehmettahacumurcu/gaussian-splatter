from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.da3 import (
    FinalPoseCamera,
    FrameContribution,
    FramePredictionArtifact,
    PinholeCamera,
    PoseConditionedDepthResult,
)
from experiments.learned_quality.depth import (
    MAX_DENSE_SEEDS,
    DenseSeedPolicy,
    DepthValidationResult,
    SupportedDepthCloud,
    ValidatedDepthFrame,
    build_supported_depth_cloud,
    build_supported_depth_cloud_from_base_evidence,
    build_supported_depth_cloud_from_validated_depth,
    validate_depth_and_fuse_seeds,
)
from experiments.learned_quality.masks import (
    FusedMaskFrame,
    MaskFusionEvidence,
    MaskFusionPolicy,
)
from tests.static_pipeline.fixtures import write_colmap_text_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _write_mask(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8) * 255, mode="L").save(
        path,
        format="PNG",
    )
    return _sha256(path)


def _inputs(
    root: Path,
    *,
    frame_count: int = 3,
    depth_values: tuple[np.ndarray, ...] | None = None,
    confidence_values: tuple[np.ndarray, ...] | None = None,
    hard_masks: tuple[np.ndarray, ...] | None = None,
    uncertain_masks: tuple[np.ndarray, ...] | None = None,
    semantic_masks: tuple[np.ndarray, ...] | None = None,
    sky_masks: tuple[np.ndarray, ...] | None = None,
    motion_masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[
    tuple[FrameArtifact, ...],
    PoseConditionedDepthResult,
    MaskFusionEvidence,
    Path,
]:
    shape = (4, 6)
    if depth_values is None:
        depth_values = tuple(
            np.full(shape, 2.0, dtype=np.float32) for _ in range(frame_count)
        )
    if confidence_values is None:
        confidence_values = tuple(
            np.ones(shape, dtype=np.float32) for _ in range(frame_count)
        )
    if hard_masks is None:
        hard_masks = tuple(np.zeros(shape, dtype=bool) for _ in range(frame_count))
    if uncertain_masks is None:
        uncertain_masks = tuple(np.zeros(shape, dtype=bool) for _ in range(frame_count))
    if semantic_masks is None:
        semantic_masks = hard_masks
    if sky_masks is None:
        sky_masks = tuple(np.zeros(shape, dtype=bool) for _ in range(frame_count))
    if motion_masks is None:
        motion_masks = tuple(np.zeros(shape, dtype=bool) for _ in range(frame_count))
    frames = []
    predictions = []
    cameras = []
    fused = []
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        (4.0, 0.0, 2.5),
        (0.0, 4.0, 1.5),
        (0.0, 0.0, 1.0),
    )
    for index in range(frame_count):
        image_name = f"frame_{index:06d}.png"
        frame_id = f"frame-{index:06d}"
        image_path = root / "frames" / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((*shape, 3), 40 + 40 * index, dtype=np.uint8),
            mode="RGB",
        ).save(image_path, format="PNG")
        frame = FrameArtifact(image_name, frame_id, image_path, _sha256(image_path))
        frames.append(frame)

        depth_path = root / "da3" / f"{frame_id}.npy"
        confidence_path = root / "da3" / f"{frame_id}.confidence.npy"
        _write_npy(depth_path, np.asarray(depth_values[index], dtype=np.float32))
        _write_npy(
            confidence_path,
            np.asarray(confidence_values[index], dtype=np.float32),
        )
        predictions.append(
            FramePredictionArtifact(
                image_name=image_name,
                frame_id=frame_id,
                source_path=image_path,
                depth_path=depth_path,
                confidence_path=confidence_path,
                sky_path=None,
            )
        )
        cameras.append(
            FinalPoseCamera(
                image_name=image_name,
                frame_id=frame_id,
                width=shape[1],
                height=shape[0],
                w2c=identity,
                intrinsics=intrinsics,
            )
        )
        hard = hard_masks[index]
        uncertain = uncertain_masks[index]
        maps = {
            "semantic_confirmed": semantic_masks[index],
            "sky_confirmed": sky_masks[index],
            "motion_confirmed": motion_masks[index],
            "uncertain": uncertain,
            "hard_exclude": hard,
            "colmap_keep": ~hard,
            "training_validity": ~hard,
        }
        map_paths = {name: root / "masks" / frame_id / f"{name}.png" for name in maps}
        digests = {
            name: _write_mask(map_paths[name], values) for name, values in maps.items()
        }
        fused.append(
            FusedMaskFrame(
                frame=frame,
                width=shape[1],
                height=shape[0],
                semantic_confirmed_path=map_paths["semantic_confirmed"],
                semantic_confirmed_sha256=digests["semantic_confirmed"],
                sky_confirmed_path=map_paths["sky_confirmed"],
                sky_confirmed_sha256=digests["sky_confirmed"],
                motion_confirmed_path=map_paths["motion_confirmed"],
                motion_confirmed_sha256=digests["motion_confirmed"],
                uncertain_path=map_paths["uncertain"],
                uncertain_sha256=digests["uncertain"],
                hard_exclude_path=map_paths["hard_exclude"],
                hard_exclude_sha256=digests["hard_exclude"],
                colmap_keep_path=map_paths["colmap_keep"],
                colmap_keep_sha256=digests["colmap_keep"],
                training_validity_path=map_paths["training_validity"],
                training_validity_sha256=digests["training_validity"],
                exclusion_fraction_before_trim=float(np.mean(hard)),
                exclusion_fraction_after_trim=float(np.mean(hard)),
                decision="accepted",
                warnings=(),
            )
        )

    metadata_path = root / "da3" / "metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    final_depth = PoseConditionedDepthResult(
        artifacts=tuple(predictions),
        cameras=tuple(cameras),
        processed_camera=PinholeCamera(
            "PINHOLE", shape[1], shape[0], 4.0, 4.0, 2.5, 1.5
        ),
        chunks=(),
        contributions=tuple(
            FrameContribution(frame.image_name, frame.frame_id, (0,))
            for frame in frames
        ),
        attempts=(),
        metadata_path=metadata_path,
    )
    mask_manifest = root / "masks" / "manifest.json"
    mask_manifest.write_text("{}\n", encoding="utf-8")
    masks = MaskFusionEvidence(
        policy=MaskFusionPolicy(0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.0, 0.0, 0.5),
        frames=tuple(fused),
        mask_set_digest="a" * 64,
        manifest_path=mask_manifest,
        manifest_sha256=_sha256(mask_manifest),
    )
    model = root / "model"
    model.mkdir()
    (model / "cameras.txt").write_text("camera\n", encoding="utf-8")
    (model / "images.txt").write_text("images\n", encoding="utf-8")
    (model / "points3D.txt").write_text("sparse-original\n", encoding="utf-8")
    return tuple(frames), final_depth, masks, model


def _policy(**overrides: float | int) -> DenseSeedPolicy:
    values: dict[str, float | int] = {
        "minimum_confidence": 0.5,
        "relative_depth_tolerance": 0.05,
        "invalid_boundary_fraction": 0.0,
        "voxel_size_fraction": 0.01,
        "minimum_neighbor_support": 2,
    }
    values.update(overrides)
    return DenseSeedPolicy(**values)


def test_policy_and_hard_seed_ceiling_are_explicit() -> None:
    assert MAX_DENSE_SEEDS == 1_000_000
    with pytest.raises(ValueError):
        _policy(minimum_confidence=-0.1)
    with pytest.raises(ValueError):
        _policy(relative_depth_tolerance=0.0)
    with pytest.raises(ValueError):
        _policy(minimum_neighbor_support=0)


def test_supported_cloud_preserves_provenance_and_ignores_semantic_masks(
    tmp_path: Path,
) -> None:
    shape = (4, 6)
    semantic = [np.zeros(shape, dtype=bool) for _ in range(3)]
    semantic[0][2, 2] = True
    frames, final_depth, masks, model = _inputs(
        tmp_path / "inputs",
        semantic_masks=tuple(semantic),
    )

    cloud = build_supported_depth_cloud(
        frames,
        final_depth,
        masks,
        model,
        policy=_policy(),
        mask_mode="motion_sky",
    )

    assert isinstance(cloud, SupportedDepthCloud)
    assert cloud.xyz.shape == (shape[0] * shape[1] * 3, 3)
    assert cloud.rgb.shape == cloud.xyz.shape
    assert cloud.source_xy.shape == (len(cloud.xyz), 2)
    assert cloud.source_frame_index.shape == (len(cloud.xyz),)
    assert np.any(
        (cloud.source_frame_index == 0)
        & (cloud.source_xy[:, 0] == 2)
        & (cloud.source_xy[:, 1] == 2)
    )
    assert np.all(cloud.view_support >= 3)
    assert cloud.camera_centers.shape == (3, 3)


def test_supported_cloud_excludes_confirmed_motion_and_sky(tmp_path: Path) -> None:
    shape = (4, 6)
    sky = [np.zeros(shape, dtype=bool) for _ in range(3)]
    motion = [np.zeros(shape, dtype=bool) for _ in range(3)]
    sky[0][0, 1] = True
    motion[0][3, 4] = True
    frames, final_depth, masks, model = _inputs(
        tmp_path / "inputs",
        sky_masks=tuple(sky),
        motion_masks=tuple(motion),
    )

    cloud = build_supported_depth_cloud(
        frames,
        final_depth,
        masks,
        model,
        policy=_policy(),
        mask_mode="motion_sky",
    )

    selected = set(
        zip(
            cloud.source_frame_index.tolist(),
            cloud.source_xy[:, 0].tolist(),
            cloud.source_xy[:, 1].tolist(),
            strict=True,
        )
    )
    assert (0, 1, 0) not in selected
    assert (0, 4, 3) not in selected
    assert (0, 2, 2) in selected


def test_cached_validated_depth_fails_closed_because_semantics_cannot_be_recovered(
    tmp_path: Path,
) -> None:
    shape = (4, 6)
    semantic = [np.zeros(shape, dtype=bool) for _ in range(3)]
    motion = [np.zeros(shape, dtype=bool) for _ in range(3)]
    semantic[0][1, 1] = True
    motion[0][2, 2] = True
    frames, final_depth, masks, _ = _inputs(
        tmp_path / "cached-inputs",
        semantic_masks=tuple(semantic),
        motion_masks=tuple(motion),
    )
    model = write_colmap_text_model(
        tmp_path / "cached-model",
        tuple(frame.image_name for frame in frames),
        (0.2, 0.3, 0.4),
        (2, 2, 2),
    )
    validated_frames = tuple(
        ValidatedDepthFrame(
            frame=frame,
            width=shape[1],
            height=shape[0],
            depth_path=artifact.depth_path,
            depth_sha256=_sha256(artifact.depth_path),
            valid_fraction=1.0,
        )
        for frame, artifact in zip(frames, final_depth.artifacts, strict=True)
    )
    validated = DepthValidationResult(
        policy=_policy(),
        frames=validated_frames,
        dense_seeds=SimpleNamespace(),
        source_model_digest="a" * 64,
        source_mask_digest="b" * 64,
        source_depth_digest="c" * 64,
        source_frame_digest="d" * 64,
    )

    with pytest.raises(ValueError, match="may contain semantic exclusions"):
        build_supported_depth_cloud_from_validated_depth(
            frames,
            validated,
            masks,
            model,
            policy=_policy(),
        )


def test_base_metric_depth_rebuilds_motion_sky_support_without_semantics(
    tmp_path: Path,
) -> None:
    from experiments.learned_quality.milestones import BaseEvidenceState

    shape = (4, 6)
    semantic = [np.zeros(shape, dtype=bool) for _ in range(3)]
    motion = [np.zeros(shape, dtype=bool) for _ in range(3)]
    semantic[0][1, 1] = True
    motion[0][2, 2] = True
    frames, final_depth, masks, _ = _inputs(
        tmp_path / "base-inputs",
        semantic_masks=tuple(semantic),
        motion_masks=tuple(motion),
    )
    model = write_colmap_text_model(
        tmp_path / "base-model",
        tuple(frame.image_name for frame in frames),
        (0.2, 0.3, 0.4),
        (2, 2, 2),
    )
    base = BaseEvidenceState(
        anchors=object(),
        depths=tuple(
            (artifact.depth_path, _sha256(artifact.depth_path))
            for artifact in final_depth.artifacts
        ),
        sky=(),
        scene=object(),
        track_audit=object(),
        colmap_ref="e" * 64,
    )

    cloud = build_supported_depth_cloud_from_base_evidence(
        frames,
        base,
        masks,
        model,
        policy=_policy(),
    )

    selected = set(
        zip(
            cloud.source_frame_index.tolist(),
            cloud.source_xy[:, 0].tolist(),
            cloud.source_xy[:, 1].tolist(),
            strict=True,
        )
    )
    assert (0, 1, 1) in selected
    assert (0, 2, 2) not in selected
    assert np.all(cloud.confidence > 0.0)
    assert np.all(cloud.confidence < 1.0)
    assert cloud.source_depth_digest != "c" * 64


def test_supported_cloud_requires_two_additional_agreeing_views(
    tmp_path: Path,
) -> None:
    depths = (
        np.full((4, 6), 2.0, dtype=np.float32),
        np.full((4, 6), 2.0, dtype=np.float32),
        np.full((4, 6), 9.0, dtype=np.float32),
    )
    frames, final_depth, masks, model = _inputs(
        tmp_path / "inputs",
        depth_values=depths,
    )

    cloud = build_supported_depth_cloud(
        frames,
        final_depth,
        masks,
        model,
        policy=_policy(),
        mask_mode="motion_sky",
    )

    assert cloud.xyz.shape == (0, 3)
    assert cloud.source_xy.shape == (0, 2)


def test_validated_depth_zeroes_low_confidence_masks_and_dilated_boundaries(
    tmp_path: Path,
) -> None:
    shape = (4, 6)
    confidence = [np.ones(shape, dtype=np.float32) for _ in range(3)]
    confidence[0][0, 0] = 0.1
    hard = [np.zeros(shape, dtype=bool) for _ in range(3)]
    uncertain = [np.zeros(shape, dtype=bool) for _ in range(3)]
    hard[0][1, 2] = True
    uncertain[0][3, 5] = True
    frames, final_depth, masks, model = _inputs(
        tmp_path / "inputs",
        confidence_values=tuple(confidence),
        hard_masks=tuple(hard),
        uncertain_masks=tuple(uncertain),
    )

    result = validate_depth_and_fuse_seeds(
        frames,
        final_depth,
        masks,
        model,
        tmp_path / "validated",
        policy=_policy(invalid_boundary_fraction=0.17),
    )

    first = np.load(result.frames[0].depth_path, allow_pickle=False)
    assert result.frames[0].depth_path.name == "frame_000000_depth.npy"
    assert first.dtype == np.float32
    assert first[0, 0] == 0.0
    assert first[1, 2] == 0.0
    assert first[3, 5] == 0.0
    assert first[1, 1] == 0.0
    assert np.any(first > 0.0)


def test_two_neighbor_support_fuses_deterministic_typed_dense_seeds(
    tmp_path: Path,
) -> None:
    frames, final_depth, masks, model = _inputs(tmp_path / "inputs")
    sparse_before = (model / "points3D.txt").read_bytes()

    first = validate_depth_and_fuse_seeds(
        frames,
        final_depth,
        masks,
        model,
        tmp_path / "first",
        policy=_policy(),
    )
    validate_depth_and_fuse_seeds(
        frames,
        final_depth,
        masks,
        model,
        tmp_path / "second",
        policy=_policy(),
    )

    with np.load(first.dense_seeds.npz_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"xyz", "rgb", "confidence", "view_support"}
        assert archive["xyz"].dtype == np.float32
        assert archive["xyz"].shape[1:] == (3,)
        assert archive["rgb"].dtype == np.uint8
        assert archive["rgb"].shape == archive["xyz"].shape
        assert archive["confidence"].dtype == np.float32
        assert archive["view_support"].dtype == np.uint16
        assert len(archive["xyz"]) > 0
        assert np.all(archive["view_support"] >= 3)
        assert np.all(archive["rgb"] == 80)
    assert first.dense_seeds.point_count <= MAX_DENSE_SEEDS
    assert (model / "points3D.txt").read_bytes() == sparse_before

    def inventory(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert inventory(tmp_path / "first") == inventory(tmp_path / "second")


def test_depth_without_two_neighbor_agreements_never_becomes_dense_seed(
    tmp_path: Path,
) -> None:
    depths = (
        np.full((4, 6), 2.0, dtype=np.float32),
        np.full((4, 6), 5.0, dtype=np.float32),
        np.full((4, 6), 9.0, dtype=np.float32),
    )
    frames, final_depth, masks, model = _inputs(
        tmp_path / "inputs",
        depth_values=depths,
    )

    result = validate_depth_and_fuse_seeds(
        frames,
        final_depth,
        masks,
        model,
        tmp_path / "validated",
        policy=_policy(),
    )

    assert all(
        not np.any(np.load(frame.depth_path, allow_pickle=False))
        for frame in result.frames
    )
    with np.load(result.dense_seeds.npz_path, allow_pickle=False) as archive:
        assert archive["xyz"].shape == (0, 3)
        assert archive["rgb"].shape == (0, 3)
        assert archive["confidence"].shape == (0,)
        assert archive["view_support"].shape == (0,)


def test_exact_join_and_digest_faults_fail_before_output(tmp_path: Path) -> None:
    frames, final_depth, masks, model = _inputs(tmp_path / "inputs")
    bad_frames = (dataclasses.replace(frames[0], sha256="f" * 64), *frames[1:])
    output = tmp_path / "validated"

    with pytest.raises(ValueError):
        validate_depth_and_fuse_seeds(
            bad_frames,
            final_depth,
            masks,
            model,
            output,
            policy=_policy(),
        )

    assert not output.exists()


def test_supported_depth_provenance_cannot_exceed_camera_count() -> None:
    with pytest.raises(ValueError, match="frame index exceeds camera count"):
        SupportedDepthCloud(
            xyz=np.zeros((1, 3), dtype=np.float32),
            rgb=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones((1,), dtype=np.float32),
            view_support=np.ones((1,), dtype=np.uint16),
            source_frame_index=np.ones((1,), dtype=np.int32),
            source_xy=np.zeros((1, 2), dtype=np.int32),
            camera_centers=np.zeros((1, 3), dtype=np.float32),
            source_model_digest="a" * 64,
            source_mask_digest="b" * 64,
            source_depth_digest="c" * 64,
            source_frame_digest="d" * 64,
        )
