from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.flow import (
    FlowPairRequest,
    FlowGatePolicy,
    MotionEvidence,
    MotionFrameEvidence,
    RigidFrameEvidence,
    RigidSceneEvidence,
    StaticTrack,
    TrackObservation,
    SeaRaftPairPrediction,
    SEA_RAFT_MODEL_REF,
    run_motion_evidence,
)
from experiments.learned_quality.lifecycle import VramReleaseRecord
from experiments.learned_quality.segmentation import (
    SemanticEvidence,
    SemanticFrameEvidence,
    SemanticPolicy,
    TRANSIENT_PROMPT,
)


def _masks_module():
    try:
        return importlib.import_module("experiments.learned_quality.masks")
    except ModuleNotFoundError as error:
        pytest.fail(f"mask fusion module is missing: {error}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_binary_png(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return _sha256(path)


def _write_npy(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return _sha256(path)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    )
    return _sha256(path)


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _frame_set_digest(frames: tuple[FrameArtifact, ...]) -> str:
    return hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "image_name": frame.image_name,
                        "sha256": frame.sha256,
                        "size_bytes": frame.path.stat().st_size,
                    }
                    for frame in frames
                ],
                "schema": "learned_quality.input_frames.v1",
            }
        )
    ).hexdigest()


def _scene_digest(frames: tuple[FrameArtifact, ...], scene: RigidSceneEvidence) -> str:
    rigid_payload = [
        {
            "depth_sha256": rigid.depth_sha256,
            "depth_size_bytes": (
                None if rigid.depth_path is None else rigid.depth_path.stat().st_size
            ),
            "frame_id": frame.frame_id,
            "height": rigid.height,
            "image_name": frame.image_name,
            "pinhole_fx_fy_cx_cy": (
                None
                if rigid.pinhole_fx_fy_cx_cy is None
                else [float(value) for value in rigid.pinhole_fx_fy_cx_cy]
            ),
            "registered": rigid.registered,
            "w2c_4x4": (
                None
                if rigid.w2c_4x4 is None
                else [float(value) for value in rigid.w2c_4x4]
            ),
            "width": rigid.width,
        }
        for frame, rigid in zip(frames, scene.frames, strict=True)
    ]
    tracks_payload = [
        {
            "mean_reprojection_error": float(track.mean_reprojection_error),
            "observations": [
                {
                    "frame_id": observation.frame_id,
                    "x": float(observation.x),
                    "y": float(observation.y),
                }
                for observation in track.observations
            ],
            "track_id": track.track_id,
            "xyz": [float(value) for value in track.xyz],
        }
        for track in scene.static_tracks
    ]
    return hashlib.sha256(
        _strict_json_bytes(
            {
                "depth_digest": scene.depth_digest,
                "frames": rigid_payload,
                "geometry_digest": scene.geometry_digest,
                "input_frame_digest": _frame_set_digest(frames),
                "static_tracks": tracks_payload,
            }
        )
    ).hexdigest()


def _mask_policy(**overrides: float):
    masks = _masks_module()
    values = {
        "boundary_dilation_fraction": 0.0,
        "opening_fraction": 0.0,
        "minimum_component_area_fraction": 0.0,
        "temporal_area_change_limit": 1.0,
        "propagation_collapse_limit": 1.0,
        "sky_far_depth_quantile": 0.75,
        "sky_track_support_radius_fraction": 0.0,
        "sky_max_component_support_fraction": 0.0,
        "sky_frame_min_fraction": 0.5,
    }
    values.update(overrides)
    return masks.MaskFusionPolicy(**values)


def _single_frame_inputs(
    root: Path,
    *,
    direct: np.ndarray,
    propagated: np.ndarray,
    direct_support: np.ndarray,
    motion_confirmed: np.ndarray,
    motion_requires_semantic: np.ndarray,
    motion_uncertain: np.ndarray,
    strength: np.ndarray,
    sky: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    image_name: str = "frame_000000.png",
    frame_id: str = "frame-000000",
    static_tracks: tuple[StaticTrack, ...] = (),
) -> tuple[
    tuple[FrameArtifact, ...],
    SemanticEvidence,
    MotionEvidence,
    tuple[FrameArtifact, ...],
    RigidSceneEvidence,
]:
    shape = direct.shape
    assert all(
        array.shape == shape
        for array in (
            propagated,
            direct_support,
            motion_confirmed,
            motion_requires_semantic,
            motion_uncertain,
            strength,
        )
    )
    height, width = shape
    source_path = root / "source" / Path(*image_name.split("/"))
    source_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB").save(
        source_path,
        format="PNG",
    )
    frame = FrameArtifact(
        image_name=image_name,
        frame_id=frame_id,
        path=source_path,
        sha256=_sha256(source_path),
    )
    frames = (frame,)

    semantic_root = root / "semantic"
    semantic_frame = SemanticFrameEvidence(
        frame=frame,
        width=width,
        height=height,
        direct_confirmed_path=semantic_root / "direct.png",
        direct_confirmed_sha256=_write_binary_png(semantic_root / "direct.png", direct),
        propagated_candidate_path=semantic_root / "propagated.png",
        propagated_candidate_sha256=_write_binary_png(
            semantic_root / "propagated.png", propagated
        ),
        direct_support_path=semantic_root / "support.png",
        direct_support_sha256=_write_binary_png(
            semantic_root / "support.png", direct_support
        ),
    )
    semantic_manifest = semantic_root / "manifest.json"
    semantic = SemanticEvidence(
        policy=SemanticPolicy(0.2, 0.2, 3, 1),
        prompt=TRANSIENT_PROMPT,
        frames=(semantic_frame,),
        stage_records=(),
        manifest_path=semantic_manifest,
        manifest_sha256=_write_json(semantic_manifest, {"stage": "semantic"}),
    )

    motion_root = root / "motion"
    motion_frame = MotionFrameEvidence(
        frame=frame,
        width=width,
        height=height,
        confirmed_without_semantic_path=motion_root / "confirmed.png",
        confirmed_without_semantic_sha256=_write_binary_png(
            motion_root / "confirmed.png", motion_confirmed
        ),
        requires_semantic_path=motion_root / "requires_semantic.png",
        requires_semantic_sha256=_write_binary_png(
            motion_root / "requires_semantic.png", motion_requires_semantic
        ),
        uncertain_path=motion_root / "uncertain.png",
        uncertain_sha256=_write_binary_png(
            motion_root / "uncertain.png", motion_uncertain
        ),
        strength_path=motion_root / "strength.npy",
        strength_sha256=_write_npy(
            motion_root / "strength.npy", np.asarray(strength, dtype=np.float32)
        ),
    )
    pair_manifest = motion_root / "pairs.json"
    motion_manifest = motion_root / "manifest.json"
    geometry_digest = "1" * 64
    depth_digest = "2" * 64
    pair_manifest_sha256 = _write_json(pair_manifest, {"stage": "pairs"})
    motion_manifest_sha256 = _write_json(motion_manifest, {"stage": "motion"})

    sky_value = np.zeros(shape, dtype=bool) if sky is None else np.asarray(sky)
    sky_path = root / "da3" / "sky.npy"
    sky_frame = FrameArtifact(
        image_name=frame.image_name,
        frame_id=frame.frame_id,
        path=sky_path,
        sha256=_write_npy(sky_path, sky_value),
    )

    depth_value = (
        np.ones(shape, dtype=np.float32)
        if depth is None
        else np.asarray(depth, dtype=np.float32)
    )
    depth_path = root / "rigid" / "depth.npy"
    rigid_frame = RigidFrameEvidence(
        frame=frame,
        width=width,
        height=height,
        registered=True,
        w2c_4x4=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        pinhole_fx_fy_cx_cy=(1.0, 1.0, 0.0, 0.0),
        depth_path=depth_path,
        depth_sha256=_write_npy(depth_path, depth_value),
    )
    scene = RigidSceneEvidence(
        frames=(rigid_frame,),
        static_tracks=static_tracks,
        geometry_digest=geometry_digest,
        depth_digest=depth_digest,
    )
    motion = MotionEvidence(
        policy=FlowGatePolicy(0.1, 0.0, 1.0, 0.1, 3.0, 2, 1.0, 3.0),
        scene_digest=_scene_digest(frames, scene),
        frames=(motion_frame,),
        stage_records=(),
        pair_manifest_path=pair_manifest,
        pair_manifest_sha256=pair_manifest_sha256,
        manifest_path=motion_manifest,
        manifest_sha256=motion_manifest_sha256,
    )
    return frames, semantic, motion, (sky_frame,), scene


def _combine_inputs(
    root: Path,
    inputs: tuple[
        tuple[
            tuple[FrameArtifact, ...],
            SemanticEvidence,
            MotionEvidence,
            tuple[FrameArtifact, ...],
            RigidSceneEvidence,
        ],
        ...,
    ],
) -> tuple[
    tuple[FrameArtifact, ...],
    SemanticEvidence,
    MotionEvidence,
    tuple[FrameArtifact, ...],
    RigidSceneEvidence,
]:
    frames = tuple(item[0][0] for item in inputs)
    semantic_manifest = root / "semantic_manifest.json"
    semantic = SemanticEvidence(
        policy=inputs[0][1].policy,
        prompt=inputs[0][1].prompt,
        frames=tuple(item[1].frames[0] for item in inputs),
        stage_records=(),
        manifest_path=semantic_manifest,
        manifest_sha256=_write_json(semantic_manifest, {"stage": "semantic"}),
    )
    geometry_digest = inputs[0][4].geometry_digest
    depth_digest = inputs[0][4].depth_digest
    pair_manifest = root / "pair_manifest.json"
    motion_manifest = root / "motion_manifest.json"
    pair_manifest_sha256 = _write_json(pair_manifest, {"stage": "pairs"})
    motion_manifest_sha256 = _write_json(motion_manifest, {"stage": "motion"})
    sky = tuple(item[3][0] for item in inputs)
    scene = RigidSceneEvidence(
        frames=tuple(item[4].frames[0] for item in inputs),
        static_tracks=tuple(
            track for item in inputs for track in item[4].static_tracks
        ),
        geometry_digest=geometry_digest,
        depth_digest=depth_digest,
    )
    motion = MotionEvidence(
        policy=inputs[0][2].policy,
        scene_digest=_scene_digest(frames, scene),
        frames=tuple(item[2].frames[0] for item in inputs),
        stage_records=(),
        pair_manifest_path=pair_manifest,
        pair_manifest_sha256=pair_manifest_sha256,
        manifest_path=motion_manifest,
        manifest_sha256=motion_manifest_sha256,
    )
    return frames, semantic, motion, sky, scene


def _load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        return np.asarray(image, dtype=np.uint8) == 255


def _empty_inputs(root: Path, count: int = 2):
    shape = (4, 5)
    empty = np.zeros(shape, dtype=bool)
    items = tuple(
        _single_frame_inputs(
            root / f"input-{index}",
            direct=empty,
            propagated=empty,
            direct_support=empty,
            motion_confirmed=empty,
            motion_requires_semantic=empty,
            motion_uncertain=empty,
            strength=np.zeros(shape, dtype=np.float32),
            image_name=f"nested/frame_{index:06d}.png",
            frame_id=f"frame-{index:06d}",
        )
        for index in range(count)
    )
    return _combine_inputs(root / "combined", items)


class _DigestBindingSeaRaft:
    model_id = SEA_RAFT_MODEL_REF.repo_id
    revision = SEA_RAFT_MODEL_REF.revision
    code_commit = SEA_RAFT_MODEL_REF.code_commit

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape

    def infer_bidirectional(
        self,
        pairs: tuple[FlowPairRequest, ...],
        *,
        batch_size: int,
    ) -> tuple[SeaRaftPairPrediction, ...]:
        del batch_size
        height, width = self.shape
        zero_flow = np.zeros((height, width, 2), dtype=np.float32)
        zero_uncertainty = np.zeros((height, width), dtype=np.float32)
        return tuple(
            SeaRaftPairPrediction(
                source_frame_id=pair.source_frame_id,
                target_frame_id=pair.target_frame_id,
                forward_flow=zero_flow,
                backward_flow=zero_flow,
                forward_uncertainty=zero_uncertainty,
                backward_uncertainty=zero_uncertainty,
            )
            for pair in pairs
        )


def _release_record(_model: object) -> VramReleaseRecord:
    return VramReleaseRecord(
        cuda_available=False,
        allocated_before_bytes=None,
        reserved_before_bytes=None,
        allocated_after_bytes=None,
        reserved_after_bytes=None,
        moved_to_cpu=True,
        gc_ran=True,
        cache_cleared=False,
    )


def test_public_mask_fusion_contract_is_frozen_and_has_no_policy_defaults() -> None:
    masks = _masks_module()

    assert masks.RUNAWAY_TRIM_FRACTION == 0.45
    assert masks.RUNAWAY_REJECT_FRACTION == 0.55
    assert tuple(
        field.name for field in dataclasses.fields(masks.MaskFusionPolicy)
    ) == (
        "boundary_dilation_fraction",
        "opening_fraction",
        "minimum_component_area_fraction",
        "temporal_area_change_limit",
        "propagation_collapse_limit",
        "sky_far_depth_quantile",
        "sky_track_support_radius_fraction",
        "sky_max_component_support_fraction",
        "sky_frame_min_fraction",
    )
    assert all(
        field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        for field in dataclasses.fields(masks.MaskFusionPolicy)
    )
    assert masks.MaskFusionPolicy.__dataclass_params__.frozen
    assert masks.FusedMaskFrame.__dataclass_params__.frozen
    assert masks.MaskFusionEvidence.__dataclass_params__.frozen
    assert tuple(inspect.signature(masks.fuse_evidence_masks).parameters) == (
        "frames",
        "semantic",
        "motion",
        "sky_proposals",
        "scene",
        "output_dir",
        "policy",
    )


def test_mask_policy_rejects_non_finite_boolean_and_out_of_range_values() -> None:
    masks = _masks_module()
    valid = dataclasses.asdict(_mask_policy())
    for field_name in valid:
        values = dict(valid)
        values[field_name] = float("nan")
        with pytest.raises(ValueError, match=field_name):
            masks.MaskFusionPolicy(**values)
    for field_name in valid:
        values = dict(valid)
        values[field_name] = True
        with pytest.raises(ValueError, match=field_name):
            masks.MaskFusionPolicy(**values)
    for field_name in valid:
        values = dict(valid)
        values[field_name] = -0.01
        with pytest.raises(ValueError, match=field_name):
            masks.MaskFusionPolicy(**values)
    for field_name in valid:
        values = dict(valid)
        values[field_name] = 1.01
        with pytest.raises(ValueError, match=field_name):
            masks.MaskFusionPolicy(**values)
    values = dict(valid)
    values["sky_frame_min_fraction"] = 0.0
    with pytest.raises(ValueError, match="sky_frame_min_fraction"):
        masks.MaskFusionPolicy(**values)


def test_fusion_promotes_only_supported_evidence_and_writes_exact_polarities(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (10, 10)
    direct = np.zeros(shape, dtype=bool)
    propagated = np.zeros(shape, dtype=bool)
    support = np.zeros(shape, dtype=bool)
    confirmed = np.zeros(shape, dtype=bool)
    requires = np.zeros(shape, dtype=bool)
    producer_uncertain = np.zeros(shape, dtype=bool)
    strength = np.zeros(shape, dtype=np.float32)
    direct[1, 1] = True
    propagated[2, 2] = True
    support[2, 2] = True
    propagated[3, 3] = True
    confirmed[4, 4] = True
    propagated[4, 4] = True
    requires[1, 1] = True
    requires[5, 5] = True
    producer_uncertain[6, 6] = True
    inputs = _single_frame_inputs(
        tmp_path,
        direct=direct,
        propagated=propagated,
        direct_support=support,
        motion_confirmed=confirmed,
        motion_requires_semantic=requires,
        motion_uncertain=producer_uncertain,
        strength=strength,
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    frame = result.frames[0]
    semantic = _load_binary(frame.semantic_confirmed_path)
    motion = _load_binary(frame.motion_confirmed_path)
    sky = _load_binary(frame.sky_confirmed_path)
    uncertain = _load_binary(frame.uncertain_path)
    hard = _load_binary(frame.hard_exclude_path)
    colmap_keep = _load_binary(frame.colmap_keep_path)
    training_validity = _load_binary(frame.training_validity_path)
    assert semantic[1, 1]
    assert semantic[2, 2]
    assert semantic[4, 4]
    assert not semantic[3, 3]
    assert motion[4, 4]
    assert motion[1, 1]
    assert not motion[5, 5]
    assert uncertain[3, 3]
    assert uncertain[5, 5]
    assert uncertain[6, 6]
    assert not sky.any()
    np.testing.assert_array_equal(hard, semantic | sky | motion)
    np.testing.assert_array_equal(colmap_keep, ~hard)
    np.testing.assert_array_equal(training_validity, ~hard)
    assert frame.colmap_keep_path.relative_to(tmp_path / "fused").as_posix() == (
        "colmap_masks/frame_000000.png.png"
    )
    assert Image.open(frame.colmap_keep_path).size == (shape[1], shape[0])
    normalized_validity = (
        np.asarray(Image.open(frame.training_validity_path), dtype=np.float32) / 255.0
    )
    assert normalized_validity[0, 0] == 1.0
    assert normalized_validity[1, 1] == 0.0


def test_sky_confirms_only_far_low_support_components_and_demotes_ceiling_shape(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (8, 8)
    empty = np.zeros(shape, dtype=bool)
    proposal = np.zeros(shape, dtype=bool)
    proposal[0:2, 3:5] = True  # near ceiling-shaped proposal
    proposal[6:8, 0:2] = True  # far but supported stable structure
    proposal[6:8, 6:8] = True  # far and unsupported
    depth = np.arange(1, 65, dtype=np.float32).reshape(shape)
    track = StaticTrack(
        track_id=7,
        xyz=(0.0, 0.0, 1.0),
        mean_reprojection_error=0.1,
        observations=(TrackObservation("frame-000000", 0.0, 6.0),),
    )
    inputs = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        sky=proposal,
        depth=depth,
        static_tracks=(track,),
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(
            sky_far_depth_quantile=0.75,
            sky_track_support_radius_fraction=0.0,
            sky_max_component_support_fraction=0.0,
            sky_frame_min_fraction=0.9,
        ),
    )

    sky_confirmed = _load_binary(result.frames[0].sky_confirmed_path)
    uncertain = _load_binary(result.frames[0].uncertain_path)
    assert not sky_confirmed[0:2, 3:5].any()
    assert uncertain[0:2, 3:5].all()
    assert not sky_confirmed[6:8, 0:2].any()
    assert uncertain[6:8, 0:2].all()
    assert sky_confirmed[6:8, 6:8].all()
    assert not uncertain[6:8, 6:8].any()


def test_morphology_demotes_candidate_pixels_but_protects_direct_semantics(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (20, 20)
    direct = np.zeros(shape, dtype=bool)
    propagated = np.zeros(shape, dtype=bool)
    support = np.zeros(shape, dtype=bool)
    motion = np.zeros(shape, dtype=bool)
    direct[1, 1] = True
    propagated[5, 5] = True
    support[5, 5] = True
    propagated[10:13, 10:13] = True
    support[10:13, 10:13] = True
    motion[15, 15] = True
    inputs = _single_frame_inputs(
        tmp_path,
        direct=direct,
        propagated=propagated,
        direct_support=support,
        motion_confirmed=motion,
        motion_requires_semantic=np.zeros(shape, dtype=bool),
        motion_uncertain=np.zeros(shape, dtype=bool),
        strength=np.ones(shape, dtype=np.float32),
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(
            opening_fraction=0.05,
            minimum_component_area_fraction=0.01,
            boundary_dilation_fraction=0.05,
        ),
    )

    frame = result.frames[0]
    semantic = _load_binary(frame.semantic_confirmed_path)
    motion_confirmed = _load_binary(frame.motion_confirmed_path)
    uncertain = _load_binary(frame.uncertain_path)
    hard = _load_binary(frame.hard_exclude_path)
    assert semantic[1, 1]
    assert not semantic[5, 5]
    assert semantic[11, 11]
    assert not motion_confirmed[15, 15]
    assert uncertain[5, 5]
    assert uncertain[15, 15]
    assert semantic[1, 2]
    assert hard[1, 2]


def test_propagation_collapse_demotes_candidate_only_pixels_to_uncertain(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (10, 10)
    direct = np.zeros(shape, dtype=bool)
    direct[0, 0] = True
    propagated = np.zeros(shape, dtype=bool)
    propagated[2:5, :] = True
    inputs = _single_frame_inputs(
        tmp_path,
        direct=direct,
        propagated=propagated,
        direct_support=propagated,
        motion_confirmed=np.zeros(shape, dtype=bool),
        motion_requires_semantic=np.zeros(shape, dtype=bool),
        motion_uncertain=np.zeros(shape, dtype=bool),
        strength=np.zeros(shape, dtype=np.float32),
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(propagation_collapse_limit=0.2),
    )

    semantic = _load_binary(result.frames[0].semantic_confirmed_path)
    uncertain = _load_binary(result.frames[0].uncertain_path)
    assert semantic[0, 0]
    assert not semantic[2:5, :].any()
    assert uncertain[2:5, :].all()


def test_sky_never_confirms_individual_static_track_support_pixels(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (8, 8)
    empty = np.zeros(shape, dtype=bool)
    proposal = np.zeros(shape, dtype=bool)
    proposal[6:8, 2:6] = True
    track = StaticTrack(
        track_id=17,
        xyz=(0.0, 0.0, 1.0),
        mean_reprojection_error=0.1,
        observations=(TrackObservation("frame-000000", 2.0, 6.0),),
    )
    inputs = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        sky=proposal,
        depth=np.ones(shape, dtype=np.float32) * 100.0,
        static_tracks=(track,),
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(sky_max_component_support_fraction=0.2),
    )

    sky = _load_binary(result.frames[0].sky_confirmed_path)
    uncertain = _load_binary(result.frames[0].uncertain_path)
    assert not sky[6, 2]
    assert uncertain[6, 2]
    assert sky[6:8, 2:6].sum() == 7


def test_temporal_area_spike_demotes_only_candidate_evidence(tmp_path: Path) -> None:
    masks = _masks_module()
    shape = (10, 10)
    empty = np.zeros(shape, dtype=bool)
    items = []
    for index in range(3):
        direct = np.zeros(shape, dtype=bool)
        motion = np.zeros(shape, dtype=bool)
        if index == 1:
            direct[0, 0] = True
            motion[2:5, :] = True
        items.append(
            _single_frame_inputs(
                tmp_path / f"input-{index}",
                direct=direct,
                propagated=empty,
                direct_support=empty,
                motion_confirmed=motion,
                motion_requires_semantic=empty,
                motion_uncertain=empty,
                strength=np.ones(shape, dtype=np.float32),
                image_name=f"frame_{index:06d}.png",
                frame_id=f"frame-{index:06d}",
            )
        )
    inputs = _combine_inputs(tmp_path / "combined", tuple(items))

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(temporal_area_change_limit=0.1),
    )

    middle = result.frames[1]
    semantic = _load_binary(middle.semantic_confirmed_path)
    motion = _load_binary(middle.motion_confirmed_path)
    uncertain = _load_binary(middle.uncertain_path)
    hard = _load_binary(middle.hard_exclude_path)
    assert semantic[0, 0]
    assert not motion.any()
    assert uncertain[2:5, :].all()
    assert hard.sum() == 1


def test_runaway_trimming_removes_weakest_motion_only_component_first(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (10, 20)
    direct = np.zeros(shape, dtype=bool)
    direct[:, 0:2] = True
    motion = np.zeros(shape, dtype=bool)
    motion[0:3, 4:14] = True
    motion[4:10, 10:20] = True
    strength = np.zeros(shape, dtype=np.float32)
    strength[0:3, 4:14] = 0.1
    strength[4:10, 10:20] = 0.9
    inputs = _single_frame_inputs(
        tmp_path,
        direct=direct,
        propagated=np.zeros(shape, dtype=bool),
        direct_support=np.zeros(shape, dtype=bool),
        motion_confirmed=motion,
        motion_requires_semantic=np.zeros(shape, dtype=bool),
        motion_uncertain=np.zeros(shape, dtype=bool),
        strength=strength,
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    frame = result.frames[0]
    semantic = _load_binary(frame.semantic_confirmed_path)
    motion_confirmed = _load_binary(frame.motion_confirmed_path)
    uncertain = _load_binary(frame.uncertain_path)
    assert frame.decision == "trimmed_motion"
    assert frame.exclusion_fraction_before_trim == pytest.approx(0.55)
    assert frame.exclusion_fraction_after_trim == pytest.approx(0.40)
    assert semantic[:, 0:2].all()
    assert not motion_confirmed[0:3, 4:14].any()
    assert uncertain[0:3, 4:14].all()
    assert motion_confirmed[4:10, 10:20].all()


def test_runaway_rejection_publishes_keep_all_and_moves_evidence_to_uncertain(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (10, 10)
    direct = np.zeros(shape, dtype=bool)
    direct[0:6, :] = True
    inputs = _single_frame_inputs(
        tmp_path,
        direct=direct,
        propagated=np.zeros(shape, dtype=bool),
        direct_support=np.zeros(shape, dtype=bool),
        motion_confirmed=np.zeros(shape, dtype=bool),
        motion_requires_semantic=np.zeros(shape, dtype=bool),
        motion_uncertain=np.zeros(shape, dtype=bool),
        strength=np.zeros(shape, dtype=np.float32),
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    frame = result.frames[0]
    assert frame.decision == "rejected_unmasked"
    assert frame.exclusion_fraction_before_trim == pytest.approx(0.60)
    assert frame.exclusion_fraction_after_trim == pytest.approx(0.60)
    assert frame.warnings
    assert not _load_binary(frame.semantic_confirmed_path).any()
    assert not _load_binary(frame.sky_confirmed_path).any()
    assert not _load_binary(frame.motion_confirmed_path).any()
    assert not _load_binary(frame.hard_exclude_path).any()
    assert _load_binary(frame.uncertain_path)[0:6, :].all()
    assert _load_binary(frame.colmap_keep_path).all()
    assert _load_binary(frame.training_validity_path).all()


def test_sky_frame_fraction_is_the_only_runaway_exemption(tmp_path: Path) -> None:
    masks = _masks_module()
    shape = (10, 10)
    empty = np.zeros(shape, dtype=bool)
    sky = np.zeros(shape, dtype=bool)
    sky[0:6, :] = True
    inputs = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        sky=sky,
        depth=np.ones(shape, dtype=np.float32) * 100.0,
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(sky_frame_min_fraction=0.5),
    )

    frame = result.frames[0]
    assert frame.decision == "accepted"
    assert frame.exclusion_fraction_before_trim == pytest.approx(0.60)
    assert frame.exclusion_fraction_after_trim == pytest.approx(0.60)
    assert _load_binary(frame.sky_confirmed_path).sum() == 60
    assert _load_binary(frame.hard_exclude_path).sum() == 60


@pytest.mark.parametrize(
    "fault",
    (
        "reordered_semantic",
        "missing_motion",
        "duplicate_canonical_frame",
        "reordered_sky",
        "reordered_scene",
        "source_digest_drift",
        "semantic_digest_drift",
        "semantic_manifest_drift",
        "motion_scene_digest_drift",
    ),
)
def test_canonical_join_and_digest_faults_fail_before_any_output_write(
    tmp_path: Path,
    fault: str,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / fault)
    if fault == "reordered_semantic":
        semantic = dataclasses.replace(
            semantic, frames=tuple(reversed(semantic.frames))
        )
    elif fault == "missing_motion":
        motion = dataclasses.replace(motion, frames=motion.frames[:-1])
    elif fault == "duplicate_canonical_frame":
        frames = (frames[0], frames[0])
    elif fault == "reordered_sky":
        sky = tuple(reversed(sky))
    elif fault == "reordered_scene":
        scene = dataclasses.replace(scene, frames=tuple(reversed(scene.frames)))
    elif fault == "source_digest_drift":
        frames[0].path.write_bytes(frames[0].path.read_bytes() + b"drift")
    elif fault == "semantic_digest_drift":
        path = semantic.frames[0].direct_confirmed_path
        path.write_bytes(path.read_bytes() + b"drift")
    elif fault == "semantic_manifest_drift":
        semantic.manifest_path.write_bytes(
            semantic.manifest_path.read_bytes() + b"drift"
        )
    elif fault == "motion_scene_digest_drift":
        motion = dataclasses.replace(motion, scene_digest="f" * 64)
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(fault)
    output_dir = tmp_path / fault / "fused"

    with pytest.raises((ValueError, FileNotFoundError)):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert not output_dir.exists()
    assert not tuple(output_dir.parent.glob(f".{output_dir.name}*.staging*"))


def test_fusion_accepts_scene_digest_published_by_current_motion_producer(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    frames, semantic, _synthetic_motion, sky, scene = _empty_inputs(
        tmp_path / "inputs", count=3
    )
    motion = run_motion_evidence(
        frames,
        scene,
        tmp_path / "real-motion",
        policy=FlowGatePolicy(0.1, 0.0, 0.05, 0.01, 3.0, 2, 1.0, 3.0),
        model_factory=lambda: _DigestBindingSeaRaft((4, 5)),
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release_record,
    )

    result = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    assert len(result.frames) == 3
    assert all(frame.decision == "accepted" for frame in result.frames)


def test_motion_scene_and_mask_digests_are_root_portable_and_bind_source_size(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    first_inputs = _empty_inputs(tmp_path / "root-a" / "inputs", count=3)
    second_inputs = _empty_inputs(tmp_path / "root-b" / "inputs", count=3)
    first_frames, first_semantic, _, first_sky, first_scene = first_inputs
    second_frames, second_semantic, _, second_sky, second_scene = second_inputs
    first_motion = run_motion_evidence(
        first_frames,
        first_scene,
        tmp_path / "root-a" / "motion",
        policy=FlowGatePolicy(0.1, 0.0, 0.05, 0.01, 3.0, 2, 1.0, 3.0),
        model_factory=lambda: _DigestBindingSeaRaft((4, 5)),
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release_record,
    )
    second_motion = run_motion_evidence(
        second_frames,
        second_scene,
        tmp_path / "root-b" / "motion",
        policy=FlowGatePolicy(0.1, 0.0, 0.05, 0.01, 3.0, 2, 1.0, 3.0),
        model_factory=lambda: _DigestBindingSeaRaft((4, 5)),
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release_record,
    )

    assert first_motion.scene_digest == second_motion.scene_digest
    first_result = masks.fuse_evidence_masks(
        first_frames,
        first_semantic,
        first_motion,
        first_sky,
        first_scene,
        tmp_path / "root-a" / "fused",
        policy=_mask_policy(),
    )
    second_result = masks.fuse_evidence_masks(
        second_frames,
        second_semantic,
        second_motion,
        second_sky,
        second_scene,
        tmp_path / "root-b" / "fused",
        policy=_mask_policy(),
    )
    assert first_result.mask_set_digest == second_result.mask_set_digest

    larger_path = tmp_path / "size-variant" / "nested" / "frame_000000.png"
    larger_path.parent.mkdir(parents=True)
    larger_path.write_bytes(first_frames[0].path.read_bytes() + b"extra-byte")
    size_variant = (
        dataclasses.replace(first_frames[0], path=larger_path),
        *first_frames[1:],
    )
    assert _frame_set_digest(size_variant) != _frame_set_digest(first_frames)


def test_publication_failure_cleans_private_staging_and_leaves_no_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=1)
    output_dir = tmp_path / "fused"
    real_write = masks._write_binary_png
    written_paths: list[Path] = []

    def fail_second_write(path: Path, mask: np.ndarray) -> str:
        written_paths.append(path)
        if len(written_paths) == 2:
            raise RuntimeError("injected publication failure")
        return real_write(path, mask)

    monkeypatch.setattr(masks, "_write_binary_png", fail_second_write)

    with pytest.raises(RuntimeError, match="injected publication failure"):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert written_paths
    assert all(path.parent != output_dir for path in written_paths)
    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}*.staging*"))


def test_no_replace_promotion_loses_race_without_replacing_competing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=1)
    output_dir = tmp_path / "fused"
    real_promote = masks.promote_directory

    def racing_promote(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "competitor.txt").write_text("winner", encoding="utf-8")
        real_promote(staging, target)

    monkeypatch.setattr(masks, "promote_directory", racing_promote)

    with pytest.raises(FileExistsError):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert (output_dir / "competitor.txt").read_text(encoding="utf-8") == "winner"
    assert not (output_dir / "manifest.json").exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}*.staging*"))


def test_upstream_digest_is_rehashed_after_staging_and_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=1)
    output_dir = tmp_path / "fused"
    real_write = masks._write_binary_png
    mutated = False

    def mutate_upstream_after_first_write(path: Path, mask: np.ndarray) -> str:
        nonlocal mutated
        digest = real_write(path, mask)
        if not mutated:
            semantic.manifest_path.write_bytes(
                semantic.manifest_path.read_bytes() + b"changed"
            )
            mutated = True
        return digest

    monkeypatch.setattr(masks, "_write_binary_png", mutate_upstream_after_first_write)

    with pytest.raises(ValueError, match="digest changed before publication"):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}*.staging*"))


def test_rerun_is_byte_deterministic_across_distinct_absent_output_dirs(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=2)

    first = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused-a",
        policy=_mask_policy(),
    )
    second = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused-b",
        policy=_mask_policy(),
    )

    assert first.mask_set_digest == second.mask_set_digest
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    for first_frame, second_frame in zip(first.frames, second.frames, strict=True):
        assert (
            first_frame.semantic_confirmed_sha256
            == second_frame.semantic_confirmed_sha256
        )
        assert first_frame.sky_confirmed_sha256 == second_frame.sky_confirmed_sha256
        assert (
            first_frame.motion_confirmed_sha256 == second_frame.motion_confirmed_sha256
        )
        assert first_frame.uncertain_sha256 == second_frame.uncertain_sha256
        assert first_frame.hard_exclude_sha256 == second_frame.hard_exclude_sha256
        assert first_frame.colmap_keep_sha256 == second_frame.colmap_keep_sha256
        assert (
            first_frame.training_validity_sha256
            == second_frame.training_validity_sha256
        )


def test_mask_set_digest_and_manifest_bind_all_policies_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=1)
    first_policy = _mask_policy(sky_far_depth_quantile=0.25)
    second_policy = _mask_policy(sky_far_depth_quantile=0.75)

    first = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused-a",
        policy=first_policy,
    )
    second = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused-b",
        policy=second_policy,
    )

    assert first.frames[0].hard_exclude_sha256 == second.frames[0].hard_exclude_sha256
    assert first.mask_set_digest != second.mask_set_digest
    payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["policy"] == dataclasses.asdict(first_policy)
    assert payload["inputs"]["input_frame_digest"]
    assert payload["inputs"]["semantic"]["manifest_sha256"] == semantic.manifest_sha256
    assert payload["inputs"]["motion"]["manifest_sha256"] == motion.manifest_sha256
    assert payload["inputs"]["motion"]["pair_manifest_sha256"] == (
        motion.pair_manifest_sha256
    )
    assert payload["inputs"]["motion"]["scene_digest"] == motion.scene_digest
    assert payload["inputs"]["scene"]["geometry_digest"] == scene.geometry_digest
    assert payload["inputs"]["scene"]["depth_digest"] == scene.depth_digest
    assert payload["inputs"]["sky_proposals"][0]["sha256"] == sky[0].sha256
    assert payload["inputs"]["sky_proposals"][0]["processed_shape"] == [4, 5]
    assert payload["inputs"]["sky_proposals"][0]["source_shape"] == [4, 5]
    assert payload["inputs"]["sky_proposals"][0]["resized"] is False
    assert payload["frames"][0]["outputs"]["hard_exclude"]["sha256"] == (
        first.frames[0].hard_exclude_sha256
    )


def test_sky_processed_shape_uses_exact_nearest_resize_and_records_it(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (4, 6)
    empty = np.zeros(shape, dtype=bool)
    processed_sky = np.array(
        [[True, False, False], [False, False, False]],
        dtype=bool,
    )
    inputs = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        sky=processed_sky,
        depth=np.ones(shape, dtype=np.float32) * 100.0,
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(sky_frame_min_fraction=0.9),
    )

    expected = np.repeat(np.repeat(processed_sky, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(
        _load_binary(result.frames[0].sky_confirmed_path), expected
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    sky_record = payload["inputs"]["sky_proposals"][0]
    assert sky_record["processed_shape"] == [2, 3]
    assert sky_record["source_shape"] == [4, 6]
    assert sky_record["resized"] is True


def test_nested_colmap_name_preserves_parents_and_appends_exact_png_suffix(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (3, 7)
    empty = np.zeros(shape, dtype=bool)
    inputs = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        image_name="nested/camera/frame.png",
    )

    result = masks.fuse_evidence_masks(
        *inputs[:4],
        inputs[4],
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    frame = result.frames[0]
    assert frame.colmap_keep_path.relative_to(tmp_path / "fused").as_posix() == (
        "colmap_masks/nested/camera/frame.png.png"
    )
    with Image.open(frame.colmap_keep_path) as image:
        assert image.size == (7, 3)
        assert image.mode == "L"
        assert image.info == {}


def test_public_result_records_are_deeply_immutable(tmp_path: Path) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / "inputs", count=1)
    result = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    def assert_deeply_immutable(value: object) -> None:
        assert not isinstance(value, (dict, list, set, np.ndarray))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            assert value.__dataclass_params__.frozen
            for field in dataclasses.fields(value):
                assert_deeply_immutable(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                assert_deeply_immutable(item)

    assert_deeply_immutable(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.frames[0].decision = "accepted"


@pytest.mark.parametrize(
    "fault",
    (
        "unsafe_image_name",
        "relative_semantic_path",
        "artifact_path_collision",
        "semantic_dimension_mismatch",
        "nonbinary_semantic_png",
        "float64_strength",
        "nonboolean_sky_npy",
    ),
)
def test_malformed_dimensions_formats_and_paths_are_rejected_before_writes(
    tmp_path: Path,
    fault: str,
) -> None:
    masks = _masks_module()
    frames, semantic, motion, sky, scene = _empty_inputs(tmp_path / fault, count=1)
    if fault == "unsafe_image_name":
        frames = (dataclasses.replace(frames[0], image_name="../escape.png"),)
    elif fault == "relative_semantic_path":
        record = dataclasses.replace(
            semantic.frames[0], direct_confirmed_path=Path("relative.png")
        )
        semantic = dataclasses.replace(semantic, frames=(record,))
    elif fault == "artifact_path_collision":
        record = dataclasses.replace(
            semantic.frames[0],
            direct_support_path=semantic.frames[0].direct_confirmed_path,
            direct_support_sha256=semantic.frames[0].direct_confirmed_sha256,
        )
        semantic = dataclasses.replace(semantic, frames=(record,))
    elif fault == "semantic_dimension_mismatch":
        record = dataclasses.replace(
            semantic.frames[0], width=semantic.frames[0].width + 1
        )
        semantic = dataclasses.replace(semantic, frames=(record,))
    elif fault == "nonbinary_semantic_png":
        path = semantic.frames[0].direct_confirmed_path
        Image.fromarray(np.full((4, 5), 127, dtype=np.uint8), mode="L").save(path)
        record = dataclasses.replace(
            semantic.frames[0], direct_confirmed_sha256=_sha256(path)
        )
        semantic = dataclasses.replace(semantic, frames=(record,))
    elif fault == "float64_strength":
        path = motion.frames[0].strength_path
        digest = _write_npy(path, np.zeros((4, 5), dtype=np.float64))
        record = dataclasses.replace(motion.frames[0], strength_sha256=digest)
        motion = dataclasses.replace(motion, frames=(record,))
    elif fault == "nonboolean_sky_npy":
        path = sky[0].path
        digest = _write_npy(path, np.zeros((4, 5), dtype=np.uint8))
        sky = (dataclasses.replace(sky[0], sha256=digest),)
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(fault)
    output_dir = tmp_path / fault / "fused"

    with pytest.raises((ValueError, FileExistsError)):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert not output_dir.exists()
    assert not tuple(output_dir.parent.glob(f".{output_dir.name}*.staging*"))


@pytest.mark.parametrize("image_name", ("./frame.png", "nested//frame.png"))
def test_noncanonical_image_name_aliases_are_rejected_before_writes(
    tmp_path: Path,
    image_name: str,
) -> None:
    masks = _masks_module()
    shape = (4, 5)
    empty = np.zeros(shape, dtype=bool)
    frames, semantic, motion, sky, scene = _single_frame_inputs(
        tmp_path / "inputs",
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        image_name=image_name,
    )
    output_dir = tmp_path / "fused"

    with pytest.raises(ValueError, match="canonical"):
        masks.fuse_evidence_masks(
            frames,
            semantic,
            motion,
            sky,
            scene,
            output_dir,
            policy=_mask_policy(),
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}*.staging*"))


def test_missing_rigid_geometry_keeps_every_sky_proposal_pixel_uncertain(
    tmp_path: Path,
) -> None:
    masks = _masks_module()
    shape = (4, 5)
    empty = np.zeros(shape, dtype=bool)
    proposal = np.ones(shape, dtype=bool)
    frames, semantic, motion, sky, scene = _single_frame_inputs(
        tmp_path,
        direct=empty,
        propagated=empty,
        direct_support=empty,
        motion_confirmed=empty,
        motion_requires_semantic=empty,
        motion_uncertain=empty,
        strength=np.zeros(shape, dtype=np.float32),
        sky=proposal,
    )
    rigid = dataclasses.replace(
        scene.frames[0],
        registered=False,
        w2c_4x4=None,
        pinhole_fx_fy_cx_cy=None,
        depth_path=None,
        depth_sha256=None,
    )
    scene = dataclasses.replace(scene, frames=(rigid,))
    motion = dataclasses.replace(motion, scene_digest=_scene_digest(frames, scene))

    result = masks.fuse_evidence_masks(
        frames,
        semantic,
        motion,
        sky,
        scene,
        tmp_path / "fused",
        policy=_mask_policy(),
    )

    assert not _load_binary(result.frames[0].sky_confirmed_path).any()
    assert _load_binary(result.frames[0].uncertain_path).all()
