from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.dependencies import CHECKPOINT_MODEL_REFS
from experiments.learned_quality.flow import (
    FlowPairRequest,
    FlowGatePolicy,
    MotionEvidence,
    MotionFrameEvidence,
    PairGateResult,
    RigidFrameEvidence,
    RigidSceneEvidence,
    SeaRaftPairPrediction,
    StaticTrack,
    TrackObservation,
    calibrate_residual_threshold,
    classify_temporal_motion,
    evaluate_flow_pair,
    project_rigid_flow,
    run_motion_evidence,
)
from experiments.learned_quality.lifecycle import VramReleaseRecord


_SEA_RAFT_REF = next(
    model
    for model in CHECKPOINT_MODEL_REFS
    if model.repo_id == "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
)


def _policy(**overrides: object) -> FlowGatePolicy:
    values: dict[str, object] = {
        "z_buffer_relative_tolerance": 0.10,
        "depth_edge_dilation_fraction": 0.0,
        "cycle_absolute_pixels": 0.05,
        "cycle_relative_fraction": 0.01,
        "uncertainty_mad_multiplier": 3.0,
        "static_track_min_length": 3,
        "static_track_max_reprojection_error": 0.25,
        "residual_mad_multiplier": 3.0,
    }
    values.update(overrides)
    return FlowGatePolicy(**values)  # type: ignore[arg-type]


def _frame(tmp_path: Path, index: int, *, width: int = 3, height: int = 3) -> FrameArtifact:
    path = (tmp_path / f"frame_{index:06d}.png").resolve()
    path.write_bytes(f"frame-{index}".encode("ascii"))
    return FrameArtifact(
        image_name=path.name,
        frame_id=f"frame-{index}",
        path=path,
        sha256="a" * 64,
    )


def _rigid_frame(
    frame: FrameArtifact,
    *,
    width: int = 3,
    height: int = 3,
    tx: float = 0.0,
    depth_path: Path | None = None,
) -> RigidFrameEvidence:
    return RigidFrameEvidence(
        frame=frame,
        width=width,
        height=height,
        registered=True,
        w2c_4x4=(
            1.0,
            0.0,
            0.0,
            tx,
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
        pinhole_fx_fy_cx_cy=(2.0, 2.0, 1.0, 1.0),
        depth_path=depth_path,
        depth_sha256=None if depth_path is None else "b" * 64,
    )


def test_policy_has_no_defaults_is_frozen_and_rejects_invalid_values() -> None:
    assert all(field.default is field.default_factory for field in fields(FlowGatePolicy))
    policy = _policy()

    with pytest.raises(FrozenInstanceError):
        policy.cycle_absolute_pixels = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="z_buffer_relative_tolerance"):
        _policy(z_buffer_relative_tolerance=0.0)
    with pytest.raises(ValueError, match="static_track_min_length"):
        _policy(static_track_min_length=True)


def test_producer_dtos_are_frozen_and_do_not_default_join_fields(tmp_path: Path) -> None:
    frame = _frame(tmp_path, 0)
    rigid = _rigid_frame(frame)
    observation = TrackObservation(frame_id=frame.frame_id, x=1.0, y=1.0)
    track = StaticTrack(
        track_id=7,
        xyz=(0.0, 0.0, 2.0),
        mean_reprojection_error=0.1,
        observations=(observation,),
    )
    scene = RigidSceneEvidence(
        frames=(rigid,),
        static_tracks=(track,),
        geometry_digest="c" * 64,
        depth_digest="d" * 64,
    )

    with pytest.raises(FrozenInstanceError):
        rigid.width = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        scene.geometry_digest = "e" * 64  # type: ignore[misc]
    assert all(field.default is field.default_factory for field in fields(MotionFrameEvidence))
    assert all(field.default is field.default_factory for field in fields(MotionEvidence))


def test_rigid_projection_uses_w2c_and_rejects_nonpositive_or_out_of_bounds(
    tmp_path: Path,
) -> None:
    source = _rigid_frame(_frame(tmp_path, 0))
    target = _rigid_frame(_frame(tmp_path, 1), tx=-0.5)
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    depth[1, 1] = 0.0

    flow, target_z, positive_in_bounds = project_rigid_flow(source, target, depth)

    np.testing.assert_allclose(flow[..., 0][positive_in_bounds], -0.5, atol=1e-6)
    np.testing.assert_allclose(flow[..., 1][positive_in_bounds], 0.0, atol=1e-6)
    np.testing.assert_allclose(target_z[positive_in_bounds], 2.0, atol=1e-6)
    assert not positive_in_bounds[:, 0].any()
    assert not positive_in_bounds[1, 1]
    assert positive_in_bounds[0, 1]


def test_pair_gate_uses_rigid_residual_not_raw_flow_magnitude(tmp_path: Path) -> None:
    source = _rigid_frame(_frame(tmp_path, 0))
    target = _rigid_frame(_frame(tmp_path, 1), tx=-0.5)
    source_depth = np.full((3, 3), 2.0, dtype=np.float32)
    target_depth = np.full((3, 3), 2.0, dtype=np.float32)
    rigid_flow, _, _ = project_rigid_flow(source, target, source_depth)
    backward = np.zeros_like(rigid_flow)
    backward[..., 0] = 0.5
    uncertainty = np.zeros((3, 3), dtype=np.float32)

    result = evaluate_flow_pair(
        source,
        target,
        source_depth,
        target_depth,
        rigid_flow,
        backward,
        uncertainty,
        uncertainty,
        policy=_policy(),
        residual_threshold=0.25,
    )

    assert not result.motion.any(), "large camera flow alone is not object motion"
    assert result.valid[1, 1]
    assert not result.uncertain[1, 1]
    assert result.strength[1, 1] == pytest.approx(0.0)


def test_static_track_calibration_is_robust_and_reports_insufficiency(
    tmp_path: Path,
) -> None:
    rigid_frames = tuple(_rigid_frame(_frame(tmp_path, index)) for index in range(4))
    observations = tuple(
        TrackObservation(
            frame_id=rigid.frame.frame_id,
            x=1.1,
            y=1.0,
        )
        for index, rigid in enumerate(rigid_frames)
    )
    long_track = StaticTrack(
        track_id=1,
        xyz=(0.0, 0.0, 2.0),
        mean_reprojection_error=0.1,
        observations=observations,
    )
    scene = RigidSceneEvidence(
        frames=rigid_frames,
        static_tracks=(long_track,),
        geometry_digest="c" * 64,
        depth_digest="d" * 64,
    )
    directional_residuals = {
        (rigid_frames[0].frame.frame_id, rigid_frames[1].frame.frame_id): np.full(
            (3, 3), 0.2, dtype=np.float64
        ),
        (rigid_frames[1].frame.frame_id, rigid_frames[0].frame.frame_id): np.full(
            (3, 3), 0.4, dtype=np.float64
        ),
        (rigid_frames[1].frame.frame_id, rigid_frames[2].frame.frame_id): np.full(
            (3, 3), 0.6, dtype=np.float64
        ),
        (rigid_frames[2].frame.frame_id, rigid_frames[1].frame.frame_id): np.full(
            (3, 3), 0.8, dtype=np.float64
        ),
        (rigid_frames[2].frame.frame_id, rigid_frames[3].frame.frame_id): np.full(
            (3, 3), 1.0, dtype=np.float64
        ),
        (rigid_frames[3].frame.frame_id, rigid_frames[2].frame.frame_id): np.full(
            (3, 3), 1.2, dtype=np.float64
        ),
    }

    assert calibrate_residual_threshold(
        scene,
        _policy(residual_mad_multiplier=1.0),
        directional_residuals,
    ) == pytest.approx(0.7 + 1.4826 * 0.3)
    assert calibrate_residual_threshold(scene, _policy()) is None
    insufficient = replace(
        scene,
        static_tracks=(replace(long_track, observations=observations[:2]),),
    )
    assert calibrate_residual_threshold(
        insufficient,
        _policy(),
        directional_residuals,
    ) is None


def _gate_inputs(tmp_path: Path) -> tuple[
    RigidFrameEvidence,
    RigidFrameEvidence,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    source = _rigid_frame(_frame(tmp_path, 10))
    target = _rigid_frame(_frame(tmp_path, 11))
    source_depth = np.full((3, 3), 2.0, dtype=np.float32)
    target_depth = np.full((3, 3), 2.0, dtype=np.float32)
    forward = np.zeros((3, 3, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    forward_uncertainty = np.zeros((3, 3), dtype=np.float32)
    backward_uncertainty = np.zeros((3, 3), dtype=np.float32)
    return (
        source,
        target,
        source_depth,
        target_depth,
        forward,
        backward,
        forward_uncertainty,
        backward_uncertainty,
    )


@pytest.mark.parametrize(
    "failed_gate",
    ("positive", "target_depth", "depth_edge", "cycle", "uncertainty"),
)
def test_each_pair_gate_independently_marks_unknown(
    tmp_path: Path,
    failed_gate: str,
) -> None:
    values = list(_gate_inputs(tmp_path))
    if failed_gate == "positive":
        values[2][1, 1] = 0.0
    elif failed_gate == "target_depth":
        values[3][1, 1] = 8.0
    elif failed_gate == "depth_edge":
        values[2][1, 2] = 8.0
        values[3][1, 2] = 8.0
    elif failed_gate == "cycle":
        values[5][..., 0] = 1.0
    elif failed_gate == "uncertainty":
        values[6][1, 1] = 100.0

    result = evaluate_flow_pair(
        *values,
        policy=_policy(),
        residual_threshold=0.25,
    )

    assert result.uncertain[1, 1]
    assert not result.motion[1, 1]


def test_z_buffer_keeps_nearest_projection_and_rejects_farther_collision(
    tmp_path: Path,
) -> None:
    values = list(_gate_inputs(tmp_path))
    source = values[0]
    target = values[1]
    values[1] = replace(target, pinhole_fx_fy_cx_cy=(0.1, 2.0, 1.0, 1.0))
    values[2][1, 0] = 2.0
    values[2][1, 1] = 4.0
    values[3][1, 1] = 2.0

    result = evaluate_flow_pair(
        *values,
        policy=_policy(z_buffer_relative_tolerance=0.9),
        residual_threshold=0.25,
    )

    assert result.valid[1, 0]
    assert result.uncertain[1, 1]


def _pair_result(
    valid: tuple[bool, ...],
    motion: tuple[bool, ...],
    strength: tuple[float, ...],
) -> PairGateResult:
    valid_array = np.asarray(valid, dtype=bool)[None, :]
    motion_array = np.asarray(motion, dtype=bool)[None, :]
    return PairGateResult(
        valid=valid_array,
        motion=motion_array,
        uncertain=~valid_array,
        strength=np.asarray(strength, dtype=np.float32)[None, :],
        uncertainty_threshold=1.0,
    )


def test_temporal_partition_distinguishes_two_sided_one_sided_and_unknown() -> None:
    incoming = _pair_result(
        (True, True, True, False),
        (True, True, True, False),
        (3.0, 2.0, 4.0, 0.0),
    )
    outgoing = _pair_result(
        (True, True, False, False),
        (True, False, False, False),
        (5.0, 1.0, 0.0, 0.0),
    )

    interior = classify_temporal_motion(
        incoming,
        outgoing,
        is_first=False,
        is_last=False,
    )
    np.testing.assert_array_equal(
        interior.confirmed_without_semantic,
        np.array([[True, False, False, False]]),
    )
    np.testing.assert_array_equal(
        interior.requires_semantic,
        np.array([[False, True, False, False]]),
    )
    np.testing.assert_array_equal(
        interior.uncertain,
        np.array([[False, False, True, True]]),
    )
    assert interior.strength[0, 0] == pytest.approx(3.0)
    assert interior.strength[0, 1] == pytest.approx(2.0)

    endpoint = classify_temporal_motion(
        None,
        incoming,
        is_first=True,
        is_last=False,
    )
    np.testing.assert_array_equal(endpoint.confirmed_without_semantic, incoming.motion)
    assert not endpoint.requires_semantic.any()


def test_missing_interior_direction_is_uncertain_not_one_sided_motion() -> None:
    outgoing = _pair_result((True, True), (True, False), (2.0, 0.0))

    result = classify_temporal_motion(
        None,
        outgoing,
        is_first=False,
        is_last=False,
    )

    assert result.uncertain.all()
    assert not result.confirmed_without_semantic.any()
    assert not result.requires_semantic.any()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_record() -> VramReleaseRecord:
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


def _motion_fixture(
    tmp_path: Path,
) -> tuple[tuple[FrameArtifact, ...], RigidSceneEvidence]:
    frames: list[FrameArtifact] = []
    rigid_frames: list[RigidFrameEvidence] = []
    observations: list[TrackObservation] = []
    for index in range(3):
        image_path = (tmp_path / f"input_{index}" / f"frame_{index:06d}.png").resolve()
        image_path.parent.mkdir()
        Image.new("RGB", (3, 3), color=(20 + index, 40, 60)).save(image_path)
        frame = FrameArtifact(
            image_name=image_path.name,
            frame_id=f"frame-{index}",
            path=image_path,
            sha256=_sha256(image_path),
        )
        depth_path = (tmp_path / f"depth-{index}.npy").resolve()
        with depth_path.open("wb") as handle:
            np.save(handle, np.full((3, 3), 2.0, dtype=np.float32), allow_pickle=False)
        frames.append(frame)
        rigid_frames.append(
            _rigid_frame(frame, depth_path=depth_path)
        )
        observations.append(
            TrackObservation(frame_id=frame.frame_id, x=1.1, y=1.0)
        )
    rigid_frames = [
        replace(rigid, depth_sha256=_sha256(rigid.depth_path))  # type: ignore[arg-type]
        for rigid in rigid_frames
    ]
    track = StaticTrack(
        track_id=1,
        xyz=(0.0, 0.0, 2.0),
        mean_reprojection_error=0.1,
        observations=tuple(observations),
    )
    return tuple(frames), RigidSceneEvidence(
        frames=tuple(rigid_frames),
        static_tracks=(track,),
        geometry_digest="c" * 64,
        depth_digest="d" * 64,
    )


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class FakeSeaRaft:
    model_id = _SEA_RAFT_REF.repo_id
    revision = _SEA_RAFT_REF.revision
    code_commit = _SEA_RAFT_REF.code_commit

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        error: Exception | None = None,
        reverse_predictions: bool = False,
        flow_x: float = 0.5,
        flow_dtype: object = np.float32,
        static_sample_x: float | None = 0.2,
        after_infer: Callable[[], None] | None = None,
        provenance_overrides: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.error = error
        self.reverse_predictions = reverse_predictions
        self.flow_x = flow_x
        self.flow_dtype = flow_dtype
        self.static_sample_x = static_sample_x
        self.after_infer = after_infer
        for field_name, value in (provenance_overrides or {}).items():
            setattr(self, field_name, value)

    def infer_bidirectional(
        self,
        pairs: tuple[FlowPairRequest, ...],
        *,
        batch_size: int,
    ) -> tuple[SeaRaftPairPrediction, ...]:
        self.events.append(f"infer:{self.name}:{batch_size}")
        if self.error is not None:
            raise self.error
        forward_x = np.full((3, 3), self.flow_x, dtype=self.flow_dtype)
        backward_x = np.full((3, 3), -self.flow_x, dtype=self.flow_dtype)
        if self.static_sample_x is not None:
            forward_x[1, :] = self.static_sample_x
            backward_x[1, :] = -self.static_sample_x
        predictions = tuple(
            SeaRaftPairPrediction(
                source_frame_id=pair.source_frame_id,
                target_frame_id=pair.target_frame_id,
                forward_flow=np.dstack(
                    (
                        forward_x.copy(),
                        np.zeros((3, 3), dtype=self.flow_dtype),
                    )
                ),
                backward_flow=np.dstack(
                    (
                        backward_x.copy(),
                        np.zeros((3, 3), dtype=self.flow_dtype),
                    )
                ),
                forward_uncertainty=np.zeros((3, 3), dtype=np.float32),
                backward_uncertainty=np.zeros((3, 3), dtype=np.float32),
            )
            for pair in pairs
        )
        if self.after_infer is not None:
            self.after_infer()
        return tuple(reversed(predictions)) if self.reverse_predictions else predictions


class FakeFactory:
    model_id = FakeSeaRaft.model_id
    revision = FakeSeaRaft.revision
    code_commit = FakeSeaRaft.code_commit
    torch_module = SimpleNamespace(
        OutOfMemoryError=FakeCudaOutOfMemoryError,
        cuda=SimpleNamespace(OutOfMemoryError=FakeCudaOutOfMemoryError),
    )

    def __init__(
        self,
        events: list[str],
        adapters: tuple[Callable[[str, list[str]], FakeSeaRaft], ...] = (),
    ) -> None:
        self.events = events
        self.adapters = list(adapters)
        self.created = 0

    def __call__(self) -> FakeSeaRaft:
        self.created += 1
        name = f"model-{self.created}"
        self.events.append(f"factory:{name}")
        if self.adapters:
            return self.adapters.pop(0)(name, self.events)
        return FakeSeaRaft(name, self.events)


def _release(events: list[str]) -> Callable[[object], VramReleaseRecord]:
    def release(model: object) -> VramReleaseRecord:
        events.append(f"release:{model.name}")  # type: ignore[attr-defined]
        return _release_record()

    return release


def _run_motion_case(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    output_dir: Path,
    events: list[str],
    *,
    factory: FakeFactory | None = None,
) -> MotionEvidence:
    return run_motion_evidence(
        frames,
        scene,
        output_dir,
        policy=_policy(),
        model_factory=FakeFactory(events) if factory is None else factory,
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release(events),
    )


def test_motion_producer_publishes_exact_pairs_maps_and_deterministic_manifests(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    first_events: list[str] = []
    first = run_motion_evidence(
        frames,
        scene,
        (tmp_path / "motion-a").resolve(),
        policy=_policy(),
        model_factory=FakeFactory(first_events),
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release(first_events),
    )

    assert first_events == ["factory:model-1", "infer:model-1:4", "release:model-1"]
    assert tuple(frame.frame for frame in first.frames) == frames
    assert all(path.is_absolute() for frame in first.frames for path in (
        frame.confirmed_without_semantic_path,
        frame.requires_semantic_path,
        frame.uncertain_path,
        frame.strength_path,
    ))
    for index, frame in enumerate(first.frames):
        binary_paths = (
            frame.confirmed_without_semantic_path,
            frame.requires_semantic_path,
            frame.uncertain_path,
        )
        assert all(path.suffix == ".png" for path in binary_paths)
        binary_maps: list[np.ndarray] = []
        for path in binary_paths:
            with Image.open(path) as image:
                assert image.format == "PNG"
                assert image.mode == "L"
                assert image.info == {}
                binary_maps.append(np.asarray(image).copy())
        confirmed, required, uncertain = binary_maps
        strength = np.load(frame.strength_path, allow_pickle=False)
        assert confirmed.dtype == np.uint8
        assert set(np.unique(confirmed)).issubset({0, 255})
        assert set(np.unique(required)).issubset({0, 255})
        assert set(np.unique(uncertain)).issubset({0, 255})
        if index == 0:
            assert confirmed[0, 0] == 255
        assert confirmed[1, 1] == 0
        assert required[1, 1] == 0
        assert uncertain[1, 1] == 0
        assert frame.strength_path.suffix == ".npy"
        assert strength.dtype == np.float32
        assert np.isfinite(strength).all()
    pair_payload = json.loads(first.pair_manifest_path.read_text(encoding="utf-8"))
    assert [(pair["source_frame_id"], pair["target_frame_id"]) for pair in pair_payload["pairs"]] == [
        ("frame-0", "frame-1"),
        ("frame-1", "frame-2"),
    ]
    assert pair_payload["model"] == {
        "code_commit": FakeSeaRaft.code_commit,
        "model_id": FakeSeaRaft.model_id,
        "revision": FakeSeaRaft.revision,
    }
    assert pair_payload["geometry_digest"] == scene.geometry_digest
    assert pair_payload["depth_digest"] == scene.depth_digest
    assert pair_payload["scene_digest"] == first.scene_digest
    assert pair_payload["residual_threshold_pixels"] == pytest.approx(0.2)
    manifest_payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["geometry_digest"] == scene.geometry_digest
    assert manifest_payload["depth_digest"] == scene.depth_digest
    assert manifest_payload["scene_digest"] == first.scene_digest

    second_events: list[str] = []
    second = run_motion_evidence(
        frames,
        scene,
        (tmp_path / "motion-b").resolve(),
        policy=_policy(),
        model_factory=FakeFactory(second_events),
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release(second_events),
    )
    assert first.pair_manifest_sha256 == second.pair_manifest_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.pair_manifest_path.read_bytes() == second.pair_manifest_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    for first_frame, second_frame in zip(first.frames, second.frames):
        assert first_frame.confirmed_without_semantic_path.read_bytes() == (
            second_frame.confirmed_without_semantic_path.read_bytes()
        )
        assert first_frame.requires_semantic_path.read_bytes() == (
            second_frame.requires_semantic_path.read_bytes()
        )
        assert first_frame.uncertain_path.read_bytes() == (
            second_frame.uncertain_path.read_bytes()
        )


def test_motion_manifest_serializes_unavailable_uncertainty_threshold_as_null(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(
                name,
                log,
                flow_x=10.0,
                static_sample_x=None,
            ),
        ),
    )

    result = _run_motion_case(
        frames,
        scene,
        (tmp_path / "motion").resolve(),
        events,
        factory=factory,
    )

    payload = json.loads(result.pair_manifest_path.read_text(encoding="utf-8"))
    assert payload["residual_threshold_pixels"] == pytest.approx(10.0)
    assert all(
        pair[direction] is None
        for pair in payload["pairs"]
        for direction in (
            "forward_uncertainty_threshold",
            "backward_uncertainty_threshold",
        )
    )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("model_id", "unapproved/sea-raft"),
        ("revision", "0" * 40),
        ("code_commit", None),
    ),
)
def test_motion_producer_rejects_unpinned_initial_model_before_inference(
    tmp_path: Path,
    field_name: str,
    wrong_value: object,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(
                name,
                log,
                provenance_overrides={field_name: wrong_value},
            ),
        ),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="pinned"):
        _run_motion_case(frames, scene, output_dir, events, factory=factory)

    assert events == ["factory:model-1", "release:model-1"]
    assert not os.path.lexists(output_dir)


def test_motion_producer_rejects_unpinned_retry_model_before_inference(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    oom = FakeCudaOutOfMemoryError("typed oom")
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(name, log, error=oom),
            lambda name, log: FakeSeaRaft(
                name,
                log,
                provenance_overrides={"revision": "0" * 40},
            ),
        ),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="pinned"):
        _run_motion_case(frames, scene, output_dir, events, factory=factory)

    assert events == [
        "factory:model-1",
        "infer:model-1:4",
        "release:model-1",
        "factory:model-2",
        "release:model-2",
    ]
    assert not os.path.lexists(output_dir)


@pytest.mark.parametrize(
    "component",
    (
        "w2c",
        "intrinsics",
        "registered",
        "depth_hash",
        "static_tracks",
        "geometry_digest",
        "depth_digest",
    ),
)
def test_scene_digest_binds_actual_ordered_scene_evidence(
    tmp_path: Path,
    component: str,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    baseline_events: list[str] = []
    baseline = _run_motion_case(
        frames,
        scene,
        (tmp_path / "baseline").resolve(),
        baseline_events,
    )
    first = scene.frames[0]
    if component == "w2c":
        w2c = list(first.w2c_4x4 or ())
        w2c[3] = 0.25
        changed = replace(
            scene,
            frames=(replace(first, w2c_4x4=tuple(w2c)), *scene.frames[1:]),
        )
    elif component == "intrinsics":
        changed = replace(
            scene,
            frames=(
                replace(first, pinhole_fx_fy_cx_cy=(2.5, 2.0, 1.0, 1.0)),
                *scene.frames[1:],
            ),
        )
    elif component == "registered":
        changed = replace(
            scene,
            frames=(
                replace(
                    first,
                    registered=False,
                    w2c_4x4=None,
                    pinhole_fx_fy_cx_cy=None,
                    depth_path=None,
                    depth_sha256=None,
                ),
                *scene.frames[1:],
            ),
        )
    elif component == "depth_hash":
        assert first.depth_path is not None
        with first.depth_path.open("wb") as handle:
            np.save(handle, np.full((3, 3), 2.5, dtype=np.float32), allow_pickle=False)
        changed = replace(
            scene,
            frames=(
                replace(first, depth_sha256=_sha256(first.depth_path)),
                *scene.frames[1:],
            ),
        )
    elif component == "static_tracks":
        track = scene.static_tracks[0]
        changed_observations = (
            replace(track.observations[0], x=0.9),
            *track.observations[1:],
        )
        changed = replace(
            scene,
            static_tracks=(replace(track, observations=changed_observations),),
        )
    elif component == "geometry_digest":
        changed = replace(scene, geometry_digest="e" * 64)
    else:
        changed = replace(scene, depth_digest="f" * 64)

    changed_events: list[str] = []
    result = _run_motion_case(
        frames,
        changed,
        (tmp_path / "changed").resolve(),
        changed_events,
    )

    assert result.scene_digest != baseline.scene_digest


def test_scene_digest_is_portable_across_identical_depth_artifact_locations(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    baseline_events: list[str] = []
    baseline = _run_motion_case(
        frames,
        scene,
        (tmp_path / "baseline").resolve(),
        baseline_events,
    )
    relocated_depth_dir = (tmp_path / "relocated-depths").resolve()
    relocated_depth_dir.mkdir()
    relocated_rigid_frames: list[RigidFrameEvidence] = []
    for index, rigid in enumerate(scene.frames):
        assert rigid.depth_path is not None
        relocated_path = relocated_depth_dir / f"depth-{index}.npy"
        relocated_path.write_bytes(rigid.depth_path.read_bytes())
        relocated_rigid_frames.append(replace(rigid, depth_path=relocated_path))
    relocated_scene = replace(scene, frames=tuple(relocated_rigid_frames))
    relocated_events: list[str] = []

    relocated = _run_motion_case(
        frames,
        relocated_scene,
        (tmp_path / "relocated").resolve(),
        relocated_events,
    )

    assert relocated.scene_digest == baseline.scene_digest


@pytest.mark.parametrize("value", (1.0e40, 1.0e-50))
def test_prediction_narrowing_rejects_float32_overflow_and_semantic_underflow(
    tmp_path: Path,
    value: float,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(
                name,
                log,
                flow_x=value,
                flow_dtype=np.float64,
                static_sample_x=None,
            ),
        ),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="float32"):
        _run_motion_case(frames, scene, output_dir, events, factory=factory)

    assert events == ["factory:model-1", "infer:model-1:4", "release:model-1"]
    assert not os.path.lexists(output_dir)


@pytest.mark.parametrize("value", (3.0e38, 1.0e-50, 1.0e-300))
def test_strength_conversion_rejects_float32_overflow_and_semantic_underflow(
    tmp_path: Path,
    value: float,
) -> None:
    values = list(_gate_inputs(tmp_path))
    forward = np.zeros((3, 3, 2), dtype=np.float64)
    forward[..., 0] = value
    forward[..., 1] = value
    backward = -forward
    values[4] = forward
    values[5] = backward

    with pytest.raises(ValueError, match="strength.*float32"):
        evaluate_flow_pair(
            *values,
            policy=_policy(),
            residual_threshold=0.0,
        )


def test_motion_producer_rejects_noncanonical_traversal_output_before_factory(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    holder = tmp_path / "holder"
    holder.mkdir()
    output_dir = holder / ".." / "escaped"
    events: list[str] = []

    with pytest.raises(ValueError, match="canonical"):
        _run_motion_case(frames, scene, output_dir, events)

    assert events == []
    assert not os.path.lexists(output_dir)


def test_motion_producer_rejects_symlinked_output_ancestor_before_factory(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    real_parent = tmp_path / "real-parent"
    (real_parent / "nested").mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    output_dir = linked_parent / "nested" / "motion"
    events: list[str] = []

    with pytest.raises(ValueError, match="canonical"):
        _run_motion_case(frames, scene, output_dir, events)

    assert events == []
    assert not os.path.lexists(output_dir)


def test_motion_producer_rejects_fewer_than_two_frames_before_callbacks(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    one_frame = frames[:1]
    one_scene = replace(
        scene,
        frames=scene.frames[:1],
        static_tracks=(
            replace(
                scene.static_tracks[0],
                observations=scene.static_tracks[0].observations[:1],
            ),
        ),
    )
    output_dir = (tmp_path / "motion").resolve()
    events: list[str] = []

    with pytest.raises(ValueError, match="at least two frames"):
        _run_motion_case(one_frame, one_scene, output_dir, events)

    assert events == []
    assert not os.path.lexists(output_dir)


@pytest.mark.parametrize("artifact", ("source_image", "depth"))
def test_motion_producer_revalidates_inputs_after_inference_before_staging(
    tmp_path: Path,
    artifact: str,
) -> None:
    frames, scene = _motion_fixture(tmp_path)

    def mutate_input() -> None:
        if artifact == "source_image":
            frames[0].path.write_bytes(b"changed-during-inference")
            return
        depth_path = scene.frames[0].depth_path
        assert depth_path is not None
        with depth_path.open("wb") as handle:
            np.save(handle, np.ones((2, 2), dtype=np.float32), allow_pickle=False)

    events: list[str] = []
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(
                name,
                log,
                after_infer=mutate_input,
            ),
        ),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="changed during SEA-RAFT inference"):
        _run_motion_case(frames, scene, output_dir, events, factory=factory)

    assert events == ["factory:model-1", "infer:model-1:4", "release:model-1"]
    assert not os.path.lexists(output_dir)


def test_motion_producer_rejects_scene_reorder_before_factory_or_filesystem(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="exact frame order"):
        run_motion_evidence(
            frames,
            replace(scene, frames=tuple(reversed(scene.frames))),
            output_dir,
            policy=_policy(),
            model_factory=FakeFactory(events),
            initial_pair_batch_size=4,
            retry_pair_batch_size=2,
            release_model=_release(events),
        )

    assert events == []
    assert not os.path.lexists(output_dir)


def test_motion_producer_rejects_prediction_pair_reorder_and_releases(
    tmp_path: Path,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    factory = FakeFactory(
        events,
        adapters=(lambda name, log: FakeSeaRaft(name, log, reverse_predictions=True),),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError, match="exact pair order"):
        run_motion_evidence(
            frames,
            scene,
            output_dir,
            policy=_policy(),
            model_factory=factory,
            initial_pair_batch_size=4,
            retry_pair_batch_size=2,
            release_model=_release(events),
        )

    assert events == ["factory:model-1", "infer:model-1:4", "release:model-1"]
    assert not os.path.lexists(output_dir)


def test_typed_cuda_oom_releases_and_recreates_exact_factory_once(tmp_path: Path) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    oom = FakeCudaOutOfMemoryError("typed oom")
    factory = FakeFactory(
        events,
        adapters=(
            lambda name, log: FakeSeaRaft(name, log, error=oom),
            lambda name, log: FakeSeaRaft(name, log),
        ),
    )

    result = run_motion_evidence(
        frames,
        scene,
        (tmp_path / "motion").resolve(),
        policy=_policy(),
        model_factory=factory,
        initial_pair_batch_size=4,
        retry_pair_batch_size=2,
        release_model=_release(events),
    )

    assert events == [
        "factory:model-1",
        "infer:model-1:4",
        "release:model-1",
        "factory:model-2",
        "infer:model-2:2",
        "release:model-2",
    ]
    assert factory.created == 2
    assert result.stage_records[0].status == "fallback"


def test_non_oom_identity_is_preserved_and_model_is_released(tmp_path: Path) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    error = ValueError("malformed flow")
    factory = FakeFactory(
        events,
        adapters=(lambda name, log: FakeSeaRaft(name, log, error=error),),
    )
    output_dir = (tmp_path / "motion").resolve()

    with pytest.raises(ValueError) as caught:
        run_motion_evidence(
            frames,
            scene,
            output_dir,
            policy=_policy(),
            model_factory=factory,
            initial_pair_batch_size=4,
            retry_pair_batch_size=2,
            release_model=_release(events),
        )

    assert caught.value is error
    assert events == ["factory:model-1", "infer:model-1:4", "release:model-1"]
    assert not os.path.lexists(output_dir)


def test_promotion_failure_leaves_no_final_or_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, scene = _motion_fixture(tmp_path)
    events: list[str] = []
    output_dir = (tmp_path / "motion").resolve()
    promotion_error = OSError("promotion failed")

    def fail_promotion(_staging: Path, _target: Path) -> None:
        raise promotion_error

    monkeypatch.setattr(
        "experiments.learned_quality.flow.promote_directory",
        fail_promotion,
    )
    with pytest.raises(OSError) as caught:
        run_motion_evidence(
            frames,
            scene,
            output_dir,
            policy=_policy(),
            model_factory=FakeFactory(events),
            initial_pair_batch_size=4,
            retry_pair_batch_size=2,
            release_model=_release(events),
        )

    assert caught.value is promotion_error
    assert not os.path.lexists(output_dir)
    assert not tuple(tmp_path.glob(".motion.staging-*"))
