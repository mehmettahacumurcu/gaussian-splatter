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
            x=1.1 if index < 3 else 11.0,
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

    assert calibrate_residual_threshold(scene, _policy()) == pytest.approx(0.1)
    insufficient = replace(
        scene,
        static_tracks=(replace(long_track, observations=observations[:2]),),
    )
    assert calibrate_residual_threshold(insufficient, _policy()) is None


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
    model_id = "facebook/sea-raft"
    revision = "1" * 40
    code_commit = "2" * 40

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        error: Exception | None = None,
        reverse_predictions: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.error = error
        self.reverse_predictions = reverse_predictions

    def infer_bidirectional(
        self,
        pairs: tuple[FlowPairRequest, ...],
        *,
        batch_size: int,
    ) -> tuple[SeaRaftPairPrediction, ...]:
        self.events.append(f"infer:{self.name}:{batch_size}")
        if self.error is not None:
            raise self.error
        predictions = tuple(
            SeaRaftPairPrediction(
                source_frame_id=pair.source_frame_id,
                target_frame_id=pair.target_frame_id,
                forward_flow=np.dstack(
                    (
                        np.full((3, 3), 0.5, dtype=np.float32),
                        np.zeros((3, 3), dtype=np.float32),
                    )
                ),
                backward_flow=np.dstack(
                    (
                        np.full((3, 3), -0.5, dtype=np.float32),
                        np.zeros((3, 3), dtype=np.float32),
                    )
                ),
                forward_uncertainty=np.zeros((3, 3), dtype=np.float32),
                backward_uncertainty=np.zeros((3, 3), dtype=np.float32),
            )
            for pair in pairs
        )
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
    for frame in first.frames:
        confirmed = np.load(frame.confirmed_without_semantic_path, allow_pickle=False)
        required = np.load(frame.requires_semantic_path, allow_pickle=False)
        uncertain = np.load(frame.uncertain_path, allow_pickle=False)
        strength = np.load(frame.strength_path, allow_pickle=False)
        assert confirmed.dtype == np.uint8
        assert confirmed[1, 1] == 255
        assert required[1, 1] == 0
        assert uncertain[1, 1] == 0
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
