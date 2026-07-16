from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.static_pipeline.contracts import (
    ColmapAttempt,
    FrameRecord,
    GateDecision,
    ModelMetrics,
    SelectionManifest,
    SelectionPolicy,
    UncoveredInterval,
)
from experiments.learned_quality.contracts import (
    FrameArtifact,
    GeometryCandidateReport,
    LearnedArtifacts,
)
from experiments.learned_quality.da3 import (
    AnchorInferenceResult,
    CameraRecord,
    FramePredictionArtifact,
    PinholeCamera,
)
from experiments.learned_quality.geometry import (
    GeometryComparisonError,
    HybridGeometryInputs,
    _rotation_matrix_to_qvec,
    closest_failure_key,
    focal_refinement_is_safe,
    geometry_frame_set_digest,
    make_classical_candidate_runner,
    make_hybrid_candidate_runner,
    run_geometry_comparison,
    winner_key,
)
from experiments.learned_quality.masks import (
    FusedMaskFrame,
    MaskFusionEvidence,
    MaskFusionPolicy,
)


def _manifest(count: int, *, digest: str = "a" * 64) -> SelectionManifest:
    frames = tuple(
        FrameRecord(
            frame_id=f"{index:024x}",
            source_relative_path="capture.mov",
            source_index=index,
            source_pts=index * 1_000,
            timestamp_s=index * 0.5,
            output_name=f"frame_{index:06d}.png",
            sha256=hashlib.sha256(f"frame-{index}".encode("ascii")).hexdigest(),
            selected=True,
            metrics=None,
            selection_score=None,
            reasons=("smart",),
        )
        for index in range(count)
    )
    return SelectionManifest(
        schema_version=1,
        source_digest="f" * 64,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=count,
            resolution_long_edge_cap=1280,
        ),
        frames=frames,
        image_set_digest=digest,
    )


def _artifacts(root: Path, manifest: SelectionManifest) -> tuple[FrameArtifact, ...]:
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index, frame in enumerate(manifest.selected_frames):
        path = root / frame.output_name
        path.write_bytes(f"frame-{index}".encode("ascii"))
        result.append(
            FrameArtifact(
                image_name=frame.output_name,
                frame_id=frame.frame_id,
                path=path,
                sha256=frame.sha256,
            )
        )
    return tuple(result)


def _image_artifacts(
    root: Path,
    manifest: SelectionManifest,
) -> tuple[SelectionManifest, tuple[FrameArtifact, ...]]:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    artifacts = []
    for index, record in enumerate(manifest.selected_frames):
        path = root / record.output_name
        pixels = np.full((20, 24, 3), 20 + index, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        updated = dataclasses.replace(record, sha256=digest)
        records.append(updated)
        artifacts.append(
            FrameArtifact(
                image_name=updated.output_name,
                frame_id=updated.frame_id,
                path=path,
                sha256=digest,
            )
        )
    updated_manifest = dataclasses.replace(
        manifest,
        frames=tuple(records),
        image_set_digest=hashlib.sha256(
            "".join(record.sha256 for record in records).encode("ascii")
        ).hexdigest(),
    )
    return updated_manifest, tuple(artifacts)


def _metrics(
    model_dir: Path,
    manifest: SelectionManifest,
    *,
    registered_ratio: float = 0.95,
    registered_count: int | None = None,
    registered_share: float = 1.0,
    max_interior_gap_s: float = 0.5,
    start_gap_s: float = 0.0,
    end_gap_s: float = 0.0,
    median_error: float = 0.4,
    p95_error: float = 1.2,
    median_track: float = 4.0,
    sparse_points: int = 1_000,
    valid: bool = True,
) -> ModelMetrics:
    selected = manifest.selected_frames
    if registered_count is None:
        registered_count = max(1, round(len(selected) * registered_ratio))
    names = frozenset(frame.output_name for frame in selected[:registered_count])
    return ModelMetrics(
        model_dir=model_dir,
        registered_names=names,
        registered_count=registered_count,
        registered_ratio=registered_ratio,
        registered_share=registered_share,
        temporal_coverage_s=max(0.0, (registered_count - 1) * 0.5),
        max_interior_gap_s=max_interior_gap_s,
        start_gap_s=start_gap_s,
        end_gap_s=end_gap_s,
        median_reprojection_error_px=median_error,
        p95_reprojection_error_px=p95_error,
        median_track_length=median_track,
        sparse_point_count=sparse_points,
        valid_names_intrinsics_and_poses=valid,
        coverage_unit="seconds",
    )


def _decision(
    metrics: ModelMetrics,
    *,
    passed: bool,
    failures: tuple[str, ...] = (),
    retry: bool = False,
) -> GateDecision:
    intervals = (
        UncoveredInterval(
            start_s=0.0,
            end_s=1.0,
            missing_frame_ids=("missing",),
            kind="interior",
            left_boundary_frame_id="left",
            right_boundary_frame_id="right",
        ),
    )
    return GateDecision(
        passed=passed,
        dominant=metrics,
        failures=failures,
        uncovered_intervals=intervals if not passed else (),
        retry_recommended=retry,
    )


def _candidate(
    root: Path,
    *,
    candidate_id: str = "classical",
    metrics: ModelMetrics | None = None,
    decision: GateDecision | None = None,
) -> GeometryCandidateReport:
    manifest = _manifest(100)
    model = root / candidate_id / "sparse" / "0"
    database = root / candidate_id / "colmap.db"
    attempt = ColmapAttempt(
        root=root / candidate_id,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.12.6",
        fingerprint="c" * 64,
    )
    dominant = metrics or _metrics(model, manifest)
    return GeometryCandidateReport(
        candidate_id=candidate_id,
        attempt=attempt,
        decision=decision or _decision(dominant, passed=True),
        model_dir=model,
        selected_manifest=manifest,
        frame_set_digest="d" * 64,
        covered_endpoint_count=2,
    )


def test_winner_key_is_exact(tmp_path: Path) -> None:
    manifest = _manifest(100)
    model = tmp_path / "model"
    dominant = _metrics(
        model,
        manifest,
        registered_ratio=0.95123456789,
        registered_count=95,
        registered_share=0.97123456789,
        max_interior_gap_s=0.41234567891,
        median_error=0.41234567891,
        p95_error=1.61234567891,
        median_track=4.61234567891,
        sparse_points=12_345,
    )
    candidate = _candidate(tmp_path, metrics=dominant)

    assert winner_key(candidate) == (
        round(-candidate.decision.dominant.registered_ratio, 9),
        -candidate.decision.dominant.registered_count,
        round(-candidate.decision.dominant.registered_share, 9),
        -candidate.covered_endpoint_count,
        round(candidate.decision.dominant.max_interior_gap_s, 9),
        round(candidate.decision.dominant.median_reprojection_error_px, 9),
        round(candidate.decision.dominant.p95_reprojection_error_px, 9),
        round(-candidate.decision.dominant.median_track_length, 9),
        -candidate.decision.dominant.sparse_point_count,
        candidate.candidate_id,
    )


def test_closest_failure_key_uses_normalized_gate_deficits(tmp_path: Path) -> None:
    manifest = _manifest(100)
    model = tmp_path / "model"
    dominant = _metrics(
        model,
        manifest,
        registered_ratio=0.81,
        registered_count=81,
        registered_share=0.855,
        max_interior_gap_s=3.0,
        median_error=1.5,
        p95_error=2.0,
        median_track=2.4,
        valid=False,
    )
    failures = (
        "registered_ratio",
        "dominant_component",
        "interior_gap",
        "median_reprojection",
        "median_track_length",
        "valid_names_intrinsics_poses",
    )
    candidate = _candidate(
        tmp_path,
        metrics=dominant,
        decision=_decision(dominant, passed=False, failures=failures),
    )
    expected = (
        (0.90 - 0.81) / 0.90
        + (0.95 - 0.855) / 0.95
        + (3.0 - 2.0) / 2.0
        + (1.5 - 1.0) / 1.0
        + (3.0 - 2.4) / 3.0
        + 1.0
    )

    assert closest_failure_key(candidate) == (
        len(failures),
        round(expected, 9),
        "classical",
    )

    invalid = dataclasses.replace(
        dominant,
        median_reprojection_error_px=float("nan"),
    )
    malformed = dataclasses.replace(
        candidate,
        decision=_decision(invalid, passed=False, failures=failures),
    )
    with pytest.raises(ValueError, match="finite"):
        closest_failure_key(malformed)


def _attempt(attempt_dir: Path, candidate_id: str) -> ColmapAttempt:
    assert not attempt_dir.exists()
    model = attempt_dir / "sparse" / "0"
    model.mkdir(parents=True)
    database = attempt_dir / "colmap.db"
    database.write_bytes(candidate_id.encode("ascii"))
    return ColmapAttempt(
        root=attempt_dir,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.12.6",
        fingerprint=("a" if candidate_id == "classical" else "b") * 64,
    )


def test_comparison_runs_fresh_isolated_branches_on_identical_frame_digest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(20)
    frames = _artifacts(tmp_path / "frames", manifest)
    calls: list[tuple[str, int, str, Path, Path]] = []
    measured: dict[Path, ModelMetrics] = {}

    def runner(candidate_id: str):
        def run(
            current: SelectionManifest,
            current_frames: tuple[FrameArtifact, ...],
            attempt_dir: Path,
            attempt_index: int,
            frame_set_digest: str,
        ) -> ColmapAttempt:
            attempt = _attempt(attempt_dir, candidate_id)
            calls.append(
                (
                    candidate_id,
                    attempt_index,
                    frame_set_digest,
                    attempt.root,
                    attempt.database_path,
                )
            )
            ratio = 0.95 if candidate_id == "learned_hybrid" else 0.80
            measured[attempt.model_dirs[0]] = _metrics(
                attempt.model_dirs[0], current, registered_ratio=ratio
            )
            assert tuple(frame.frame_id for frame in current_frames) == tuple(
                frame.frame_id for frame in current.selected_frames
            )
            return attempt

        return run

    def evaluate(
        models: tuple[ModelMetrics, ...],
        current: SelectionManifest,
        attempt_index: int,
    ) -> GateDecision:
        model = models[0]
        passed = model.registered_ratio >= 0.90
        return _decision(
            model,
            passed=passed,
            failures=() if passed else ("registered_ratio",),
        )

    result = run_geometry_comparison(
        manifest,
        frames,
        output_root=tmp_path / "geometry",
        artifacts=LearnedArtifacts(),
        classical_runner=runner("classical"),
        hybrid_runner=runner("learned_hybrid"),
        materialize_backfill=lambda *args, **kwargs: pytest.fail("unexpected backfill"),
        measure_models_fn=lambda paths, current: tuple(
            measured[path] for path in paths
        ),
        evaluate_fn=evaluate,
        publish_model_fn=lambda source, target: source,
    )

    assert result.decision.passed is True
    assert result.decision.dominant.registered_ratio == 0.95
    assert result.accepted_model_dir == result.geometry_candidates[1].model_dir
    assert len(calls) == 2
    assert calls[0][2] == calls[1][2]
    assert calls[0][3] != calls[1][3]
    assert calls[0][4] != calls[1][4]
    assert calls[0][3].name == "classical"
    assert calls[1][3].name == "learned_hybrid"
    assert result.bundle.attempts == tuple(
        candidate.attempt for candidate in result.geometry_candidates
    )


def test_one_shared_backfill_reruns_both_branches_once_with_limit_two(
    tmp_path: Path,
) -> None:
    first_manifest = _manifest(10, digest="a" * 64)
    second_manifest = dataclasses.replace(
        _manifest(12, digest="b" * 64),
        policy=first_manifest.policy,
    )
    shared_frames = tmp_path / "frames"
    first_frames = _artifacts(shared_frames, first_manifest)
    second_frames = _artifacts(shared_frames, second_manifest)
    calls: list[tuple[str, int, str]] = []
    measured: dict[Path, ModelMetrics] = {}

    def runner(candidate_id: str):
        def run(
            current: SelectionManifest,
            current_frames: tuple[FrameArtifact, ...],
            attempt_dir: Path,
            attempt_index: int,
            frame_set_digest: str,
        ) -> ColmapAttempt:
            attempt = _attempt(attempt_dir, candidate_id)
            calls.append((candidate_id, attempt_index, frame_set_digest))
            passed = attempt_index == 1 and candidate_id == "learned_hybrid"
            ratio = (
                0.95
                if passed
                else (
                    0.89 if attempt_index == 0 and candidate_id == "classical" else 0.80
                )
            )
            measured[attempt.model_dirs[0]] = _metrics(
                attempt.model_dirs[0],
                current,
                registered_ratio=ratio,
            )
            return attempt

        return run

    def evaluate(
        models: tuple[ModelMetrics, ...],
        current: SelectionManifest,
        attempt_index: int,
    ) -> GateDecision:
        model = models[0]
        passed = model.registered_ratio >= 0.90
        decision = _decision(
            model,
            passed=passed,
            failures=() if passed else ("registered_ratio",),
            retry=(
                not passed and attempt_index == 0 and model.registered_ratio <= 0.80
            ),
        )
        if passed:
            return decision
        interval = dataclasses.replace(
            decision.uncovered_intervals[0],
            missing_frame_ids=(model.model_dir.parent.parent.name,),
        )
        return dataclasses.replace(decision, uncovered_intervals=(interval,))

    backfill_calls: list[tuple[SelectionManifest, int]] = []

    def backfill(
        current: SelectionManifest,
        intervals: tuple[UncoveredInterval, ...],
        *,
        max_per_interval: int,
    ) -> tuple[SelectionManifest, tuple[FrameArtifact, ...]]:
        assert intervals
        assert intervals[0].missing_frame_ids == ("learned_hybrid",)
        backfill_calls.append((current, max_per_interval))
        return second_manifest, second_frames

    result = run_geometry_comparison(
        first_manifest,
        first_frames,
        output_root=tmp_path / "geometry",
        artifacts=LearnedArtifacts(),
        classical_runner=runner("classical"),
        hybrid_runner=runner("learned_hybrid"),
        materialize_backfill=backfill,
        measure_models_fn=lambda paths, current: tuple(
            measured[path] for path in paths
        ),
        evaluate_fn=evaluate,
        publish_model_fn=lambda source, target: source,
    )

    assert backfill_calls == [(first_manifest, 2)]
    assert [call[:2] for call in calls] == [
        ("classical", 0),
        ("learned_hybrid", 0),
        ("classical", 1),
        ("learned_hybrid", 1),
    ]
    assert calls[0][2] == calls[1][2]
    assert calls[2][2] == calls[3][2]
    assert calls[0][2] != calls[2][2]
    assert result.selected_manifest == second_manifest
    assert result.decision.passed is True
    assert len(result.geometry_candidates) == 4


def test_all_failed_candidates_raise_with_normalized_closest_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest(10)
    frames = _artifacts(tmp_path / "frames", manifest)
    measured: dict[Path, ModelMetrics] = {}

    def runner(candidate_id: str, ratio: float):
        def run(
            current: SelectionManifest,
            current_frames: tuple[FrameArtifact, ...],
            attempt_dir: Path,
            attempt_index: int,
            frame_set_digest: str,
        ) -> ColmapAttempt:
            attempt = _attempt(attempt_dir, candidate_id)
            measured[attempt.model_dirs[0]] = _metrics(
                attempt.model_dirs[0], current, registered_ratio=ratio
            )
            return attempt

        return run

    def evaluate(
        models: tuple[ModelMetrics, ...],
        current: SelectionManifest,
        attempt_index: int,
    ) -> GateDecision:
        model = models[0]
        return _decision(
            model,
            passed=False,
            failures=("registered_ratio",),
            retry=False,
        )

    with pytest.raises(GeometryComparisonError) as raised:
        run_geometry_comparison(
            manifest,
            frames,
            output_root=tmp_path / "geometry",
            artifacts=LearnedArtifacts(),
            classical_runner=runner("classical", 0.80),
            hybrid_runner=runner("learned_hybrid", 0.89),
            materialize_backfill=lambda *args, **kwargs: pytest.fail(
                "unexpected backfill"
            ),
            measure_models_fn=lambda paths, current: tuple(
                measured[path] for path in paths
            ),
            evaluate_fn=evaluate,
            publish_model_fn=lambda source, target: source,
        )

    assert raised.value.closest_candidate.candidate_id == "learned_hybrid"
    assert all(not candidate.decision.passed for candidate in raised.value.candidates)


def test_classical_runner_wraps_production_attempt_with_exact_policy_and_digest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(20)
    frames = _artifacts(tmp_path / "frames", manifest)
    calls: list[tuple[Path, Path, object, Path | None]] = []

    def fake_run(
        frames_root: Path,
        attempt_dir: Path,
        policy: object,
        colmap_exe: Path | None,
    ) -> ColmapAttempt:
        calls.append((frames_root, attempt_dir, policy, colmap_exe))
        return _attempt(attempt_dir, "classical")

    runner = make_classical_candidate_runner(
        use_gpu=True,
        colmap_exe=tmp_path / "colmap",
        run_attempt_fn=fake_run,
    )
    digest = geometry_frame_set_digest(manifest, frames)
    result = runner(manifest, frames, tmp_path / "attempt", 1, digest)

    assert result.root == tmp_path / "attempt"
    assert len(calls) == 1
    frames_root, attempt_root, policy, executable = calls[0]
    assert frames_root == tmp_path / "frames"
    assert attempt_root == tmp_path / "attempt"
    assert executable == tmp_path / "colmap"
    assert policy.use_gpu is True
    assert policy.matcher == "exhaustive"


def test_focal_refinement_requires_improvement_no_gate_regression_and_five_percent_cap(
    tmp_path: Path,
) -> None:
    manifest = _manifest(100)
    before = _metrics(
        tmp_path / "before",
        manifest,
        median_error=0.8,
        p95_error=2.0,
    )
    improved = dataclasses.replace(
        before,
        model_dir=tmp_path / "improved",
        median_reprojection_error_px=0.7,
        p95_reprojection_error_px=1.9,
    )

    assert focal_refinement_is_safe(
        before,
        improved,
        initial_focal_xy=(1000.0, 1000.0),
        refined_focal_xy=(1049.0, 951.0),
    )
    assert not focal_refinement_is_safe(
        before,
        improved,
        initial_focal_xy=(1000.0, 1000.0),
        refined_focal_xy=(1051.0, 1000.0),
    )
    assert not focal_refinement_is_safe(
        before,
        dataclasses.replace(
            improved,
            median_track_length=2.9,
        ),
        initial_focal_xy=(1000.0, 1000.0),
        refined_focal_xy=(1000.0, 1000.0),
    )
    assert not focal_refinement_is_safe(
        before,
        dataclasses.replace(
            improved, median_reprojection_error_px=0.8, p95_reprojection_error_px=2.0
        ),
        initial_focal_xy=(1000.0, 1000.0),
        refined_focal_xy=(1000.0, 1000.0),
    )


def _hybrid_inputs(
    root: Path,
    frames: tuple[FrameArtifact, ...],
) -> HybridGeometryInputs:
    anchor_indices = tuple(range(min(3, len(frames))))
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    cameras = tuple(
        CameraRecord(
            image_name=frames[index].image_name,
            frame_id=frames[index].frame_id,
            w2c=identity,
        )
        for index in anchor_indices
    )
    predictions = tuple(
        FramePredictionArtifact(
            image_name=frames[index].image_name,
            frame_id=frames[index].frame_id,
            source_path=frames[index].path,
            depth_path=root / "depth" / f"{frames[index].frame_id}.npy",
            confidence_path=None,
            sky_path=None,
        )
        for index in anchor_indices
    )
    anchors = AnchorInferenceResult(
        artifacts=predictions,
        cameras=cameras,
        shared_camera=PinholeCamera("PINHOLE", 24, 20, 20.0, 20.0, 12.0, 10.0),
        anchor_indices=anchor_indices,
        attempts=(),
        metadata_path=root / "anchors.json",
    )
    policy = MaskFusionPolicy(0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.0, 0.0, 0.5)
    fused = []
    for frame in frames:
        mask_path = (
            root
            / "colmap_masks"
            / Path(*frame.image_name.split("/")).with_name(
                f"{Path(frame.image_name).name}.png"
            )
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(b"mask")
        digest = hashlib.sha256(b"mask").hexdigest()
        fused.append(
            FusedMaskFrame(
                frame=frame,
                width=24,
                height=20,
                semantic_confirmed_path=mask_path,
                semantic_confirmed_sha256=digest,
                sky_confirmed_path=mask_path,
                sky_confirmed_sha256=digest,
                motion_confirmed_path=mask_path,
                motion_confirmed_sha256=digest,
                uncertain_path=mask_path,
                uncertain_sha256=digest,
                hard_exclude_path=mask_path,
                hard_exclude_sha256=digest,
                colmap_keep_path=mask_path,
                colmap_keep_sha256=digest,
                training_validity_path=mask_path,
                training_validity_sha256=digest,
                exclusion_fraction_before_trim=0.0,
                exclusion_fraction_after_trim=0.0,
                decision="accepted",
                warnings=(),
            )
        )
    masks = MaskFusionEvidence(
        policy=policy,
        frames=tuple(fused),
        mask_set_digest="e" * 64,
        manifest_path=root / "mask-manifest.json",
        manifest_sha256="f" * 64,
    )
    return HybridGeometryInputs(anchors=anchors, masks=masks)


def test_hybrid_runner_forwards_exact_known_poses_masks_and_fresh_database(
    tmp_path: Path,
) -> None:
    manifest, frames = _image_artifacts(tmp_path / "frames", _manifest(5))
    inputs = _hybrid_inputs(tmp_path / "evidence", frames)
    calls: list[dict[str, object]] = []

    class FakeBackend:
        version = "PyCOLMAP 3.12.6-fake"

        def reconstruct(self, **kwargs: object) -> tuple[Path, ...]:
            calls.append(kwargs)
            database_path = kwargs["database_path"]
            sparse_root = kwargs["sparse_root"]
            assert isinstance(database_path, Path)
            assert isinstance(sparse_root, Path)
            database_path.write_bytes(b"database")
            model = sparse_root / "final" / "0"
            model.mkdir(parents=True)
            return (model,)

    runner = make_hybrid_candidate_runner(
        inputs,
        use_gpu=True,
        backend=FakeBackend(),
    )
    digest = geometry_frame_set_digest(manifest, frames)
    result = runner(manifest, frames, tmp_path / "hybrid", 0, digest)

    assert result.root == tmp_path / "hybrid"
    assert result.database_path == tmp_path / "hybrid" / "colmap.db"
    assert result.model_dirs == (tmp_path / "hybrid" / "sparse" / "final" / "0",)
    assert result.colmap_version == "PyCOLMAP 3.12.6-fake"
    assert len(calls) == 1
    call = calls[0]
    assert call["frames_root"] == tmp_path / "frames"
    assert call["mask_root"] == tmp_path / "evidence" / "colmap_masks"
    assert call["image_names"] == tuple(frame.image_name for frame in frames)
    assert call["anchor_cameras"] == inputs.anchors.cameras
    assert call["shared_camera"] == inputs.anchors.shared_camera
    assert call["use_gpu"] is True


def test_rotation_to_colmap_qvec_preserves_identity_and_right_handed_rotation() -> None:
    np.testing.assert_allclose(
        _rotation_matrix_to_qvec(np.eye(3)),
        (1.0, 0.0, 0.0, 0.0),
        atol=1e-12,
    )
    rotation = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    quaternion = _rotation_matrix_to_qvec(rotation)
    np.testing.assert_allclose(
        quaternion,
        (2**-0.5, 0.0, 0.0, 2**-0.5),
        atol=1e-12,
    )
