from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from backend.static_pipeline.progress import (
    StageDefinition,
    StageReporter,
    use_stage_reporter,
)
from backend.static_pipeline.contracts import ColmapAttempt
from backend.static_pipeline.runner import HardwareInfo, SelectionOutput
from experiments.learned_quality import runtime as runtime_module
from experiments.learned_quality.cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCheckpointStore,
    checkpoint_fingerprint,
)
from experiments.learned_quality.contracts import FrameArtifact, LearnedArtifacts
from experiments.learned_quality.runtime import (
    _colmap_cache_fingerprint,
    _run_evidence_cycle,
    run_learned_reconstruction,
)
from experiments.learned_quality.tracks import (
    TrackAuditReport,
    TrackQualificationError,
)


EVIDENCE_STAGES = tuple(
    StageDefinition(stage_id, label)
    for stage_id, label in (
        ("da3_anchor", "DA3 anchor inference"),
        ("classical_colmap", "Classical COLMAP prepass"),
        ("da3_metric_sky", "DA3 metric depth and sky"),
        ("semantic_masks", "Semantic masks"),
        ("optical_flow", "Optical flow"),
        ("mask_fusion", "Evidence mask fusion"),
    )
)

FINAL_STAGES = tuple(
    StageDefinition(stage_id, label)
    for stage_id, label in (
        ("geometry_comparison", "Geometry comparison"),
        ("photometric_validation", "Photometric validation"),
        ("final_pose_depth", "Final-pose depth"),
        ("dense_seed_fusion", "Dense seed fusion"),
    )
)


def _mock_evidence_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SelectionOutput, tuple[FrameArtifact, ...], HardwareInfo]:
    model = tmp_path / "model"
    model.mkdir()
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"png")
    selection = cast(SelectionOutput, SimpleNamespace(manifest=object()))
    frames = (FrameArtifact("frame.png", "frame-1", frame_path, "a" * 64),)
    hardware = HardwareInfo("NVIDIA A100", 80.0, True, 100.0, True)
    anchors = SimpleNamespace(shared_camera=object(), anchor_indices=(0,))
    metric = SimpleNamespace(artifacts=(object(),))
    semantic = SimpleNamespace(stage_records=())
    flow = SimpleNamespace(stage_records=())
    masks = SimpleNamespace(mask_set_digest="b" * 64)

    monkeypatch.setattr(runtime_module, "load_da3_model", lambda *_args: object())
    monkeypatch.setattr(runtime_module, "release_cuda_model", lambda _model: None)
    monkeypatch.setattr(
        runtime_module,
        "run_anchor_inference",
        lambda *_args, **_kwargs: anchors,
    )
    monkeypatch.setattr(
        runtime_module,
        "make_classical_candidate_runner",
        lambda **_kwargs: (
            lambda *_args, **_inner: SimpleNamespace(model_dirs=(model,))
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "geometry_frame_set_digest",
        lambda *_args: "c" * 64,
    )
    monkeypatch.setattr(
        runtime_module,
        "measure_models",
        lambda *_args: (
            SimpleNamespace(
                registered_count=1,
                sparse_point_count=10,
                model_dir=model,
            ),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "run_metric_sky",
        lambda *_args, **_kwargs: metric,
    )
    monkeypatch.setattr(
        runtime_module,
        "_materialize_metric",
        lambda *_args: (((tmp_path / "depth.npy", "d" * 64),), object()),
    )
    monkeypatch.setattr(
        runtime_module, "_rigid_scene", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runtime_module,
        "qualify_colmap_static_tracks",
        lambda *_args, **_kwargs: SimpleNamespace(tracks=()),
    )
    monkeypatch.setattr(
        runtime_module,
        "run_semantic_evidence",
        lambda *_args, **_kwargs: semantic,
    )
    monkeypatch.setattr(
        runtime_module,
        "run_motion_evidence",
        lambda *_args, **_kwargs: flow,
    )
    monkeypatch.setattr(
        runtime_module,
        "fuse_evidence_masks",
        lambda *_args, **_kwargs: masks,
    )
    monkeypatch.setattr(
        runtime_module,
        "_validate_model_manifest",
        lambda: manifest_path,
    )
    return selection, frames, hardware


def test_colmap_cache_fingerprint_binds_preprocessing_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_bytes(b"verified manifest")
    selection = cast(
        SelectionOutput,
        SimpleNamespace(
            inventory=SimpleNamespace(digest="a" * 64),
            manifest=SimpleNamespace(image_set_digest="b" * 64),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "producer_code_digest",
        lambda *_args: "c" * 64,
    )

    actual = _colmap_cache_fingerprint(
        selection,
        HardwareInfo("NVIDIA A100", 80.0, True, 100.0, True),
        model_manifest,
        version_probe=lambda: "COLMAP 3.11.1",
    )

    assert actual == checkpoint_fingerprint(
        CheckpointKind.COLMAP,
        CheckpointInputs(
            source_digest="a" * 64,
            settings={
                "selection_digest": "b" * 64,
                "use_gpu": True,
            },
            model_manifest_sha256=runtime_module._sha256(model_manifest),
            tool_versions={"colmap": "COLMAP 3.11.1"},
            producer_code_sha256="c" * 64,
        ),
    )


def test_evidence_cycle_prints_every_expensive_substage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, frames, hardware = _mock_evidence_cycle(tmp_path, monkeypatch)
    output = StringIO()
    reporter = StageReporter(EVIDENCE_STAGES, stream=output)

    with use_stage_reporter(reporter):
        _run_evidence_cycle(selection, frames, hardware, tmp_path / "evidence")

    starts = [
        line.split(" START ", 1)[1]
        for line in output.getvalue().splitlines()
        if " START " in line
    ]
    assert starts == [definition.label for definition in EVIDENCE_STAGES]


def test_evidence_cycle_reports_failure_and_preserves_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, frames, hardware = _mock_evidence_cycle(tmp_path, monkeypatch)
    error = RuntimeError("flow failed")

    def fail_flow(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(runtime_module, "run_motion_evidence", fail_flow)
    output = StringIO()
    reporter = StageReporter(EVIDENCE_STAGES, stream=output)

    with pytest.raises(RuntimeError) as caught:
        with use_stage_reporter(reporter):
            _run_evidence_cycle(selection, frames, hardware, tmp_path / "evidence")

    assert caught.value is error
    assert any("FAIL  Optical flow" in line for line in output.getvalue().splitlines())


def test_evidence_cycle_qualifies_tracks_before_metric_or_semantic_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, frames, hardware = _mock_evidence_cycle(tmp_path, monkeypatch)
    audit_path = tmp_path / "evidence" / "track_audit.json"
    report = TrackAuditReport(
        schema_version=1,
        raw_track_count=1,
        accepted_track_count=0,
        rejected_track_count=1,
        raw_observation_count=1,
        accepted_observation_count=0,
        rejected_observation_count=1,
        rejected_tracks_by_reason={"insufficient_observations": 1},
        rejected_observations_by_reason={"out_of_bounds": 1},
        examples=(),
        accepted_tracks_sha256="a" * 64,
    )

    def reject_tracks(
        _model: Path,
        _frames: tuple[FrameArtifact, ...],
        target: Path,
        **_kwargs: object,
    ) -> object:
        target.write_text('{"schema_version":1}\n', encoding="utf-8")
        raise TrackQualificationError(
            "no qualified COLMAP static tracks remain",
            report=report,
            audit_path=target,
        )

    def expensive_model_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-COLMAP learned model ran before track qualification")

    monkeypatch.setattr(runtime_module, "qualify_colmap_static_tracks", reject_tracks)
    monkeypatch.setattr(runtime_module, "run_metric_sky", expensive_model_must_not_run)
    monkeypatch.setattr(
        runtime_module, "run_semantic_evidence", expensive_model_must_not_run
    )

    with pytest.raises(TrackQualificationError, match="no qualified"):
        _run_evidence_cycle(selection, frames, hardware, tmp_path / "evidence")

    assert audit_path.is_file()


def test_final_reconstruction_prints_geometry_depth_and_validation_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"png")
    frame = FrameArtifact("frame.png", "frame-1", frame_path, "a" * 64)
    manifest = object()
    selection = cast(
        SelectionOutput,
        SimpleNamespace(
            manifest=manifest,
            inventory=object(),
            frames_dir=tmp_path,
            source_manifest_path=tmp_path / "selection_manifest.json",
        ),
    )
    model = tmp_path / "model"
    model.mkdir()
    output_root = tmp_path / "reconstruction"
    metric = tmp_path / "learned-evidence-0" / "metric-native" / "frame.depth.npy"
    metric.parent.mkdir(parents=True)
    metric.write_bytes(b"depth")
    flow = SimpleNamespace(frames=())
    artifacts = LearnedArtifacts(da3=object(), flow=flow, masks=object())
    checkpoint_store = LearnedCheckpointStore(
        tmp_path / "drive-cache",
        input_identity="a" * 64,
    )
    observed_stores: list[LearnedCheckpointStore | None] = []

    def evidence_cycle(
        *_args: object,
        checkpoint_store: LearnedCheckpointStore | None = None,
    ) -> LearnedArtifacts:
        observed_stores.append(checkpoint_store)
        return artifacts

    comparison = SimpleNamespace(
        selected_manifest=manifest,
        accepted_model_dir=model,
        geometry_candidates=(object(),),
        bundle=cast(object, SimpleNamespace()),
        frames_dir=tmp_path,
    )
    photometric = SimpleNamespace(
        decision="accepted",
        training_rgb_digest="b" * 64,
    )
    final_depth = SimpleNamespace(artifacts=(object(),))
    dense_seeds = SimpleNamespace(point_count=100)
    validated = SimpleNamespace(dense_seeds=dense_seeds)
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        runtime_module, "_validate_model_manifest", lambda: manifest_path
    )
    monkeypatch.setattr(runtime_module, "_frame_artifacts", lambda _selection: (frame,))
    monkeypatch.setattr(
        runtime_module,
        "_run_evidence_cycle",
        evidence_cycle,
    )
    monkeypatch.setattr(
        runtime_module,
        "make_classical_candidate_runner",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_module,
        "run_geometry_comparison",
        lambda *_args, **_kwargs: comparison,
    )
    monkeypatch.setattr(
        runtime_module, "_rigid_scene", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runtime_module,
        "qualify_colmap_static_tracks",
        lambda *_args, **_kwargs: SimpleNamespace(tracks=()),
    )
    monkeypatch.setattr(
        runtime_module,
        "fit_and_validate_photometric_transforms",
        lambda *_args, **_kwargs: photometric,
    )
    monkeypatch.setattr(runtime_module, "load_da3_model", lambda *_args: object())
    monkeypatch.setattr(runtime_module, "release_cuda_model", lambda _model: None)
    monkeypatch.setattr(runtime_module, "_final_cameras", lambda *_args: ())
    monkeypatch.setattr(
        runtime_module,
        "run_pose_conditioned_depth",
        lambda *_args, **_kwargs: final_depth,
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_depth_and_fuse_seeds",
        lambda *_args, **_kwargs: validated,
    )
    output = StringIO()

    with use_stage_reporter(StageReporter(FINAL_STAGES, stream=output)):
        run_learned_reconstruction(
            selection,
            spec=object(),
            hardware=HardwareInfo("NVIDIA A100", 80.0, True, 100.0, True),
            output_root=output_root,
            checkpoint_store=checkpoint_store,
        )

    starts = [
        line.split(" START ", 1)[1]
        for line in output.getvalue().splitlines()
        if " START " in line
    ]
    assert starts == [definition.label for definition in FINAL_STAGES]
    assert observed_stores == [checkpoint_store]


def test_evidence_cycle_reuses_drive_colmap_checkpoint_in_fresh_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, frames, hardware = _mock_evidence_cycle(tmp_path, monkeypatch)
    cache_store = LearnedCheckpointStore(
        tmp_path / "drive-cache",
        input_identity="a" * 64,
    )
    calls: list[Path] = []
    measured_models: list[tuple[Path, ...]] = []

    def classical_factory(**_kwargs: object):
        def run(
            _manifest: object,
            _frames: object,
            attempt_root: Path,
            _attempt_index: int,
            _frame_digest: str,
        ) -> ColmapAttempt:
            calls.append(attempt_root)
            model = attempt_root / "sparse" / "0"
            model.mkdir(parents=True)
            database = attempt_root / "colmap.db"
            database.write_bytes(b"database")
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text(name, encoding="utf-8")
            return ColmapAttempt(
                root=attempt_root,
                database_path=database,
                model_dirs=(model,),
                colmap_version="COLMAP 3.11.1",
                fingerprint="d" * 64,
            )

        return run

    def measure(model_dirs: tuple[Path, ...], _manifest: object):
        measured_models.append(tuple(model_dirs))
        return (
            SimpleNamespace(
                registered_count=1,
                sparse_point_count=10,
                model_dir=model_dirs[0],
            ),
        )

    monkeypatch.setattr(
        runtime_module,
        "make_classical_candidate_runner",
        classical_factory,
    )
    monkeypatch.setattr(runtime_module, "measure_models", measure)
    monkeypatch.setattr(
        runtime_module,
        "_colmap_cache_fingerprints",
        lambda *_args, **_kwargs: ("e" * 64,),
    )
    output = StringIO()
    reporter = StageReporter(EVIDENCE_STAGES, stream=output)

    with use_stage_reporter(reporter):
        _run_evidence_cycle(
            selection,
            frames,
            hardware,
            tmp_path / "first-run" / "evidence",
            checkpoint_store=cache_store,
        )
        _run_evidence_cycle(
            selection,
            frames,
            hardware,
            tmp_path / "second-run" / "evidence",
            checkpoint_store=cache_store,
        )

    assert calls == [tmp_path / "first-run" / "evidence" / "classical-prepass"]
    assert len(measured_models) == 2
    assert measured_models[1][0].is_relative_to(tmp_path / "second-run")
    assert "[CACHE MISS] COLMAP checkpoint" in output.getvalue()
    assert "[CACHE SAVE] COLMAP checkpoint" in output.getvalue()
    assert "[CACHE HIT] COLMAP checkpoint" in output.getvalue()


def test_evidence_cycle_migrates_allowlisted_colmap_without_rerunning_colmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, frames, hardware = _mock_evidence_cycle(tmp_path, monkeypatch)
    cache_store = LearnedCheckpointStore(
        tmp_path / "drive-cache",
        input_identity="a" * 64,
    )
    legacy_fingerprint = "d" * 64
    current_fingerprint = "e" * 64
    attempt_root = tmp_path / "legacy-attempt"
    model = attempt_root / "sparse" / "0"
    model.mkdir(parents=True)
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (model / name).write_text(name, encoding="utf-8")
    cache_store.publish_colmap(
        ColmapAttempt(
            root=attempt_root,
            database_path=database,
            model_dirs=(model,),
            colmap_version="COLMAP 3.11.1",
            fingerprint="c" * 64,
        ),
        fingerprint=legacy_fingerprint,
        run_id="legacy",
    )
    validation_calls: list[str] = []

    monkeypatch.setattr(
        runtime_module,
        "_colmap_cache_fingerprints",
        lambda *_args, **_kwargs: (current_fingerprint, legacy_fingerprint),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "_colmap_cache_fingerprint",
        lambda *_args, **_kwargs: current_fingerprint,
    )
    monkeypatch.setattr(
        runtime_module,
        "make_classical_candidate_runner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("COLMAP reran despite a compatible legacy checkpoint")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "qualify_colmap_static_tracks",
        lambda *_args, **_kwargs: (
            validation_calls.append("tracks") or SimpleNamespace(tracks=())
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_rigid_scene",
        lambda *_args, **_kwargs: validation_calls.append("scene") or object(),
    )

    _run_evidence_cycle(
        selection,
        frames,
        hardware,
        tmp_path / "run" / "evidence",
        checkpoint_store=cache_store,
    )

    assert validation_calls == ["tracks", "scene"]
    assert (
        cache_store.find_generation(CheckpointKind.COLMAP, current_fingerprint)
        is not None
    )
