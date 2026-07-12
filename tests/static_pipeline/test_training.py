from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import backend.config as backend_config
from backend.notebooks.models import NotebookQualityProfile, StaticNotebookRunSpec
from backend.preprocess.cache_utils import compute_settings_hash
from backend.static_pipeline.contracts import (
    FrameRecord,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
    SelectionPolicy,
)
from backend.static_pipeline.training import (
    PreparedTrainingInput,
    TrainingResult,
    run_validated_training,
)
from tests.static_pipeline.fixtures import write_colmap_text_model


_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selection_digest(frames: tuple[FrameRecord, ...]) -> str:
    payload = [
        [frame.frame_id, frame.output_name, frame.sha256]
        for frame in frames
        if frame.selected
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _prepared(tmp_path: Path) -> PreparedTrainingInput:
    frames_dir = tmp_path / "validated-frames"
    frames_dir.mkdir()
    frame_bytes = (b"selected-frame-zero", b"selected-frame-one")
    records = []
    for index, content in enumerate(frame_bytes):
        output_name = f"frame_{index:06d}.png"
        (frames_dir / output_name).write_bytes(content)
        records.append(
            FrameRecord(
                frame_id=f"{index + 1:024x}",
                source_relative_path="capture.mov",
                source_index=index,
                source_pts=index * 1_000,
                timestamp_s=index / 4,
                output_name=output_name,
                sha256=_sha256_bytes(content),
                selected=True,
                metrics=None,
                selection_score=None,
                reasons=("smart",),
            )
        )
    (frames_dir / "ignored-extra.bin").write_bytes(b"do-not-copy")
    selected = tuple(records)
    source_digest = "a" * 64
    selection_digest = _selection_digest(selected)
    manifest = SelectionManifest(
        schema_version=1,
        source_digest=source_digest,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=300,
            resolution_long_edge_cap=1280,
        ),
        frames=selected,
        image_set_digest=selection_digest,
    )

    model_dir = write_colmap_text_model(
        tmp_path / "validated-model",
        tuple(frame.output_name for frame in selected),
        (0.2, 0.3, 0.4, 0.5),
        (4, 4, 4, 4),
    )
    (model_dir / "ignored-extra.db").write_bytes(b"do-not-copy")
    metrics = ModelMetrics(
        model_dir=model_dir,
        registered_names=frozenset(frame.output_name for frame in selected),
        registered_count=len(selected),
        registered_ratio=1.0,
        registered_share=1.0,
        temporal_coverage_s=0.25,
        max_interior_gap_s=0.0,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=0.2,
        p95_reprojection_error_px=0.4,
        median_track_length=4.0,
        sparse_point_count=4,
        valid_names_intrinsics_and_poses=True,
    )
    decision = GateDecision(
        passed=True,
        dominant=metrics,
        failures=(),
        uncovered_intervals=(),
        retry_recommended=False,
    )
    reconstruction = ReconstructionBundle(
        selected_manifest=manifest,
        accepted_model_dir=model_dir,
        decision=decision,
        attempts=(),
        decisions=(decision,),
    )
    return PreparedTrainingInput(
        run_id="run-001",
        data_root=tmp_path / "data-root",
        scene_name="room-001",
        frames_dir=frames_dir,
        reconstruction=reconstruction,
        source_digest=source_digest,
        selection_digest=selection_digest,
    )


def _spec(
    *,
    foundation: bool | None = None,
    profile: NotebookQualityProfile = NotebookQualityProfile.BALANCED_L4,
) -> StaticNotebookRunSpec:
    advanced: dict[str, object] = {}
    if foundation is not None:
        advanced["foundation"] = foundation
    return StaticNotebookRunSpec.model_validate(
        {
            "input_folder": "captures/room",
            "quality": {"profile": profile.value, "advanced": advanced},
        }
    )


class RecordingRunner:
    def __init__(
        self,
        *,
        create_ply: bool = True,
        failure: Exception | None = None,
        status: Mapping[str, object] | None = None,
    ) -> None:
        self.create_ply = create_ply
        self.failure = failure
        self.status = status or {"training": "ok", "export": "ok"}
        self.calls: list[dict[str, Any]] = []
        self.data_root_seen: Path | None = None

    def __call__(self, **kwargs: Any) -> Mapping[str, object]:
        self.calls.append(kwargs)
        self.data_root_seen = Path(backend_config.DATA_ROOT)
        scene = self.data_root_seen / str(kwargs["scene_name"])
        if self.create_ply:
            raw_ply = scene / "output" / "ply" / "frame_0000.ply"
            raw_ply.parent.mkdir(parents=True, exist_ok=True)
            raw_ply.write_bytes(b"fresh-inria-ply")
        if self.failure is not None:
            raise self.failure
        return self.status


@pytest.fixture(autouse=True)
def _restore_process_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOURDGS_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        backend_config,
        "DATA_ROOT",
        Path(backend_config.PROJECT_ROOT) / "data",
    )


def test_gate_failure_precedes_runner_and_data_root_mutation(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    failed = replace(
        prepared.reconstruction.decision,
        passed=False,
        failures=("registered_ratio",),
    )
    prepared = replace(
        prepared,
        reconstruction=replace(
            prepared.reconstruction,
            decision=failed,
            decisions=(failed,),
        ),
    )
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="passing reconstruction"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()
    assert "FOURDGS_DATA_ROOT" not in os.environ


def test_success_copies_exact_validated_inputs_binds_root_and_calls_legacy_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    runner = RecordingRunner(status={"training": "done", "export": "done"})

    import backend.notebooks.training_config as training_config

    real_resolver = training_config.resolve_static_training_config
    resolver_roots: list[Path] = []

    def checking_resolver(
        spec: StaticNotebookRunSpec,
        source_long_edge: int | None = None,
        *,
        native_image_size: tuple[int, int] | None = None,
    ) -> tuple[object, object]:
        resolver_roots.append(Path(os.environ["FOURDGS_DATA_ROOT"]))
        return real_resolver(
            spec,
            source_long_edge=source_long_edge,
            native_image_size=native_image_size,
        )

    monkeypatch.setattr(
        training_config,
        "resolve_static_training_config",
        checking_resolver,
    )

    result = run_validated_training(
        prepared,
        _spec(),
        source_long_edge=1600,
        pipeline_runner=runner,
    )

    scene = prepared.data_root / prepared.scene_name
    expected_root = prepared.data_root.resolve()
    assert isinstance(result, TrainingResult)
    assert result.raw_ply_path == scene / "output" / "ply" / "frame_0000.ply"
    assert result.raw_ply_path.read_bytes() == b"fresh-inria-ply"
    assert result.status == {"training": "done", "export": "done"}
    assert result.run_manifest_path == scene / "run_manifest.json"
    assert resolver_roots == [expected_root]
    assert Path(os.environ["FOURDGS_DATA_ROOT"]) == expected_root
    assert runner.data_root_seen == expected_root

    copied_frames = scene / "frames"
    assert sorted(path.name for path in copied_frames.iterdir()) == [
        "frame_000000.png",
        "frame_000001.png",
    ]
    for frame in prepared.reconstruction.selected_manifest.selected_frames:
        assert (copied_frames / frame.output_name).read_bytes() == (
            prepared.frames_dir / frame.output_name
        ).read_bytes()

    copied_model = scene / "colmap" / "sparse" / "0"
    assert sorted(path.name for path in copied_model.iterdir()) == sorted(_MODEL_FILES)
    for name in _MODEL_FILES:
        assert (copied_model / name).read_bytes() == (
            prepared.reconstruction.accepted_model_dir / name
        ).read_bytes()

    assert len(runner.calls) == 1
    kwargs = runner.calls[0]
    assert kwargs["video_path"] == scene / "video.mp4"
    assert kwargs["scene_name"] == prepared.scene_name
    assert kwargs["force_preprocess"] is False
    assert kwargs["skip_training"] is False
    assert kwargs["skip_export"] is False
    assert kwargs["skip_foundation"] is False
    assert kwargs["cfg"].train.static_mode is True

    for step in ("frames", "colmap"):
        marker = json.loads(
            (scene / ".cache_markers" / f"{step}.json").read_text(encoding="utf-8")
        )
        assert marker["settings_hash"] == compute_settings_hash(kwargs["cfg"], step)

    run_manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest == {
        "schema_version": 1,
        "run_id": prepared.run_id,
        "scene_name": prepared.scene_name,
        "selection_mode": "smart",
        "source_digest": prepared.source_digest,
        "image_set_digest": prepared.selection_digest,
        "profile": "balanced_l4",
        "legacy_preset": "balanced",
        "foundation": True,
        "run_eval": False,
        "config_digest": result.resolved_config.config_digest,
    }


def test_explicit_foundation_false_is_forwarded_to_runner(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    runner = RecordingRunner()

    result = run_validated_training(
        prepared,
        _spec(foundation=False),
        pipeline_runner=runner,
    )

    assert result.resolved_config.foundation is False
    assert result.resolved_config.lambda_depth == 0.0
    assert runner.calls[0]["cfg"].train.lambda_depth == 0.0
    assert runner.calls[0]["skip_foundation"] is True
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["foundation"] is False


def test_ultra_snapshot_uses_accepted_colmap_native_resolution(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    runner = RecordingRunner()

    result = run_validated_training(
        prepared,
        _spec(profile=NotebookQualityProfile.ULTRA),
        pipeline_runner=runner,
    )

    cfg = runner.calls[0]["cfg"]
    assert cfg.train.image_resolution == (1920, 1080)
    assert result.resolved_config.image_resolution == (1920, 1080)


@pytest.mark.parametrize("stale_kind", ["scene", "output"])
def test_preexisting_scene_or_output_is_never_merged(
    tmp_path: Path,
    stale_kind: str,
) -> None:
    prepared = _prepared(tmp_path)
    scene = prepared.data_root / prepared.scene_name
    scene.mkdir(parents=True)
    if stale_kind == "output":
        raw_ply = scene / "output" / "ply" / "frame_0000.ply"
        raw_ply.parent.mkdir(parents=True)
        raw_ply.write_bytes(b"stale")
    runner = RecordingRunner()

    with pytest.raises(FileExistsError, match="fresh"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    if stale_kind == "output":
        assert raw_ply.read_bytes() == b"stale"


def test_runner_must_create_a_new_regular_ply(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    runner = RecordingRunner(create_ply=False)

    with pytest.raises(FileNotFoundError, match="frame_0000.ply"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert len(runner.calls) == 1
    assert (prepared.data_root / prepared.scene_name).is_dir()


@pytest.mark.parametrize(
    "mismatch",
    ["source_digest", "selection_digest", "manifest_digest", "frame_hash"],
)
def test_digest_or_frame_hash_mismatch_is_rejected_before_scene_creation(
    tmp_path: Path,
    mismatch: str,
) -> None:
    prepared = _prepared(tmp_path)
    manifest = prepared.reconstruction.selected_manifest
    if mismatch == "source_digest":
        prepared = replace(prepared, source_digest="b" * 64)
    elif mismatch == "selection_digest":
        prepared = replace(prepared, selection_digest="b" * 64)
    elif mismatch == "manifest_digest":
        manifest = replace(manifest, image_set_digest="b" * 64)
        prepared = replace(
            prepared,
            selection_digest="b" * 64,
            reconstruction=replace(
                prepared.reconstruction,
                selected_manifest=manifest,
            ),
        )
    else:
        first, *rest = manifest.frames
        bad_frames = (replace(first, sha256="b" * 64), *rest)
        manifest = replace(
            manifest,
            frames=bad_frames,
            image_set_digest=_selection_digest(bad_frames),
        )
        prepared = replace(
            prepared,
            selection_digest=manifest.image_set_digest,
            reconstruction=replace(
                prepared.reconstruction,
                selected_manifest=manifest,
            ),
        )
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="digest|hash"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()


@pytest.mark.parametrize("unsafe_source", ["frame", "model"])
def test_symlinked_selected_or_model_source_is_rejected_where_supported(
    tmp_path: Path,
    unsafe_source: str,
) -> None:
    prepared = _prepared(tmp_path)
    if unsafe_source == "frame":
        link = prepared.frames_dir / "frame_000000.png"
    else:
        link = prepared.reconstruction.accepted_model_dir / "cameras.txt"
    content = link.read_bytes()
    target = tmp_path / f"{unsafe_source}-target"
    target.write_bytes(content)
    link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="symlink|regular|unsafe"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("run_id", "../escape"), ("scene_name", "nested/scene")],
)
def test_run_and_scene_names_must_be_safe_single_components(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    prepared = replace(_prepared(tmp_path), **{field: value})
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="safe single component"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()


def test_missing_model_file_is_rejected_before_scene_creation(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    (prepared.reconstruction.accepted_model_dir / "points3D.txt").unlink()
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="points3D.txt"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()


def test_mutated_accepted_model_must_pass_the_gate_again_before_training(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    (prepared.reconstruction.accepted_model_dir / "points3D.txt").write_text(
        "# emptied after the original gate\n",
        encoding="utf-8",
    )
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="no longer passes.*gate"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    assert runner.calls == []
    assert not prepared.data_root.exists()


def test_runner_failure_preserves_fresh_scene_for_diagnostics(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    runner = RecordingRunner(
        create_ply=False,
        failure=RuntimeError("injected training failure"),
    )

    with pytest.raises(RuntimeError, match="injected training failure"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)

    scene = prepared.data_root / prepared.scene_name
    assert len(runner.calls) == 1
    assert (scene / "run_manifest.json").is_file()
    assert (scene / "frames" / "frame_000000.png").is_file()
    assert not (scene / "output" / "ply" / "frame_0000.ply").exists()
