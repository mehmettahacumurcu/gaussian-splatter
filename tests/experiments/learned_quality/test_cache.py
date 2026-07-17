from __future__ import annotations

import errno
import json
import shutil
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pytest

from backend.static_pipeline.contracts import (
    ColmapAttempt,
    FrameRecord,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
    SelectionPolicy,
    SourceFile,
    SourceInventory,
)
from backend.static_pipeline.runner import SelectionOutput
from backend.static_pipeline.runner import PretrainingRestore

from experiments.learned_quality.cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCacheOwnershipError,
    LearnedCheckpointStore,
    checkpoint_fingerprint,
    decode_checkpoint_state,
    encode_checkpoint_state,
    producer_code_digest,
    _checkpoint_types,
)
from experiments.learned_quality.contracts import (
    FrameArtifact,
    GENERATOR_ID,
    GeometryCandidateReport,
    LearnedArtifacts,
    LearnedReconstructionOutput,
    StageRecord,
)
from experiments.learned_quality.da3 import (
    AnchorInferenceResult,
    CameraRecord,
    FramePredictionArtifact,
    PinholeCamera,
)
from experiments.learned_quality.depth import (
    DenseSeedArtifact,
    DenseSeedPolicy,
    DepthValidationResult,
    ValidatedDepthFrame,
)
from experiments.learned_quality.flow import (
    FlowGatePolicy,
    MotionEvidence,
    MotionFrameEvidence,
)
from experiments.learned_quality.lifecycle import BatchAttemptRecord
from experiments.learned_quality.masks import (
    FusedMaskFrame,
    MaskFusionEvidence,
    MaskFusionPolicy,
)
from experiments.learned_quality.photometric import (
    PhotometricEvidence,
    RgbAffineTransform,
)
from experiments.learned_quality.segmentation import (
    SemanticEvidence,
    SemanticFrameEvidence,
    SemanticPolicy,
)


def _payload(root: Path, value: bytes = b"model") -> Path:
    source = root / "payload"
    model = source / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_bytes(value)
    (model / "images.txt").write_bytes(b"images")
    (model / "points3D.txt").write_bytes(b"points")
    return source


def test_store_publishes_owned_hash_verified_generation(tmp_path: Path) -> None:
    cache_root = tmp_path / "myroom_test_learned_test_cache"
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)

    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="run-1",
        source_root=_payload(tmp_path / "source"),
    )

    assert generation == cache_root / "colmap" / ("b" * 64)
    ownership = json.loads((cache_root / "_OWNERSHIP.json").read_text(encoding="utf-8"))
    assert ownership == {
        "generator_id": GENERATOR_ID,
        "input_identity": "a" * 64,
        "schema_version": 1,
    }
    assert (generation / "manifest.json").is_file()
    assert (generation / "_SUCCESS.json").is_file()
    assert store.find_generation(CheckpointKind.COLMAP, "b" * 64) == generation


def test_drive_fuse_unsupported_atomic_rename_uses_verified_marker_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "myroom_test_learned_test_cache"
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)

    def reject_renameat2(_staged: Path, destination: Path) -> None:
        raise OSError(errno.EINVAL, "Invalid argument", str(destination))

    monkeypatch.setattr(
        "experiments.learned_quality.cache._atomic_promote_no_replace",
        reject_renameat2,
    )

    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="run-1",
        source_root=_payload(tmp_path / "source"),
    )

    assert generation == cache_root / "colmap" / ("b" * 64)
    assert store.find_generation(CheckpointKind.COLMAP, "b" * 64) == generation
    assert (generation / "_SUCCESS.json").is_file()
    assert not (cache_root / "staging" / "colmap-run-1").exists()


def test_store_rejects_an_unowned_existing_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "myroom_test_learned_test_cache"
    cache_root.mkdir()
    (cache_root / "user-file.txt").write_text("keep me", encoding="utf-8")
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)

    with pytest.raises(LearnedCacheOwnershipError, match="not owned"):
        store.publish_generation(
            CheckpointKind.COLMAP,
            fingerprint="b" * 64,
            run_id="run-1",
            source_root=_payload(tmp_path / "source"),
        )

    assert (cache_root / "user-file.txt").read_text(encoding="utf-8") == "keep me"


def test_store_rejects_mismatched_input_identity(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first = LearnedCheckpointStore(cache_root, input_identity="a" * 64)
    first.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="run-1",
        source_root=_payload(tmp_path / "source"),
    )

    second = LearnedCheckpointStore(cache_root, input_identity="c" * 64)
    with pytest.raises(LearnedCacheOwnershipError, match="different input"):
        second.find_generation(CheckpointKind.COLMAP, "b" * 64)


def test_missing_marker_or_changed_bytes_never_produces_a_hit(tmp_path: Path) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="run-1",
        source_root=_payload(tmp_path / "source"),
    )

    (generation / "payload" / "sparse" / "0" / "cameras.txt").write_bytes(b"changed")
    assert store.find_generation(CheckpointKind.COLMAP, "b" * 64) is None

    (generation / "_SUCCESS.json").unlink()
    assert store.find_generation(CheckpointKind.COLMAP, "b" * 64) is None


def test_checkpoint_types_are_independent_and_old_generation_is_cleaned(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)
    old_colmap = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="run-1",
        source_root=_payload(tmp_path / "source-1", b"old"),
    )
    pretraining = store.publish_generation(
        CheckpointKind.PRETRAINING,
        fingerprint="c" * 64,
        run_id="run-2",
        source_root=_payload(tmp_path / "source-2", b"pretraining"),
    )
    current_colmap = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="d" * 64,
        run_id="run-3",
        source_root=_payload(tmp_path / "source-3", b"new"),
    )

    assert not old_colmap.exists()
    assert pretraining.is_dir()
    assert current_colmap.is_dir()
    assert store.find_generation(CheckpointKind.PRETRAINING, "c" * 64) == pretraining
    assert store.find_generation(CheckpointKind.COLMAP, "d" * 64) == current_colmap


def test_checkpoint_fingerprint_invalidates_every_preprocessing_dependency() -> None:
    inputs = CheckpointInputs(
        source_digest="a" * 64,
        settings={"selection": {"mode": "smart", "cap": 1280}},
        model_manifest_sha256="b" * 64,
        tool_versions={"colmap": "COLMAP 3.11.1"},
        producer_code_sha256="c" * 64,
        upstream_fingerprint=None,
    )
    original = checkpoint_fingerprint(CheckpointKind.COLMAP, inputs)
    changed = {
        checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(inputs, source_digest="d" * 64),
        ),
        checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(inputs, settings={"selection": {"mode": "smart", "cap": 960}}),
        ),
        checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(inputs, model_manifest_sha256="e" * 64),
        ),
        checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(inputs, tool_versions={"colmap": "COLMAP 3.12.0"}),
        ),
        checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(inputs, producer_code_sha256="f" * 64),
        ),
        checkpoint_fingerprint(
            CheckpointKind.PRETRAINING,
            replace(inputs, upstream_fingerprint="1" * 64),
        ),
    }

    assert len(original) == 64
    assert original not in changed
    assert len(changed) == 6


def test_producer_digest_ignores_training_only_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    preprocessing = repository / "runtime.py"
    training = repository / "training.py"
    preprocessing.write_text("PREPROCESSING_VERSION = 1\n", encoding="utf-8")
    training.write_text("TRAINING_VERSION = 1\n", encoding="utf-8")

    original = producer_code_digest(repository, ("runtime.py",))
    training.write_text("TRAINING_VERSION = 2\n", encoding="utf-8")
    training_only_change = producer_code_digest(repository, ("runtime.py",))
    preprocessing.write_text("PREPROCESSING_VERSION = 2\n", encoding="utf-8")
    preprocessing_change = producer_code_digest(repository, ("runtime.py",))

    assert training_only_change == original
    assert preprocessing_change != original


def _portable_state(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], SourceInventory]:
    snapshot = tmp_path / "snapshot"
    frames = snapshot / "selection" / "selected"
    frames.mkdir(parents=True)
    (frames / "frame_000001.png").write_bytes(b"png")
    selection_manifest_path = frames / "selection_manifest.json"
    selection_manifest_path.write_text("{}", encoding="utf-8")
    attempt_root = snapshot / "reconstruction" / "classical"
    model = attempt_root / "sparse" / "0"
    model.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (model / name).write_text(name, encoding="utf-8")
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    model_manifest = snapshot / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")

    input_root = snapshot / "input"
    input_root.mkdir()
    (input_root / "room.mov").write_bytes(b"video")
    source_file = SourceFile("room.mov", 5, "9" * 64)
    original_inventory = SourceInventory(
        schema_version=1,
        root=input_root,
        kind="video",
        media_files=(source_file,),
        all_files=(source_file,),
        digest="a" * 64,
    )
    policy = SelectionPolicy(
        mode="smart",
        frame_budget=800,
        resolution_long_edge_cap=1920,
    )
    frame = FrameRecord(
        frame_id="frame-1",
        source_relative_path="room.mov",
        source_index=1,
        source_pts=100,
        timestamp_s=1.0,
        output_name="frame_000001.png",
        sha256="1" * 64,
        selected=True,
        metrics=None,
        selection_score=1.0,
        reasons=("selected",),
    )
    manifest = SelectionManifest(
        schema_version=1,
        source_digest="a" * 64,
        effective_mode="smart",
        policy=policy,
        frames=(frame,),
        image_set_digest="2" * 64,
    )
    selection = SelectionOutput(
        inventory=original_inventory,
        manifest=manifest,
        frames_dir=frames,
        source_manifest_path=selection_manifest_path,
    )
    metrics = ModelMetrics(
        model_dir=model,
        registered_names=frozenset({"frame_000001.png"}),
        registered_count=1,
        registered_ratio=1.0,
        registered_share=1.0,
        temporal_coverage_s=0.0,
        max_interior_gap_s=0.0,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=0.1,
        p95_reprojection_error_px=0.2,
        median_track_length=3.0,
        sparse_point_count=10,
        valid_names_intrinsics_and_poses=True,
    )
    decision = GateDecision(
        passed=True,
        dominant=metrics,
        failures=(),
        uncovered_intervals=(),
        retry_recommended=False,
    )
    attempt = ColmapAttempt(
        root=attempt_root,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.11.1",
        fingerprint="3" * 64,
    )
    bundle = ReconstructionBundle(
        selected_manifest=manifest,
        accepted_model_dir=model,
        decision=decision,
        attempts=(attempt,),
        decisions=(decision,),
    )
    candidate = GeometryCandidateReport(
        candidate_id="classical",
        attempt=attempt,
        decision=decision,
        model_dir=model,
        selected_manifest=manifest,
        frame_set_digest="2" * 64,
        covered_endpoint_count=2,
    )
    reconstruction = LearnedReconstructionOutput(
        bundle=bundle,
        frames_dir=frames,
        artifacts=LearnedArtifacts(
            model_manifest_path=model_manifest,
            stage_records=(StageRecord("classical_prepass", "accepted"),),
        ),
        geometry_candidates=(candidate,),
    )

    current_input = tmp_path / "current-input"
    current_input.mkdir()
    (current_input / "room.mov").write_bytes(b"video")
    current_inventory = replace(original_inventory, root=current_input)
    return (
        snapshot,
        {"selection": selection, "reconstruction": reconstruction},
        current_inventory,
    )


def test_portable_state_round_trip_rebases_paths_and_injects_inventory(
    tmp_path: Path,
) -> None:
    snapshot, state, current_inventory = _portable_state(tmp_path)
    encoded = encode_checkpoint_state(state, snapshot_root=snapshot)
    json.dumps(encoded, allow_nan=False)
    restore = tmp_path / "restore"
    shutil.copytree(snapshot, restore)

    decoded = decode_checkpoint_state(
        encoded,
        restore_root=restore,
        source_inventory=current_inventory,
    )

    selection = decoded["selection"]
    reconstruction = decoded["reconstruction"]
    assert isinstance(selection, SelectionOutput)
    assert selection.inventory is current_inventory
    assert selection.frames_dir == restore / "selection" / "selected"
    assert isinstance(reconstruction, LearnedReconstructionOutput)
    assert selection.manifest is reconstruction.selected_manifest
    assert reconstruction.geometry_candidates[0].decision is reconstruction.decision
    assert reconstruction.accepted_model_dir == (
        restore / "reconstruction" / "classical" / "sparse" / "0"
    )
    assert reconstruction.decision.dominant.registered_names == frozenset(
        {"frame_000001.png"}
    )
    assert reconstruction.artifacts.stage_records == (
        StageRecord("classical_prepass", "accepted"),
    )


@dataclass(frozen=True)
class UnregisteredState:
    value: str


def test_portable_state_rejects_unregistered_types_and_unsafe_values(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="not registered"):
        encode_checkpoint_state(UnregisteredState("value"), snapshot_root=snapshot)
    with pytest.raises(ValueError, match="finite"):
        encode_checkpoint_state(float("nan"), snapshot_root=snapshot)
    with pytest.raises(ValueError, match="unknown checkpoint type"):
        decode_checkpoint_state(
            {
                "__kind__": "dataclass",
                "type": "unknown.Type",
                "object_id": "o1",
                "fields": {},
            },
            restore_root=snapshot,
            source_inventory=object(),
        )
    for unsafe in ("../escape", "/absolute/path", r"C:\escape"):
        with pytest.raises(ValueError, match="safe POSIX relative path"):
            decode_checkpoint_state(
                {"__kind__": "path", "relative_path": unsafe},
                restore_root=snapshot,
                source_inventory=object(),
            )


def test_colmap_checkpoint_restores_attempt_into_fresh_local_root(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "first-run" / "classical-prepass"
    model = attempt_root / "sparse" / "0"
    model.mkdir(parents=True)
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (model / name).write_text(name, encoding="utf-8")
    attempt = ColmapAttempt(
        root=attempt_root,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.11.1",
        fingerprint="4" * 64,
    )
    store = LearnedCheckpointStore(tmp_path / "drive-cache", input_identity="a" * 64)

    store.publish_colmap(
        attempt,
        fingerprint="5" * 64,
        run_id="run-1",
    )
    shutil.rmtree(tmp_path / "first-run")
    destination = tmp_path / "second-run" / "classical-prepass"
    restored = store.restore_colmap("5" * 64, destination=destination)

    assert isinstance(restored, ColmapAttempt)
    assert restored.root == destination / "attempt"
    assert restored.database_path == destination / "attempt" / "colmap.db"
    assert restored.model_dirs == (destination / "attempt" / "sparse" / "0",)
    assert restored.colmap_version == "COLMAP 3.11.1"
    assert restored.fingerprint == "4" * 64


def test_complete_pretraining_checkpoint_restores_into_fresh_local_root(
    tmp_path: Path,
) -> None:
    run_root, state, current_inventory = _portable_state(tmp_path)
    store = LearnedCheckpointStore(tmp_path / "drive-cache", input_identity="a" * 64)

    generation = store.publish_pretraining(
        source_inventory=current_inventory,
        selection=state["selection"],
        reconstruction=state["reconstruction"],
        fingerprint="6" * 64,
        run_id="run-1",
        run_root=run_root,
    )

    assert not (generation / "payload" / "input").exists()
    shutil.rmtree(run_root)
    destination = tmp_path / "second-run" / "pretraining-restored"
    restored = store.restore_pretraining(
        source_inventory=current_inventory,
        destination=destination,
        expected_fingerprint=lambda selection: "6" * 64,
    )

    assert isinstance(restored, PretrainingRestore)
    assert restored.selection.inventory is current_inventory
    assert restored.selection.frames_dir == destination / "selection" / "selected"
    assert restored.reconstruction.frames_dir == (
        destination / "selection" / "selected"
    )
    assert restored.reconstruction.accepted_model_dir == (
        destination / "reconstruction" / "classical" / "sparse" / "0"
    )
    assert restored.reconstruction.artifacts.model_manifest_path == (
        destination / "model-manifest" / "model_manifest.json"
    )


def test_corrupt_complete_checkpoint_is_a_miss_without_local_restore(
    tmp_path: Path,
) -> None:
    run_root, state, current_inventory = _portable_state(tmp_path)
    store = LearnedCheckpointStore(tmp_path / "drive-cache", input_identity="a" * 64)
    generation = store.publish_pretraining(
        source_inventory=current_inventory,
        selection=state["selection"],
        reconstruction=state["reconstruction"],
        fingerprint="6" * 64,
        run_id="run-1",
        run_root=run_root,
    )
    artifact = generation / "payload" / "selection" / "selected" / "frame_000001.png"
    artifact.write_bytes(b"corrupt")
    destination = tmp_path / "restore"

    restored = store.restore_pretraining(
        source_inventory=current_inventory,
        destination=destination,
        expected_fingerprint=lambda selection: "6" * 64,
    )

    assert restored is None
    assert not destination.exists()


def test_complete_checkpoint_restore_copy_corruption_is_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, state, current_inventory = _portable_state(tmp_path)
    store = LearnedCheckpointStore(tmp_path / "drive-cache", input_identity="a" * 64)
    store.publish_pretraining(
        source_inventory=current_inventory,
        selection=state["selection"],
        reconstruction=state["reconstruction"],
        fingerprint="6" * 64,
        run_id="run-1",
        run_root=run_root,
    )
    original_copytree = shutil.copytree

    destination = tmp_path / "restore"

    def corrupt_copy(
        source: Path,
        copied_destination: Path,
        *args: object,
        **kwargs: object,
    ) -> Path:
        copied = original_copytree(source, copied_destination, *args, **kwargs)
        if Path(copied_destination) == destination:
            (destination / "state.json").write_text("{}", encoding="utf-8")
        return copied

    monkeypatch.setattr(shutil, "copytree", corrupt_copy)

    with pytest.raises(ValueError, match="differs from Drive"):
        store.restore_pretraining(
            source_inventory=current_inventory,
            destination=destination,
            expected_fingerprint=lambda selection: "6" * 64,
        )

    assert not destination.exists()


def test_complete_checkpoint_registry_covers_every_persisted_learned_type() -> None:
    registered = set(_checkpoint_types().values())
    persisted = {
        FrameArtifact,
        BatchAttemptRecord,
        PinholeCamera,
        FramePredictionArtifact,
        CameraRecord,
        AnchorInferenceResult,
        SemanticPolicy,
        SemanticFrameEvidence,
        SemanticEvidence,
        FlowGatePolicy,
        MotionFrameEvidence,
        MotionEvidence,
        MaskFusionPolicy,
        FusedMaskFrame,
        MaskFusionEvidence,
        RgbAffineTransform,
        PhotometricEvidence,
        DenseSeedPolicy,
        ValidatedDepthFrame,
        DenseSeedArtifact,
        DepthValidationResult,
    }

    assert persisted <= registered
