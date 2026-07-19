from __future__ import annotations

import errno
import json
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.learned_quality import cache as cache_module

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
from experiments.learned_quality.milestones import (
    FinalPretrainingState,
    GeometryMilestoneState,
    MilestoneRef,
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


def test_legacy_complete_restore_ignores_terminal_milestone_generation(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    store.publish_generation(
        CheckpointKind.PRETRAINING,
        fingerprint="b" * 64,
        run_id="terminal",
        source_root=_payload(tmp_path / "terminal-source"),
        upstream={
            CheckpointKind.GEOMETRY: "c" * 64,
            CheckpointKind.MASKS: "d" * 64,
        },
        artifact_roots={"terminal": "artifacts/terminal"},
    )

    assert store._valid_pretraining_generations() == ()


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


def test_drive_publication_probe_uses_real_cache_operations_and_cleans_up(
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

    store.probe_drive_publication(run_id="run-1")

    assert (cache_root / "_OWNERSHIP.json").is_file()
    assert not (cache_root / "probe").exists()
    assert not (cache_root / "colmap").exists()
    assert not (cache_root / "pretraining").exists()


def test_verified_staging_survives_promotion_error_and_recovers_on_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "myroom_test_learned_test_cache"
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)
    original_promote = cache_module._atomic_promote_no_replace

    def fail_promotion(_staged: Path, destination: Path) -> None:
        raise OSError(errno.EIO, "Drive unavailable", str(destination))

    monkeypatch.setattr(
        "experiments.learned_quality.cache._atomic_promote_no_replace",
        fail_promotion,
    )

    with pytest.raises(OSError, match="Drive unavailable"):
        store.publish_generation(
            CheckpointKind.COLMAP,
            fingerprint="b" * 64,
            run_id="run-1",
            source_root=_payload(tmp_path / "source"),
        )

    staged = cache_root / "staging" / "colmap-run-1"
    assert staged.is_dir()
    assert (staged / "_SUCCESS.json").is_file()

    monkeypatch.setattr(
        "experiments.learned_quality.cache._atomic_promote_no_replace",
        original_promote,
    )
    generation = store.find_generation(CheckpointKind.COLMAP, "b" * 64)

    assert generation == cache_root / "colmap" / ("b" * 64)
    assert store.find_generation(CheckpointKind.COLMAP, "b" * 64) == generation
    assert not staged.exists()


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


def test_terminal_milestone_states_round_trip_with_local_paths(tmp_path: Path) -> None:
    snapshot, state, current_inventory = _portable_state(tmp_path)
    reconstruction = state["reconstruction"]
    geometry = GeometryMilestoneState(
        bundle=reconstruction.bundle,
        frames_dir=reconstruction.frames_dir,
        geometry_candidates=reconstruction.geometry_candidates,
    )
    final = FinalPretrainingState(
        photometric=None,
        depth=None,
        dense_seeds=None,
        final_stage_records=(StageRecord("dense_seed_fusion", "accepted"),),
        model_manifest_path=reconstruction.artifacts.model_manifest_path,
    )

    encoded = encode_checkpoint_state(
        {"geometry": geometry, "final": final},
        snapshot_root=snapshot,
    )
    restore = tmp_path / "terminal-restore"
    shutil.copytree(snapshot, restore)
    decoded = decode_checkpoint_state(
        encoded,
        restore_root=restore,
        source_inventory=current_inventory,
    )

    assert isinstance(decoded["geometry"], GeometryMilestoneState)
    assert isinstance(decoded["final"], FinalPretrainingState)
    assert decoded["geometry"].bundle.accepted_model_dir.is_relative_to(restore)
    assert decoded["geometry"].frames_dir.is_relative_to(restore)
    assert decoded["final"].model_manifest_path.is_relative_to(restore)


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


def test_colmap_restore_does_not_prehash_drive_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    generation = store.publish_colmap(
        attempt,
        fingerprint="5" * 64,
        run_id="run-1",
    )
    payload_root = (generation / "payload").resolve()
    hashed_paths: list[Path] = []
    payload_reads: dict[Path, int] = {}
    original_sha256 = cache_module._sha256
    original_open = Path.open

    def record_sha256(path: Path) -> str:
        hashed_paths.append(Path(path).resolve())
        return original_sha256(path)

    def record_open(path: Path, *args: object, **kwargs: object):
        candidate = Path(path).resolve()
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if candidate.is_relative_to(payload_root) and "r" in mode:
            payload_reads[candidate] = payload_reads.get(candidate, 0) + 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(cache_module, "_sha256", record_sha256)
    monkeypatch.setattr(Path, "open", record_open)
    restored = store.restore_colmap(
        "5" * 64,
        destination=tmp_path / "second-run" / "classical-prepass",
    )

    assert isinstance(restored, ColmapAttempt)
    assert not any(path.is_relative_to(payload_root) for path in hashed_paths)
    assert payload_reads
    assert set(payload_reads.values()) == {1}


@pytest.mark.parametrize("mutation", ["size", "digest", "missing", "unexpected"])
def test_colmap_stream_restore_rejects_corruption_and_cleans_target(
    tmp_path: Path,
    mutation: str,
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
    generation = store.publish_colmap(
        attempt,
        fingerprint="5" * 64,
        run_id="run-1",
    )
    artifact = generation / "payload" / "attempt" / "colmap.db"
    if mutation == "size":
        artifact.write_bytes(artifact.read_bytes() + b"x")
    elif mutation == "digest":
        artifact.write_bytes(b"databasa")
    elif mutation == "missing":
        artifact.unlink()
    else:
        (generation / "payload" / "unexpected.bin").write_bytes(b"unexpected")
    destination = tmp_path / "second-run" / "classical-prepass"

    with pytest.raises(ValueError, match="metadata|size mismatch|hash mismatch"):
        store.restore_colmap("5" * 64, destination=destination)

    assert not destination.exists()


def _published_parallel_payload(
    tmp_path: Path,
    *,
    file_count: int,
    file_size: int = 8,
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    source = tmp_path / "parallel-source"
    source.mkdir()
    expected: dict[str, bytes] = {}
    for index in range(file_count):
        relative = Path(f"group-{index % 2}") / f"file-{index:02d}.bin"
        payload = bytes([index + 1]) * file_size
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[relative.as_posix()] = payload
    store = LearnedCheckpointStore(tmp_path / "parallel-cache", input_identity="a" * 64)
    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="parallel-test",
        source_root=source,
    )
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    return generation, manifest, expected


def test_copy_verified_payload_runs_file_workers_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation, manifest, _expected = _published_parallel_payload(
        tmp_path, file_count=4
    )
    barrier = threading.Barrier(4, timeout=3)
    thread_ids: set[int] = set()
    real_worker = cache_module._copy_verified_file

    def wrapped_worker(*args: object) -> int:
        thread_ids.add(threading.get_ident())
        barrier.wait()
        return real_worker(*args)

    monkeypatch.setattr(cache_module, "_copy_verified_file", wrapped_worker)

    cache_module._copy_verified_payload(
        generation, tmp_path / "restored", manifest, max_workers=4
    )

    assert len(thread_ids) == 4


@pytest.mark.parametrize("corrupt_index", range(4))
def test_copy_verified_payload_verifies_sha256_for_every_manifest_row(
    tmp_path: Path,
    corrupt_index: int,
) -> None:
    generation, manifest, expected = _published_parallel_payload(tmp_path, file_count=4)
    relative = sorted(expected)[corrupt_index]
    payload = generation / "payload" / relative
    payload.write_bytes(b"x" * len(expected[relative]))

    with pytest.raises(ValueError, match=f"hash mismatch.*{relative}"):
        cache_module._copy_verified_payload(
            generation, tmp_path / "restored", manifest, max_workers=4
        )


@pytest.mark.parametrize(
    ("file_count", "configured", "expected"),
    ((12, 4, 4), (3, 16, 3), (1, 8, 1)),
)
def test_copy_verified_payload_clamps_workers_to_file_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_count: int,
    configured: int,
    expected: int,
) -> None:
    generation, manifest, _payloads = _published_parallel_payload(
        tmp_path, file_count=file_count
    )
    observed: list[int] = []
    real_executor = cache_module.ThreadPoolExecutor

    def record_executor(*args: object, **kwargs: object):
        observed.append(int(kwargs["max_workers"] if kwargs else args[0]))
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(cache_module, "ThreadPoolExecutor", record_executor)

    cache_module._copy_verified_payload(
        generation, tmp_path / "restored", manifest, max_workers=configured
    )

    assert observed == [expected]


@pytest.mark.parametrize("value", (True, False, None, 0, -1, 17, 1.0, "4"))
def test_copy_verified_payload_rejects_invalid_worker_bounds_before_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    generation, manifest, _payloads = _published_parallel_payload(tmp_path, file_count=1)
    destination = tmp_path / "restored"

    def fail_executor(*args: object, **kwargs: object):
        raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(cache_module, "ThreadPoolExecutor", fail_executor)

    with pytest.raises(
        ValueError, match="max_workers must be a plain integer from 1 through 16"
    ):
        cache_module._copy_verified_payload(
            generation, destination, manifest, max_workers=value  # type: ignore[arg-type]
        )

    assert not destination.exists()


def test_copy_verified_payload_default_worker_count_is_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation, manifest, _payloads = _published_parallel_payload(tmp_path, file_count=8)
    observed: list[int] = []
    real_executor = cache_module.ThreadPoolExecutor

    def record_executor(*args: object, **kwargs: object):
        observed.append(int(kwargs["max_workers"] if kwargs else args[0]))
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(cache_module, "ThreadPoolExecutor", record_executor)

    cache_module._copy_verified_payload(generation, tmp_path / "restored", manifest)

    assert observed == [4]


def test_copy_verified_payload_one_worker_preserves_single_pass_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "single-source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"payload")
    (nested / "empty.bin").write_bytes(b"")
    store = LearnedCheckpointStore(tmp_path / "single-cache", input_identity="a" * 64)
    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="single-pass",
        source_root=source,
    )
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    payload_root = (generation / "payload").resolve()
    reads: dict[Path, int] = {}
    original_open = Path.open
    original_sha256 = cache_module._sha256

    def record_open(path: Path, *args: object, **kwargs: object):
        candidate = Path(path).resolve()
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if candidate.is_relative_to(payload_root) and "r" in mode:
            reads[candidate] = reads.get(candidate, 0) + 1
        return original_open(path, *args, **kwargs)

    def fail_sha256(path: Path) -> str:
        if Path(path).resolve().is_relative_to(payload_root):
            raise AssertionError("Drive payload must not be prehashed")
        return original_sha256(path)

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(cache_module, "_sha256", fail_sha256)
    destination = tmp_path / "restored"

    cache_module._copy_verified_payload(
        generation, destination, manifest, max_workers=1
    )

    assert (destination / "nested" / "payload.bin").read_bytes() == b"payload"
    assert (destination / "nested" / "empty.bin").read_bytes() == b""
    assert set(reads.values()) == {1}


def _colmap_restore_generation(tmp_path: Path) -> tuple[LearnedCheckpointStore, Path]:
    attempt_root = tmp_path / "first-run" / "classical-prepass"
    attempt_root.mkdir(parents=True)
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    attempt = ColmapAttempt(
        root=attempt_root,
        database_path=database,
        model_dirs=(),
        colmap_version="COLMAP 3.11.1",
        fingerprint="4" * 64,
    )
    store = LearnedCheckpointStore(tmp_path / "drive-cache", input_identity="a" * 64)
    store.publish_colmap(attempt, fingerprint="5" * 64, run_id="run-1")
    return store, attempt_root


def test_colmap_restore_propagates_first_worker_failure_after_quiescence_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _attempt_root = _colmap_restore_generation(tmp_path)
    barrier = threading.Barrier(2, timeout=3)
    worker_completed = threading.Event()
    failure = RuntimeError("first worker failure")

    def fail_one_worker(
        source_root: Path,
        target: Path,
        row: cache_module._ManifestEntry,
        cancellation: threading.Event,
        progress: object,
    ) -> int:
        barrier.wait()
        if row.relative_path.name == "state.json":
            raise failure
        assert cancellation.wait(timeout=3)
        worker_completed.set()
        return 0

    monkeypatch.setattr(cache_module, "_copy_verified_file", fail_one_worker)
    destination = tmp_path / "restored"

    with pytest.raises(RuntimeError) as raised:
        store.restore_colmap("5" * 64, destination=destination)

    assert raised.value is failure
    assert worker_completed.is_set()
    assert not destination.exists()


def test_copy_verified_payload_progress_is_serialized_monotonic_and_completes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation, manifest, _payloads = _published_parallel_payload(tmp_path, file_count=4)
    monkeypatch.setattr(cache_module, "_RESTORE_PROGRESS_STEP_BYTES", 8)
    observed: list[int] = []
    original_add_verified = cache_module._RestoreProgress.add_verified
    output: list[str] = []

    def record_progress(self: object, byte_count: int) -> None:
        original_add_verified(self, byte_count)
        observed.append(self._verified_bytes)  # type: ignore[attr-defined]

    def record_print(*args: object, **kwargs: object) -> None:
        output.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(cache_module._RestoreProgress, "add_verified", record_progress)
    monkeypatch.setattr("builtins.print", record_print)

    cache_module._copy_verified_payload(
        generation, tmp_path / "restored", manifest, max_workers=4
    )

    assert observed == sorted(observed)
    assert len(set(observed)) == len(observed)
    assert observed[-1] == 32
    assert sum("Streaming" in line for line in output) == 1
    assert output[-1] == "[CACHE RESTORE] Verified streaming copy complete."


def test_restore_milestone_interrupt_cleans_destination_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _attempt_root = _colmap_restore_generation(tmp_path)
    destination = tmp_path / "restored"
    interrupt = KeyboardInterrupt()

    def interrupt_copy(
        generation: Path,
        target: Path,
        manifest: Mapping[str, object],
    ) -> None:
        target.mkdir()
        raise interrupt

    monkeypatch.setattr(cache_module, "_copy_verified_payload", interrupt_copy)

    with pytest.raises(KeyboardInterrupt) as raised:
        store.restore_milestone(
            MilestoneRef(CheckpointKind.COLMAP, "5" * 64),
            destination=destination,
            source_inventory=object(),
        )

    assert raised.value is interrupt
    assert not destination.exists()


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
    original_copy = cache_module._copy_verified_payload

    destination = tmp_path / "restore"

    def corrupt_copy(
        generation: Path,
        copied_destination: Path,
        manifest: Mapping[str, object],
    ) -> None:
        original_copy(generation, copied_destination, manifest)
        (destination / "state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cache_module, "_copy_verified_payload", corrupt_copy)

    with pytest.raises(ValueError, match="malformed|unknown checkpoint value"):
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
