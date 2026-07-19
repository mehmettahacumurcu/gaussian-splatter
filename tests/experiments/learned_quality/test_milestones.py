from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.learned_quality.cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCheckpointStore,
)
from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.da3 import (
    AnchorInferenceResult,
    CameraRecord,
    FramePredictionArtifact,
    PinholeCamera,
)
from experiments.learned_quality.flow import RigidFrameEvidence, RigidSceneEvidence
from experiments.learned_quality.milestones import (
    BaseEvidenceState,
    FinalPretrainingState,
    GeometryMilestoneState,
    LEGACY_COLMAP_PRODUCER_DIGESTS,
    MasksMilestoneState,
    MilestoneInputs,
    MilestoneRef,
    MilestoneSession,
    MilestoneState,
    MotionMilestoneState,
    SemanticMilestoneState,
    compatible_colmap_fingerprints,
    assemble_learned_reconstruction,
    milestone_fingerprint,
)
from experiments.learned_quality.tracks import (
    QualifiedStaticTracks,
    TrackAuditReport,
)


_COLMAP_PRODUCER_PATHS = (
    "backend/static_pipeline/colmap.py",
    "experiments/learned_quality/geometry.py",
    "experiments/learned_quality/runtime.py",
)


def _producer_digest_at_commit(repository_root: Path, commit: str) -> str:
    rows = []
    for relative in sorted(_COLMAP_PRODUCER_PATHS):
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        rows.append((relative, hashlib.sha256(content).hexdigest()))
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inputs(
    upstream: dict[CheckpointKind, str],
    *,
    producer: str = "e" * 64,
) -> MilestoneInputs:
    return MilestoneInputs(
        source_digest="a" * 64,
        selection_digest="b" * 64,
        settings={"policy": {"threshold": 0.25}},
        model_manifest_sha256="c" * 64,
        tool_versions={"python": "3.12.13"},
        producer_code_sha256=producer,
        upstream=upstream,
    )


def _artifact_root(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, Path]:
    root = tmp_path / name
    root.mkdir()
    artifact = root / "frame.bin"
    artifact.write_bytes(payload)
    return root, artifact


def test_assemble_learned_reconstruction_rehydrates_terminal_state(
    tmp_path: Path,
) -> None:
    bundle = object()
    candidate = object()
    acceptance = object()
    anchors = object()
    semantic = object()
    motion = object()
    masks = object()
    photometric = object()
    depth = object()
    dense_seeds = object()
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    evidence_record = object()
    final_record = object()
    base = BaseEvidenceState(
        anchors=anchors,
        depths=(),
        sky=(),
        scene=object(),
        track_audit=object(),
        colmap_ref="a" * 64,
    )
    geometry = GeometryMilestoneState(
        bundle=bundle,
        frames_dir=tmp_path,
        geometry_candidates=(candidate,),
        acceptance=acceptance,
    )
    final = FinalPretrainingState(
        photometric=photometric,
        depth=depth,
        dense_seeds=dense_seeds,
        final_stage_records=(final_record,),
        model_manifest_path=manifest,
    )

    output = assemble_learned_reconstruction(
        geometry=geometry,
        base=base,
        semantic=SemanticMilestoneState(semantic),
        motion=MotionMilestoneState(motion),
        masks=MasksMilestoneState(masks),
        evidence_stage_records=(evidence_record,),
        final=final,
    )

    assert output.bundle is bundle
    assert output.frames_dir == tmp_path
    assert output.geometry_candidates == (candidate,)
    assert output.acceptance is acceptance
    assert output.artifacts.da3 is anchors
    assert output.artifacts.base_evidence is base
    assert output.artifacts.semantic is semantic
    assert output.artifacts.flow is motion
    assert output.artifacts.masks is masks
    assert output.artifacts.photometric is photometric
    assert output.artifacts.depth is depth
    assert output.artifacts.dense_seeds is dense_seeds
    assert output.artifacts.model_manifest_path == manifest
    assert output.artifacts.stage_records == (evidence_record, final_record)


def _state(
    tmp_path: Path,
    *,
    kind: CheckpointKind,
    fingerprint: str,
    upstream: dict[CheckpointKind, str],
    name: str,
    payload: bytes,
) -> MilestoneState:
    root, artifact = _artifact_root(tmp_path, name, payload)
    return MilestoneState(
        ref=MilestoneRef(kind, fingerprint),
        upstream=upstream,
        value=FrameArtifact(
            image_name=f"{name}.png",
            frame_id=f"frame-{name}",
            path=artifact.resolve(),
            sha256="f" * 64,
        ),
        artifact_roots={name: root.resolve()},
    )


def test_milestone_fingerprint_sorts_upstream_and_invalidates_transitively() -> None:
    ordered = _inputs(
        {
            CheckpointKind.SEMANTIC: "c" * 64,
            CheckpointKind.BASE_EVIDENCE: "b" * 64,
        }
    )
    reversed_order = _inputs(
        {
            CheckpointKind.BASE_EVIDENCE: "b" * 64,
            CheckpointKind.SEMANTIC: "c" * 64,
        }
    )

    first = milestone_fingerprint(CheckpointKind.MASKS, ordered)
    second = milestone_fingerprint(CheckpointKind.MASKS, reversed_order)
    changed = milestone_fingerprint(
        CheckpointKind.MASKS,
        _inputs(
            {
                CheckpointKind.BASE_EVIDENCE: "b" * 64,
                CheckpointKind.SEMANTIC: "d" * 64,
            }
        ),
    )

    assert first == second
    assert first != changed


def test_milestone_session_scopes_stage_refs_and_state_contracts(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    session = MilestoneSession(
        store=store,
        source_inventory=object(),
        source_digest="a" * 64,
        selection_digest="b" * 64,
        model_manifest_sha256="c" * 64,
        tool_versions={"python": "3.12.13"},
        selection_ref=MilestoneRef(CheckpointKind.SELECTION, "1" * 64),
        repository_root=tmp_path,
        run_id="run-1",
    )

    semantic_ref = session.make_ref(
        CheckpointKind.SEMANTIC,
        upstream={CheckpointKind.BASE_EVIDENCE: "2" * 64},
        settings={"threshold": 0.3},
        producer_code_sha256="d" * 64,
    )

    assert semantic_ref.kind is CheckpointKind.SEMANTIC
    assert BaseEvidenceState.__dataclass_params__.frozen is True
    assert SemanticMilestoneState.__dataclass_params__.frozen is True
    assert MotionMilestoneState.__dataclass_params__.frozen is True
    assert MasksMilestoneState.__dataclass_params__.frozen is True

    original_selection_ref = session.make_ref(
        CheckpointKind.SELECTION,
        upstream={},
        settings={"mode": "geometry_backfill"},
        producer_code_sha256="e" * 64,
    )
    branch_selection_ref = session.make_ref(
        CheckpointKind.SELECTION,
        upstream={},
        settings={"mode": "geometry_backfill"},
        producer_code_sha256="e" * 64,
        selection_digest="f" * 64,
    )
    branch_root = tmp_path / "backfill-selection"
    branch_root.mkdir()
    branch = session.branch(
        selection_digest="f" * 64,
        selection_ref=branch_selection_ref,
        selection_root=branch_root,
    )

    assert branch_selection_ref != original_selection_ref
    assert session.selection_digest == "b" * 64
    assert branch.selection_digest == "f" * 64
    assert branch.selection_ref == branch_selection_ref
    assert branch.external_roots == {"selection": branch_root.resolve()}

    restored_colmap = tmp_path / "restored-colmap"
    restored_colmap.mkdir()
    rebound = branch.bind_external_root("round0_colmap", restored_colmap)

    assert branch.external_roots == {"selection": branch_root.resolve()}
    assert rebound.external_roots == {
        "round0_colmap": restored_colmap.resolve(),
        "selection": branch_root.resolve(),
    }


def test_semantic_change_does_not_invalidate_motion_sibling() -> None:
    base = "a" * 64
    semantic_v1 = milestone_fingerprint(
        CheckpointKind.SEMANTIC,
        _inputs({CheckpointKind.BASE_EVIDENCE: base}, producer="b" * 64),
    )
    semantic_v2 = milestone_fingerprint(
        CheckpointKind.SEMANTIC,
        _inputs({CheckpointKind.BASE_EVIDENCE: base}, producer="c" * 64),
    )
    motion_v1 = milestone_fingerprint(
        CheckpointKind.MOTION,
        _inputs({CheckpointKind.BASE_EVIDENCE: base}, producer="d" * 64),
    )
    motion_again = milestone_fingerprint(
        CheckpointKind.MOTION,
        _inputs({CheckpointKind.BASE_EVIDENCE: base}, producer="d" * 64),
    )

    assert semantic_v1 != semantic_v2
    assert motion_v1 == motion_again


def test_colmap_keeps_legacy_fingerprint_function() -> None:
    current = CheckpointInputs(
        source_digest="a" * 64,
        settings={"selection_digest": "b" * 64, "use_gpu": True},
        model_manifest_sha256="c" * 64,
        tool_versions={"colmap": "COLMAP 3.11.1"},
        producer_code_sha256="d" * 64,
    )

    fingerprints = compatible_colmap_fingerprints(
        current,
        legacy_producer_digests=("e" * 64, "d" * 64),
    )

    assert len(fingerprints) == 2
    assert fingerprints[0] != fingerprints[1]
    with pytest.raises(ValueError, match="legacy checkpoint"):
        milestone_fingerprint(CheckpointKind.COLMAP, _inputs({}))


def test_pycolmap_only_fix_prefers_compatible_classical_colmap_fingerprint() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    previous_producer = _producer_digest_at_commit(
        repository_root,
        "eca3bc222d907cf3130a31f3f72b777937693051",
    )
    current_producer = _producer_digest_at_commit(
        repository_root,
        "afa0c8384b9a1639f191804875c8bb186f422e57",
    )
    current = CheckpointInputs(
        source_digest="a" * 64,
        settings={"selection_digest": "b" * 64, "use_gpu": True},
        model_manifest_sha256="c" * 64,
        tool_versions={"colmap": "COLMAP 3.11.1"},
        producer_code_sha256=current_producer,
    )
    previous = CheckpointInputs(
        source_digest=current.source_digest,
        settings=current.settings,
        model_manifest_sha256=current.model_manifest_sha256,
        tool_versions=current.tool_versions,
        producer_code_sha256=previous_producer,
    )
    expected = compatible_colmap_fingerprints(
        previous,
        legacy_producer_digests=(),
    )[0]

    actual = compatible_colmap_fingerprints(
        current,
        legacy_producer_digests=(previous_producer,),
    )

    assert actual[0] == expected


def test_allowlisted_colmap_producer_digests_match_verified_legacy_commits() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    expected = tuple(
        _producer_digest_at_commit(repository_root, commit)
        for commit in (
            "eca3bc222d907cf3130a31f3f72b777937693051",
            "1684964",
        )
    )

    assert LEGACY_COLMAP_PRODUCER_DIGESTS == expected


def test_store_publishes_and_restores_portable_milestone_graph(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    colmap_source, _ = _artifact_root(tmp_path, "colmap-payload", b"colmap")
    colmap = MilestoneRef(CheckpointKind.COLMAP, "1" * 64)
    store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint=colmap.fingerprint,
        run_id="colmap",
        source_root=colmap_source,
    )
    selection = _state(
        tmp_path,
        kind=CheckpointKind.SELECTION,
        fingerprint="2" * 64,
        upstream={},
        name="selection",
        payload=b"selection",
    )
    store.publish_milestone(selection, run_id="selection")
    base = _state(
        tmp_path,
        kind=CheckpointKind.BASE_EVIDENCE,
        fingerprint="3" * 64,
        upstream={
            CheckpointKind.COLMAP: colmap.fingerprint,
            CheckpointKind.SELECTION: selection.ref.fingerprint,
        },
        name="base",
        payload=b"base",
    )
    store.publish_milestone(base, run_id="base")
    semantic = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="4" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: base.ref.fingerprint},
        name="semantic",
        payload=b"semantic",
    )

    generation = store.publish_milestone(semantic, run_id="semantic")
    graph = store.validate_milestone_graph(semantic.ref)
    restored = store.restore_milestone(
        semantic.ref,
        destination=tmp_path / "restored",
        source_inventory=object(),
    )

    assert tuple(item.kind for item in graph) == (
        CheckpointKind.COLMAP,
        CheckpointKind.SELECTION,
        CheckpointKind.BASE_EVIDENCE,
        CheckpointKind.SEMANTIC,
    )
    assert restored is not None
    assert isinstance(restored.value, FrameArtifact)
    assert restored.value.path.read_bytes() == b"semantic"
    assert restored.value.path.is_relative_to(tmp_path / "restored")
    assert (
        restored.artifact_roots["semantic"]
        == (tmp_path / "restored" / "artifacts" / "semantic").resolve()
    )
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    success = json.loads((generation / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert manifest["upstream"] == {"base_evidence": base.ref.fingerprint}
    assert success["upstream"] == manifest["upstream"]
    assert not (generation / "payload" / "artifacts" / "base").exists()


def test_latest_complete_lineage_ignores_newer_incomplete_semantic_branch(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    colmap_source, _ = _artifact_root(tmp_path, "colmap-lineage", b"colmap")
    colmap = MilestoneRef(CheckpointKind.COLMAP, "1" * 64)
    store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint=colmap.fingerprint,
        run_id="colmap",
        source_root=colmap_source,
    )
    round0_selection = _state(
        tmp_path,
        kind=CheckpointKind.SELECTION,
        fingerprint="2" * 64,
        upstream={},
        name="round0-selection",
        payload=b"selection-0",
    )
    store.publish_milestone(round0_selection, run_id="round0-selection")
    round0_base = _state(
        tmp_path,
        kind=CheckpointKind.BASE_EVIDENCE,
        fingerprint="3" * 64,
        upstream={
            CheckpointKind.COLMAP: colmap.fingerprint,
            CheckpointKind.SELECTION: round0_selection.ref.fingerprint,
        },
        name="round0-base",
        payload=b"base-0",
    )
    store.publish_milestone(round0_base, run_id="round0-base")
    round0_semantic = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="4" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: round0_base.ref.fingerprint},
        name="round0-semantic",
        payload=b"semantic-0",
    )
    store.publish_milestone(round0_semantic, run_id="round0-semantic")
    round0_motion = _state(
        tmp_path,
        kind=CheckpointKind.MOTION,
        fingerprint="5" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: round0_base.ref.fingerprint},
        name="round0-motion",
        payload=b"motion-0",
    )
    store.publish_milestone(round0_motion, run_id="round0-motion")
    round0_masks = _state(
        tmp_path,
        kind=CheckpointKind.MASKS,
        fingerprint="6" * 64,
        upstream={
            CheckpointKind.BASE_EVIDENCE: round0_base.ref.fingerprint,
            CheckpointKind.SEMANTIC: round0_semantic.ref.fingerprint,
            CheckpointKind.MOTION: round0_motion.ref.fingerprint,
        },
        name="round0-masks",
        payload=b"masks-0",
    )
    store.publish_milestone(round0_masks, run_id="round0-masks")

    round1_selection = _state(
        tmp_path,
        kind=CheckpointKind.SELECTION,
        fingerprint="7" * 64,
        upstream={},
        name="round1-selection",
        payload=b"selection-1",
    )
    store.publish_milestone(round1_selection, run_id="round1-selection")
    round1_base = _state(
        tmp_path,
        kind=CheckpointKind.BASE_EVIDENCE,
        fingerprint="8" * 64,
        upstream={
            CheckpointKind.COLMAP: colmap.fingerprint,
            CheckpointKind.SELECTION: round1_selection.ref.fingerprint,
        },
        name="round1-base",
        payload=b"base-1",
    )
    store.publish_milestone(round1_base, run_id="round1-base")
    round1_semantic = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="9" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: round1_base.ref.fingerprint},
        name="round1-semantic",
        payload=b"semantic-1",
    )
    store.publish_milestone(round1_semantic, run_id="round1-semantic")

    refs = store.find_latest_complete_lineage(
        CheckpointKind.MASKS,
        selection_fingerprint=round0_selection.ref.fingerprint,
    )

    assert refs is not None
    assert refs == {
        CheckpointKind.COLMAP: colmap,
        CheckpointKind.SELECTION: round0_selection.ref,
        CheckpointKind.BASE_EVIDENCE: round0_base.ref,
        CheckpointKind.SEMANTIC: round0_semantic.ref,
        CheckpointKind.MOTION: round0_motion.ref,
        CheckpointKind.MASKS: round0_masks.ref,
    }
    assert round1_semantic.ref not in refs.values()
    assert (
        store.find_latest_complete_lineage(
            CheckpointKind.MASKS,
            selection_fingerprint=round1_selection.ref.fingerprint,
        )
        is None
    )


def test_missing_or_wrong_upstream_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    missing = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="4" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: "3" * 64},
        name="semantic",
        payload=b"semantic",
    )

    with pytest.raises(ValueError, match="missing upstream"):
        store.publish_milestone(missing, run_id="missing")

    wrong = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="5" * 64,
        upstream={CheckpointKind.MOTION: "6" * 64},
        name="wrong",
        payload=b"wrong",
    )
    with pytest.raises(ValueError, match="upstream kinds"):
        store.publish_milestone(wrong, run_id="wrong")


def test_cycle_is_rejected_even_when_each_generation_is_hash_valid(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    first_source, _ = _artifact_root(tmp_path, "first", b"first")
    second_source, _ = _artifact_root(tmp_path, "second", b"second")
    semantic = MilestoneRef(CheckpointKind.SEMANTIC, "4" * 64)
    motion = MilestoneRef(CheckpointKind.MOTION, "5" * 64)
    store.publish_generation(
        semantic.kind,
        fingerprint=semantic.fingerprint,
        run_id="semantic",
        source_root=first_source,
        upstream={motion.kind: motion.fingerprint},
        artifact_roots={},
    )
    store.publish_generation(
        motion.kind,
        fingerprint=motion.fingerprint,
        run_id="motion",
        source_root=second_source,
        upstream={semantic.kind: semantic.fingerprint},
        artifact_roots={},
    )

    with pytest.raises(ValueError, match="cycle"):
        store.validate_milestone_graph(semantic)


def test_corrupt_success_marker_never_restores_a_milestone(tmp_path: Path) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    selection = _state(
        tmp_path,
        kind=CheckpointKind.SELECTION,
        fingerprint="2" * 64,
        upstream={},
        name="selection",
        payload=b"selection",
    )
    generation = store.publish_milestone(selection, run_id="selection")
    (generation / "_SUCCESS.json").write_text("{}\n", encoding="utf-8")

    assert (
        store.restore_milestone(
            selection.ref,
            destination=tmp_path / "restore",
            source_inventory=object(),
        )
        is None
    )
    assert not (tmp_path / "restore").exists()


def test_state_path_outside_direct_artifact_roots_is_rejected(tmp_path: Path) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    root, _ = _artifact_root(tmp_path, "selection", b"selection")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    state = MilestoneState(
        ref=MilestoneRef(CheckpointKind.SELECTION, "2" * 64),
        upstream={},
        value=FrameArtifact(
            "outside.png", "frame-outside", outside.resolve(), "f" * 64
        ),
        artifact_roots={"selection": root.resolve()},
    )

    with pytest.raises(ValueError, match="exactly one root"):
        store.publish_milestone(state, run_id="selection")

    assert store.find_generation(CheckpointKind.SELECTION, "2" * 64) is None


def test_external_selection_paths_are_rebound_without_copying_upstream_frames(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    direct_root, _ = _artifact_root(tmp_path, "direct", b"direct")
    selected_root, selected_frame = _artifact_root(
        tmp_path, "selected-source", b"selected"
    )
    fresh_selected_root, fresh_selected_frame = _artifact_root(
        tmp_path, "selected-restored", b"selected"
    )
    state = MilestoneState(
        ref=MilestoneRef(CheckpointKind.SELECTION, "2" * 64),
        upstream={},
        value=FrameArtifact("frame.png", "frame-1", selected_frame.resolve(), "f" * 64),
        artifact_roots={"direct": direct_root.resolve()},
    )

    generation = store.publish_milestone(
        state,
        run_id="selection",
        external_roots={"selection": selected_root.resolve()},
    )
    restored = store.restore_milestone(
        state.ref,
        destination=tmp_path / "external-restore",
        source_inventory=object(),
        external_roots={"selection": fresh_selected_root.resolve()},
    )

    assert restored is not None
    assert restored.value.path == fresh_selected_frame.resolve()
    assert not (generation / "payload" / "selected-source").exists()


def test_base_evidence_round_trip_rebinds_canonical_frames_and_keeps_depth_local(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    selected_root, selected_path = _artifact_root(
        tmp_path, "selected-base", b"selected"
    )
    restored_selected_root, restored_selected_path = _artifact_root(
        tmp_path, "selected-base-restored", b"selected"
    )
    source_frame = FrameArtifact(
        "frame.png", "frame-1", selected_path.resolve(), "f" * 64
    )
    colmap_root, _ = _artifact_root(tmp_path, "colmap-base", b"colmap")
    colmap_ref = MilestoneRef(CheckpointKind.COLMAP, "1" * 64)
    store.publish_generation(
        colmap_ref.kind,
        fingerprint=colmap_ref.fingerprint,
        run_id="colmap-base",
        source_root=colmap_root,
    )
    selection_state = MilestoneState(
        ref=MilestoneRef(CheckpointKind.SELECTION, "2" * 64),
        upstream={},
        value=source_frame,
        artifact_roots={"selection": selected_root.resolve()},
    )
    store.publish_milestone(selection_state, run_id="selection-base")

    direct_root = tmp_path / "base-direct"
    direct_root.mkdir()
    depth_path = direct_root / "frame.depth.npy"
    depth_path.write_bytes(b"depth")
    sky_path = direct_root / "frame.sky.npy"
    sky_path.write_bytes(b"sky")
    metadata_path = direct_root / "anchors.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    audit_path = direct_root / "track_audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    anchors = AnchorInferenceResult(
        artifacts=(
            FramePredictionArtifact(
                "frame.png",
                "frame-1",
                selected_path.resolve(),
                depth_path.resolve(),
                None,
                None,
            ),
        ),
        cameras=(
            CameraRecord(
                "frame.png",
                "frame-1",
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
        shared_camera=PinholeCamera("PINHOLE", 8, 6, 4.0, 4.0, 4.0, 3.0),
        anchor_indices=(0,),
        attempts=(),
        metadata_path=metadata_path.resolve(),
    )
    scene = RigidSceneEvidence(
        frames=(
            RigidFrameEvidence(
                source_frame,
                8,
                6,
                False,
                None,
                None,
                depth_path.resolve(),
                "d" * 64,
            ),
        ),
        static_tracks=(),
        geometry_digest="e" * 64,
        depth_digest="d" * 64,
    )
    report = TrackAuditReport(
        1,
        1,
        1,
        0,
        3,
        3,
        0,
        {},
        {},
        (),
        "a" * 64,
    )
    base = BaseEvidenceState(
        anchors=anchors,
        depths=((depth_path.resolve(), "d" * 64),),
        sky=(FrameArtifact("frame.png", "frame-1", sky_path.resolve(), "s" * 64),),
        scene=scene,
        track_audit=QualifiedStaticTracks((), report, audit_path.resolve(), "b" * 64),
        colmap_ref=colmap_ref.fingerprint,
    )
    base_ref = MilestoneRef(CheckpointKind.BASE_EVIDENCE, "3" * 64)

    generation = store.publish_milestone(
        MilestoneState(
            ref=base_ref,
            upstream={
                CheckpointKind.COLMAP: colmap_ref.fingerprint,
                CheckpointKind.SELECTION: selection_state.ref.fingerprint,
            },
            value=base,
            artifact_roots={"base": direct_root.resolve()},
        ),
        run_id="base",
        external_roots={"selection": selected_root.resolve()},
    )
    restored = store.restore_milestone(
        base_ref,
        destination=tmp_path / "base-restored",
        source_inventory=object(),
        external_roots={"selection": restored_selected_root.resolve()},
    )

    assert restored is not None
    assert isinstance(restored.value, BaseEvidenceState)
    assert restored.value.anchors.artifacts[0].source_path == (
        restored_selected_path.resolve()
    )
    assert restored.value.scene.frames[0].frame.path == restored_selected_path.resolve()
    assert restored.value.depths[0][0].is_relative_to(tmp_path / "base-restored")
    assert not (generation / "payload" / "selected-base").exists()


def test_unknown_checkpoint_tag_is_ignored_without_leaking_restore_tree(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    source = tmp_path / "source"
    artifacts = source / "artifacts" / "selection"
    artifacts.mkdir(parents=True)
    (artifacts / "frame.bin").write_bytes(b"selection")
    (source / "state.json").write_text(
        '{"__kind__":"not-registered"}\n', encoding="utf-8"
    )
    ref = MilestoneRef(CheckpointKind.SELECTION, "2" * 64)
    store.publish_generation(
        ref.kind,
        fingerprint=ref.fingerprint,
        run_id="selection",
        source_root=source,
        upstream={},
        artifact_roots={"selection": "artifacts/selection"},
    )

    assert (
        store.restore_milestone(
            ref,
            destination=tmp_path / "restore",
            source_inventory=object(),
        )
        is None
    )
    assert not (tmp_path / "restore").exists()


def test_reference_aware_cleanup_keeps_only_reachable_or_latest_generation(
    tmp_path: Path,
) -> None:
    store = LearnedCheckpointStore(tmp_path / "cache", input_identity="a" * 64)
    colmap_source, _ = _artifact_root(tmp_path, "colmap", b"colmap")
    colmap = MilestoneRef(CheckpointKind.COLMAP, "1" * 64)
    store.publish_generation(
        colmap.kind,
        fingerprint=colmap.fingerprint,
        run_id="colmap",
        source_root=colmap_source,
    )
    selection = _state(
        tmp_path,
        kind=CheckpointKind.SELECTION,
        fingerprint="2" * 64,
        upstream={},
        name="selection-cleanup",
        payload=b"selection",
    )
    store.publish_milestone(selection, run_id="selection")
    base = _state(
        tmp_path,
        kind=CheckpointKind.BASE_EVIDENCE,
        fingerprint="3" * 64,
        upstream={
            CheckpointKind.COLMAP: colmap.fingerprint,
            CheckpointKind.SELECTION: selection.ref.fingerprint,
        },
        name="base-cleanup",
        payload=b"base",
    )
    store.publish_milestone(base, run_id="base")
    motion = _state(
        tmp_path,
        kind=CheckpointKind.MOTION,
        fingerprint="6" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: base.ref.fingerprint},
        name="motion-cleanup",
        payload=b"motion",
    )
    store.publish_milestone(motion, run_id="motion")
    semantic_v1 = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="4" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: base.ref.fingerprint},
        name="semantic-v1",
        payload=b"semantic-v1",
    )
    first = store.publish_milestone(semantic_v1, run_id="semantic-v1")
    masks = _state(
        tmp_path,
        kind=CheckpointKind.MASKS,
        fingerprint="7" * 64,
        upstream={
            CheckpointKind.BASE_EVIDENCE: base.ref.fingerprint,
            CheckpointKind.SEMANTIC: semantic_v1.ref.fingerprint,
            CheckpointKind.MOTION: motion.ref.fingerprint,
        },
        name="masks-cleanup",
        payload=b"masks",
    )
    masks_generation = store.publish_milestone(masks, run_id="masks")
    semantic_v2 = _state(
        tmp_path,
        kind=CheckpointKind.SEMANTIC,
        fingerprint="5" * 64,
        upstream={CheckpointKind.BASE_EVIDENCE: base.ref.fingerprint},
        name="semantic-v2",
        payload=b"semantic-v2",
    )
    second = store.publish_milestone(semantic_v2, run_id="semantic-v2")

    assert first.is_dir()
    assert second.is_dir()
    (masks_generation / "_SUCCESS.json").write_text("{}\n", encoding="utf-8")
    store.cleanup_unreferenced_milestones(semantic_v2.ref)

    assert not first.exists()
    assert second.is_dir()
