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
from experiments.learned_quality.milestones import (
    LEGACY_COLMAP_PRODUCER_DIGESTS,
    MilestoneInputs,
    MilestoneRef,
    MilestoneState,
    compatible_colmap_fingerprints,
    milestone_fingerprint,
)


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


def test_allowlisted_colmap_producer_digest_matches_verified_legacy_commit() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    paths = (
        "backend/static_pipeline/colmap.py",
        "experiments/learned_quality/geometry.py",
        "experiments/learned_quality/runtime.py",
    )
    rows = []
    for relative in sorted(paths):
        content = subprocess.run(
            ["git", "show", f"1684964:{relative}"],
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

    assert LEGACY_COLMAP_PRODUCER_DIGESTS == (hashlib.sha256(encoded).hexdigest(),)


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
