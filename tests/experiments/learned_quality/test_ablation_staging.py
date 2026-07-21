from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.static_pipeline.runner import SelectionOutput
from experiments.learned_quality.ablation_staging import (
    RestoredAblationPretraining,
    freeze_staged_tree,
    materialize_experiment_workspace,
    restore_output_first_pretraining,
    stage_ablation_inputs,
)
from experiments.learned_quality.cache import CheckpointKind
from experiments.learned_quality.milestones import (
    BaseEvidenceState,
    FinalPretrainingState,
    GeometryMilestoneState,
    MasksMilestoneState,
    MilestoneRef,
    MilestoneState,
    MotionMilestoneState,
    SemanticMilestoneState,
)


PIN = "a" * 40
FINGERPRINT = "b" * 64
GRAPH_FINGERPRINT = "f" * 64


class FakeStore:
    def __init__(self, *, escaped_path: Path | None = None) -> None:
        self.probes = 0
        self.restores = 0
        self.escaped_path = escaped_path

    def probe_drive_publication(self, *, run_id: str) -> None:
        assert run_id == "ablation-run"
        self.probes += 1

    def restore_pretraining(
        self,
        *,
        source_inventory: object,
        destination: Path,
        expected_fingerprint: object,
    ) -> object:
        self.restores += 1
        assert getattr(source_inventory, "digest") == "c" * 64
        assert callable(expected_fingerprint)
        frames = destination / "artifacts" / "selection" / "frames"
        model = destination / "artifacts" / "reconstruction" / "model"
        frames.mkdir(parents=True)
        model.mkdir(parents=True)
        (frames / "frame_000001.png").write_bytes(b"rgb")
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            (model / name).write_text(name, encoding="utf-8")
        source_manifest = destination / "artifacts" / "selection" / "source.json"
        source_manifest.write_text("{}", encoding="utf-8")
        frame_path = self.escaped_path or frames
        selection = SimpleNamespace(
            inventory=source_inventory,
            manifest=SimpleNamespace(image_set_digest="d" * 64),
            frames_dir=frame_path,
            source_manifest_path=source_manifest,
        )
        reconstruction = SimpleNamespace(
            frames_dir=frame_path,
            accepted_model_dir=model,
            artifacts=SimpleNamespace(model_manifest_path=model / "cameras.txt"),
        )
        return SimpleNamespace(selection=selection, reconstruction=reconstruction)


def _stage(
    tmp_path: Path,
    store: FakeStore,
    *,
    freeze: bool = False,
    restored_validator=None,
):
    inventory = SimpleNamespace(digest="c" * 64)
    audit_calls: list[object] = []
    staged = stage_ablation_inputs(
        store=store,
        source_inventory=inventory,
        destination=tmp_path / "local" / "inputs",
        drive_root=tmp_path / "drive",
        expected_fingerprint=lambda _selection: FINGERPRINT,
        expected_source_revision=PIN,
        actual_source_revision=PIN,
        audit_validator=lambda value: audit_calls.append(value),
        minimum_free_bytes=1_000,
        available_free_bytes=2_000,
        run_id="ablation-run",
        freeze=freeze,
        restored_validator=restored_validator,
    )
    return staged, inventory, audit_calls


def test_staging_restores_and_verifies_the_drive_payload_exactly_once(
    tmp_path: Path,
) -> None:
    store = FakeStore()

    staged, inventory, audit_calls = _stage(tmp_path, store)

    assert store.probes == 1
    assert store.restores == 1
    assert audit_calls == [inventory]
    assert staged.source_digest == "c" * 64
    assert staged.pretraining_fingerprint == FINGERPRINT
    assert staged.selection.frames_dir.is_dir()
    assert staged.reconstruction.accepted_model_dir.is_dir()
    assert staged.manifest_path.is_file()
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_revision"] == PIN
    assert manifest["source_digest"] == "c" * 64
    assert manifest["pretraining_fingerprint"] == FINGERPRINT
    assert manifest["drive_reads_permitted_after_staging"] is False


def test_staging_accepts_a_graph_restorer_and_records_its_real_fingerprint(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    calls: list[Path] = []

    def restore_graph(destination: Path) -> object:
        calls.append(destination)
        frames = destination / "graph" / "frames"
        model = destination / "graph" / "model"
        frames.mkdir(parents=True)
        model.mkdir(parents=True)
        (frames / "frame_000001.png").write_bytes(b"rgb")
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            (model / name).write_text(name, encoding="utf-8")
        source_manifest = destination / "graph" / "source.json"
        source_manifest.write_text("{}", encoding="utf-8")
        inventory = SimpleNamespace(digest="c" * 64)
        selection = SimpleNamespace(
            inventory=inventory,
            manifest=SimpleNamespace(image_set_digest="d" * 64),
            frames_dir=frames,
            source_manifest_path=source_manifest,
        )
        reconstruction = SimpleNamespace(
            frames_dir=frames,
            accepted_model_dir=model,
            artifacts=SimpleNamespace(model_manifest_path=model / "cameras.txt"),
        )
        return RestoredAblationPretraining(
            selection=selection,
            reconstruction=reconstruction,
            fingerprint=GRAPH_FINGERPRINT,
        )

    inventory = SimpleNamespace(digest="c" * 64)
    staged = stage_ablation_inputs(
        store=store,
        source_inventory=inventory,
        destination=tmp_path / "local" / "inputs",
        drive_root=tmp_path / "drive",
        expected_fingerprint=lambda _selection: FINGERPRINT,
        expected_source_revision=PIN,
        actual_source_revision=PIN,
        audit_validator=lambda _inventory: None,
        minimum_free_bytes=1_000,
        available_free_bytes=2_000,
        run_id="ablation-run",
        freeze=False,
        restore_pretraining=restore_graph,
    )

    assert calls == [tmp_path / "local" / "inputs"]
    assert store.restores == 0
    assert staged.pretraining_fingerprint == GRAPH_FINGERPRINT
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pretraining_fingerprint"] == GRAPH_FINGERPRINT


def test_output_first_restore_uses_the_complete_graph_without_legacy_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = SimpleNamespace(digest="c" * 64)
    selection_ref = MilestoneRef(CheckpointKind.SELECTION, "1" * 64)
    base_ref = MilestoneRef(CheckpointKind.BASE_EVIDENCE, "2" * 64)
    semantic_ref = MilestoneRef(CheckpointKind.SEMANTIC, "3" * 64)
    motion_ref = MilestoneRef(CheckpointKind.MOTION, "4" * 64)
    masks_ref = MilestoneRef(CheckpointKind.MASKS, "5" * 64)
    colmap_ref = MilestoneRef(CheckpointKind.COLMAP, "6" * 64)
    geometry_ref = MilestoneRef(CheckpointKind.GEOMETRY, "7" * 64)
    pretraining_ref = MilestoneRef(CheckpointKind.PRETRAINING, GRAPH_FINGERPRINT)
    restored_kinds: list[CheckpointKind] = []

    class FakeGraphStore:
        def restore_pretraining(self, **_kwargs: object) -> object:
            raise AssertionError("legacy cumulative restore must not be used")

        def restore_milestone(
            self,
            ref: MilestoneRef,
            *,
            destination: Path,
            source_inventory: object,
            external_roots: object = None,
        ) -> MilestoneState | None:
            del external_roots
            assert source_inventory is inventory
            restored_kinds.append(ref.kind)
            destination.mkdir(parents=True)
            artifact = destination / "artifact"
            artifact.mkdir()
            if ref.kind is CheckpointKind.SELECTION:
                frames = artifact / "frames"
                frames.mkdir()
                source = artifact / "source.json"
                source.write_text("{}", encoding="utf-8")
                value = SelectionOutput(
                    inventory=inventory,
                    manifest=SimpleNamespace(image_set_digest="9" * 64),
                    frames_dir=frames,
                    source_manifest_path=source,
                )
            elif ref.kind is CheckpointKind.BASE_EVIDENCE:
                value = BaseEvidenceState(
                    anchors=object(),
                    depths=(),
                    sky=(),
                    scene=object(),
                    track_audit=object(),
                    colmap_ref=colmap_ref.fingerprint,
                )
            elif ref.kind is CheckpointKind.SEMANTIC:
                value = SemanticMilestoneState(object())
            elif ref.kind is CheckpointKind.MOTION:
                value = MotionMilestoneState(object())
            elif ref.kind is CheckpointKind.MASKS:
                value = MasksMilestoneState(object())
            elif ref.kind is CheckpointKind.GEOMETRY:
                value = GeometryMilestoneState(
                    bundle=object(),
                    frames_dir=artifact,
                    geometry_candidates=(),
                )
            elif ref.kind is CheckpointKind.PRETRAINING:
                manifest = artifact / "model_manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                value = FinalPretrainingState(
                    photometric=object(),
                    depth=object(),
                    dense_seeds=object(),
                    final_stage_records=(),
                    model_manifest_path=manifest,
                )
            else:
                raise AssertionError(ref.kind)
            return MilestoneState(ref, {}, value, {"artifact": artifact})

        def find_latest_complete_lineage(
            self,
            kind: CheckpointKind,
            *,
            selection_fingerprint: str,
        ) -> dict[CheckpointKind, MilestoneRef]:
            assert kind is CheckpointKind.MASKS
            assert selection_fingerprint == selection_ref.fingerprint
            return {
                CheckpointKind.SELECTION: selection_ref,
                CheckpointKind.COLMAP: colmap_ref,
                CheckpointKind.BASE_EVIDENCE: base_ref,
                CheckpointKind.SEMANTIC: semantic_ref,
                CheckpointKind.MOTION: motion_ref,
                CheckpointKind.MASKS: masks_ref,
            }

        def restore_colmap(self, fingerprint: str, *, destination: Path) -> object:
            assert fingerprint == colmap_ref.fingerprint
            destination.mkdir(parents=True)
            return SimpleNamespace(root=destination)

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            self.store = kwargs["store"]
            self.source_inventory = kwargs["source_inventory"]
            self.external_roots = kwargs["external_roots"]

        def restore(self, ref: MilestoneRef, destination: Path) -> object:
            return self.store.restore_milestone(
                ref,
                destination=destination,
                source_inventory=self.source_inventory,
                external_roots=self.external_roots,
            )

        def bind_external_root(self, label: str, root: Path) -> "FakeSession":
            assert label == "round0_colmap"
            return FakeSession(
                store=self.store,
                source_inventory=self.source_inventory,
                external_roots={**self.external_roots, label: root},
            )

    expected_reconstruction = SimpleNamespace(done=True)
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.MilestoneSession",
        FakeSession,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging._geometry_milestone_ref",
        lambda *_args, **_kwargs: geometry_ref,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging._final_pretraining_milestone_ref",
        lambda *_args, **_kwargs: pretraining_ref,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.assemble_learned_reconstruction",
        lambda **_kwargs: expected_reconstruction,
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")

    restored = restore_output_first_pretraining(
        store=FakeGraphStore(),
        source_inventory=inventory,
        destination=tmp_path / "restore",
        selection_refs=(selection_ref,),
        hardware=object(),
        model_manifest_path=model_manifest,
        repository_root=tmp_path,
        run_id="ablation-run",
    )

    assert restored is not None
    assert restored.reconstruction is expected_reconstruction
    assert restored.fingerprint == GRAPH_FINGERPRINT
    assert restored_kinds == [
        CheckpointKind.SELECTION,
        CheckpointKind.BASE_EVIDENCE,
        CheckpointKind.SEMANTIC,
        CheckpointKind.MOTION,
        CheckpointKind.MASKS,
        CheckpointKind.GEOMETRY,
        CheckpointKind.PRETRAINING,
    ]


def test_output_first_restore_rejects_unaudited_candidate_before_graph_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = SimpleNamespace(digest="c" * 64)
    rejected_ref = MilestoneRef(CheckpointKind.SELECTION, "1" * 64)
    accepted_ref = MilestoneRef(CheckpointKind.SELECTION, "2" * 64)
    base_ref = MilestoneRef(CheckpointKind.BASE_EVIDENCE, "3" * 64)
    semantic_ref = MilestoneRef(CheckpointKind.SEMANTIC, "4" * 64)
    motion_ref = MilestoneRef(CheckpointKind.MOTION, "5" * 64)
    masks_ref = MilestoneRef(CheckpointKind.MASKS, "6" * 64)
    colmap_ref = MilestoneRef(CheckpointKind.COLMAP, "7" * 64)
    geometry_ref = MilestoneRef(CheckpointKind.GEOMETRY, "8" * 64)
    pretraining_ref = MilestoneRef(CheckpointKind.PRETRAINING, GRAPH_FINGERPRINT)
    restored_selections: list[SelectionOutput] = []
    lineage_queries: list[str] = []
    non_selection_restores: list[tuple[str, CheckpointKind]] = []

    class FakeGraphStore:
        def restore_milestone(
            self,
            ref: MilestoneRef,
            *,
            destination: Path,
            source_inventory: object,
        ) -> MilestoneState:
            assert ref.kind is CheckpointKind.SELECTION
            assert source_inventory is inventory
            assert not destination.exists()
            frames = destination / "frames"
            frames.mkdir(parents=True)
            source = destination / "source.json"
            source.write_text("{}", encoding="utf-8")
            selection = SelectionOutput(
                inventory=inventory,
                manifest=SimpleNamespace(image_set_digest=ref.fingerprint),
                frames_dir=frames,
                source_manifest_path=source,
            )
            restored_selections.append(selection)
            return MilestoneState(ref, {}, selection, {"frames": frames})

        def find_latest_complete_lineage(
            self,
            kind: CheckpointKind,
            *,
            selection_fingerprint: str,
        ) -> dict[CheckpointKind, MilestoneRef]:
            assert kind is CheckpointKind.MASKS
            lineage_queries.append(selection_fingerprint)
            return {
                CheckpointKind.SELECTION: accepted_ref,
                CheckpointKind.COLMAP: colmap_ref,
                CheckpointKind.BASE_EVIDENCE: base_ref,
                CheckpointKind.SEMANTIC: semantic_ref,
                CheckpointKind.MOTION: motion_ref,
                CheckpointKind.MASKS: masks_ref,
            }

        def restore_colmap(self, fingerprint: str, *, destination: Path) -> object:
            assert fingerprint == colmap_ref.fingerprint
            destination.mkdir(parents=True)
            return SimpleNamespace(root=destination)

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            self.selection_ref = kwargs["selection_ref"]

        def bind_external_root(self, _label: str, _root: Path) -> "FakeSession":
            return self

    def fake_restored_value(
        session: FakeSession,
        ref: MilestoneRef,
        _destination: Path,
        expected_type: type,
    ) -> object:
        non_selection_restores.append((session.selection_ref.fingerprint, ref.kind))
        if expected_type is BaseEvidenceState:
            return BaseEvidenceState(
                anchors=object(),
                depths=(),
                sky=(),
                scene=object(),
                track_audit=object(),
                colmap_ref=colmap_ref.fingerprint,
            )
        if expected_type is SemanticMilestoneState:
            return SemanticMilestoneState(object())
        if expected_type is MotionMilestoneState:
            return MotionMilestoneState(object())
        if expected_type is MasksMilestoneState:
            return MasksMilestoneState(object())
        if expected_type is GeometryMilestoneState:
            return GeometryMilestoneState(
                bundle=object(),
                frames_dir=tmp_path,
                geometry_candidates=(),
            )
        if expected_type is FinalPretrainingState:
            return FinalPretrainingState(
                photometric=object(),
                depth=object(),
                dense_seeds=object(),
                final_stage_records=(),
                model_manifest_path=tmp_path / "model_manifest.json",
            )
        raise AssertionError(expected_type)

    expected_reconstruction = SimpleNamespace(selection="second")
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.MilestoneSession",
        FakeSession,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging._restored_value",
        fake_restored_value,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging._geometry_milestone_ref",
        lambda *_args, **_kwargs: geometry_ref,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging._final_pretraining_milestone_ref",
        lambda *_args, **_kwargs: pretraining_ref,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.assemble_learned_reconstruction",
        lambda **_kwargs: expected_reconstruction,
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")

    def validate_selection(selection: object) -> None:
        assert isinstance(selection, SelectionOutput)
        if selection.manifest.image_set_digest == rejected_ref.fingerprint:
            raise RuntimeError("restored selection does not have an exact CPU audit")

    restored = restore_output_first_pretraining(
        store=FakeGraphStore(),
        source_inventory=inventory,
        destination=tmp_path / "restore",
        selection_refs=(rejected_ref, accepted_ref),
        selection_validator=validate_selection,
        hardware=object(),
        model_manifest_path=model_manifest,
        repository_root=tmp_path,
        run_id="ablation-run",
    )

    assert restored is not None
    assert restored.selection is restored_selections[1]
    assert restored.reconstruction is expected_reconstruction
    assert lineage_queries == [accepted_ref.fingerprint]
    assert all(
        fingerprint == accepted_ref.fingerprint
        for fingerprint, _kind in non_selection_restores
    )


def test_staging_rejects_revision_disk_and_existing_destination_before_restore(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    inventory = SimpleNamespace(digest="c" * 64)
    common = {
        "store": store,
        "source_inventory": inventory,
        "drive_root": tmp_path / "drive",
        "expected_fingerprint": lambda _selection: FINGERPRINT,
        "audit_validator": lambda _inventory: None,
        "minimum_free_bytes": 1_000,
        "run_id": "ablation-run",
    }

    with pytest.raises(RuntimeError, match="source revision"):
        stage_ablation_inputs(
            **common,
            destination=tmp_path / "revision" / "inputs",
            expected_source_revision=PIN,
            actual_source_revision="e" * 40,
            available_free_bytes=2_000,
        )
    with pytest.raises(RuntimeError, match="free local disk"):
        stage_ablation_inputs(
            **common,
            destination=tmp_path / "disk" / "inputs",
            expected_source_revision=PIN,
            actual_source_revision=PIN,
            available_free_bytes=999,
        )
    destination = tmp_path / "exists" / "inputs"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="destination"):
        stage_ablation_inputs(
            **common,
            destination=destination,
            expected_source_revision=PIN,
            actual_source_revision=PIN,
            available_free_bytes=2_000,
        )

    assert store.probes == 0
    assert store.restores == 0


def test_staging_rejects_any_training_payload_that_resolves_under_drive(
    tmp_path: Path,
) -> None:
    drive = tmp_path / "drive"
    escaped = drive / "cache" / "frames"
    escaped.mkdir(parents=True)
    (escaped / "frame.png").write_bytes(b"drive")
    store = FakeStore(escaped_path=escaped)

    with pytest.raises(ValueError, match="Drive-backed"):
        _stage(tmp_path, store)


def test_experiment_workspaces_share_inputs_but_keep_mutable_state_private(
    tmp_path: Path,
) -> None:
    staged, _, _ = _stage(tmp_path, FakeStore())

    control = materialize_experiment_workspace(
        staged,
        experiments_root=tmp_path / "experiments",
        experiment_id="legacy_control",
    )
    learned = materialize_experiment_workspace(
        staged,
        experiments_root=tmp_path / "experiments",
        experiment_id="full_learned",
    )

    assert control.inputs_root == staged.root
    assert learned.inputs_root == staged.root
    assert control.scene_root != learned.scene_root
    assert control.output_root != learned.output_root
    assert control.scene_root.is_dir() and learned.scene_root.is_dir()
    assert control.output_root.is_dir() and learned.output_root.is_dir()
    with pytest.raises(FileExistsError):
        materialize_experiment_workspace(
            staged,
            experiments_root=tmp_path / "experiments",
            experiment_id="legacy_control",
        )
    with pytest.raises(ValueError, match="experiment_id"):
        materialize_experiment_workspace(
            staged,
            experiments_root=tmp_path / "experiments",
            experiment_id="../escape",
        )


def test_freeze_marks_the_staged_payload_read_only(tmp_path: Path) -> None:
    staged, _, _ = _stage(tmp_path, FakeStore())

    freeze_staged_tree(staged.root)

    payload = staged.selection.frames_dir / "frame_000001.png"
    assert payload.stat().st_mode & 0o222 == 0


def test_staging_validates_the_exact_restored_selection_before_freezing(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, object]] = []
    staged, _, _ = _stage(
        tmp_path,
        FakeStore(),
        restored_validator=lambda selection, reconstruction: calls.append(
            (selection, reconstruction)
        ),
    )

    assert calls == [(staged.selection, staged.reconstruction)]


def test_failed_restored_selection_validation_removes_local_payload(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "local" / "inputs"

    with pytest.raises(RuntimeError, match="audit mismatch"):
        _stage(
            tmp_path,
            FakeStore(),
            restored_validator=lambda _selection, _reconstruction: (_ for _ in ()).throw(
                RuntimeError("audit mismatch")
            ),
        )

    assert not destination.exists()
