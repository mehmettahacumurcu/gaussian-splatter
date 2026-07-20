from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.learned_quality.ablation_staging import (
    freeze_staged_tree,
    materialize_experiment_workspace,
    stage_ablation_inputs,
)


PIN = "a" * 40
FINGERPRINT = "b" * 64


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


def _stage(tmp_path: Path, store: FakeStore, *, freeze: bool = False):
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
