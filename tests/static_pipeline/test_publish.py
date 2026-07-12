from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.static_pipeline.publish import (
    GENERATOR_ID,
    FileOps,
    PublishError,
    PublishOwnershipError,
    derive_result_path,
    inventory_bundle,
    is_owned_result,
    publish_diagnostics,
    publish_result,
    validate_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _make_bundle(root: Path, run_id: str = "run-2") -> Path:
    root.mkdir()
    (root / "logs").mkdir()
    (root / "logs" / "pipeline.log").write_text("complete\n", encoding="utf-8")
    (root / "splat.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    (root / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\npreview")
    _write_json(root / "scene_metadata.json", {"schema_version": 1})
    _write_json(root / "quality_report.json", {"status": "passed"})

    artifacts = [
        {
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in inventory_bundle(root)
    ]
    _write_json(
        root / "run_manifest.json",
        {
            "generator_id": GENERATOR_ID,
            "schema_version": 1,
            "run_id": run_id,
            "status": "success",
            "source": {"digest": "a" * 64},
            "artifacts": artifacts,
            "repository_commit": "f54d1e5",
            "tool_versions": {"python": "3.12"},
            "timing": {"total_seconds": 10.0},
            "hardware": {"gpu": "test"},
        },
    )
    return root


def _mark_success(root: Path) -> None:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    _write_json(
        root / "_SUCCESS",
        {
            "generator_id": GENERATOR_ID,
            "run_id": manifest["run_id"],
            "manifest_sha256": _sha256(root / "run_manifest.json"),
            "published_at": "2026-07-12T00:00:00+00:00",
        },
    )


def _make_owned_result(root: Path, run_id: str = "previous") -> Path:
    _make_bundle(root, run_id=run_id)
    _mark_success(root)
    return root


class FailingFileOps(FileOps):
    def __init__(self, failure_point: str) -> None:
        self.failure_point = failure_point

    def checkpoint(self, name: str) -> None:
        if name == self.failure_point:
            raise OSError(f"injected {name}")


class PartialFinalMoveFileOps(FileOps):
    def move_tree(self, source: Path, destination: Path) -> None:
        if ".__tmp__" in source.name:
            destination.mkdir()
            (destination / "partial").write_text("incomplete", encoding="utf-8")
            raise OSError("injected partial final move")
        super().move_tree(source, destination)


def test_result_path_is_exact_sibling(tmp_path: Path) -> None:
    source = tmp_path / "captures" / "room"
    assert derive_result_path(source) == tmp_path / "captures" / "room_result"


def test_publish_creates_verified_success_marker(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")

    receipt = publish_result(bundle, input_folder, run_id="run-2")

    assert receipt.final_path == tmp_path / "room_result"
    assert receipt.run_id == "run-2"
    assert receipt.manifest_sha256 == _sha256(receipt.final_path / "run_manifest.json")
    assert is_owned_result(receipt.final_path)
    assert not (tmp_path / "room_result.__tmp__run-2").exists()


def test_publish_rejects_a_local_success_marker(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    _mark_success(bundle)

    with pytest.raises(ValueError, match="pre-publish _SUCCESS"):
        publish_result(bundle, input_folder, run_id="run-2")

    assert not (tmp_path / "room_result").exists()


def test_owned_result_is_replaced_without_merging_stale_files(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    final = _make_owned_result(tmp_path / "room_result")
    (final / "stale.bin").write_bytes(b"old")

    receipt = publish_result(
        bundle,
        input_folder,
        run_id="run-2",
        replace_owned_result=True,
    )

    assert receipt.final_path == final
    assert receipt.replaced_previous
    assert not (final / "stale.bin").exists()
    assert (final / "_SUCCESS").is_file()


def test_unowned_result_is_never_deleted(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    final = tmp_path / "room_result"
    final.mkdir()
    (final / "mine.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(PublishOwnershipError):
        publish_result(
            bundle,
            input_folder,
            run_id="run-2",
            replace_owned_result=True,
        )

    assert (final / "mine.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "failure_point",
    ["stage_verify", "backup_move", "final_move", "final_verify"],
)
def test_failure_restores_previous_good_result(
    tmp_path: Path,
    failure_point: str,
) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    final = _make_owned_result(tmp_path / "room_result", run_id="previous")

    with pytest.raises(PublishError, match=failure_point):
        publish_result(
            bundle,
            input_folder,
            run_id="run-2",
            replace_owned_result=True,
            file_ops=FailingFileOps(failure_point),
        )

    marker = json.loads((final / "_SUCCESS").read_text(encoding="utf-8"))
    assert marker["run_id"] == "previous"
    assert is_owned_result(final)


def test_partial_final_move_is_removed_before_backup_restore(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    final = _make_owned_result(tmp_path / "room_result", run_id="previous")

    with pytest.raises(PublishError, match="final_move"):
        publish_result(
            bundle,
            input_folder,
            run_id="run-2",
            replace_owned_result=True,
            file_ops=PartialFinalMoveFileOps(),
        )

    assert not (final / "partial").exists()
    assert json.loads((final / "_SUCCESS").read_text(encoding="utf-8"))["run_id"] == (
        "previous"
    )
    assert is_owned_result(final)


def test_validation_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    validate_bundle(bundle, run_id="run-2")
    (bundle / "preview.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="preview|artifact"):
        validate_bundle(bundle, run_id="run-2")

    symlink_bundle = _make_bundle(tmp_path / "symlink-bundle")
    target = tmp_path / "external.log"
    target.write_text("external", encoding="utf-8")
    try:
        (symlink_bundle / "logs" / "linked.log").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        validate_bundle(symlink_bundle, run_id="run-2")


def test_manifest_inventory_covers_optional_and_nested_manifest_named_files(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    (bundle / "world").mkdir()
    (bundle / "world" / "run_manifest.json").write_text("world", encoding="utf-8")
    (bundle / "orbit.mp4").write_bytes(b"orbit")
    records = inventory_bundle(bundle)
    paths = {record.relative_path for record in records}
    assert "world/run_manifest.json" in paths
    assert "orbit.mp4" in paths
    assert "run_manifest.json" not in paths


def test_validation_parses_optional_json_files(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    (bundle / "world").mkdir()
    (bundle / "world" / "collider.json").write_text("{broken", encoding="utf-8")
    manifest_path = bundle / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in inventory_bundle(bundle)
    ]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid JSON"):
        validate_bundle(bundle, run_id="run-2")


def test_validation_rejects_wrong_run_and_missing_provenance(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    with pytest.raises(ValueError, match="run_id"):
        validate_bundle(bundle, run_id="another-run")

    manifest_path = bundle / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["hardware"]
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="hardware"):
        validate_bundle(bundle, run_id="run-2")


def test_owned_result_requires_valid_manifest_hash(tmp_path: Path) -> None:
    result = _make_owned_result(tmp_path / "room_result")
    assert is_owned_result(result)
    marker = json.loads((result / "_SUCCESS").read_text(encoding="utf-8"))
    marker["manifest_sha256"] = "0" * 64
    _write_json(result / "_SUCCESS", marker)
    assert not is_owned_result(result)


def test_owned_result_with_added_symlink_is_not_safe_to_replace(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = _make_bundle(tmp_path / "bundle")
    result = _make_owned_result(tmp_path / "room_result")
    external = tmp_path / "external.txt"
    external.write_text("keep", encoding="utf-8")
    try:
        (result / "external-link").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(PublishOwnershipError, match="unsafe"):
        publish_result(
            bundle,
            input_folder,
            run_id="run-2",
            replace_owned_result=True,
        )

    assert external.read_text(encoding="utf-8") == "keep"
    assert is_owned_result(result)


def test_diagnostics_never_claim_success_or_touch_result(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    final = tmp_path / "room_result"
    final.mkdir()
    (final / "mine.txt").write_text("keep", encoding="utf-8")
    log = tmp_path / "pipeline.log"
    log.write_text("failed", encoding="utf-8")

    diagnostics = publish_diagnostics(
        input_folder,
        "run-failed",
        {"logs/pipeline.log": log},
    )

    assert diagnostics == tmp_path / "room_result_diagnostics" / "run-failed"
    assert (diagnostics / "logs" / "pipeline.log").read_text(
        encoding="utf-8"
    ) == "failed"
    assert not (diagnostics / "_SUCCESS").exists()
    assert (final / "mine.txt").read_text(encoding="utf-8") == "keep"


def test_diagnostics_reject_splat_ply(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    splat = tmp_path / "splat.ply"
    splat.write_bytes(b"ply")

    with pytest.raises(ValueError, match="splat.ply"):
        publish_diagnostics(input_folder, "run-failed", {"splat.ply": splat})


def test_diagnostics_reject_unapproved_artifacts(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    artifact = tmp_path / "viewer.html"
    artifact.write_text("viewer", encoding="utf-8")

    with pytest.raises(ValueError, match="approved diagnostic"):
        publish_diagnostics(input_folder, "run-failed", {"viewer.html": artifact})
