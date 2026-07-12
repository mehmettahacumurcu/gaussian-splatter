from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping

from .contracts import ArtifactRecord, PublishReceipt

GENERATOR_ID = "4dgs-studio.static-notebook"
MANIFEST_SCHEMA_VERSION = 1
_REQUIRED_FILES = frozenset(
    {
        "splat.ply",
        "scene_metadata.json",
        "preview.png",
        "quality_report.json",
        "run_manifest.json",
    }
)
_JSON_FILES = frozenset(
    {"scene_metadata.json", "quality_report.json", "run_manifest.json"}
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_DIAGNOSTIC_ROOT_FILES = frozenset(
    {
        "contact_sheet.png",
        "quality_report.json",
        "uncovered_intervals.json",
        "uncovered_interval_report.json",
    }
)


class PublishError(RuntimeError):
    """A result could not be published without weakening atomicity."""


class PublishOwnershipError(PublishError):
    """An existing result is not authenticated as one of our outputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact relative_path must be a portable relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("artifact relative_path escapes the bundle")
    return relative


def _assert_plain_tree(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("bundle root must not be a symlink")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle contains a symlink: {path.relative_to(root)}")


def inventory_bundle(root: Path) -> tuple[ArtifactRecord, ...]:
    """Return the deterministic non-self-referential bundle inventory."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError("bundle must be an existing directory")
    _assert_plain_tree(root)
    records: list[ArtifactRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"run_manifest.json", "_SUCCESS"}:
            continue
        _safe_relative_path(relative)
        records.append(
            ArtifactRecord(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return tuple(records)


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc


def _read_json_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path.name}")
    return value


def _validate_ply(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(64 * 1024)
    normalized = header.replace(b"\r\n", b"\n")
    end = normalized.find(b"end_header\n")
    if (
        not normalized.startswith(b"ply\n")
        or end < 0
        or b"format " not in normalized[:end]
        or b"element vertex " not in normalized[:end]
        or not all(
            token in normalized[:end]
            for token in (b"property float x", b"property float y", b"property float z")
        )
    ):
        raise ValueError("splat.ply has an invalid PLY header")


def _manifest_artifacts(value: object) -> tuple[ArtifactRecord, ...]:
    if not isinstance(value, list):
        raise ValueError("run manifest artifacts must be a list")
    records: list[ArtifactRecord] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("run manifest contains an invalid artifact record")
        relative_path = item["relative_path"]
        if not isinstance(relative_path, str):
            raise ValueError("artifact relative_path must be a string")
        _safe_relative_path(relative_path)
        size_bytes = item["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be a non-negative integer")
        sha256 = item["sha256"]
        if not _is_sha256(sha256):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        if relative_path in seen:
            raise ValueError("run manifest contains duplicate artifact paths")
        seen.add(relative_path)
        records.append(ArtifactRecord(relative_path, size_bytes, sha256))
    return tuple(sorted(records, key=lambda record: record.relative_path))


def validate_bundle(
    root: Path, *, run_id: str | None = None
) -> tuple[ArtifactRecord, ...]:
    """Validate a complete local or copied result bundle and its checksums."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError("bundle must be an existing directory")
    _assert_plain_tree(root)
    missing = sorted(name for name in _REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"bundle is missing required files: {', '.join(missing)}")
    logs = root / "logs"
    if not logs.is_dir() or not any(path.is_file() for path in logs.rglob("*")):
        raise ValueError("bundle logs directory must contain at least one file")
    for name in _JSON_FILES:
        _read_json_object(root / name)
    for json_path in root.rglob("*.json"):
        _read_json(json_path)
    _validate_ply(root / "splat.ply")
    if not (root / "preview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("preview.png has an invalid PNG signature")

    manifest = _read_json_object(root / "run_manifest.json")
    if manifest.get("generator_id") != GENERATOR_ID:
        raise ValueError("run manifest has an unexpected generator_id")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest schema is unsupported")
    manifest_run_id = manifest.get("run_id")
    if not isinstance(manifest_run_id, str) or not manifest_run_id:
        raise ValueError("run manifest run_id must be a non-empty string")
    if run_id is not None and manifest_run_id != run_id:
        raise ValueError("run manifest does not match the requested run_id")
    if manifest.get("status") != "success":
        raise ValueError("run manifest status must be success")
    source = manifest.get("source")
    if not isinstance(source, dict) or not _is_sha256(source.get("digest")):
        raise ValueError("run manifest source digest is invalid")
    repository_commit = manifest.get("repository_commit")
    if not isinstance(repository_commit, str) or not repository_commit.strip():
        raise ValueError("run manifest repository_commit is required")
    for key in ("tool_versions", "timing", "hardware"):
        value = manifest.get(key)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"run manifest {key} must be a non-empty object")

    expected = _manifest_artifacts(manifest.get("artifacts"))
    actual = inventory_bundle(root)
    if expected != actual:
        raise ValueError(
            "run manifest artifact inventory does not match bundle contents"
        )
    return actual


def derive_result_path(input_folder: Path) -> Path:
    input_folder = Path(input_folder)
    if not input_folder.name:
        raise ValueError("input folder must have a name")
    return input_folder.with_name(f"{input_folder.name}_result")


def _valid_run_id(run_id: str) -> bool:
    return (
        isinstance(run_id, str)
        and bool(run_id)
        and run_id not in {".", ".."}
        and len(run_id) <= 128
        and all(
            character.isalnum() or character in {"-", "_", "."} for character in run_id
        )
    )


def is_owned_result(path: Path) -> bool:
    """Authenticate ownership without trusting a filename alone."""

    path = Path(path)
    try:
        if not path.is_dir() or path.is_symlink():
            return False
        manifest = _read_json_object(path / "run_manifest.json")
        success = _read_json_object(path / "_SUCCESS")
        if (
            manifest.get("generator_id") != GENERATOR_ID
            or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("status") != "success"
            or not isinstance(manifest.get("run_id"), str)
        ):
            return False
        return (
            success.get("generator_id") == GENERATOR_ID
            and success.get("run_id") == manifest["run_id"]
            and success.get("manifest_sha256") == _sha256(path / "run_manifest.json")
        )
    except (OSError, ValueError, KeyError):
        return False


def _raw_inventory(root: Path) -> tuple[tuple[str, int, str], ...]:
    _assert_plain_tree(root)
    return tuple(
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )


class FileOps:
    """Injectable filesystem boundary used to verify rollback behavior."""

    def checkpoint(self, name: str) -> None:
        del name

    def copy_tree(self, source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)

    def move_tree(self, source: Path, destination: Path) -> None:
        try:
            os.replace(source, destination)
            return
        except OSError:
            if destination.exists():
                raise
        shutil.copytree(source, destination)
        if _raw_inventory(source) != _raw_inventory(destination):
            shutil.rmtree(destination)
            raise OSError("cross-device directory copy verification failed")
        shutil.rmtree(source)

    def remove_tree(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)


def _write_success(path: Path, run_id: str, manifest_sha256: str) -> None:
    value = {
        "generator_id": GENERATOR_ID,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / "_SUCCESS").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def publish_result(
    bundle: Path,
    input_folder: Path,
    *,
    run_id: str,
    replace_owned_result: bool = False,
    file_ops: FileOps | None = None,
) -> PublishReceipt:
    """Publish through a verified sibling stage with authenticated rollback."""

    if not _valid_run_id(run_id):
        raise ValueError("run_id is not safe for a result path")
    bundle = Path(bundle)
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError("input folder must be an existing directory")
    artifacts = validate_bundle(bundle, run_id=run_id)
    if (bundle / "_SUCCESS").exists():
        raise ValueError("local bundle must not contain a pre-publish _SUCCESS marker")
    manifest_sha256 = _sha256(bundle / "run_manifest.json")
    final = derive_result_path(input_folder)
    staged = final.with_name(f"{final.name}.__tmp__{run_id}")
    backup = final.with_name(f"{final.name}.__backup__{run_id}")
    ops = file_ops or FileOps()

    replaced_previous = final.exists()
    if replaced_previous:
        if not replace_owned_result:
            raise FileExistsError(f"result folder already exists: {final}")
        if not is_owned_result(final):
            raise PublishOwnershipError(
                f"refusing to replace an unowned result folder: {final}"
            )
        try:
            _assert_plain_tree(final)
        except ValueError as exc:
            raise PublishOwnershipError(
                f"refusing to replace an unsafe result folder: {final}"
            ) from exc
    if staged.exists() or backup.exists():
        raise PublishError("publish staging or backup path already exists")

    backup_created = False
    current_point = "stage_copy"
    try:
        ops.copy_tree(bundle, staged)
        current_point = "stage_verify"
        ops.checkpoint(current_point)
        copied_artifacts = validate_bundle(staged, run_id=run_id)
        if copied_artifacts != artifacts:
            raise ValueError("staged artifact inventory changed during copy")
        _write_success(staged, run_id, manifest_sha256)
        if not is_owned_result(staged):
            raise ValueError("staged success marker could not be verified")

        if replaced_previous:
            current_point = "backup_move"
            ops.checkpoint(current_point)
            ops.move_tree(final, backup)
            backup_created = True
            if not is_owned_result(backup):
                raise ValueError("backup ownership verification failed")

        current_point = "final_move"
        ops.checkpoint(current_point)
        ops.move_tree(staged, final)
        current_point = "final_verify"
        ops.checkpoint(current_point)
        final_artifacts = validate_bundle(final, run_id=run_id)
        if final_artifacts != artifacts or not is_owned_result(final):
            raise ValueError("published result verification failed")

        if backup_created:
            try:
                ops.remove_tree(backup)
            except OSError:
                # The new result is already fully verified. Preserve it rather
                # than risking data loss while cleaning an obsolete backup.
                pass
        return PublishReceipt(
            run_id=run_id,
            final_path=final,
            artifacts=artifacts,
            manifest_sha256=manifest_sha256,
            replaced_previous=replaced_previous,
        )
    except Exception as exc:
        try:
            if final.exists() and is_owned_result(final):
                marker = _read_json_object(final / "_SUCCESS")
                if marker.get("run_id") == run_id:
                    ops.remove_tree(final)
            if backup.exists():
                if not is_owned_result(backup):
                    raise PublishError("rollback backup ownership verification failed")
                if final.exists() and is_owned_result(final):
                    # A failed cross-device backup can leave two good copies.
                    # In that case the original final remains authoritative.
                    ops.remove_tree(backup)
                else:
                    if final.exists():
                        ops.remove_tree(final)
                    ops.move_tree(backup, final)
            if staged.exists():
                ops.remove_tree(staged)
        except Exception as restore_exc:
            raise PublishError(
                f"publish failed at {current_point}; rollback also failed: {restore_exc}"
            ) from exc
        if isinstance(exc, PublishError):
            raise
        raise PublishError(f"publish failed at {current_point}: {exc}") from exc


def publish_diagnostics(
    input_folder: Path,
    run_id: str,
    files: Mapping[str, Path],
) -> Path:
    """Copy non-success diagnostics without touching the canonical result."""

    if not _valid_run_id(run_id):
        raise ValueError("run_id is not safe for a diagnostics path")
    if not Path(input_folder).is_dir():
        raise ValueError("input folder must be an existing directory")
    destination = (
        derive_result_path(Path(input_folder)).with_name(
            f"{Path(input_folder).name}_result_diagnostics"
        )
        / run_id
    )
    if destination.exists():
        raise FileExistsError(f"diagnostics folder already exists: {destination}")

    planned: list[tuple[Path, Path]] = []
    for relative_value, source_value in files.items():
        relative = _safe_relative_path(relative_value)
        if relative.name == "splat.ply" or relative.name == "_SUCCESS":
            raise ValueError(f"diagnostics cannot publish {relative.name}")
        if not (
            relative.parts[0] == "logs"
            or (len(relative.parts) == 1 and relative.name in _DIAGNOSTIC_ROOT_FILES)
        ):
            raise ValueError(
                "diagnostics may contain only logs and approved diagnostic reports"
            )
        source = Path(source_value)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"diagnostic source must be a plain file: {source}")
        planned.append((source, destination.joinpath(*relative.parts)))

    try:
        for source, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return destination
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
