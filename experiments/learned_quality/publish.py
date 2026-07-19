from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from backend.static_pipeline.contracts import PublishReceipt

from .contracts import (
    GENERATOR_ID,
    derive_learned_diagnostics_root,
    derive_learned_result_path,
)
from .reports import (
    MANIFEST_SCHEMA_VERSION,
    _json_object,
    _plain_tree,
    _safe_relative,
    _sha256,
    validate_learned_bundle,
)


class LearnedPublishError(RuntimeError):
    pass


class LearnedPublishOwnershipError(LearnedPublishError):
    pass


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


def _raw_inventory(root: Path) -> tuple[tuple[str, int, str], ...]:
    _plain_tree(root)
    return tuple(
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )


class LearnedFileOps:
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
            raise OSError("cross-device learned-result copy verification failed")
        shutil.rmtree(source)

    def remove_tree(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)


def _write_success(path: Path, run_id: str, manifest_sha256: str) -> None:
    manifest = _json_object(path / "run_manifest.json")
    payload = {
        "generator_id": GENERATOR_ID,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "geometry_acceptance_mode": manifest.get("geometry_acceptance_mode"),
        "geometry_policy_version": manifest.get("geometry_policy_version"),
        "geometry_strict_failures": manifest.get("geometry_strict_failures"),
        "geometry_colmap_fingerprint": manifest.get(
            "geometry_colmap_fingerprint"
        ),
    }
    (path / "_SUCCESS").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def is_owned_learned_result(path: Path) -> bool:
    path = Path(path)
    try:
        if not path.is_dir() or path.is_symlink():
            return False
        manifest = _json_object(path / "run_manifest.json")
        success = _json_object(path / "_SUCCESS")
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
            and success.get("geometry_acceptance_mode")
            == manifest.get("geometry_acceptance_mode")
            and success.get("geometry_policy_version")
            == manifest.get("geometry_policy_version")
            and success.get("geometry_strict_failures")
            == manifest.get("geometry_strict_failures")
            and success.get("geometry_colmap_fingerprint")
            == manifest.get("geometry_colmap_fingerprint")
        )
    except (OSError, ValueError, KeyError):
        return False


def publish_learned_result(
    bundle: Path,
    input_folder: Path,
    *,
    run_id: str,
    replace_owned_result: bool = False,
    file_ops: LearnedFileOps | None = None,
) -> PublishReceipt:
    if not _valid_run_id(run_id):
        raise ValueError("run_id is not safe for a learned result path")
    bundle = Path(bundle)
    input_folder = Path(input_folder)
    if not input_folder.is_dir() or input_folder.is_symlink():
        raise ValueError("input folder must be an existing regular directory")
    artifacts = validate_learned_bundle(bundle, run_id=run_id)
    if (bundle / "_SUCCESS").exists():
        raise ValueError("local learned bundle must not contain _SUCCESS")
    manifest_sha256 = _sha256(bundle / "run_manifest.json")
    final = derive_learned_result_path(input_folder)
    staged = final.with_name(f"{final.name}.__tmp__{run_id}")
    backup = final.with_name(f"{final.name}.__backup__{run_id}")
    ops = file_ops or LearnedFileOps()

    replaced = final.exists()
    if replaced:
        if not replace_owned_result:
            raise FileExistsError(f"learned result already exists: {final}")
        if not is_owned_learned_result(final):
            raise LearnedPublishOwnershipError(
                f"refusing to replace an unowned learned result: {final}"
            )
        try:
            _plain_tree(final)
        except ValueError as exc:
            raise LearnedPublishOwnershipError(
                f"refusing to replace an unsafe learned result: {final}"
            ) from exc
    if staged.exists() or backup.exists():
        raise LearnedPublishError("learned publish staging or backup already exists")

    backup_created = False
    current_point = "stage_copy"
    try:
        ops.copy_tree(bundle, staged)
        current_point = "stage_verify"
        ops.checkpoint(current_point)
        if validate_learned_bundle(staged, run_id=run_id) != artifacts:
            raise ValueError("staged learned inventory changed during copy")
        _write_success(staged, run_id, manifest_sha256)
        if not is_owned_learned_result(staged):
            raise ValueError("staged learned success marker is invalid")

        if replaced:
            current_point = "backup_move"
            ops.checkpoint(current_point)
            ops.move_tree(final, backup)
            backup_created = True
            if not is_owned_learned_result(backup):
                raise ValueError("learned backup ownership verification failed")

        current_point = "final_move"
        ops.checkpoint(current_point)
        ops.move_tree(staged, final)
        current_point = "final_verify"
        ops.checkpoint(current_point)
        if validate_learned_bundle(
            final, run_id=run_id
        ) != artifacts or not is_owned_learned_result(final):
            raise ValueError("published learned result verification failed")
        if backup_created:
            try:
                ops.remove_tree(backup)
            except OSError:
                pass
        return PublishReceipt(
            run_id=run_id,
            final_path=final,
            artifacts=artifacts,
            manifest_sha256=manifest_sha256,
            replaced_previous=replaced,
        )
    except Exception as exc:
        try:
            if final.exists() and is_owned_learned_result(final):
                marker = _json_object(final / "_SUCCESS")
                if marker.get("run_id") == run_id:
                    ops.remove_tree(final)
            if backup.exists():
                if not is_owned_learned_result(backup):
                    raise LearnedPublishError(
                        "learned rollback backup ownership verification failed"
                    )
                if final.exists() and is_owned_learned_result(final):
                    ops.remove_tree(backup)
                else:
                    if final.exists():
                        ops.remove_tree(final)
                    ops.move_tree(backup, final)
            if staged.exists():
                ops.remove_tree(staged)
        except Exception as restore_exc:
            raise LearnedPublishError(
                f"learned publish failed at {current_point}; rollback failed: "
                f"{restore_exc}"
            ) from exc
        if isinstance(exc, LearnedPublishError):
            raise
        raise LearnedPublishError(
            f"learned publish failed at {current_point}: {exc}"
        ) from exc


_DIAGNOSTIC_ROOT_FILES = frozenset(
    {
        "quality_report.json",
        "experiment_report.json",
        "geometry_candidates.json",
        "model_manifest.json",
    }
)
_DIAGNOSTIC_FILES = frozenset(
    {
        "masks_contact_sheet.png",
        "depth_contact_sheet.png",
        "geometry_contact_sheet.png",
        "final_render_contact_sheet.png",
        "density_history.json",
        "photometric_report.json",
    }
)


def publish_learned_diagnostics(
    input_folder: Path,
    run_id: str,
    files: Mapping[str, Path],
) -> Path:
    if not _valid_run_id(run_id):
        raise ValueError("run_id is not safe for learned diagnostics")
    input_folder = Path(input_folder)
    if not input_folder.is_dir() or input_folder.is_symlink():
        raise ValueError("input folder must be an existing regular directory")
    destination = derive_learned_diagnostics_root(input_folder) / run_id
    if destination.exists():
        raise FileExistsError(f"learned diagnostics already exist: {destination}")

    planned: list[tuple[Path, Path]] = []
    for relative_value, source_value in files.items():
        relative = _safe_relative(relative_value)
        if relative.suffix.casefold() == ".ply" or relative.name == "_SUCCESS":
            raise ValueError(f"learned diagnostics cannot publish {relative.name}")
        if relative.suffix.casefold() in {".html", ".htm", ".js", ".mjs", ".wasm"}:
            raise ValueError(
                f"learned diagnostics cannot publish web asset {relative.name}"
            )
        allowed = (
            relative.parts[0] == "logs"
            or (len(relative.parts) == 1 and relative.name in _DIAGNOSTIC_ROOT_FILES)
            or (
                len(relative.parts) == 2
                and relative.parts[0] == "diagnostics"
                and relative.name in _DIAGNOSTIC_FILES
            )
        )
        if not allowed:
            raise ValueError(f"learned diagnostic path is not allowlisted: {relative}")
        source = Path(source_value)
        if not source.is_file() or source.is_symlink():
            raise ValueError(
                f"learned diagnostic source must be a plain file: {source}"
            )
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
