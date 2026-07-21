from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class StagedAblationInputs:
    root: Path
    selection: Any
    reconstruction: Any
    source_digest: str
    pretraining_fingerprint: str
    source_revision: str
    manifest_path: Path


@dataclass(frozen=True)
class ExperimentWorkspace:
    experiment_id: str
    root: Path
    inputs_root: Path
    scene_root: Path
    output_root: Path


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    base = root.resolve(strict=False)
    return resolved == base or resolved.is_relative_to(base)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing parent for {path}")
        candidate = parent
    return candidate


def _object_paths(
    value: object,
    *,
    skip_fields: frozenset[str] = frozenset(),
    seen: set[int] | None = None,
) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return ()
    active_seen = seen if seen is not None else set()
    identity = id(value)
    if identity in active_seen:
        return ()
    active_seen.add(identity)
    if isinstance(value, Mapping):
        return tuple(
            path
            for item in value.values()
            for path in _object_paths(item, skip_fields=skip_fields, seen=active_seen)
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(
            path
            for item in value
            for path in _object_paths(item, skip_fields=skip_fields, seen=active_seen)
        )
    if is_dataclass(value) and not isinstance(value, type):
        items = (
            getattr(value, field.name)
            for field in fields(value)
            if field.name not in skip_fields
        )
    elif hasattr(value, "__dict__"):
        items = (
            item
            for name, item in vars(value).items()
            if name not in skip_fields
        )
    else:
        return ()
    return tuple(
        path
        for item in items
        for path in _object_paths(item, skip_fields=skip_fields, seen=active_seen)
    )


def _validate_payload_paths(
    selection: object,
    reconstruction: object,
    *,
    staged_root: Path,
    drive_root: Path,
) -> tuple[Path, ...]:
    paths = (
        *_object_paths(selection, skip_fields=frozenset({"inventory"})),
        *_object_paths(reconstruction),
    )
    unique = tuple(sorted({path.resolve(strict=True) for path in paths}, key=str))
    if not unique:
        raise ValueError("restored pretraining payload contains no local artifacts")
    for path in unique:
        if _is_within(path, drive_root):
            raise ValueError(f"Drive-backed training payload is forbidden: {path}")
        if not _is_within(path, staged_root):
            raise ValueError(f"staged training payload escapes the local root: {path}")
    return unique


def freeze_staged_tree(root: Path) -> None:
    base = Path(root).resolve(strict=True)
    for path in sorted(base.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    mode = stat.S_IMODE(base.stat().st_mode)
    base.chmod(mode & ~0o222)


def stage_ablation_inputs(
    *,
    store: object,
    source_inventory: object,
    destination: Path,
    drive_root: Path,
    expected_fingerprint: Callable[[object], str],
    expected_source_revision: str,
    actual_source_revision: str,
    audit_validator: Callable[[object], None],
    minimum_free_bytes: int,
    run_id: str,
    available_free_bytes: int | None = None,
    freeze: bool = True,
    restored_validator: Callable[[object, object], None] | None = None,
) -> StagedAblationInputs:
    if actual_source_revision != expected_source_revision:
        raise RuntimeError(
            "source revision differs from the notebook pin: "
            f"{actual_source_revision} != {expected_source_revision}"
        )
    target = Path(destination).resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"staging destination already exists: {target}")
    free_bytes = available_free_bytes
    if free_bytes is None:
        free_bytes = shutil.disk_usage(_nearest_existing_parent(target)).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            "insufficient free local disk for the ablation snapshot: "
            f"{free_bytes} < {minimum_free_bytes}"
        )
    drive = Path(drive_root).resolve(strict=False)
    if _is_within(target, drive):
        raise ValueError("ablation staging destination cannot be on Drive")

    audit_validator(source_inventory)
    probe = getattr(store, "probe_drive_publication", None)
    if probe is not None:
        probe(run_id=run_id)
    try:
        restored = store.restore_pretraining(
            source_inventory=source_inventory,
            destination=target,
            expected_fingerprint=expected_fingerprint,
        )
        if restored is None:
            raise RuntimeError("verified final pretraining checkpoint is missing")
        selection = restored.selection
        reconstruction = restored.reconstruction
        if restored_validator is not None:
            restored_validator(selection, reconstruction)
        source_digest = getattr(source_inventory, "digest", None)
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError("source inventory digest is invalid")
        fingerprint = expected_fingerprint(selection)
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("pretraining fingerprint is invalid")
        local_paths = _validate_payload_paths(
            selection,
            reconstruction,
            staged_root=target,
            drive_root=drive,
        )
        manifest_path = target / "ablation_staging_manifest.json"
        manifest = {
            "schema_version": 1,
            "source_revision": actual_source_revision,
            "source_digest": source_digest,
            "pretraining_fingerprint": fingerprint,
            "drive_reads_permitted_after_staging": False,
            "verified_local_paths": [
                path.relative_to(target).as_posix() for path in local_paths
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        staged = StagedAblationInputs(
            root=target,
            selection=selection,
            reconstruction=reconstruction,
            source_digest=source_digest,
            pretraining_fingerprint=fingerprint,
            source_revision=actual_source_revision,
            manifest_path=manifest_path,
        )
        if freeze:
            freeze_staged_tree(target)
        return staged
    except BaseException:
        if os.path.lexists(target):
            shutil.rmtree(target, ignore_errors=True)
        raise


def materialize_experiment_workspace(
    staged: StagedAblationInputs,
    *,
    experiments_root: Path,
    experiment_id: str,
) -> ExperimentWorkspace:
    if (
        not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id)
        or experiment_id in {".", ".."}
        or ".." in experiment_id
    ):
        raise ValueError("experiment_id must be one safe path component")
    inputs_root = staged.root.resolve(strict=True)
    root = Path(experiments_root).resolve(strict=False) / experiment_id
    root.mkdir(parents=True, exist_ok=False)
    scene_root = root / "scene"
    output_root = root / "output"
    scene_root.mkdir()
    output_root.mkdir()
    return ExperimentWorkspace(
        experiment_id=experiment_id,
        root=root,
        inputs_root=inputs_root,
        scene_root=scene_root,
        output_root=output_root,
    )
