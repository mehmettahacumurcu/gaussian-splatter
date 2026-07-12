from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sources import _atomic_promote_no_replace


def stage_fingerprint(
    stage: str,
    inputs: Mapping[str, Any],
    settings: Any,
    policy_version: str,
    tools: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "stage": stage,
        "inputs": dict(sorted(inputs.items())),
        "settings": settings,
        "policy_version": policy_version,
        "tools": dict(sorted((tools or {}).items())),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_matches(
    marker_path: str | Path,
    expected_fingerprint: str,
    required_paths: Sequence[str | Path],
) -> bool:
    marker = Path(marker_path)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return False
    if payload.get("fingerprint") != expected_fingerprint:
        return False
    return all(Path(path).exists() for path in required_paths)


def promote_directory(staging: str | Path, target: str | Path) -> None:
    staging_path = Path(staging)
    target_path = Path(target)
    if not os.path.lexists(staging_path):
        raise FileNotFoundError(f"staging directory does not exist: {staging_path}")
    if staging_path.is_symlink() or not staging_path.is_dir():
        raise NotADirectoryError(f"staging path is not a directory: {staging_path}")
    if os.path.lexists(target_path):
        raise FileExistsError(f"promotion target already exists: {target_path}")
    if staging_path.parent.resolve() != target_path.parent.resolve():
        raise ValueError("staging and target must have the same parent")
    _atomic_promote_no_replace(staging_path, target_path)
