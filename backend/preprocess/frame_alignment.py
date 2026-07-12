"""Join registered cameras to physical frames by immutable image name."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.static_pipeline.contracts import FrameRecord, SelectionManifest


@dataclass(frozen=True)
class RegisteredFrame:
    frame_id: str | None
    image_name: str
    image_path: Path
    depth_path: Path | None
    K: np.ndarray
    w2c: np.ndarray
    timestamp_s: float | None


def _is_physical_regular_file(path: Path, root: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return False
        if current == root:
            break
        current = current.parent
    return path.is_file()


def _index_physical_frames(frames_dir: Path) -> dict[str, Path]:
    if frames_dir.is_symlink() or not frames_dir.is_dir():
        raise FileNotFoundError(f"frame directory is missing or unsafe: {frames_dir}")

    indexed: dict[str, Path] = {}
    for path in sorted(frames_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not _is_physical_regular_file(path, frames_dir):
            continue
        previous = indexed.get(path.name)
        if previous is not None:
            raise ValueError(
                f"duplicate physical frame basename {path.name!r}: "
                f"{previous} and {path}"
            )
        indexed[path.name] = path
    return indexed


def _validated_matrix(
    camera: Mapping[str, object],
    field: str,
    shape: tuple[int, int],
    image_name: str,
) -> np.ndarray:
    if field not in camera:
        raise ValueError(f"camera {image_name!r} is missing {field}")
    raw = np.asarray(camera[field])
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"camera {image_name!r} has malformed {field}; expected {shape}"
        )
    matrix = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError(f"camera {image_name!r} has non-finite {field}")
    return matrix


def _natural_key(value: str) -> tuple[tuple[tuple[int, str | int], ...], str]:
    pieces = tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    )
    return pieces, value


def _selected_manifest_records(
    manifest: SelectionManifest,
) -> tuple[tuple[FrameRecord, ...], dict[str, FrameRecord]]:
    ordered = manifest.selected_frames
    indexed: dict[str, FrameRecord] = {}
    for record in ordered:
        name = record.output_name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("selected manifest contains an empty output_name")
        if name in indexed:
            raise ValueError(f"duplicate selected manifest output_name {name!r}")
        indexed[name] = record
    return ordered, indexed


def join_registered_frames(
    frames_dir: str | Path,
    cameras: Mapping[str, Mapping[str, object]],
    *,
    depth_dir: str | Path | None = None,
    manifest: SelectionManifest | None = None,
) -> tuple[RegisteredFrame, ...]:
    """Return registered frames aligned by exact COLMAP image name."""

    if not cameras:
        raise ValueError("camera map must not be empty")

    camera_names: list[str] = []
    for name in cameras:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("camera name must be a non-empty string")
        camera_names.append(name)

    root = Path(frames_dir)
    physical_frames = _index_physical_frames(root)
    missing_physical = [name for name in camera_names if name not in physical_frames]
    if missing_physical:
        missing = ", ".join(repr(name) for name in sorted(missing_physical))
        raise FileNotFoundError(f"registered camera frame is missing: {missing}")

    manifest_records: dict[str, FrameRecord] = {}
    if manifest is None:
        ordered_names = sorted(camera_names, key=_natural_key)
    else:
        selected, manifest_records = _selected_manifest_records(manifest)
        missing_manifest = [
            name for name in camera_names if name not in manifest_records
        ]
        if missing_manifest:
            missing = ", ".join(repr(name) for name in sorted(missing_manifest))
            raise ValueError(f"registered cameras are absent from manifest: {missing}")
        camera_name_set = set(camera_names)
        ordered_names = [
            record.output_name
            for record in selected
            if record.output_name in camera_name_set
        ]

    depth_root = Path(depth_dir) if depth_dir is not None else None
    joined: list[RegisteredFrame] = []
    for name in ordered_names:
        camera = cameras[name]
        if not isinstance(camera, Mapping):
            raise ValueError(f"camera {name!r} must be a mapping")
        image_path = physical_frames[name]
        depth_path = None
        if depth_root is not None and not depth_root.is_symlink():
            candidate = depth_root / f"{image_path.stem}_depth.npy"
            if _is_physical_regular_file(candidate, depth_root):
                depth_path = candidate

        record = manifest_records.get(name)
        joined.append(
            RegisteredFrame(
                frame_id=record.frame_id if record is not None else None,
                image_name=name,
                image_path=image_path,
                depth_path=depth_path,
                K=_validated_matrix(camera, "K", (3, 3), name),
                w2c=_validated_matrix(camera, "w2c", (4, 4), name),
                timestamp_s=record.timestamp_s if record is not None else None,
            )
        )
    return tuple(joined)
