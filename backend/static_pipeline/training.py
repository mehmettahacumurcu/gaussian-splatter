from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import ReconstructionBundle, SelectionManifest

if TYPE_CHECKING:
    from backend.notebooks.models import StaticNotebookRunSpec
    from backend.notebooks.training_config import ResolvedStaticConfig


_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class PreparedTrainingInput:
    run_id: str
    data_root: Path
    scene_name: str
    frames_dir: Path
    reconstruction: ReconstructionBundle
    source_digest: str
    selection_digest: str


@dataclass(frozen=True)
class TrainingResult:
    raw_ply_path: Path
    status: Mapping[str, object]
    resolved_config: ResolvedStaticConfig
    run_manifest_path: Path


@dataclass(frozen=True)
class _ValidatedInputs:
    data_root: Path
    scene_dir: Path
    manifest: SelectionManifest
    frame_sources: tuple[tuple[Path, str, str], ...]
    model_sources: tuple[tuple[Path, str, str], ...]


PipelineRunner = Callable[..., Mapping[str, object]]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_digest(manifest: SelectionManifest) -> str:
    payload = [
        [frame.frame_id, frame.output_name, frame.sha256]
        for frame in manifest.frames
        if frame.selected
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_safe_component(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a safe single component")
    reserved_stem = value.split(".", 1)[0].upper()
    if (
        not value
        or value in {".", ".."}
        or value != value.strip()
        or value.endswith(".")
        or any(character in _WINDOWS_FORBIDDEN for character in value)
        or any(ord(character) < 32 for character in value)
        or reserved_stem in _WINDOWS_RESERVED
        or Path(value).name != value
    ):
        raise ValueError(f"{label} must be a safe single component")
    return value


def _require_regular_directory(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is not a regular directory: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a regular directory or is a symlink: {path}")


def _require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is not a regular file: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file or is a symlink: {path}")


def _validate_manifest(
    prepared: PreparedTrainingInput,
) -> tuple[SelectionManifest, tuple[tuple[Path, str, str], ...]]:
    reconstruction = prepared.reconstruction
    decision = reconstruction.decision
    if not decision.passed or decision.failures:
        raise ValueError("training requires a passing reconstruction decision")

    manifest = reconstruction.selected_manifest
    if manifest.schema_version != 1:
        raise ValueError("selection manifest must use schema version 1")
    if manifest.policy.mode not in {"smart", "fixed_fps"}:
        raise ValueError("training requires a Smart/Fixed selected manifest")
    if manifest.effective_mode not in {"smart", "fixed_fps", "photo_set_all"}:
        raise ValueError("training requires a Smart/Fixed selected manifest")
    if (manifest.effective_mode == "smart" and manifest.policy.mode != "smart") or (
        manifest.effective_mode == "fixed_fps" and manifest.policy.mode != "fixed_fps"
    ):
        raise ValueError("selection policy and effective mode are inconsistent")

    if not _is_sha256(prepared.source_digest) or not _is_sha256(
        prepared.selection_digest
    ):
        raise ValueError("prepared source/selection digest is invalid")
    if prepared.source_digest != manifest.source_digest:
        raise ValueError("prepared source digest does not match the selection")
    if prepared.selection_digest != manifest.image_set_digest:
        raise ValueError("prepared selection digest does not match the selection")
    if _selection_digest(manifest) != manifest.image_set_digest:
        raise ValueError("selection image-set digest does not match its records")

    frames_dir = Path(prepared.frames_dir)
    _require_regular_directory(frames_dir, "selected frames directory")
    selected = manifest.selected_frames
    if not selected:
        raise ValueError("selection manifest has no selected frames")

    frame_sources: list[tuple[Path, str, str]] = []
    for index, frame in enumerate(selected):
        expected_name = f"frame_{index:06d}.png"
        if frame.output_name != expected_name:
            raise ValueError(
                "selected frame output names must be canonical and contiguous"
            )
        if not _is_sha256(frame.sha256):
            raise ValueError(f"selected frame hash is invalid: {frame.output_name}")
        source = frames_dir / frame.output_name
        _require_regular_file(source, f"selected frame {frame.output_name}")
        if _sha256_file(source) != frame.sha256:
            raise ValueError(f"selected frame hash mismatch: {frame.output_name}")
        frame_sources.append((source, frame.output_name, frame.sha256))
    return manifest, tuple(frame_sources)


def _validate_inputs(prepared: PreparedTrainingInput) -> _ValidatedInputs:
    run_id = _require_safe_component(prepared.run_id, "run_id")
    scene_name = _require_safe_component(prepared.scene_name, "scene_name")
    if run_id != prepared.run_id or scene_name != prepared.scene_name:
        raise AssertionError("validated component changed unexpectedly")

    manifest, frame_sources = _validate_manifest(prepared)
    model_dir = Path(prepared.reconstruction.accepted_model_dir)
    _require_regular_directory(model_dir, "accepted COLMAP model")
    model_sources: list[tuple[Path, str, str]] = []
    for name in _MODEL_FILES:
        source = model_dir / name
        _require_regular_file(source, f"accepted COLMAP model {name}")
        model_sources.append((source, name, _sha256_file(source)))

    from .reconstruction import evaluate_reconstruction, measure_models

    try:
        current_decision = evaluate_reconstruction(
            measure_models((model_dir,), manifest),
            manifest,
            attempt_index=1,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "accepted COLMAP model no longer passes the reconstruction gate"
        ) from exc
    if not current_decision.passed:
        failures = ", ".join(current_decision.failures) or "unknown"
        raise ValueError(
            "accepted COLMAP model no longer passes the reconstruction gate: "
            f"{failures}"
        )

    raw_data_root = Path(prepared.data_root)
    if os.path.lexists(raw_data_root):
        _require_regular_directory(raw_data_root, "data root")
    data_root = raw_data_root.resolve(strict=False)
    scene_dir = data_root / scene_name
    if os.path.lexists(scene_dir):
        raise FileExistsError(f"training requires a fresh scene: {scene_dir}")

    return _ValidatedInputs(
        data_root=data_root,
        scene_dir=scene_dir,
        manifest=manifest,
        frame_sources=frame_sources,
        model_sources=tuple(model_sources),
    )


def _copy_verified(source: Path, destination: Path, expected_digest: str) -> None:
    _require_regular_file(source, f"copy source {source.name}")
    if _sha256_file(source) != expected_digest:
        raise ValueError(f"source hash changed before copy: {source.name}")
    shutil.copyfile(source, destination, follow_symlinks=False)
    _require_regular_file(destination, f"copied file {destination.name}")
    if _sha256_file(destination) != expected_digest:
        raise ValueError(f"copied file hash mismatch: {destination.name}")


def _accepted_native_image_size(validated: _ValidatedInputs) -> tuple[int, int]:
    from backend.preprocess.parse_colmap import parse_cameras_from_model

    model_dir = validated.model_sources[0][0].parent
    cameras = parse_cameras_from_model(model_dir)
    first_camera = cameras[sorted(cameras)[0]]
    return int(first_camera["width"]), int(first_camera["height"])


def _profile_value(profile: object) -> str:
    value = getattr(profile, "value", profile)
    if not isinstance(value, str) or not value:
        raise ValueError("resolved profile has an invalid identifier")
    return value


def _write_run_manifest(
    path: Path,
    prepared: PreparedTrainingInput,
    manifest: SelectionManifest,
    resolved: ResolvedStaticConfig,
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": prepared.run_id,
        "scene_name": prepared.scene_name,
        "selection_mode": manifest.effective_mode,
        "source_digest": prepared.source_digest,
        "image_set_digest": prepared.selection_digest,
        "profile": _profile_value(resolved.profile),
        "legacy_preset": resolved.legacy_preset,
        "foundation": resolved.foundation,
        "run_eval": resolved.run_eval,
        "config_digest": resolved.config_digest,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def run_validated_training(
    prepared: PreparedTrainingInput,
    spec: StaticNotebookRunSpec,
    *,
    source_long_edge: int | None = None,
    pipeline_runner: PipelineRunner | None = None,
) -> TrainingResult:
    validated = _validate_inputs(prepared)
    native_image_size = _accepted_native_image_size(validated)

    os.environ["FOURDGS_DATA_ROOT"] = str(validated.data_root)
    from backend.notebooks.training_config import resolve_static_training_config

    cfg, resolved = resolve_static_training_config(
        spec,
        source_long_edge=source_long_edge,
        native_image_size=native_image_size,
    )

    import backend.config as backend_config

    backend_config.DATA_ROOT = validated.data_root
    runner = pipeline_runner
    if runner is None:
        from backend.pipeline import run_pipeline

        runner = run_pipeline

    validated.data_root.mkdir(parents=True, exist_ok=True)
    validated.scene_dir.mkdir(exist_ok=False)
    frames_dir = validated.scene_dir / "frames"
    model_dir = validated.scene_dir / "colmap" / "sparse" / "0"
    frames_dir.mkdir()
    model_dir.mkdir(parents=True)

    for source, name, digest in validated.frame_sources:
        _copy_verified(source, frames_dir / name, digest)
    for source, name, digest in validated.model_sources:
        _copy_verified(source, model_dir / name, digest)

    from backend.preprocess.cache_utils import write_cache_marker

    selected_count = len(validated.frame_sources)
    write_cache_marker(
        validated.scene_dir,
        "frames",
        {
            "n_frames": selected_count,
            "source": "validated_selection",
            "source_digest": prepared.source_digest,
            "image_set_digest": prepared.selection_digest,
        },
        cfg=cfg,
    )
    write_cache_marker(
        validated.scene_dir,
        "colmap",
        {
            "n_cameras": prepared.reconstruction.decision.dominant.registered_count,
            "n_points": prepared.reconstruction.decision.dominant.sparse_point_count,
            "source": "validated_reconstruction",
            "image_set_digest": prepared.selection_digest,
        },
        cfg=cfg,
    )

    run_manifest_path = validated.scene_dir / "run_manifest.json"
    _write_run_manifest(
        run_manifest_path,
        prepared,
        validated.manifest,
        resolved,
    )

    raw_ply_path = validated.scene_dir / "output" / "ply" / "frame_0000.ply"
    if os.path.lexists(raw_ply_path):
        raise FileExistsError(
            f"training output must be absent before run: {raw_ply_path}"
        )

    status = runner(
        video_path=validated.scene_dir / "video.mp4",
        scene_name=prepared.scene_name,
        cfg=cfg,
        force_preprocess=False,
        skip_training=False,
        skip_export=False,
        skip_foundation=not resolved.foundation,
    )
    try:
        output_metadata = os.lstat(raw_ply_path)
    except OSError as exc:
        raise FileNotFoundError(
            f"training did not create a new frame_0000.ply: {raw_ply_path}"
        ) from exc
    if not stat.S_ISREG(output_metadata.st_mode):
        raise ValueError(
            f"training output is not a regular non-symlink file: {raw_ply_path}"
        )
    if not isinstance(status, Mapping):
        raise TypeError("pipeline runner must return a status mapping")

    return TrainingResult(
        raw_ply_path=raw_ply_path,
        status=dict(status),
        resolved_config=resolved,
        run_manifest_path=run_manifest_path,
    )
