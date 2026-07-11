from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .contracts import ColmapAttempt, ColmapPolicy
from .stage_cache import stage_fingerprint

LOGGER = logging.getLogger(__name__)
ColmapProgressCallback = Callable[[float, str], None]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FRAME_ID_PATTERN = re.compile(r"[0-9a-f]{24}")
_CONVERTED_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def choose_colmap_policy(
    *,
    use_gpu: bool,
    selected_count: int,
    attempt_index: int,
) -> ColmapPolicy:
    if selected_count < 1 or selected_count > 800:
        raise ValueError("COLMAP policy accepts 1..800 selected frames")
    if attempt_index not in {0, 1}:
        raise ValueError("Only the initial attempt and one backfill retry are allowed")
    if use_gpu:
        return ColmapPolicy("PINHOLE", "exhaustive", 0, True)
    if attempt_index == 0:
        return ColmapPolicy("PINHOLE", "sequential", 20, False)
    if selected_count <= 300:
        return ColmapPolicy("PINHOLE", "exhaustive", 0, False)
    return ColmapPolicy("PINHOLE", "sequential", 40, False)


def _resolve_colmap_executable(colmap_exe: str | Path | None) -> str:
    if colmap_exe is not None:
        return str(colmap_exe)
    resolved = shutil.which("colmap")
    if resolved is None:
        raise RuntimeError("COLMAP executable was not found")
    return resolved


def _write_checkerboards(images_dir: Path) -> None:
    images_dir.mkdir(parents=True)
    for image_index, offset in enumerate((0, 1)):
        image = Image.new("RGB", (32, 32))
        pixels = image.load()
        for y in range(32):
            for x in range(32):
                value = 255 if ((x // 4) + (y // 4) + offset) % 2 else 0
                pixels[x, y] = (value, value, value)
        image.save(images_dir / f"checkerboard_{image_index}.png")


def probe_colmap_gpu_support(
    colmap_exe: str | Path,
    probe_root: str | Path,
) -> bool:
    probe = Path(probe_root)
    if os.path.lexists(probe):
        raise FileExistsError(probe)
    images = probe / "images"
    database = probe / "probe.db"
    created = False
    succeeded = False
    try:
        probe.mkdir(parents=True, exist_ok=False)
        created = True
        _write_checkerboards(images)
        command = (
            str(colmap_exe),
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            "1",
        )
        subprocess.run(command, check=True)
        if not database.is_file():
            raise RuntimeError("COLMAP GPU probe completed without a database")
        succeeded = True
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        LOGGER.warning("COLMAP GPU SIFT probe failed: %s", exc)
    finally:
        if created:
            shutil.rmtree(probe, ignore_errors=True)
    return succeeded


def build_colmap_commands(
    frames_dir: str | Path,
    attempt_dir: str | Path,
    colmap_exe: str | Path,
    policy: ColmapPolicy,
) -> tuple[tuple[str, ...], ...]:
    frames = Path(frames_dir)
    attempt = Path(attempt_dir)
    database = attempt / "colmap.db"
    sparse = attempt / "sparse"

    feature = (
        str(colmap_exe),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(frames),
        "--ImageReader.camera_model",
        policy.camera_model,
        "--ImageReader.single_camera",
        "1",
    )
    matcher = (
        str(colmap_exe),
        f"{policy.matcher}_matcher",
        "--database_path",
        str(database),
    )
    mapper = (
        str(colmap_exe),
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(frames),
        "--output_path",
        str(sparse),
    )
    if not policy.use_gpu:
        feature += ("--SiftExtraction.use_gpu", "0")
        matcher += ("--SiftMatching.use_gpu", "0")
    if policy.matcher == "sequential":
        matcher += (
            "--SequentialMatching.overlap",
            str(policy.sequential_overlap),
        )
    return feature, matcher, mapper


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _safe_output_name(value: object) -> str | None:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        return None
    if "/" in value or "\\" in value or Path(value).name != value:
        return None
    if not value.endswith(".png"):
        return None
    return value


def _read_selection_digest(frames_dir: Path) -> str:
    manifest_path = frames_dir / "selection_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("selection manifest cannot be a symlink")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection manifest: {manifest_path}")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("selection manifest must use schema version 1")
    if not _is_sha256(payload.get("source_digest")):
        raise ValueError("selection manifest has an invalid source digest")
    if payload.get("effective_mode") not in {"smart", "fixed_fps", "photo_set_all"}:
        raise ValueError("selection manifest has an invalid effective mode")
    if not isinstance(payload.get("policy"), dict):
        raise ValueError("selection manifest is missing its policy")
    records = payload.get("frames")
    if not isinstance(records, list):
        raise ValueError("selection manifest is missing frame records")

    digest = payload.get("image_set_digest")
    if not _is_sha256(digest):
        raise ValueError("selection manifest has an invalid image_set_digest")

    frame_ids: set[str] = set()
    output_names: set[str] = set()
    selected_payload: list[list[str]] = []
    selected_files: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("selection manifest contains an invalid frame record")
        frame_id = record.get("frame_id")
        if (
            not isinstance(frame_id, str)
            or _FRAME_ID_PATTERN.fullmatch(frame_id) is None
            or frame_id in frame_ids
        ):
            raise ValueError("selection manifest contains an invalid frame ID")
        frame_ids.add(frame_id)
        selected = record.get("selected")
        if type(selected) is not bool:
            raise ValueError("selection manifest contains an invalid selected flag")
        output_name = record.get("output_name")
        recorded_sha256 = record.get("sha256")
        if not selected:
            if output_name != "" or recorded_sha256 != "":
                raise ValueError(
                    "unselected frame records cannot reference output bytes"
                )
            continue
        safe_name = _safe_output_name(output_name)
        if safe_name is None or safe_name in output_names:
            raise ValueError(
                "selection manifest contains an unsafe or duplicate output"
            )
        if not _is_sha256(recorded_sha256):
            raise ValueError("selection manifest contains an invalid frame hash")
        output_names.add(safe_name)
        output_path = frames_dir / safe_name
        if output_path.is_symlink() or not output_path.is_file():
            raise ValueError(f"selected frame is missing or unsafe: {safe_name}")
        actual_sha256 = _sha256_file(output_path)
        if actual_sha256 != recorded_sha256:
            raise ValueError(f"selected frame hash mismatch: {safe_name}")
        selected_payload.append([frame_id, safe_name, actual_sha256])
        selected_files[safe_name] = output_path

    if not 1 <= len(selected_payload) <= 800:
        raise ValueError("selection manifest must contain 1..800 selected frames")
    discovered_names: set[str] = set()
    for path in frames_dir.rglob("*"):
        if path.suffix.casefold() != ".png":
            continue
        relative = path.relative_to(frames_dir)
        if len(relative.parts) != 1 or path.is_symlink() or not path.is_file():
            raise ValueError(
                f"selected frame directory contains an unsafe PNG: {relative}"
            )
        discovered_names.add(path.name)
    if discovered_names != set(selected_files):
        raise ValueError("selected frame directory does not match its manifest")

    recomputed = hashlib.sha256(
        json.dumps(
            selected_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if recomputed != digest:
        raise ValueError("selection manifest image_set_digest mismatch")
    return digest


def run_colmap_attempt(
    frames_dir: str | Path,
    attempt_dir: str | Path,
    policy: ColmapPolicy,
    colmap_exe: str | Path | None = None,
    on_progress: ColmapProgressCallback | None = None,
) -> ColmapAttempt:
    """Run one fresh attempt; progress is monotonic and ends at 1.0 on success."""
    attempt = Path(attempt_dir)
    if os.path.lexists(attempt):
        raise FileExistsError(attempt)

    frames = Path(frames_dir)
    selection_digest = _read_selection_digest(frames)
    executable = _resolve_colmap_executable(colmap_exe)
    version_result = subprocess.run(
        (executable, "--version"),
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_version = (version_result.stdout or "").strip()
    stderr_version = (version_result.stderr or "").strip()
    colmap_version = stdout_version or stderr_version
    if not colmap_version:
        raise RuntimeError("COLMAP did not report a version")

    attempt.mkdir(parents=True, exist_ok=False)
    sparse = attempt / "sparse"
    sparse.mkdir()
    commands = build_colmap_commands(frames, attempt, executable, policy)
    for command_index, command in enumerate(commands):
        if on_progress is not None:
            on_progress(command_index / len(commands), command[1])
        subprocess.run(command, check=True)

    database_path = attempt / "colmap.db"
    if database_path.is_symlink() or not database_path.is_file():
        raise RuntimeError("COLMAP produced no database")
    model_dirs = tuple(
        sorted(
            path for path in sparse.iterdir() if path.is_dir() and not path.is_symlink()
        )
    )
    if not model_dirs:
        raise RuntimeError("COLMAP produced no sparse model")
    for model_index, model_dir in enumerate(model_dirs):
        subprocess.run(
            (
                executable,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "TXT",
            ),
            check=True,
        )
        missing_outputs = [
            name for name in _CONVERTED_MODEL_FILES if not (model_dir / name).is_file()
        ]
        if missing_outputs:
            raise RuntimeError(
                f"COLMAP model conversion is incomplete for {model_dir}: "
                f"{', '.join(missing_outputs)}"
            )
        if on_progress is not None:
            on_progress(
                (len(commands) + model_index + 1) / (len(commands) + len(model_dirs)),
                f"model_converter:{model_dir.name}",
            )

    fingerprint = stage_fingerprint(
        "colmap",
        inputs={"selection": selection_digest},
        settings=asdict(policy),
        policy_version=policy.version,
        tools={"colmap": colmap_version},
    )
    return ColmapAttempt(
        root=attempt,
        database_path=database_path,
        model_dirs=model_dirs,
        colmap_version=colmap_version,
        fingerprint=fingerprint,
    )
