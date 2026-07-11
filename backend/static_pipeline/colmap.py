from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from PIL import Image

from .contracts import ColmapAttempt, ColmapPolicy
from .sources import _atomic_promote_no_replace
from .stage_cache import stage_fingerprint

LOGGER = logging.getLogger(__name__)
ColmapProgressCallback = Callable[[float, str], None]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FRAME_ID_PATTERN = re.compile(r"[0-9a-f]{24}")
_OUTPUT_NAME_PATTERN = re.compile(r"frame_[0-9]{6}\.png")
_CONVERTED_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")
_MANIFEST_KEYS = {
    "schema_version",
    "source_digest",
    "effective_mode",
    "policy",
    "frames",
    "image_set_digest",
    "reconstruction_guardrail",
}
_POLICY_KEYS = {
    "mode",
    "frame_budget",
    "resolution_long_edge_cap",
    "fixed_fps",
    "candidate_fps",
    "candidate_long_edge",
    "version",
}
_FRAME_RECORD_KEYS = {
    "frame_id",
    "source_relative_path",
    "source_index",
    "source_pts",
    "timestamp_s",
    "output_name",
    "sha256",
    "selected",
    "metrics",
    "selection_score",
    "reasons",
}
_METRIC_KEYS = {
    "sharpness",
    "exposure_score",
    "duplicate_similarity",
    "overlap_score",
}
_DirectoryIdentity = tuple[int, int]
_FileSignature = tuple[int, int, int, int, int, str]


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


def _path_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        metadata = os.lstat(path)
    except OSError:
        return None
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _directory_identity(path: Path) -> _DirectoryIdentity:
    identity = _path_identity(path)
    if identity is None or identity[2] != stat.S_IFDIR:
        raise RuntimeError(f"directory identity is invalid: {path}")
    return identity[0], identity[1]


def _require_directory_identity(
    path: Path,
    expected: _DirectoryIdentity,
    label: str,
) -> None:
    current = _path_identity(path)
    if current != (expected[0], expected[1], stat.S_IFDIR):
        raise RuntimeError(f"{label} directory identity changed: {path}")


def _write_checkerboards(images_dir: Path) -> tuple[Path, ...]:
    written: list[Path] = []
    for image_index, offset in enumerate((0, 1)):
        image = Image.new("RGB", (32, 32))
        pixels = image.load()
        for y in range(32):
            for x in range(32):
                value = 255 if ((x // 4) + (y // 4) + offset) % 2 else 0
                pixels[x, y] = (value, value, value)
        output = images_dir / f"checkerboard_{image_index}.png"
        image.save(output)
        written.append(output)
    return tuple(written)


def _restore_quarantined_path(quarantine: Path, original: Path) -> None:
    try:
        _atomic_promote_no_replace(quarantine, original)
    except (OSError, RuntimeError):
        pass


def _claim_owned_path(
    path: Path,
    expected: tuple[int, int, int],
) -> Path | None:
    if _path_identity(path) != expected:
        return None
    quarantine = path.with_name(f".{path.name}.cleanup-{uuid.uuid4().hex}")
    try:
        _atomic_promote_no_replace(path, quarantine)
    except (OSError, RuntimeError):
        return None
    if _path_identity(quarantine) != expected:
        _restore_quarantined_path(quarantine, path)
        return None
    return quarantine


def _unlink_owned_file(path: Path, expected: tuple[int, int, int]) -> None:
    quarantine = _claim_owned_path(path, expected)
    if quarantine is None:
        return
    try:
        if _path_identity(quarantine) == expected:
            quarantine.unlink()
    except OSError:
        pass


def _remove_owned_empty_directory(
    path: Path,
    expected: _DirectoryIdentity,
) -> None:
    expected_path_identity = (expected[0], expected[1], stat.S_IFDIR)
    try:
        if next(path.iterdir(), None) is not None:
            return
    except OSError:
        return
    quarantine = _claim_owned_path(path, expected_path_identity)
    if quarantine is None:
        return
    try:
        quarantine.rmdir()
    except OSError:
        _restore_quarantined_path(quarantine, path)


def _cleanup_owned_probe(
    probe: Path,
    probe_identity: _DirectoryIdentity | None,
    images: Path,
    images_identity: _DirectoryIdentity | None,
    owned_files: dict[Path, tuple[int, int, int]],
) -> None:
    if probe_identity is None:
        return
    try:
        _require_directory_identity(probe, probe_identity, "probe")
    except RuntimeError:
        return
    for path, expected in owned_files.items():
        try:
            _require_directory_identity(probe, probe_identity, "probe")
            if path.parent == images and images_identity is not None:
                _require_directory_identity(images, images_identity, "probe images")
        except RuntimeError:
            return
        _unlink_owned_file(path, expected)
    if images_identity is not None:
        try:
            _require_directory_identity(probe, probe_identity, "probe")
        except RuntimeError:
            return
        _remove_owned_empty_directory(images, images_identity)
    _remove_owned_empty_directory(probe, probe_identity)


def probe_colmap_gpu_support(
    colmap_exe: str | Path,
    probe_root: str | Path,
) -> bool:
    probe = Path(probe_root)
    if os.path.lexists(probe):
        LOGGER.warning("COLMAP GPU SIFT probe path already exists: %s", probe)
        return False
    images = probe / "images"
    database = probe / "probe.db"
    probe_identity: _DirectoryIdentity | None = None
    images_identity: _DirectoryIdentity | None = None
    owned_files: dict[Path, tuple[int, int, int]] = {}
    succeeded = False
    try:
        probe.mkdir(parents=True, exist_ok=False)
        probe_identity = _directory_identity(probe)
        _require_directory_identity(probe, probe_identity, "probe")
        images.mkdir()
        images_identity = _directory_identity(images)
        _require_directory_identity(probe, probe_identity, "probe")
        _require_directory_identity(images, images_identity, "probe images")
        checkerboards = tuple(
            images / f"checkerboard_{image_index}.png" for image_index in range(2)
        )
        for checkerboard in checkerboards:
            checkerboard.touch(exist_ok=False)
            identity = _path_identity(checkerboard)
            if identity is None or identity[2] != stat.S_IFREG:
                raise RuntimeError("COLMAP GPU probe checkerboard is invalid")
            owned_files[checkerboard] = identity
        database.touch(exist_ok=False)
        database_identity = _path_identity(database)
        if database_identity is None or database_identity[2] != stat.S_IFREG:
            raise RuntimeError("COLMAP GPU probe database is invalid")
        owned_files[database] = database_identity
        try:
            _write_checkerboards(images)
        finally:
            _require_directory_identity(probe, probe_identity, "probe")
            _require_directory_identity(images, images_identity, "probe images")
            for checkerboard in checkerboards:
                if _path_identity(checkerboard) != owned_files[checkerboard]:
                    raise RuntimeError("COLMAP GPU probe checkerboard was replaced")
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
        _require_directory_identity(probe, probe_identity, "probe")
        _require_directory_identity(images, images_identity, "probe images")
        database_identity = _path_identity(database)
        if database_identity != owned_files[database] or database.stat().st_size <= 0:
            raise RuntimeError("COLMAP GPU probe completed without a valid database")
        succeeded = True
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        LOGGER.warning("COLMAP GPU SIFT probe failed: %s", exc)
    finally:
        _cleanup_owned_probe(
            probe,
            probe_identity,
            images,
            images_identity,
            owned_files,
        )
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


def _regular_file_signature(path: Path, label: str) -> _FileSignature:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if stat.S_IFMT(before.st_mode) != stat.S_IFREG:
        raise RuntimeError(f"{label} is not a regular file: {path}")
    digest = _sha256_file(path)
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} changed while it was read: {path}") from exc
    before_metadata = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_metadata = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stat.S_IFMT(after.st_mode) != stat.S_IFREG or after_metadata != before_metadata:
        raise RuntimeError(f"{label} changed while it was read: {path}")
    return (*after_metadata, digest)


def _require_file_signature(
    path: Path,
    expected: _FileSignature,
    label: str,
) -> None:
    if _regular_file_signature(path, label) != expected:
        raise RuntimeError(f"{label} contents changed during COLMAP execution")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _is_finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _is_nonempty_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(ord(char) < 32 for char in value)
    )


def _safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or value == ".":
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    if path.as_posix() != value:
        return None
    return value


def _validate_policy_payload(policy: object, effective_mode: str) -> None:
    if not isinstance(policy, dict) or set(policy) != _POLICY_KEYS:
        raise ValueError("selection manifest has an invalid policy schema")
    mode = policy["mode"]
    if mode not in ("smart", "fixed_fps"):
        raise ValueError("selection manifest has an invalid policy mode")
    expected_mode = "smart" if effective_mode == "smart" else "fixed_fps"
    if mode != expected_mode:
        raise ValueError("selection policy mode does not match its effective mode")

    frame_budget = policy["frame_budget"]
    if not _is_exact_int(frame_budget) or frame_budget <= 0:
        raise ValueError("selection policy has an invalid frame budget")
    long_edge_cap = policy["resolution_long_edge_cap"]
    if long_edge_cap is not None and (
        not _is_exact_int(long_edge_cap) or long_edge_cap <= 0
    ):
        raise ValueError("selection policy has an invalid resolution cap")

    fixed_fps = policy["fixed_fps"]
    candidate_fps = policy["candidate_fps"]
    candidate_long_edge = policy["candidate_long_edge"]
    if not all(
        _is_exact_int(value)
        for value in (fixed_fps, candidate_fps, candidate_long_edge)
    ):
        raise ValueError("selection policy rates and dimensions must be integers")
    if mode == "fixed_fps" and fixed_fps <= 0:
        raise ValueError("selection policy fixed FPS must be positive")
    if mode == "smart" and (
        candidate_fps <= 0
        or candidate_fps > 12
        or candidate_long_edge <= 0
        or candidate_long_edge > 320
    ):
        raise ValueError("selection policy smart analysis settings are invalid")
    if not isinstance(policy["version"], str):
        raise ValueError("selection policy has an invalid version")


def _validate_frame_metadata(record: dict[str, object], effective_mode: str) -> None:
    if set(record) != _FRAME_RECORD_KEYS:
        raise ValueError("selection manifest has an invalid frame record schema")
    if _safe_relative_path(record["source_relative_path"]) is None:
        raise ValueError("selection manifest has an unsafe source path")

    source_index = record["source_index"]
    source_pts = record["source_pts"]
    timestamp_s = record["timestamp_s"]
    if source_index is not None and (
        not _is_exact_int(source_index) or source_index < 0
    ):
        raise ValueError("selection manifest has an invalid source index")
    if source_pts is not None and not _is_exact_int(source_pts):
        raise ValueError("selection manifest has an invalid source timestamp")
    if timestamp_s is not None and not _is_finite_number(timestamp_s):
        raise ValueError("selection manifest has an invalid frame timestamp")
    if (source_index is None) != (timestamp_s is None):
        raise ValueError("selection manifest has inconsistent source metadata")
    if source_index is None and source_pts is not None:
        raise ValueError("selection manifest has inconsistent source metadata")
    if effective_mode == "fixed_fps" and source_index is None:
        raise ValueError("fixed-FPS frames require video source metadata")
    if effective_mode == "photo_set_all" and any(
        value is not None for value in (source_index, source_pts, timestamp_s)
    ):
        raise ValueError("photo-set frames cannot contain video source metadata")

    metrics = record["metrics"]
    selection_score = record["selection_score"]
    if effective_mode == "smart":
        if not isinstance(metrics, dict) or set(metrics) != _METRIC_KEYS:
            raise ValueError("smart frame metrics have an invalid schema")
        if not all(_is_finite_number(value) for value in metrics.values()):
            raise ValueError("smart frame metrics must be finite numbers")
        if not _is_finite_number(selection_score):
            raise ValueError("smart frame selection score must be finite")
    elif metrics is not None or selection_score is not None:
        raise ValueError("non-smart frames cannot contain smart selection metrics")

    reasons = record["reasons"]
    if (
        not isinstance(reasons, list)
        or not all(_is_nonempty_text(reason) for reason in reasons)
        or len(set(reasons)) != len(reasons)
    ):
        raise ValueError("selection manifest has invalid frame reasons")


def _safe_output_name(value: object) -> str | None:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        return None
    if "/" in value or "\\" in value or Path(value).name != value:
        return None
    if _OUTPUT_NAME_PATTERN.fullmatch(value) is None:
        return None
    return value


def _read_selection_digest(frames_dir: Path) -> str:
    if frames_dir.is_symlink() or not frames_dir.is_dir():
        raise ValueError(f"selected frame directory is missing or unsafe: {frames_dir}")
    manifest_path = frames_dir / "selection_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("selection manifest is missing or unsafe")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ValueError(f"Invalid selection manifest: {manifest_path}")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("selection manifest must use schema version 1")
    if not _is_sha256(payload.get("source_digest")):
        raise ValueError("selection manifest has an invalid source digest")
    effective_mode = payload.get("effective_mode")
    if effective_mode not in ("smart", "fixed_fps", "photo_set_all"):
        raise ValueError("selection manifest has an invalid effective mode")
    policy_payload = payload.get("policy")
    _validate_policy_payload(policy_payload, effective_mode)
    records = payload.get("frames")
    if not isinstance(records, list):
        raise ValueError("selection manifest is missing frame records")

    digest = payload.get("image_set_digest")
    if not _is_sha256(digest):
        raise ValueError("selection manifest has an invalid image_set_digest")
    reconstruction_guardrail = payload.get("reconstruction_guardrail")
    if reconstruction_guardrail is not None and (
        reconstruction_guardrail != "uniform_profile_cap"
        or effective_mode != "photo_set_all"
    ):
        raise ValueError("selection manifest has an invalid reconstruction guardrail")

    frame_ids: set[str] = set()
    output_names: set[str] = set()
    selected_payload: list[list[str]] = []
    unselected_count = 0
    source_kinds: set[bool] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("selection manifest contains an invalid frame record")
        _validate_frame_metadata(record, effective_mode)
        source_kinds.add(record["source_index"] is not None)
        frame_id = record.get("frame_id")
        expected_frame_id = hashlib.sha256(
            (
                f"{payload['source_digest']}|{record['source_index']}|"
                f"{record['timestamp_s']}|{record['source_relative_path']}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        if (
            not isinstance(frame_id, str)
            or _FRAME_ID_PATTERN.fullmatch(frame_id) is None
            or frame_id != expected_frame_id
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
            unselected_count += 1
            if output_name != "" or recorded_sha256 != "":
                raise ValueError(
                    "unselected frame records cannot reference output bytes"
                )
            continue
        safe_name = _safe_output_name(output_name)
        expected_name = f"frame_{len(selected_payload):06d}.png"
        if safe_name is None or safe_name != expected_name or safe_name in output_names:
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

    if not 1 <= len(selected_payload) <= 800:
        raise ValueError("selection manifest must contain 1..800 selected frames")
    if (
        effective_mode in {"smart", "fixed_fps"}
        and len(selected_payload) > policy_payload["frame_budget"]
    ):
        raise ValueError("selection manifest exceeds its frame budget")
    if effective_mode == "smart" and len(source_kinds) != 1:
        raise ValueError("smart selection cannot mix photo and video frame metadata")
    if effective_mode == "fixed_fps" and unselected_count:
        raise ValueError("fixed-FPS selection cannot contain unselected records")
    if effective_mode == "photo_set_all":
        if unselected_count and reconstruction_guardrail != "uniform_profile_cap":
            raise ValueError("bounded photo selection requires its guardrail marker")
        if not unselected_count and reconstruction_guardrail is not None:
            raise ValueError("photo guardrail marker requires bounded records")

    expected_entries = {"selection_manifest.json", *output_names}
    actual_entries = {path.name for path in frames_dir.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError("selected frame directory does not match its manifest")
    for entry_name in expected_entries:
        identity = _path_identity(frames_dir / entry_name)
        if identity is None or identity[2] != stat.S_IFREG:
            raise ValueError(
                f"selected frame directory contains an unsafe entry: {entry_name}"
            )

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


def _require_selection_digest(frames_dir: Path, expected: str) -> None:
    if _read_selection_digest(frames_dir) != expected:
        raise RuntimeError("selected frame digest changed during COLMAP execution")


def _require_attempt_layout(
    attempt: Path,
    attempt_identity: _DirectoryIdentity,
    sparse: Path,
    sparse_identity: _DirectoryIdentity,
    database_path: Path,
    database_identity: tuple[int, int, int] | None,
    *,
    require_empty_sparse: bool,
) -> None:
    _require_directory_identity(attempt, attempt_identity, "COLMAP attempt")
    _require_directory_identity(sparse, sparse_identity, "COLMAP sparse")
    expected_entries = {"sparse"}
    if database_identity is not None:
        expected_entries.add("colmap.db")
    actual_entries = {path.name for path in attempt.iterdir()}
    if actual_entries != expected_entries:
        raise RuntimeError(
            "fresh COLMAP attempt contains unexpected entries: "
            f"{sorted(actual_entries)}"
        )
    if (
        database_identity is not None
        and _path_identity(database_path) != database_identity
    ):
        raise RuntimeError("COLMAP database identity changed during execution")
    if require_empty_sparse and next(sparse.iterdir(), None) is not None:
        raise RuntimeError("fresh COLMAP sparse directory is not empty")


def _require_sparse_model_set(
    sparse: Path,
    model_entries: list[tuple[Path, _DirectoryIdentity]],
) -> None:
    expected = {
        path.name: (identity[0], identity[1], stat.S_IFDIR)
        for path, identity in model_entries
    }
    actual = {path.name: _path_identity(path) for path in sparse.iterdir()}
    if actual != expected:
        raise RuntimeError("COLMAP sparse model set changed during conversion")


def _capture_converted_outputs(model_dir: Path) -> dict[Path, _FileSignature]:
    return {
        model_dir / name: _regular_file_signature(
            model_dir / name,
            f"COLMAP converted output {name}",
        )
        for name in _CONVERTED_MODEL_FILES
    }


def _require_converted_outputs(
    converted: dict[Path, dict[Path, _FileSignature]],
) -> None:
    for outputs in converted.values():
        for path, signature in outputs.items():
            _require_file_signature(
                path,
                signature,
                f"COLMAP converted output {path.name}",
            )


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
    attempt_identity = _directory_identity(attempt)
    _require_directory_identity(attempt, attempt_identity, "COLMAP attempt")
    sparse = attempt / "sparse"
    sparse.mkdir()
    _require_directory_identity(attempt, attempt_identity, "COLMAP attempt")
    sparse_identity = _directory_identity(sparse)
    commands = build_colmap_commands(frames, attempt, executable, policy)
    expected_stages = (
        "feature_extractor",
        f"{policy.matcher}_matcher",
        "mapper",
    )
    if (
        len(commands) != 3
        or any(len(command) < 2 for command in commands)
        or tuple(command[1] for command in commands) != expected_stages
    ):
        raise RuntimeError("COLMAP attempt requires exactly three core commands")
    database_path = attempt / "colmap.db"
    database_identity: tuple[int, int, int] | None = None
    database_signature: _FileSignature | None = None
    for command_index, command in enumerate(commands):
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=True,
        )
        if database_signature is not None:
            _require_file_signature(
                database_path,
                database_signature,
                "COLMAP database",
            )
        if on_progress is not None:
            on_progress(0.6 * command_index / len(commands), command[1])
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=True,
        )
        if on_progress is not None and database_signature is not None:
            _require_file_signature(
                database_path,
                database_signature,
                "COLMAP database",
            )
        if on_progress is not None or command_index in {0, 2}:
            _require_selection_digest(frames, selection_digest)
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=True,
        )
        if database_signature is not None:
            _require_file_signature(
                database_path,
                database_signature,
                "COLMAP database",
            )
        subprocess.run(command, check=True)
        if command_index in {0, 2}:
            _require_selection_digest(frames, selection_digest)
        if command_index == 0:
            database_identity = _path_identity(database_path)
            if database_identity is None or database_identity[2] != stat.S_IFREG:
                raise RuntimeError("COLMAP produced no database")
        if database_identity is None:
            raise RuntimeError("COLMAP produced no database")
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=command_index < 2,
        )
        database_signature = _regular_file_signature(
            database_path,
            "COLMAP database",
        )

    if (
        database_identity is None or database_signature is None
    ):  # pragma: no cover - guarded by the loop above
        raise RuntimeError("COLMAP produced no database")
    model_entries: list[tuple[Path, _DirectoryIdentity]] = []
    for path in sparse.iterdir():
        identity = _path_identity(path)
        if identity is None or identity[2] != stat.S_IFDIR:
            raise RuntimeError(f"COLMAP produced an unsafe sparse entry: {path}")
        model_entries.append((path, (identity[0], identity[1])))
    model_entries.sort(key=lambda item: item[0].name)
    if not model_entries:
        raise RuntimeError("COLMAP produced no sparse model")
    _require_sparse_model_set(sparse, model_entries)
    converted_outputs: dict[Path, dict[Path, _FileSignature]] = {}
    for model_index, (model_dir, model_identity) in enumerate(model_entries):
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=False,
        )
        _require_sparse_model_set(sparse, model_entries)
        _require_directory_identity(model_dir, model_identity, "COLMAP model")
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
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=False,
        )
        _require_sparse_model_set(sparse, model_entries)
        _require_directory_identity(model_dir, model_identity, "COLMAP model")
        converted_outputs[model_dir] = _capture_converted_outputs(model_dir)
        if on_progress is not None:
            on_progress(
                0.6 + 0.4 * (model_index + 1) / len(model_entries),
                f"model_converter:{model_dir.name}",
            )
            _require_selection_digest(frames, selection_digest)
            _require_file_signature(
                database_path,
                database_signature,
                "COLMAP database",
            )
            _require_converted_outputs(converted_outputs)
        _require_attempt_layout(
            attempt,
            attempt_identity,
            sparse,
            sparse_identity,
            database_path,
            database_identity,
            require_empty_sparse=False,
        )
        _require_sparse_model_set(sparse, model_entries)
        _require_directory_identity(model_dir, model_identity, "COLMAP model")

    _require_attempt_layout(
        attempt,
        attempt_identity,
        sparse,
        sparse_identity,
        database_path,
        database_identity,
        require_empty_sparse=False,
    )
    _require_sparse_model_set(sparse, model_entries)
    _require_file_signature(
        database_path,
        database_signature,
        "COLMAP database",
    )
    _require_converted_outputs(converted_outputs)
    for model_dir, model_identity in model_entries:
        _require_directory_identity(model_dir, model_identity, "COLMAP model")
    model_dirs = tuple(path for path, _identity in model_entries)

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
