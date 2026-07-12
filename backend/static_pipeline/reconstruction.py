from __future__ import annotations

import math
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.preprocess import parse_colmap as colmap_parser

from .colmap import choose_colmap_policy
from .contracts import (
    ColmapAttempt,
    ColmapPolicy,
    FrameRecord,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
    UncoveredInterval,
)
from .stage_cache import promote_directory


RunAttempt = Callable[[SelectionManifest, Path, ColmapPolicy], ColmapAttempt]
MaterializeBackfill = Callable[
    [SelectionManifest, tuple[UncoveredInterval, ...]], SelectionManifest
]

_COVERAGE_FAILURES = {
    "registered_ratio",
    "dominant_component",
    "interior_gap",
    "start_endpoint",
    "end_endpoint",
}
_MODEL_TEXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")


class ReconstructionGateError(RuntimeError):
    def __init__(
        self,
        decision: GateDecision,
        attempts: tuple[ColmapAttempt, ...],
        decisions: tuple[GateDecision, ...] | None = None,
    ) -> None:
        self.decision = decision
        self.attempts = attempts
        self.decisions = decisions or (decision,)
        failures = ", ".join(decision.failures) or "unknown"
        super().__init__(f"COLMAP reconstruction failed quality gate: {failures}")


def _selected_axis(
    manifest: SelectionManifest,
) -> tuple[tuple[FrameRecord, ...], tuple[float, ...], str]:
    selected = manifest.selected_frames
    if not selected:
        raise ValueError("selection manifest has no selected frames")
    frame_ids = [frame.frame_id for frame in selected]
    output_names = [frame.output_name for frame in selected]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("selected frame IDs must be unique")
    if any(not name for name in output_names) or len(set(output_names)) != len(
        output_names
    ):
        raise ValueError("selected output names must be non-empty and unique")

    has_timestamp = [frame.timestamp_s is not None for frame in selected]
    if all(has_timestamp):
        if any(
            frame.timestamp_s is None or not math.isfinite(frame.timestamp_s)
            for frame in selected
        ):
            raise ValueError("selected video timestamps must be finite")
        ordered = tuple(selected)
        coordinates = tuple(float(frame.timestamp_s) for frame in ordered)
        if any(
            right < left
            for left, right in zip(coordinates, coordinates[1:], strict=False)
        ):
            raise ValueError(
                "selected video timestamps must be nondecreasing in manifest order"
            )
        return ordered, coordinates, "seconds"
    if any(has_timestamp):
        raise ValueError("selected frame timestamps cannot mix video and photo records")
    ordered = tuple(selected)
    coordinates = tuple(float(index) for index in range(len(ordered)))
    return ordered, coordinates, "photo_order"


def _coverage_metrics(
    manifest: SelectionManifest,
    registered_names: frozenset[str],
) -> tuple[
    float,
    float,
    float,
    float,
    str,
    tuple[UncoveredInterval, ...],
]:
    selected, coordinates, coverage_unit = _selected_axis(manifest)
    registered = [frame.output_name in registered_names for frame in selected]
    registered_indices = [index for index, value in enumerate(registered) if value]

    intervals: list[UncoveredInterval] = []
    cursor = 0
    while cursor < len(selected):
        if registered[cursor]:
            cursor += 1
            continue
        run_start = cursor
        while cursor + 1 < len(selected) and not registered[cursor + 1]:
            cursor += 1
        run_end = cursor
        left_index = (
            run_start - 1 if run_start > 0 and registered[run_start - 1] else None
        )
        right_index = (
            run_end + 1
            if run_end + 1 < len(selected) and registered[run_end + 1]
            else None
        )
        if left_index is None and right_index is None:
            kind = "all"
            interval_start = coordinates[0]
            interval_end = coordinates[-1]
        elif left_index is None:
            kind = "start"
            interval_start = coordinates[0]
            interval_end = coordinates[right_index]
        elif right_index is None:
            kind = "end"
            interval_start = coordinates[left_index]
            interval_end = coordinates[-1]
        else:
            kind = "interior"
            interval_start = coordinates[left_index]
            interval_end = coordinates[right_index]
        intervals.append(
            UncoveredInterval(
                start_s=interval_start,
                end_s=interval_end,
                missing_frame_ids=tuple(
                    selected[index].frame_id for index in range(run_start, run_end + 1)
                ),
                coverage_unit=coverage_unit,
                kind=kind,
                left_boundary_frame_id=(
                    selected[left_index].frame_id if left_index is not None else None
                ),
                right_boundary_frame_id=(
                    selected[right_index].frame_id if right_index is not None else None
                ),
            )
        )
        cursor += 1

    if not registered_indices:
        return 0.0, math.inf, math.inf, math.inf, coverage_unit, tuple(intervals)

    first_registered = registered_indices[0]
    last_registered = registered_indices[-1]
    coverage = coordinates[last_registered] - coordinates[first_registered]
    start_gap = coordinates[first_registered] - coordinates[0]
    end_gap = coordinates[-1] - coordinates[last_registered]
    interior_gaps = [
        interval.end_s - interval.start_s
        for interval in intervals
        if interval.kind == "interior"
    ]
    max_interior_gap = max(interior_gaps, default=0.0)
    return (
        coverage,
        max_interior_gap,
        start_gap,
        end_gap,
        coverage_unit,
        tuple(intervals),
    )


def _camera_payload_is_valid(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    try:
        image_id = payload["image_id"]
        width = payload["width"]
        height = payload["height"]
        intrinsic = np.asarray(payload["K"], dtype=np.float64)
        rotation = np.asarray(payload["R"], dtype=np.float64)
        translation = np.asarray(payload["t"], dtype=np.float64)
        world_to_camera = np.asarray(payload["w2c"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if type(image_id) is not int or image_id <= 0:
        return False
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return False
    if (
        intrinsic.shape != (3, 3)
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or world_to_camera.shape != (4, 4)
    ):
        return False
    if not all(
        np.isfinite(value).all()
        for value in (intrinsic, rotation, translation, world_to_camera)
    ):
        return False
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        return False
    if not np.allclose(
        world_to_camera[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8, rtol=0
    ):
        return False
    if not np.allclose(world_to_camera[:3, :3], rotation, atol=1e-8, rtol=0):
        return False
    if not np.allclose(world_to_camera[:3, 3], translation, atol=1e-8, rtol=0):
        return False
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=1e-5):
        return False
    determinant = float(np.linalg.det(rotation))
    return math.isfinite(determinant) and abs(determinant - 1.0) <= 1e-5


def _read_cameras(
    model_dir: Path, expected_names: frozenset[str]
) -> tuple[frozenset[str], bool]:
    try:
        parsed = colmap_parser.parse_cameras_from_model(model_dir)
    except Exception:
        return frozenset(), False
    if not isinstance(parsed, Mapping) or not parsed:
        return frozenset(), False
    parsed_names = frozenset(name for name in parsed if isinstance(name, str) and name)
    known_names = parsed_names & expected_names
    names_are_valid = (
        len(parsed_names) == len(parsed)
        and not (parsed_names - expected_names)
        and all(_camera_payload_is_valid(payload) for payload in parsed.values())
    )
    return known_names, names_are_valid


def _read_point_metrics(model_dir: Path) -> tuple[float, float, float, int]:
    try:
        xyz, rgb, track_lengths, reprojection_errors = (
            colmap_parser.load_points3d_from_model(model_dir)
        )
        xyz_array = np.asarray(xyz, dtype=np.float64)
        rgb_array = np.asarray(rgb, dtype=np.float64)
        tracks_array = np.asarray(track_lengths, dtype=np.float64)
        errors_array = np.asarray(reprojection_errors, dtype=np.float64)
    except Exception:
        return math.inf, math.inf, 0.0, 0
    point_count = len(xyz_array) if xyz_array.ndim >= 1 else 0
    if (
        point_count == 0
        or xyz_array.shape != (point_count, 3)
        or rgb_array.shape != (point_count, 3)
        or tracks_array.shape != (point_count,)
        or errors_array.shape != (point_count,)
        or not np.isfinite(xyz_array).all()
        or not np.isfinite(rgb_array).all()
        or not np.isfinite(tracks_array).all()
        or not np.isfinite(errors_array).all()
        or np.any(tracks_array < 0)
        or np.any(errors_array < 0)
    ):
        return math.inf, math.inf, 0.0, 0
    return (
        float(np.median(errors_array)),
        float(np.percentile(errors_array, 95)),
        float(np.median(tracks_array)),
        point_count,
    )


def measure_models(
    model_dirs: Sequence[str | Path], manifest: SelectionManifest
) -> tuple[ModelMetrics, ...]:
    selected, _coordinates, _coverage_unit = _selected_axis(manifest)
    expected_names = frozenset(frame.output_name for frame in selected)
    measured: list[ModelMetrics] = []
    for raw_model_dir in model_dirs:
        model_dir = Path(raw_model_dir)
        registered_names, camera_valid = _read_cameras(model_dir, expected_names)
        median_error, p95_error, median_track, point_count = _read_point_metrics(
            model_dir
        )
        (
            coverage,
            max_interior_gap,
            start_gap,
            end_gap,
            coverage_unit,
            _intervals,
        ) = _coverage_metrics(manifest, registered_names)
        registered_count = len(registered_names)
        measured.append(
            ModelMetrics(
                model_dir=model_dir,
                registered_names=registered_names,
                registered_count=registered_count,
                registered_ratio=registered_count / len(selected),
                registered_share=0.0,
                temporal_coverage_s=coverage,
                max_interior_gap_s=max_interior_gap,
                start_gap_s=start_gap,
                end_gap_s=end_gap,
                median_reprojection_error_px=median_error,
                p95_reprojection_error_px=p95_error,
                median_track_length=median_track,
                sparse_point_count=point_count,
                valid_names_intrinsics_and_poses=camera_valid,
                coverage_unit=coverage_unit,
            )
        )
    registered_union = frozenset().union(
        *(model.registered_names for model in measured)
    )
    union_count = len(registered_union)
    return tuple(
        replace(
            model,
            registered_share=(
                model.registered_count / union_count if union_count else 0.0
            ),
        )
        for model in measured
    )


def rank_models(models: Sequence[ModelMetrics]) -> tuple[ModelMetrics, ...]:
    return tuple(
        sorted(
            models,
            key=lambda model: (
                model.registered_count,
                model.temporal_coverage_s,
                -model.max_interior_gap_s,
                model.registered_share,
                -model.median_reprojection_error_px,
                -model.p95_reprojection_error_px,
                model.median_track_length,
                model.sparse_point_count,
            ),
            reverse=True,
        )
    )


def evaluate_reconstruction(
    models: Sequence[ModelMetrics],
    manifest: SelectionManifest,
    attempt_index: int = 0,
) -> GateDecision:
    if attempt_index not in {0, 1}:
        raise ValueError("attempt_index must be 0 or 1")
    ranked = rank_models(models)
    if not ranked:
        raise ValueError("at least one measured COLMAP model is required")
    dominant = ranked[0]
    (
        _coverage,
        _max_interior_gap,
        _start_gap,
        _end_gap,
        coverage_unit,
        uncovered_intervals,
    ) = _coverage_metrics(manifest, dominant.registered_names)
    if dominant.coverage_unit != coverage_unit:
        raise ValueError("model coverage unit does not match selection manifest")
    checks = {
        "registered_ratio": dominant.registered_ratio >= 0.90,
        "dominant_component": dominant.registered_share >= 0.95,
        "interior_gap": dominant.max_interior_gap_s <= 2.0,
        "start_endpoint": dominant.start_gap_s <= 1.0,
        "end_endpoint": dominant.end_gap_s <= 1.0,
        "median_reprojection": dominant.median_reprojection_error_px <= 1.0,
        "p95_reprojection": dominant.p95_reprojection_error_px <= 2.5,
        "median_track_length": dominant.median_track_length >= 3.0,
        "valid_names_intrinsics_poses": dominant.valid_names_intrinsics_and_poses,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    warnings = (
        ("below_95_percent_registration_target",)
        if 0.90 <= dominant.registered_ratio < 0.95
        else ()
    )
    smart_manifest = (
        manifest.effective_mode == "smart" and manifest.policy.mode == "smart"
    )
    retry_recommended = (
        bool(failures)
        and smart_manifest
        and attempt_index == 0
        and bool(uncovered_intervals)
        and bool(_COVERAGE_FAILURES.intersection(failures))
    )
    return GateDecision(
        passed=not failures,
        dominant=dominant,
        failures=failures,
        uncovered_intervals=uncovered_intervals,
        retry_recommended=retry_recommended,
        warnings=warnings,
    )


def _validate_model_tree(model_dir: Path) -> None:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError(
            f"accepted COLMAP model is not a regular directory: {model_dir}"
        )
    for root, directories, files in os.walk(model_dir):
        root_path = Path(root)
        if any((root_path / name).is_symlink() for name in (*directories, *files)):
            raise ValueError(f"accepted COLMAP model contains a symlink: {model_dir}")
    for name in _MODEL_TEXT_FILES:
        path = model_dir / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"accepted COLMAP model is missing {name}: {model_dir}"
            )


def _directory_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = os.lstat(path)
    except OSError:
        return None
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR:
        return None
    return metadata.st_dev, metadata.st_ino


def _restore_quarantined_stage(quarantine: Path, original: Path) -> None:
    try:
        promote_directory(quarantine, original)
    except (OSError, RuntimeError):
        pass


def _claim_owned_stage(path: Path, identity: tuple[int, int]) -> Path | None:
    if _directory_identity(path) != identity:
        return None
    quarantine = path.with_name(f".{path.name}.cleanup-{uuid.uuid4().hex}")
    try:
        promote_directory(path, quarantine)
    except (OSError, RuntimeError):
        return None
    if _directory_identity(quarantine) != identity:
        _restore_quarantined_stage(quarantine, path)
        return None
    return quarantine


def _clear_directory_contents(path: Path) -> None:
    for entry in path.iterdir():
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            raise OSError(f"cleanup staging tree is unexpectedly nested: {entry}")
        else:
            entry.unlink()


def _posix_directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _clear_directory_fd(directory_fd: int) -> None:
    with os.scandir(directory_fd) as iterator:
        entries = list(iterator)
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"cleanup staging tree is unexpectedly nested: {entry.name}")
        try:
            os.unlink(entry.name, dir_fd=directory_fd)
        except FileNotFoundError:
            continue


def _remove_owned_directory_posix(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    directory_fd: int | None = None
    try:
        directory_fd = os.open(path, _posix_directory_open_flags())
        metadata = os.fstat(directory_fd)
        if (metadata.st_dev, metadata.st_ino) != identity or stat.S_IFMT(
            metadata.st_mode
        ) != stat.S_IFDIR:
            return False
        _clear_directory_fd(directory_fd)
        if _directory_identity(path) != identity:
            return False
        os.rmdir(path)
        return True
    except (NotImplementedError, OSError, TypeError):
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _remove_owned_directory_windows(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    import ctypes
    from ctypes import wintypes

    delete_access = 0x00010000
    file_read_attributes = 0x00000080
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    file_disposition_info = 4

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_file_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        delete_access | file_read_attributes,
        file_share_read | file_share_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        return False

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", ctypes.c_ubyte),)

    delete_marked = False
    try:
        if _directory_identity(path) != identity:
            return False
        _clear_directory_contents(path)
        disposition = FileDispositionInfo(1)
        delete_marked = bool(
            set_file_information(
                handle,
                file_disposition_info,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            )
        )
    except OSError:
        return False
    finally:
        close_handle(handle)
    return delete_marked and not os.path.lexists(path)


def _remove_owned_stage(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    quarantine = _claim_owned_stage(path, identity)
    if quarantine is None:
        return
    if os.name == "nt":
        _remove_owned_directory_windows(quarantine, identity)
        return
    _remove_owned_directory_posix(quarantine, identity)


def _publish_validated_model(model_dir: Path, target: Path) -> Path:
    source = Path(model_dir)
    destination = Path(target)
    _validate_model_tree(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    staging_identity: tuple[int, int] | None = None
    try:
        staging.mkdir(exist_ok=False)
        staging_identity = _directory_identity(staging)
        if staging_identity is None:
            raise RuntimeError("validated model staging directory identity is invalid")
        for name in _MODEL_TEXT_FILES:
            shutil.copy2(
                source / name,
                staging / name,
                follow_symlinks=False,
            )
        _validate_model_tree(staging)
        colmap_parser.parse_cameras_from_model(staging)
        colmap_parser.load_points3d_from_model(staging)
        promote_directory(staging, destination)
        staging_identity = None
        return destination
    finally:
        _remove_owned_stage(staging, staging_identity)


def _backfill_preserves_contract(
    before: SelectionManifest, after: SelectionManifest
) -> bool:
    return (
        after.schema_version == before.schema_version
        and after.source_digest == before.source_digest
        and after.policy == before.policy
        and after.policy.mode == "smart"
        and after.effective_mode == "smart"
        and after.image_set_digest != before.image_set_digest
    )


def reconstruct_with_gate(
    manifest: SelectionManifest,
    *,
    use_gpu: bool,
    attempt_root: Path,
    run_attempt: RunAttempt,
    materialize_backfill: MaterializeBackfill,
) -> ReconstructionBundle:
    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"attempt root is not a regular directory: {root}")

    attempts: list[ColmapAttempt] = []
    decisions: list[GateDecision] = []
    current = manifest
    for attempt_index in range(2):
        policy = choose_colmap_policy(
            use_gpu=use_gpu,
            selected_count=len(current.selected_frames),
            attempt_index=attempt_index,
        )
        attempt = run_attempt(current, root / f"attempt-{attempt_index}", policy)
        attempts.append(attempt)
        decision = evaluate_reconstruction(
            measure_models(attempt.model_dirs, current),
            current,
            attempt_index=attempt_index,
        )
        decisions.append(decision)
        if decision.passed:
            accepted_model = _publish_validated_model(
                decision.dominant.model_dir, root / "validated"
            )
            return ReconstructionBundle(
                selected_manifest=current,
                accepted_model_dir=accepted_model,
                decision=decision,
                attempts=tuple(attempts),
                decisions=tuple(decisions),
            )
        if not decision.retry_recommended or attempt_index == 1:
            raise ReconstructionGateError(decision, tuple(attempts), tuple(decisions))
        backfilled = materialize_backfill(current, decision.uncovered_intervals)
        if not _backfill_preserves_contract(current, backfilled):
            raise ReconstructionGateError(decision, tuple(attempts), tuple(decisions))
        current = backfilled
    raise AssertionError("bounded reconstruction loop exhausted")
