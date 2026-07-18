from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from backend.static_pipeline.contracts import (
    ColmapAttempt,
    ColmapPolicy,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
    UncoveredInterval,
)
from backend.static_pipeline.colmap import choose_colmap_policy, run_colmap_attempt
from backend.static_pipeline.reconstruction import (
    _publish_validated_model,
    evaluate_reconstruction,
    measure_models,
)

from .contracts import (
    FrameArtifact,
    GeometryCandidateReport,
    LearnedArtifacts,
    LearnedReconstructionOutput,
)
from .da3 import AnchorInferenceResult, CameraRecord, PinholeCamera
from .masks import MaskFusionEvidence


CandidateId = Literal["classical", "learned_hybrid"]
CandidateRunner = Callable[
    [SelectionManifest, tuple[FrameArtifact, ...], Path, int, str],
    ColmapAttempt,
]
MaterializeBackfill = Callable[
    [SelectionManifest, tuple[UncoveredInterval, ...]],
    tuple[SelectionManifest, tuple[FrameArtifact, ...]],
]
MeasureModels = Callable[
    [Sequence[Path], SelectionManifest],
    Sequence[ModelMetrics],
]
EvaluateCandidate = Callable[
    [Sequence[ModelMetrics], SelectionManifest, int],
    GateDecision,
]
PublishModel = Callable[[Path, Path], Path]
RunColmapAttempt = Callable[[Path, Path, ColmapPolicy, Path | None], ColmapAttempt]


@dataclass(frozen=True)
class HybridGeometryInputs:
    anchors: AnchorInferenceResult
    masks: MaskFusionEvidence


class HybridGeometryBackend(Protocol):
    version: str

    def reconstruct(
        self,
        *,
        manifest: SelectionManifest,
        frames_root: Path,
        image_names: tuple[str, ...],
        mask_root: Path,
        database_path: Path,
        sparse_root: Path,
        shared_camera: PinholeCamera,
        anchor_cameras: tuple[CameraRecord, ...],
        use_gpu: bool,
    ) -> tuple[Path, ...]: ...


_MINIMUM_GATES = {
    "registered_ratio": ("registered_ratio", 0.90),
    "dominant_component": ("registered_share", 0.95),
    "median_track_length": ("median_track_length", 3.0),
}
_MAXIMUM_GATES = {
    "interior_gap": ("max_interior_gap_s", 2.0),
    "start_endpoint": ("start_gap_s", 1.0),
    "end_endpoint": ("end_gap_s", 1.0),
    "median_reprojection": ("median_reprojection_error_px", 1.0),
    "p95_reprojection": ("p95_reprojection_error_px", 2.5),
}
_BOOLEAN_GATES = {"valid_names_intrinsics_poses": "valid_names_intrinsics_and_poses"}


class GeometryComparisonError(RuntimeError):
    def __init__(
        self,
        candidates: tuple[GeometryCandidateReport, ...],
        closest_candidate: GeometryCandidateReport,
    ) -> None:
        self.candidates = candidates
        self.closest_candidate = closest_candidate
        failures = ", ".join(closest_candidate.decision.failures) or "unknown"
        super().__init__(
            "all geometry candidates failed; closest candidate "
            f"{closest_candidate.candidate_id} failed: {failures}"
        )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric metric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def winner_key(candidate: GeometryCandidateReport) -> tuple[object, ...]:
    if not isinstance(candidate, GeometryCandidateReport):
        raise ValueError("candidate must be a GeometryCandidateReport")
    if not candidate.decision.passed:
        raise ValueError("only gate-passing candidates have a winner key")
    dominant = candidate.decision.dominant
    if type(dominant.registered_count) is not int or dominant.registered_count < 0:
        raise ValueError("registered_count must be a nonnegative plain integer")
    if type(candidate.covered_endpoint_count) is not int or not (
        0 <= candidate.covered_endpoint_count <= 2
    ):
        raise ValueError("covered_endpoint_count must be 0, 1, or 2")
    if type(dominant.sparse_point_count) is not int or dominant.sparse_point_count < 0:
        raise ValueError("sparse_point_count must be a nonnegative plain integer")
    registered_ratio = _finite(dominant.registered_ratio, "registered_ratio")
    registered_share = _finite(dominant.registered_share, "registered_share")
    max_gap = _finite(dominant.max_interior_gap_s, "max_interior_gap_s")
    median_error = _finite(
        dominant.median_reprojection_error_px,
        "median_reprojection_error_px",
    )
    p95_error = _finite(
        dominant.p95_reprojection_error_px,
        "p95_reprojection_error_px",
    )
    median_track = _finite(dominant.median_track_length, "median_track_length")
    return (
        round(-registered_ratio, 9),
        -dominant.registered_count,
        round(-registered_share, 9),
        -candidate.covered_endpoint_count,
        round(max_gap, 9),
        round(median_error, 9),
        round(p95_error, 9),
        round(-median_track, 9),
        -dominant.sparse_point_count,
        candidate.candidate_id,
    )


def closest_failure_key(candidate: GeometryCandidateReport) -> tuple[int, float, str]:
    if not isinstance(candidate, GeometryCandidateReport):
        raise ValueError("candidate must be a GeometryCandidateReport")
    if candidate.decision.passed:
        raise ValueError("only failed candidates have a closest-failure key")
    failures = candidate.decision.failures
    if type(failures) is not tuple or not failures:
        raise ValueError("failed candidate must enumerate failed gates")
    dominant = candidate.decision.dominant
    deficits: list[float] = []
    for failure in failures:
        if failure in _MINIMUM_GATES:
            field_name, threshold = _MINIMUM_GATES[failure]
            value = _finite(getattr(dominant, field_name), field_name)
            deficits.append(max(0.0, (threshold - value) / max(abs(threshold), 1e-9)))
        elif failure in _MAXIMUM_GATES:
            field_name, threshold = _MAXIMUM_GATES[failure]
            value = _finite(getattr(dominant, field_name), field_name)
            deficits.append(max(0.0, (value - threshold) / max(abs(threshold), 1e-9)))
        elif failure in _BOOLEAN_GATES:
            field_name = _BOOLEAN_GATES[failure]
            value = getattr(dominant, field_name)
            if type(value) is not bool:
                raise ValueError(f"{field_name} must be a plain boolean")
            deficits.append(0.0 if value else 1.0)
        else:
            raise ValueError(f"unknown reconstruction gate: {failure}")
    return len(failures), round(sum(deficits), 9), candidate.candidate_id


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _safe_image_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("frame image_name must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("frame image_name must be canonical and traversal-free")
    return pure.parts


def _validate_frame_set(
    manifest: SelectionManifest,
    frames: tuple[FrameArtifact, ...],
) -> tuple[str, Path]:
    if not isinstance(manifest, SelectionManifest):
        raise ValueError("manifest must be a SelectionManifest")
    selected = manifest.selected_frames
    if type(frames) is not tuple or not frames or len(frames) != len(selected):
        raise ValueError("frames must exactly match the selected manifest")
    payload: list[dict[str, object]] = []
    roots: list[Path] = []
    occupied: set[Path] = set()
    for record, frame in zip(selected, frames, strict=True):
        if not isinstance(frame, FrameArtifact):
            raise ValueError("frames must contain FrameArtifact records")
        parts = _safe_image_parts(frame.image_name)
        if (
            frame.frame_id != record.frame_id
            or frame.image_name != record.output_name
            or frame.sha256 != record.sha256
        ):
            raise ValueError("frame artifact does not preserve the manifest join")
        path = frame.path
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or not os.path.lexists(path)
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise ValueError("frame artifact path must be a canonical regular file")
        if tuple(path.parts[-len(parts) :]) != parts:
            raise ValueError("frame artifact path suffix must match image_name")
        if _sha256_path(path) != frame.sha256:
            raise ValueError("frame artifact digest does not match source bytes")
        resolved = path.resolve(strict=True)
        if resolved in occupied:
            raise ValueError("frame artifact paths must be unique")
        occupied.add(resolved)
        root = path
        for _part in parts:
            root = root.parent
        roots.append(root)
        payload.append(
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sha256": frame.sha256,
                "size_bytes": path.stat().st_size,
            }
        )
    if len(set(roots)) != 1:
        raise ValueError("selected frames must share one canonical image root")
    digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": payload,
                "image_set_digest": manifest.image_set_digest,
                "schema": "learned_quality.geometry_frame_set.v1",
                "source_digest": manifest.source_digest,
            }
        )
    ).hexdigest()
    return digest, roots[0]


def geometry_frame_set_digest(
    manifest: SelectionManifest,
    frames: tuple[FrameArtifact, ...],
) -> str:
    digest, _root = _validate_frame_set(manifest, frames)
    return digest


def _covered_endpoint_count(
    manifest: SelectionManifest,
    metrics: ModelMetrics,
) -> int:
    selected = manifest.selected_frames
    if not selected:
        raise ValueError("selected manifest must not be empty")
    endpoint_names = {selected[0].output_name, selected[-1].output_name}
    return len(endpoint_names.intersection(metrics.registered_names))


def _validate_attempt(attempt: ColmapAttempt, branch_root: Path) -> None:
    if not isinstance(attempt, ColmapAttempt):
        raise ValueError("candidate runner must return ColmapAttempt")
    if (
        attempt.root != branch_root
        or not branch_root.is_dir()
        or branch_root.is_symlink()
    ):
        raise ValueError("candidate attempt must own its exact fresh branch directory")
    if (
        attempt.database_path != branch_root / "colmap.db"
        or not attempt.database_path.is_file()
        or attempt.database_path.is_symlink()
    ):
        raise ValueError("candidate attempt must own a fresh exact database")
    if type(attempt.model_dirs) is not tuple or not attempt.model_dirs:
        raise ValueError("candidate attempt must publish at least one model directory")
    for model_dir in attempt.model_dirs:
        if (
            not isinstance(model_dir, Path)
            or not model_dir.is_dir()
            or model_dir.is_symlink()
            or not model_dir.is_relative_to(branch_root)
        ):
            raise ValueError("candidate model directories must stay inside the branch")


def _candidate_report(
    candidate_id: CandidateId,
    runner: CandidateRunner,
    manifest: SelectionManifest,
    frames: tuple[FrameArtifact, ...],
    branch_root: Path,
    attempt_index: int,
    frame_set_digest: str,
    measure_models_fn: MeasureModels,
    evaluate_fn: EvaluateCandidate,
) -> GeometryCandidateReport:
    if os.path.lexists(branch_root):
        raise FileExistsError(branch_root)
    attempt = runner(
        manifest,
        frames,
        branch_root,
        attempt_index,
        frame_set_digest,
    )
    _validate_attempt(attempt, branch_root)
    measured = tuple(measure_models_fn(attempt.model_dirs, manifest))
    if not measured:
        raise ValueError("candidate measurement must return at least one model")
    decision = evaluate_fn(measured, manifest, attempt_index)
    if not isinstance(decision, GateDecision):
        raise ValueError("candidate evaluation must return GateDecision")
    if decision.dominant not in measured:
        raise ValueError("gate decision dominant model must come from measured models")
    return GeometryCandidateReport(
        candidate_id=candidate_id,
        attempt=attempt,
        decision=decision,
        model_dir=decision.dominant.model_dir,
        selected_manifest=manifest,
        frame_set_digest=frame_set_digest,
        covered_endpoint_count=_covered_endpoint_count(manifest, decision.dominant),
    )


def _validate_backfill(
    before: SelectionManifest,
    before_frames: tuple[FrameArtifact, ...],
    after: SelectionManifest,
    after_frames: tuple[FrameArtifact, ...],
) -> tuple[str, Path]:
    if (
        not isinstance(after, SelectionManifest)
        or after.schema_version != before.schema_version
        or after.source_digest != before.source_digest
        or after.policy != before.policy
        or after.effective_mode != "smart"
        or after.image_set_digest == before.image_set_digest
    ):
        raise ValueError(
            "backfill must preserve the canonical Smart selection contract"
        )
    before_ids = {frame.frame_id for frame in before.selected_frames}
    after_ids = {frame.frame_id for frame in after.selected_frames}
    if len(after_ids) <= len(before_ids) or not before_ids.issubset(after_ids):
        raise ValueError("backfill must retain every selected frame and add new frames")
    digest, root = _validate_frame_set(after, after_frames)
    old_artifacts = {frame.frame_id: frame for frame in before_frames}
    new_artifacts = {frame.frame_id: frame for frame in after_frames}
    if any(
        new_artifacts[frame_id] != artifact
        for frame_id, artifact in old_artifacts.items()
    ):
        raise ValueError("backfill must preserve existing frame artifacts exactly")
    return digest, root


def make_classical_candidate_runner(
    *,
    use_gpu: bool,
    colmap_exe: Path | None = None,
    run_attempt_fn: RunColmapAttempt = run_colmap_attempt,
) -> CandidateRunner:
    if type(use_gpu) is not bool:
        raise ValueError("use_gpu must be a plain boolean")
    if colmap_exe is not None and (
        not isinstance(colmap_exe, Path) or not colmap_exe.is_absolute()
    ):
        raise ValueError("colmap_exe must be an absolute Path when supplied")

    def run(
        manifest: SelectionManifest,
        frames: tuple[FrameArtifact, ...],
        attempt_dir: Path,
        attempt_index: int,
        frame_set_digest: str,
    ) -> ColmapAttempt:
        expected_digest, frames_root = _validate_frame_set(manifest, frames)
        if frame_set_digest != expected_digest:
            raise ValueError("classical candidate received a mismatched frame digest")
        policy = choose_colmap_policy(
            use_gpu=use_gpu,
            selected_count=len(manifest.selected_frames),
            attempt_index=attempt_index,
        )
        return run_attempt_fn(frames_root, attempt_dir, policy, colmap_exe)

    return run


def focal_refinement_is_safe(
    before: ModelMetrics,
    after: ModelMetrics,
    *,
    initial_focal_xy: tuple[float, float],
    refined_focal_xy: tuple[float, float],
) -> bool:
    if not isinstance(before, ModelMetrics) or not isinstance(after, ModelMetrics):
        raise ValueError("focal refinement metrics must be ModelMetrics")
    if type(initial_focal_xy) is not tuple or type(refined_focal_xy) is not tuple:
        raise ValueError("focal pairs must be immutable tuples")
    if len(initial_focal_xy) != 2 or len(refined_focal_xy) != 2:
        raise ValueError("focal pairs must contain fx and fy")
    initial = tuple(_finite(value, "initial focal") for value in initial_focal_xy)
    refined = tuple(_finite(value, "refined focal") for value in refined_focal_xy)
    if any(value <= 0.0 for value in (*initial, *refined)):
        raise ValueError("focal values must be positive")
    if any(
        abs(current - original) / original > 0.05
        for original, current in zip(initial, refined, strict=True)
    ):
        return False

    def gates(metrics: ModelMetrics) -> dict[str, bool]:
        return {
            "registered_ratio": _finite(metrics.registered_ratio, "registered_ratio")
            >= 0.90,
            "dominant_component": _finite(metrics.registered_share, "registered_share")
            >= 0.95,
            "interior_gap": _finite(metrics.max_interior_gap_s, "max_interior_gap_s")
            <= 2.0,
            "start_endpoint": _finite(metrics.start_gap_s, "start_gap_s") <= 1.0,
            "end_endpoint": _finite(metrics.end_gap_s, "end_gap_s") <= 1.0,
            "median_reprojection": _finite(
                metrics.median_reprojection_error_px,
                "median_reprojection_error_px",
            )
            <= 1.0,
            "p95_reprojection": _finite(
                metrics.p95_reprojection_error_px,
                "p95_reprojection_error_px",
            )
            <= 2.5,
            "median_track_length": _finite(
                metrics.median_track_length, "median_track_length"
            )
            >= 3.0,
            "valid_names_intrinsics_poses": (
                metrics.valid_names_intrinsics_and_poses is True
            ),
        }

    before_gates = gates(before)
    after_gates = gates(after)
    if any(before_gates[name] and not after_gates[name] for name in before_gates):
        return False
    before_median = _finite(
        before.median_reprojection_error_px,
        "before median_reprojection_error_px",
    )
    after_median = _finite(
        after.median_reprojection_error_px,
        "after median_reprojection_error_px",
    )
    before_p95 = _finite(
        before.p95_reprojection_error_px,
        "before p95_reprojection_error_px",
    )
    after_p95 = _finite(
        after.p95_reprojection_error_px,
        "after p95_reprojection_error_px",
    )
    return (
        after_median <= before_median
        and after_p95 <= before_p95
        and (after_median < before_median or after_p95 < before_p95)
    )


def _mask_root_for_frames(
    frames: tuple[FrameArtifact, ...],
    masks: MaskFusionEvidence,
) -> Path:
    if not isinstance(masks, MaskFusionEvidence) or type(masks.frames) is not tuple:
        raise ValueError("hybrid masks must be MaskFusionEvidence")
    if tuple(mask.frame for mask in masks.frames) != frames:
        raise ValueError("hybrid masks must preserve exact canonical frame joins")
    roots: list[Path] = []
    for frame, mask in zip(frames, masks.frames, strict=True):
        parts = _safe_image_parts(frame.image_name)
        expected_parts = (*parts[:-1], f"{parts[-1]}.png")
        path = mask.colmap_keep_path
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or tuple(path.parts[-len(expected_parts) :]) != expected_parts
            or _sha256_path(path) != mask.colmap_keep_sha256
        ):
            raise ValueError(
                "COLMAP keep mask must be an exact canonical .png artifact"
            )
        root = path
        for _part in expected_parts:
            root = root.parent
        roots.append(root)
    if len(set(roots)) != 1:
        raise ValueError("COLMAP keep masks must share one exact mask root")
    return roots[0]


def _anchor_cameras_for_frames(
    frames: tuple[FrameArtifact, ...],
    anchors: AnchorInferenceResult,
) -> tuple[CameraRecord, ...]:
    if not isinstance(anchors, AnchorInferenceResult):
        raise ValueError("anchors must be AnchorInferenceResult")
    if (
        type(anchors.anchor_indices) is not tuple
        or not anchors.anchor_indices
        or tuple(sorted(set(anchors.anchor_indices))) != anchors.anchor_indices
        or any(index < 0 or index >= len(frames) for index in anchors.anchor_indices)
        or type(anchors.cameras) is not tuple
        or len(anchors.cameras) != len(anchors.anchor_indices)
        or len(anchors.artifacts) != len(anchors.anchor_indices)
    ):
        raise ValueError("anchor indices, cameras, and artifacts must join exactly")
    result: list[CameraRecord] = []
    for position, frame_index in enumerate(anchors.anchor_indices):
        frame = frames[frame_index]
        camera = anchors.cameras[position]
        artifact = anchors.artifacts[position]
        if (
            not isinstance(camera, CameraRecord)
            or camera.frame_id != frame.frame_id
            or camera.image_name != frame.image_name
            or artifact.frame_id != frame.frame_id
            or artifact.image_name != frame.image_name
            or artifact.source_path != frame.path
        ):
            raise ValueError("DA3 anchor cameras must preserve exact frame joins")
        matrix = np.asarray(camera.w2c, dtype=np.float64)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], np.array((0.0, 0.0, 0.0, 1.0)))
            or not np.allclose(
                matrix[:3, :3] @ matrix[:3, :3].T,
                np.eye(3),
                rtol=0.0,
                atol=1e-6,
            )
            or not math.isclose(
                float(np.linalg.det(matrix[:3, :3])),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("DA3 anchor camera must be a proper finite OpenCV W2C")
        result.append(camera)
    return tuple(result)


def _feature_camera(
    frames: tuple[FrameArtifact, ...],
    processed: PinholeCamera,
) -> PinholeCamera:
    if not isinstance(processed, PinholeCamera) or processed.model != "PINHOLE":
        raise ValueError("DA3 shared camera must use PINHOLE")
    values = (processed.fx, processed.fy, processed.cx, processed.cy)
    if (
        type(processed.width) is not int
        or type(processed.height) is not int
        or processed.width <= 0
        or processed.height <= 0
        or any(not math.isfinite(float(value)) for value in values)
        or processed.fx <= 0.0
        or processed.fy <= 0.0
        or not (0.0 <= processed.cx < processed.width)
        or not (0.0 <= processed.cy < processed.height)
    ):
        raise ValueError("DA3 shared PINHOLE camera is malformed")
    sizes: list[tuple[int, int]] = []
    for frame in frames:
        try:
            with Image.open(frame.path) as image:
                width, height = image.size
                image.verify()
        except Exception as error:
            raise ValueError("hybrid feature image must be readable") from error
        sizes.append((int(width), int(height)))
    if len(set(sizes)) != 1:
        raise ValueError("hybrid shared PINHOLE requires one exact image size")
    width, height = sizes[0]
    scale_x = width / processed.width
    scale_y = height / processed.height
    return PinholeCamera(
        model="PINHOLE",
        width=width,
        height=height,
        fx=float(processed.fx) * scale_x,
        fy=float(processed.fy) * scale_y,
        cx=float(processed.cx) * scale_x,
        cy=float(processed.cy) * scale_y,
    )


def make_hybrid_candidate_runner(
    inputs: HybridGeometryInputs,
    *,
    use_gpu: bool,
    backend: HybridGeometryBackend | None = None,
) -> CandidateRunner:
    if not isinstance(inputs, HybridGeometryInputs):
        raise ValueError("inputs must be HybridGeometryInputs")
    if type(use_gpu) is not bool:
        raise ValueError("use_gpu must be a plain boolean")
    active_backend: HybridGeometryBackend = backend or PycolmapHybridBackend()
    version = getattr(active_backend, "version", None)
    if not isinstance(version, str) or not version:
        raise ValueError("hybrid backend must report a non-empty version")

    def run(
        manifest: SelectionManifest,
        frames: tuple[FrameArtifact, ...],
        attempt_dir: Path,
        attempt_index: int,
        frame_set_digest: str,
    ) -> ColmapAttempt:
        expected_digest, frames_root = _validate_frame_set(manifest, frames)
        if frame_set_digest != expected_digest:
            raise ValueError("hybrid candidate received a mismatched frame digest")
        mask_root = _mask_root_for_frames(frames, inputs.masks)
        anchor_cameras = _anchor_cameras_for_frames(frames, inputs.anchors)
        feature_camera = _feature_camera(frames, inputs.anchors.shared_camera)
        if os.path.lexists(attempt_dir):
            raise FileExistsError(attempt_dir)
        attempt_dir.mkdir(parents=True)
        database_path = attempt_dir / "colmap.db"
        sparse_root = attempt_dir / "sparse"
        sparse_root.mkdir()
        model_dirs = active_backend.reconstruct(
            manifest=manifest,
            frames_root=frames_root,
            image_names=tuple(frame.image_name for frame in frames),
            mask_root=mask_root,
            database_path=database_path,
            sparse_root=sparse_root,
            shared_camera=feature_camera,
            anchor_cameras=anchor_cameras,
            use_gpu=use_gpu,
        )
        if not database_path.is_file() or database_path.is_symlink():
            raise ValueError("hybrid backend must create the exact fresh database")
        if type(model_dirs) is not tuple or not model_dirs:
            raise ValueError("hybrid backend must return final model directories")
        if any(
            not isinstance(path, Path)
            or not path.is_dir()
            or path.is_symlink()
            or not path.is_relative_to(sparse_root)
            for path in model_dirs
        ):
            raise ValueError("hybrid final models must stay inside sparse_root")
        fingerprint = hashlib.sha256(
            _strict_json_bytes(
                {
                    "attempt_index": attempt_index,
                    "backend_version": version,
                    "frame_set_digest": frame_set_digest,
                    "mask_set_digest": inputs.masks.mask_set_digest,
                    "schema": "learned_quality.hybrid_geometry.v1",
                }
            )
        ).hexdigest()
        return ColmapAttempt(
            root=attempt_dir,
            database_path=database_path,
            model_dirs=model_dirs,
            colmap_version=version,
            fingerprint=fingerprint,
        )

    return run


def _rotation_matrix_to_qvec(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        first = (axis + 1) % 3
        second = (axis + 2) % 3
        scale = (
            math.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[axis, axis]
                    - matrix[first, first]
                    - matrix[second, second],
                )
            )
            * 2.0
        )
        if scale <= 0.0:
            raise ValueError("rotation cannot be converted to a stable quaternion")
        xyz = np.zeros(3, dtype=np.float64)
        xyz[axis] = 0.25 * scale
        xyz[first] = (matrix[first, axis] + matrix[axis, first]) / scale
        xyz[second] = (matrix[second, axis] + matrix[axis, second]) / scale
        quaternion = np.array(
            (
                (matrix[second, first] - matrix[first, second]) / scale,
                *xyz,
            ),
            dtype=np.float64,
        )
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _write_known_pose_model(
    root: Path,
    *,
    camera_id: int,
    shared_camera: PinholeCamera,
    image_ids: dict[str, int],
    anchor_cameras: tuple[CameraRecord, ...],
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    camera_line = (
        f"{camera_id} PINHOLE {shared_camera.width} {shared_camera.height} "
        f"{shared_camera.fx:.17g} {shared_camera.fy:.17g} "
        f"{shared_camera.cx:.17g} {shared_camera.cy:.17g}\n"
    )
    (root / "cameras.txt").write_text(camera_line, encoding="utf-8", newline="\n")
    image_lines: list[str] = []
    for camera in anchor_cameras:
        matrix = np.asarray(camera.w2c, dtype=np.float64)
        qvec = _rotation_matrix_to_qvec(matrix[:3, :3])
        translation = tuple(float(value) for value in matrix[:3, 3])
        image_lines.append(
            " ".join(
                (
                    str(image_ids[camera.image_name]),
                    *(f"{value:.17g}" for value in qvec),
                    *(f"{value:.17g}" for value in translation),
                    str(camera_id),
                    camera.image_name,
                )
            )
        )
        image_lines.append("")
    (root / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "points3D.txt").write_text("", encoding="utf-8", newline="\n")


class PycolmapHybridBackend:
    def __init__(self, pycolmap_module: object | None = None) -> None:
        if pycolmap_module is None:
            import pycolmap as pycolmap_module

        self._pycolmap = pycolmap_module
        raw_version = getattr(pycolmap_module, "__version__", None)
        if not isinstance(raw_version, str) or not raw_version:
            raise ValueError("PyCOLMAP must report a non-empty version")
        self.version = f"PyCOLMAP {raw_version}"

    def reconstruct(
        self,
        *,
        manifest: SelectionManifest,
        frames_root: Path,
        image_names: tuple[str, ...],
        mask_root: Path,
        database_path: Path,
        sparse_root: Path,
        shared_camera: PinholeCamera,
        anchor_cameras: tuple[CameraRecord, ...],
        use_gpu: bool,
    ) -> tuple[Path, ...]:
        pycolmap = self._pycolmap
        device = pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu
        # The pinned wheel can expose CUDA enums without CUDA-compiled solvers.
        # Device.auto falls back safely; the BA boolean has no such fallback.
        bundle_adjustment_use_gpu = False
        reader_options = pycolmap.ImageReaderOptions(
            mask_path=str(mask_root),
            camera_model="PINHOLE",
            camera_params=(
                f"{shared_camera.fx:.17g},{shared_camera.fy:.17g},"
                f"{shared_camera.cx:.17g},{shared_camera.cy:.17g}"
            ),
        )
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(frames_root),
            image_names=list(image_names),
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE",
            reader_options=reader_options,
            device=device,
        )
        pycolmap.match_exhaustive(
            database_path=str(database_path),
            device=device,
        )
        database = pycolmap.Database(str(database_path))
        try:
            database_images = tuple(database.read_all_images())
            image_ids = {image.name: int(image.image_id) for image in database_images}
            if set(image_ids) != set(image_names) or len(image_ids) != len(image_names):
                raise ValueError(
                    "PyCOLMAP database image IDs must join exact filenames"
                )
            camera_ids = {int(image.camera_id) for image in database_images}
            if len(camera_ids) != 1:
                raise ValueError("hybrid geometry requires one shared PINHOLE camera")
            camera_id = next(iter(camera_ids))
            database_camera = database.read_camera(camera_id)
            database_camera.params = np.asarray(
                (
                    shared_camera.fx,
                    shared_camera.fy,
                    shared_camera.cx,
                    shared_camera.cy,
                ),
                dtype=np.float64,
            )
            database.update_camera(database_camera)
        finally:
            database.close()

        known_root = sparse_root / "known"
        _write_known_pose_model(
            known_root,
            camera_id=camera_id,
            shared_camera=shared_camera,
            image_ids=image_ids,
            anchor_cameras=anchor_cameras,
        )
        known = pycolmap.Reconstruction(str(known_root))
        fixed_binary = sparse_root / "fixed-binary"
        fixed = pycolmap.triangulate_points(
            known,
            str(database_path),
            str(frames_root),
            str(fixed_binary),
            clear_points=True,
            refine_intrinsics=False,
        )
        fixed_options = pycolmap.BundleAdjustmentOptions(
            refine_focal_length=False,
            refine_principal_point=False,
            refine_extra_params=False,
            print_summary=False,
            use_gpu=bundle_adjustment_use_gpu,
        )
        pycolmap.bundle_adjustment(fixed, fixed_options)
        fixed_text = sparse_root / "fixed"
        fixed_text.mkdir()
        fixed.write_text(str(fixed_text))

        chosen = fixed_text
        try:
            focal = pycolmap.Reconstruction(str(fixed_text))
            focal_options = pycolmap.BundleAdjustmentOptions(
                refine_focal_length=True,
                refine_principal_point=False,
                refine_extra_params=False,
                refine_rig_from_world=False,
                refine_sensor_from_rig=False,
                print_summary=False,
                use_gpu=bundle_adjustment_use_gpu,
            )
            pycolmap.bundle_adjustment(focal, focal_options)
            focal_text = sparse_root / "focal"
            focal_text.mkdir()
            focal.write_text(str(focal_text))
            fixed_metrics = measure_models((fixed_text,), manifest)[0]
            focal_metrics = measure_models((focal_text,), manifest)[0]
            focal_camera = next(iter(focal.cameras.values()))
            refined_focal = tuple(float(value) for value in focal_camera.params[:2])
            if focal_refinement_is_safe(
                fixed_metrics,
                focal_metrics,
                initial_focal_xy=(shared_camera.fx, shared_camera.fy),
                refined_focal_xy=refined_focal,
            ):
                chosen = focal_text
        except (ArithmeticError, OSError, RuntimeError, ValueError):
            chosen = fixed_text

        if len(anchor_cameras) == len(image_names):
            final_root = sparse_root / "final" / "0"
            final_root.mkdir(parents=True)
            pycolmap.Reconstruction(str(chosen)).write_text(str(final_root))
            return (final_root,)

        mapping_root = sparse_root / "registered-binary"
        pipeline_options = pycolmap.IncrementalPipelineOptions(
            multiple_models=False,
        )
        for options in (
            pipeline_options.get_local_bundle_adjustment(),
            pipeline_options.get_global_bundle_adjustment(),
        ):
            options.refine_focal_length = False
            options.refine_principal_point = False
            options.refine_extra_params = False
            options.use_gpu = bundle_adjustment_use_gpu
        mapped = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(frames_root),
            output_path=str(mapping_root),
            options=pipeline_options,
            input_path=str(chosen),
        )
        if not isinstance(mapped, dict) or not mapped:
            raise RuntimeError("PyCOLMAP failed to register remaining hybrid frames")
        final_dirs: list[Path] = []
        for index, reconstruction in enumerate(mapped.values()):
            final_root = sparse_root / "final" / str(index)
            final_root.mkdir(parents=True)
            reconstruction.write_text(str(final_root))
            final_dirs.append(final_root)
        return tuple(final_dirs)


def _raise_all_failed(
    candidates: tuple[GeometryCandidateReport, ...],
) -> None:
    failed = tuple(
        candidate for candidate in candidates if not candidate.decision.passed
    )
    if not failed:
        raise AssertionError("all-failed path requires failed candidates")
    closest = min(failed, key=closest_failure_key)
    raise GeometryComparisonError(candidates, closest)


def run_geometry_comparison(
    manifest: SelectionManifest,
    frames: tuple[FrameArtifact, ...],
    *,
    output_root: Path,
    artifacts: LearnedArtifacts,
    classical_runner: CandidateRunner,
    hybrid_runner: CandidateRunner,
    materialize_backfill: MaterializeBackfill,
    measure_models_fn: MeasureModels = measure_models,
    evaluate_fn: EvaluateCandidate = evaluate_reconstruction,
    publish_model_fn: PublishModel = _publish_validated_model,
) -> LearnedReconstructionOutput:
    if not isinstance(artifacts, LearnedArtifacts):
        raise ValueError("artifacts must be LearnedArtifacts")
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("output_root must be an absolute Path")
    if output_root.resolve(strict=False) != output_root:
        raise ValueError("output_root must be canonical")
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise ValueError("output_root parent must be an existing regular directory")
    frame_set_digest, frames_root = _validate_frame_set(manifest, frames)
    output_root.mkdir()

    current_manifest = manifest
    current_frames = frames
    current_frames_root = frames_root
    candidates: list[GeometryCandidateReport] = []
    runners: tuple[tuple[CandidateId, CandidateRunner], ...] = (
        ("classical", classical_runner),
        ("learned_hybrid", hybrid_runner),
    )
    for attempt_index in range(2):
        round_candidates = tuple(
            _candidate_report(
                candidate_id,
                runner,
                current_manifest,
                current_frames,
                output_root / f"round-{attempt_index}" / candidate_id,
                attempt_index,
                frame_set_digest,
                measure_models_fn,
                evaluate_fn,
            )
            for candidate_id, runner in runners
        )
        candidates.extend(round_candidates)
        passing = tuple(
            candidate for candidate in round_candidates if candidate.decision.passed
        )
        if passing:
            winner = min(passing, key=winner_key)
            accepted_model = publish_model_fn(
                winner.model_dir,
                output_root / "validated",
            )
            if not isinstance(accepted_model, Path) or not accepted_model.is_absolute():
                raise ValueError("published winner model must be an absolute Path")
            all_candidates = tuple(candidates)
            bundle = ReconstructionBundle(
                selected_manifest=current_manifest,
                accepted_model_dir=accepted_model,
                decision=winner.decision,
                attempts=tuple(candidate.attempt for candidate in all_candidates),
                decisions=tuple(candidate.decision for candidate in all_candidates),
            )
            return LearnedReconstructionOutput(
                bundle=bundle,
                frames_dir=current_frames_root,
                artifacts=artifacts,
                geometry_candidates=all_candidates,
            )
        retry_candidates = tuple(
            candidate
            for candidate in round_candidates
            if candidate.decision.retry_recommended
        )
        if attempt_index == 1 or not retry_candidates:
            _raise_all_failed(tuple(candidates))
        closest = min(retry_candidates, key=closest_failure_key)
        backfilled = materialize_backfill(
            current_manifest,
            closest.decision.uncovered_intervals,
            max_per_interval=2,
        )
        if (
            type(backfilled) is not tuple
            or len(backfilled) != 2
            or not isinstance(backfilled[0], SelectionManifest)
            or type(backfilled[1]) is not tuple
        ):
            raise ValueError(
                "materialize_backfill must return manifest and frame tuple"
            )
        next_manifest, next_frames = backfilled
        frame_set_digest, current_frames_root = _validate_backfill(
            current_manifest,
            current_frames,
            next_manifest,
            next_frames,
        )
        current_manifest = next_manifest
        current_frames = next_frames
    raise AssertionError("bounded geometry comparison exhausted")
