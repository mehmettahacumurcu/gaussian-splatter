from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np

from backend.static_pipeline.stage_cache import promote_directory

from .lifecycle import (
    BatchAttemptRecord,
    VramReleaseRecord,
    _is_cuda_oom,
    run_with_smaller_batch_retry,
)


ANCHOR_PROCESS_RESOLUTION = 504
ANCHOR_BUDGET_BELOW_70_GIB = 96
ANCHOR_BUDGET_AT_LEAST_70_GIB = 120
MAX_CHUNK_FRAMES = 48
CHUNK_STRIDE = 24
_MAX_ABSOLUTE_CAMERA_VALUE = 1.0e9
_ROTATION_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class FrameChunk:
    index: int
    start: int
    stop: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class PinholeCamera:
    model: Literal["PINHOLE"]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        return (
            (self.fx, 0.0, self.cx),
            (0.0, self.fy, self.cy),
            (0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class DA3Frame:
    image_name: str
    frame_id: str
    path: Path


@dataclass(frozen=True)
class FramePredictionArtifact:
    image_name: str
    frame_id: str
    source_path: Path
    depth_path: Path
    confidence_path: Path | None
    sky_path: Path | None


@dataclass(frozen=True)
class CameraRecord:
    image_name: str
    frame_id: str
    w2c: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class AnchorInferenceResult:
    artifacts: tuple[FramePredictionArtifact, ...]
    cameras: tuple[CameraRecord, ...]
    shared_camera: PinholeCamera
    anchor_indices: tuple[int, ...]
    attempts: tuple[BatchAttemptRecord, ...]
    metadata_path: Path


@dataclass(frozen=True)
class ChunkProvenance:
    index: int
    start: int
    stop: int
    image_names: tuple[str, ...]
    frame_ids: tuple[str, ...]
    requested_size: int


@dataclass(frozen=True)
class MetricSkyResult:
    artifacts: tuple[FramePredictionArtifact, ...]
    processed_camera: PinholeCamera
    chunks: tuple[ChunkProvenance, ...]
    attempts: tuple[BatchAttemptRecord, ...]
    release_record: VramReleaseRecord | None
    retry_model_release_record: VramReleaseRecord | None
    metadata_path: Path


@dataclass(frozen=True)
class FinalPoseCamera:
    image_name: str
    frame_id: str
    width: int
    height: int
    w2c: tuple[tuple[float, float, float, float], ...]
    intrinsics: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class FrameContribution:
    image_name: str
    frame_id: str
    chunk_indices: tuple[int, ...]


@dataclass(frozen=True)
class PoseConditionedDepthResult:
    artifacts: tuple[FramePredictionArtifact, ...]
    cameras: tuple[FinalPoseCamera, ...]
    processed_camera: PinholeCamera
    chunks: tuple[ChunkProvenance, ...]
    contributions: tuple[FrameContribution, ...]
    attempts: tuple[BatchAttemptRecord, ...]
    metadata_path: Path


class AnchorStageFailure(RuntimeError):
    def __init__(self, locked_anchor_count: int, process_resolution: int) -> None:
        self.locked_anchor_count = locked_anchor_count
        self.process_resolution = process_resolution
        super().__init__(
            "DA3 anchor inference exhausted CUDA memory at the locked "
            f"{locked_anchor_count} anchors and {process_resolution}px resolution"
        )


class FinalPoseStageFailure(RuntimeError):
    def __init__(
        self,
        *,
        chunk_index: int,
        chunk_start: int,
        chunk_stop: int,
        actual_chunk_size: int,
    ) -> None:
        self.chunk_index = chunk_index
        self.chunk_start = chunk_start
        self.chunk_stop = chunk_stop
        self.actual_chunk_size = actual_chunk_size
        self.locked_context_size = MAX_CHUNK_FRAMES
        self.process_resolution = ANCHOR_PROCESS_RESOLUTION
        super().__init__(
            "DA3 final-pose depth exhausted CUDA memory for locked context "
            f"chunk {chunk_index} [{chunk_start}:{chunk_stop}] at "
            f"{ANCHOR_PROCESS_RESOLUTION}px; smaller context is forbidden"
        )


@dataclass(frozen=True)
class _ValidatedPrediction:
    depth: np.ndarray
    confidence: np.ndarray | None
    sky: np.ndarray | None
    cameras: tuple[np.ndarray, ...]
    shared_camera: PinholeCamera | None


@dataclass(frozen=True)
class _MetricOperationResult:
    prediction: _ValidatedPrediction
    chunks: tuple[ChunkProvenance, ...]


def select_anchor_indices(frame_count: int, vram_gb: float) -> tuple[int, ...]:
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive plain integer")
    if (
        isinstance(vram_gb, bool)
        or not isinstance(vram_gb, Real)
        or not math.isfinite(float(vram_gb))
        or float(vram_gb) < 0.0
    ):
        raise ValueError("vram_gb must be finite and nonnegative")

    budget = (
        ANCHOR_BUDGET_AT_LEAST_70_GIB
        if float(vram_gb) >= 70.0
        else ANCHOR_BUDGET_BELOW_70_GIB
    )
    selected_count = min(frame_count, budget)
    if selected_count == frame_count:
        return tuple(range(frame_count))
    denominator = selected_count - 1
    return tuple(
        (index * (frame_count - 1) + denominator // 2) // denominator
        for index in range(selected_count)
    )


def normalize_w2c(matrix: object) -> np.ndarray:
    try:
        source = np.asarray(matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("W2C must be a finite numeric 3x4 or 4x4 matrix") from error
    if source.shape not in {(3, 4), (4, 4)} or source.dtype.kind not in "iuf":
        raise ValueError("W2C must be a finite numeric 3x4 or 4x4 matrix")
    values = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("W2C values must be finite")
    if np.max(np.abs(values)) > _MAX_ABSOLUTE_CAMERA_VALUE:
        raise ValueError("W2C values are implausibly large")
    if values.shape == (3, 4):
        values = np.vstack((values, np.array((0.0, 0.0, 0.0, 1.0))))
    elif not np.array_equal(values[3], np.array((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("W2C must have the exact affine bottom row")

    rotation = values[:3, :3]
    if not np.allclose(
        rotation @ rotation.T,
        np.eye(3),
        rtol=0.0,
        atol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("W2C rotation must be orthonormal without scale or shear")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not math.isclose(
        determinant,
        1.0,
        rel_tol=0.0,
        abs_tol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("W2C rotation must be proper and non-reflecting")
    values.setflags(write=False)
    return values


def _validate_image_size(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError("image_sizes must contain positive integer width/height pairs")
    return value[0], value[1]


def _shared_image_size(image_sizes: object, frame_count: int) -> tuple[int, int]:
    if not isinstance(image_sizes, (tuple, list)):
        raise ValueError("image_sizes must be one size or one size per frame")
    if len(image_sizes) == 2 and all(type(item) is int for item in image_sizes):
        return _validate_image_size(image_sizes)
    if len(image_sizes) != frame_count:
        raise ValueError("image_sizes count must match intrinsics")
    sizes = tuple(_validate_image_size(value) for value in image_sizes)
    if len(set(sizes)) != 1:
        raise ValueError("image_sizes must agree for one video")
    return sizes[0]


def _has_coherent_majority(values: np.ndarray, tolerance: float) -> bool:
    center = float(np.median(values))
    required = 1 if len(values) == 1 else math.ceil(2 * len(values) / 3)
    return int(np.count_nonzero(np.abs(values - center) <= tolerance)) >= required


def robust_shared_pinhole(
    intrinsics: object,
    image_sizes: object,
) -> PinholeCamera:
    try:
        source = np.asarray(intrinsics)
    except (TypeError, ValueError) as error:
        raise ValueError("intrinsics must be finite numeric N x 3 x 3 matrices") from error
    if (
        source.ndim != 3
        or source.shape[0] == 0
        or source.shape[1:] != (3, 3)
        or source.dtype.kind not in "iuf"
    ):
        raise ValueError("intrinsics must be finite numeric N x 3 x 3 matrices")
    values = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("intrinsics values must be finite")
    if np.max(np.abs(values)) > _MAX_ABSOLUTE_CAMERA_VALUE:
        raise ValueError("intrinsics values are implausibly large")

    width, height = _shared_image_size(image_sizes, len(values))
    canonical_last_row = np.array((0.0, 0.0, 1.0))
    if not all(np.array_equal(matrix[2], canonical_last_row) for matrix in values):
        raise ValueError("intrinsics must have the exact canonical last row")
    if np.any(values[:, 0, 1] != 0.0) or np.any(values[:, 1, 0] != 0.0):
        raise ValueError("intrinsics must have zero skew")

    fx = values[:, 0, 0]
    fy = values[:, 1, 1]
    cx = values[:, 0, 2]
    cy = values[:, 1, 2]
    if np.any(fx <= 0.0) or np.any(fy <= 0.0):
        raise ValueError("intrinsics focal lengths must be positive")
    if np.any(cx < 0.0) or np.any(cx >= width) or np.any(cy < 0.0) or np.any(cy >= height):
        raise ValueError("intrinsics principal points must be in image bounds")

    median_fx = float(np.median(fx))
    median_fy = float(np.median(fy))
    if not _has_coherent_majority(fx, 0.25 * median_fx) or not _has_coherent_majority(
        fy,
        0.25 * median_fy,
    ):
        raise ValueError("intrinsics focal estimates severely disagree")
    if not _has_coherent_majority(cx, 0.1 * width) or not _has_coherent_majority(
        cy,
        0.1 * height,
    ):
        raise ValueError("intrinsics principal-point estimates severely disagree")
    return PinholeCamera(
        model="PINHOLE",
        width=width,
        height=height,
        fx=median_fx,
        fy=median_fy,
        cx=float(np.median(cx)),
        cy=float(np.median(cy)),
    )


def _safe_frame_component(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", ":"))
        or any(unicodedata.category(character) == "Cc" for character in value)
        or Path(value).is_absolute()
    ):
        raise ValueError(f"frame {label} must be a safe path component")
    return value


def _artifact_token(frame: DA3Frame) -> str:
    return (
        f"{quote(frame.frame_id, safe='-_.')}--"
        f"{quote(frame.image_name, safe='-_.')}"
    )


def _validated_frames(frames: Sequence[DA3Frame]) -> tuple[DA3Frame, ...]:
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames must be a non-empty sequence of DA3Frame records")
    try:
        values = tuple(frames)
    except TypeError as error:
        raise ValueError("frames must be a non-empty sequence of DA3Frame records") from error
    if not values:
        raise ValueError("frames must be a non-empty sequence of DA3Frame records")
    image_names: set[str] = set()
    frame_ids: set[str] = set()
    source_paths: set[str] = set()
    artifact_tokens: set[str] = set()
    for frame in values:
        if not isinstance(frame, DA3Frame):
            raise ValueError("frames must contain only DA3Frame records")
        image_name = _safe_frame_component(frame.image_name, "image_name")
        frame_id = _safe_frame_component(frame.frame_id, "frame_id")
        if not isinstance(frame.path, Path) or not frame.path.is_absolute():
            raise ValueError("frame path must be absolute")
        if (
            not os.path.lexists(frame.path)
            or frame.path.is_symlink()
            or not frame.path.is_file()
        ):
            raise ValueError("frame path must be a regular non-symlink file")
        if frame.path.name != image_name:
            raise ValueError("frame image_name must exactly match the source basename")
        resolved_path = str(frame.path.resolve(strict=True)).casefold()
        token = _artifact_token(frame).casefold()
        if image_name.casefold() in image_names:
            raise ValueError("frame image names must be unique")
        if frame_id.casefold() in frame_ids:
            raise ValueError("frame ids must be unique")
        if resolved_path in source_paths:
            raise ValueError("frame source paths must be unique")
        if token in artifact_tokens:
            raise ValueError("frame artifact names collide")
        image_names.add(image_name.casefold())
        frame_ids.add(frame_id.casefold())
        source_paths.add(resolved_path)
        artifact_tokens.add(token)
    return values


def _prepare_output_directory(output_dir: Path) -> None:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output directory must be absolute")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output directory parent must be a plain directory")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"output directory already exists: {output_dir}")


def _required_prediction_attribute(prediction: object, name: str) -> object:
    try:
        value = getattr(prediction, name)
    except AttributeError as error:
        raise ValueError(f"prediction is missing required field {name!r}") from error
    if value is None:
        raise ValueError(f"prediction field {name!r} cannot be None")
    return value


def _optional_prediction_attribute(prediction: object, name: str) -> object | None:
    try:
        return getattr(prediction, name)
    except AttributeError:
        return None


def _prediction_array(
    value: object,
    name: str,
    *,
    allow_boolean: bool = False,
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"prediction {name} must be a numeric array") from error
    allowed_kinds = "biuf" if allow_boolean else "iuf"
    if source.dtype.kind not in allowed_kinds:
        raise ValueError(f"prediction {name} must be a real numeric array")
    values = np.array(source, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"prediction {name} must contain only finite values")
    return values


def _finite_float32(
    values: object,
    label: str,
    *,
    positive_mask: object | None = None,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{label} must be representable as finite float32")
    if positive_mask is not None:
        mask = np.asarray(positive_mask, dtype=bool)
        if mask.shape != converted.shape:
            raise AssertionError("float32 positive mask must match converted values")
        if np.any(converted[mask] <= 0.0):
            raise ValueError(f"{label} must preserve positive values in float32")
    return converted


def _validated_prediction(
    prediction: object,
    expected_count: int,
    *,
    require_cameras: bool,
) -> _ValidatedPrediction:
    depth = _prediction_array(
        _required_prediction_attribute(prediction, "depth"),
        "depth",
    )
    if depth.ndim != 3 or depth.shape[0] != expected_count or min(depth.shape[1:]) <= 0:
        raise ValueError("prediction depth must have exact shape N x H x W")

    confidence_value = _optional_prediction_attribute(prediction, "conf")
    confidence: np.ndarray | None = None
    if confidence_value is not None:
        confidence = _prediction_array(confidence_value, "conf")
        if confidence.shape != depth.shape:
            raise ValueError("prediction conf shape must exactly match depth")
        if np.any(confidence < 0.0):
            raise ValueError("prediction conf must be nonnegative")

    if np.any(depth < 0.0):
        raise ValueError("prediction depth cannot be negative")
    valid = np.ones(depth.shape, dtype=bool) if confidence is None else confidence > 0.0
    if np.any(depth[valid] <= 0.0):
        raise ValueError("prediction depth must be positive where valid")

    sky_value = _optional_prediction_attribute(prediction, "sky")
    sky: np.ndarray | None = None
    if sky_value is not None:
        sky_source = np.asarray(sky_value)
        if sky_source.dtype.kind != "b":
            raise ValueError("prediction sky must be a thresholded boolean mask")
        sky = np.array(sky_source, dtype=bool, copy=True)
        if sky.shape != depth.shape:
            raise ValueError("prediction sky shape must exactly match depth")

    cameras: tuple[np.ndarray, ...] = ()
    shared_camera: PinholeCamera | None = None
    if require_cameras:
        extrinsics = _prediction_array(
            _required_prediction_attribute(prediction, "extrinsics"),
            "extrinsics",
        )
        if extrinsics.shape not in {
            (expected_count, 3, 4),
            (expected_count, 4, 4),
        }:
            raise ValueError("prediction extrinsics must have exact shape N x 3 x 4 or N x 4 x 4")
        try:
            cameras = tuple(normalize_w2c(matrix) for matrix in extrinsics)
        except ValueError as error:
            raise ValueError(f"prediction extrinsics are invalid: {error}") from error

        intrinsics = _required_prediction_attribute(prediction, "intrinsics")
        try:
            shared_camera = robust_shared_pinhole(
                intrinsics,
                (depth.shape[2], depth.shape[1]),
            )
        except ValueError as error:
            raise ValueError(f"prediction intrinsics are invalid: {error}") from error
        try:
            intrinsic_count = np.asarray(intrinsics).shape[0]
        except (AttributeError, IndexError, TypeError):
            intrinsic_count = -1
        if intrinsic_count != expected_count:
            raise ValueError("prediction intrinsics count must exactly match frames")

    depth_output = _finite_float32(
        depth,
        "prediction depth",
        positive_mask=valid,
    )
    confidence_output = (
        None
        if confidence is None
        else _finite_float32(
            confidence,
            "prediction conf",
            positive_mask=confidence > 0.0,
        )
    )
    if sky is None:
        sky_output = None
    elif sky.dtype.kind == "b":
        sky_output = np.ascontiguousarray(sky, dtype=bool)
    else:
        sky_output = np.ascontiguousarray(sky, dtype=np.float32)
    return _ValidatedPrediction(
        depth=depth_output,
        confidence=confidence_output,
        sky=sky_output,
        cameras=cameras,
        shared_camera=shared_camera,
    )


def _matrix_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _attempt_payload(attempt: BatchAttemptRecord) -> dict[str, object]:
    return {
        "error_message": attempt.error_message,
        "error_type": attempt.error_type,
        "outcome": attempt.outcome,
        "size": attempt.size,
    }


def _camera_payload(camera: PinholeCamera) -> dict[str, object]:
    return {
        "cx": camera.cx,
        "cy": camera.cy,
        "fx": camera.fx,
        "fy": camera.fy,
        "height": camera.height,
        "model": camera.model,
        "width": camera.width,
    }


def _release_payload(record: VramReleaseRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "allocated_after_bytes": record.allocated_after_bytes,
        "allocated_before_bytes": record.allocated_before_bytes,
        "cache_cleared": record.cache_cleared,
        "cuda_available": record.cuda_available,
        "gc_ran": record.gc_ran,
        "moved_to_cpu": record.moved_to_cpu,
        "reserved_after_bytes": record.reserved_after_bytes,
        "reserved_before_bytes": record.reserved_before_bytes,
    }


def _chunk_payload(chunk: ChunkProvenance) -> dict[str, object]:
    return {
        "frame_ids": chunk.frame_ids,
        "image_names": chunk.image_names,
        "index": chunk.index,
        "requested_size": chunk.requested_size,
        "start": chunk.start,
        "stop": chunk.stop,
    }


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _publish_anchor_artifacts(
    frames: tuple[DA3Frame, ...],
    prediction: _ValidatedPrediction,
    output_dir: Path,
    anchor_indices: tuple[int, ...],
    attempts: tuple[BatchAttemptRecord, ...],
    promote: Callable[[Path, Path], None],
) -> AnchorInferenceResult:
    if prediction.shared_camera is None:
        raise AssertionError("anchor prediction must include a shared camera")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    artifacts: list[FramePredictionArtifact] = []
    artifact_payload: list[dict[str, object]] = []
    cameras: list[CameraRecord] = []
    camera_payload: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        token = _artifact_token(frame)
        depth_name = f"{token}.depth.npy"
        confidence_name = (
            None if prediction.confidence is None else f"{token}.confidence.npy"
        )
        with (staging / depth_name).open("wb") as handle:
            np.save(handle, prediction.depth[index], allow_pickle=False)
        if confidence_name is not None:
            with (staging / confidence_name).open("wb") as handle:
                np.save(handle, prediction.confidence[index], allow_pickle=False)
        artifacts.append(
            FramePredictionArtifact(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                source_path=frame.path,
                depth_path=output_dir / depth_name,
                confidence_path=(
                    None if confidence_name is None else output_dir / confidence_name
                ),
                sky_path=None,
            )
        )
        artifact_payload.append(
            {
                "confidence_path": confidence_name,
                "depth_path": depth_name,
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sky_path": None,
                "source_path": str(frame.path),
            }
        )
        w2c = _matrix_tuple(prediction.cameras[index])
        cameras.append(
            CameraRecord(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                w2c=w2c,
            )
        )
        camera_payload.append(
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "w2c": w2c,
            }
        )
    metadata = {
        "anchor_indices": anchor_indices,
        "artifacts": artifact_payload,
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
        "cameras": camera_payload,
        "process_resolution": ANCHOR_PROCESS_RESOLUTION,
        "schema_version": 1,
        "shared_camera": _camera_payload(prediction.shared_camera),
        "stage": "da3_anchor",
    }
    (staging / "metadata.json").write_bytes(_strict_json_bytes(metadata))
    promote(staging, output_dir)
    return AnchorInferenceResult(
        artifacts=tuple(artifacts),
        cameras=tuple(cameras),
        shared_camera=prediction.shared_camera,
        anchor_indices=anchor_indices,
        attempts=attempts,
        metadata_path=output_dir / "metadata.json",
    )


def _is_injected_cuda_oom(
    error: BaseException,
    torch_module: object | None,
) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            return False
    return _is_cuda_oom(error, torch_module)


def run_anchor_inference(
    model: object,
    frames: Sequence[DA3Frame],
    output_dir: Path,
    *,
    vram_gb: float,
    torch_module: object | None = None,
    promote: Callable[[Path, Path], None] = promote_directory,
) -> AnchorInferenceResult:
    validated_frames = _validated_frames(frames)
    anchor_indices = select_anchor_indices(len(validated_frames), vram_gb)
    _prepare_output_directory(output_dir)
    inference = getattr(model, "inference", None)
    if not callable(inference):
        raise TypeError("model must expose a callable inference method")
    anchor_frames = tuple(validated_frames[index] for index in anchor_indices)
    try:
        raw_prediction = inference(
            image=[str(frame.path) for frame in anchor_frames],
            process_res=ANCHOR_PROCESS_RESOLUTION,
            process_res_method="upper_bound_resize",
            use_ray_pose=True,
            ref_view_strategy="middle",
        )
    except Exception as error:
        if _is_injected_cuda_oom(error, torch_module):
            raise AnchorStageFailure(
                locked_anchor_count=len(anchor_frames),
                process_resolution=ANCHOR_PROCESS_RESOLUTION,
            ) from error
        raise
    prediction = _validated_prediction(
        raw_prediction,
        len(anchor_frames),
        require_cameras=True,
    )
    attempts = (
        BatchAttemptRecord(size=len(anchor_frames), outcome="succeeded"),
    )
    return _publish_anchor_artifacts(
        anchor_frames,
        prediction,
        output_dir,
        anchor_indices,
        attempts,
        promote,
    )


def _validated_processed_camera(camera: PinholeCamera) -> PinholeCamera:
    if not isinstance(camera, PinholeCamera) or camera.model != "PINHOLE":
        raise ValueError("processed_camera must be a PINHOLE camera")
    try:
        validated = robust_shared_pinhole(
            np.asarray(camera.matrix, dtype=np.float64)[None],
            (camera.width, camera.height),
        )
    except ValueError as error:
        raise ValueError(f"processed_camera is invalid: {error}") from error
    if validated != camera:
        raise ValueError("processed_camera must contain canonical finite calibration")
    return camera


def _run_metric_batches(
    model: object,
    frames: tuple[DA3Frame, ...],
    processed_camera: PinholeCamera,
    batch_size: int,
) -> _MetricOperationResult:
    inference = getattr(model, "inference", None)
    if not callable(inference):
        raise TypeError("model must expose a callable inference method")
    predictions: list[_ValidatedPrediction] = []
    chunks: list[ChunkProvenance] = []
    confidence_presence: bool | None = None
    for chunk_index, start in enumerate(range(0, len(frames), batch_size)):
        stop = min(start + batch_size, len(frames))
        batch_frames = frames[start:stop]
        raw_prediction = inference(
            image=[str(frame.path) for frame in batch_frames],
            process_res=ANCHOR_PROCESS_RESOLUTION,
            process_res_method="upper_bound_resize",
        )
        try:
            prediction = _validated_prediction(
                raw_prediction,
                len(batch_frames),
                require_cameras=False,
            )
        except ValueError as error:
            raise ValueError(f"metric prediction is invalid: {error}") from error
        if prediction.sky is None:
            raise ValueError("metric prediction must include a boolean sky mask")
        if prediction.depth.shape[1:] != (
            processed_camera.height,
            processed_camera.width,
        ):
            raise ValueError(
                "metric prediction processed H/W must match processed_camera"
            )
        has_confidence = prediction.confidence is not None
        if confidence_presence is None:
            confidence_presence = has_confidence
        elif confidence_presence != has_confidence:
            raise ValueError(
                "metric prediction confidence presence must agree across batches"
            )
        predictions.append(prediction)
        chunks.append(
            ChunkProvenance(
                index=chunk_index,
                start=start,
                stop=stop,
                image_names=tuple(frame.image_name for frame in batch_frames),
                frame_ids=tuple(frame.frame_id for frame in batch_frames),
                requested_size=batch_size,
            )
        )
    depth_values = np.concatenate(
        [prediction.depth for prediction in predictions],
        axis=0,
    )
    confidence_values = (
        None
        if not confidence_presence
        else np.concatenate(
            [prediction.confidence for prediction in predictions],
            axis=0,
        )
    )
    valid = (
        np.ones(depth_values.shape, dtype=bool)
        if confidence_values is None
        else confidence_values > 0.0
    )
    depth = _finite_float32(
        depth_values,
        "metric batched depth",
        positive_mask=valid,
    )
    confidence = (
        None
        if confidence_values is None
        else _finite_float32(
            confidence_values,
            "metric batched confidence",
            positive_mask=confidence_values > 0.0,
        )
    )
    sky = np.ascontiguousarray(
        np.concatenate([prediction.sky for prediction in predictions], axis=0),
        dtype=bool,
    )
    return _MetricOperationResult(
        prediction=_ValidatedPrediction(
            depth=depth,
            confidence=confidence,
            sky=sky,
            cameras=(),
            shared_camera=None,
        ),
        chunks=tuple(chunks),
    )


def _publish_metric_artifacts(
    frames: tuple[DA3Frame, ...],
    operation: _MetricOperationResult,
    processed_camera: PinholeCamera,
    attempts: tuple[BatchAttemptRecord, ...],
    release_record: VramReleaseRecord | None,
    retry_model_release_record: VramReleaseRecord | None,
    output_dir: Path,
    promote: Callable[[Path, Path], None],
) -> MetricSkyResult:
    if operation.prediction.sky is None:
        raise AssertionError("metric prediction must include sky")
    focal_scale = ((processed_camera.fx + processed_camera.fy) * 0.5) / 300.0
    valid = (
        np.ones(operation.prediction.depth.shape, dtype=bool)
        if operation.prediction.confidence is None
        else operation.prediction.confidence > 0.0
    )
    metric_depth = _finite_float32(
        operation.prediction.depth.astype(np.float64) * focal_scale,
        "metric depth",
        positive_mask=valid,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    artifacts: list[FramePredictionArtifact] = []
    artifact_payload: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        token = _artifact_token(frame)
        depth_name = f"{token}.metric_depth.npy"
        confidence_name = (
            None
            if operation.prediction.confidence is None
            else f"{token}.confidence.npy"
        )
        sky_name = f"{token}.sky.npy"
        with (staging / depth_name).open("wb") as handle:
            np.save(handle, metric_depth[index], allow_pickle=False)
        if confidence_name is not None:
            with (staging / confidence_name).open("wb") as handle:
                np.save(
                    handle,
                    operation.prediction.confidence[index],
                    allow_pickle=False,
                )
        with (staging / sky_name).open("wb") as handle:
            np.save(handle, operation.prediction.sky[index], allow_pickle=False)
        artifacts.append(
            FramePredictionArtifact(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                source_path=frame.path,
                depth_path=output_dir / depth_name,
                confidence_path=(
                    None if confidence_name is None else output_dir / confidence_name
                ),
                sky_path=output_dir / sky_name,
            )
        )
        artifact_payload.append(
            {
                "confidence_path": confidence_name,
                "depth_path": depth_name,
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sky_path": sky_name,
                "source_path": str(frame.path),
            }
        )
    metadata = {
        "artifacts": artifact_payload,
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
        "chunks": [_chunk_payload(chunk) for chunk in operation.chunks],
        "depth": {
            "conversion": "mean_processed_focal_times_network_depth_div_300",
            "processed_height": processed_camera.height,
            "processed_width": processed_camera.width,
            "units": "meters",
        },
        "process_resolution": ANCHOR_PROCESS_RESOLUTION,
        "processed_camera": _camera_payload(processed_camera),
        "release_record": _release_payload(release_record),
        "retry_model_release_record": _release_payload(
            retry_model_release_record
        ),
        "schema_version": 1,
        "stage": "da3_metric_sky",
    }
    (staging / "metadata.json").write_bytes(_strict_json_bytes(metadata))
    promote(staging, output_dir)
    return MetricSkyResult(
        artifacts=tuple(artifacts),
        processed_camera=processed_camera,
        chunks=operation.chunks,
        attempts=attempts,
        release_record=release_record,
        retry_model_release_record=retry_model_release_record,
        metadata_path=output_dir / "metadata.json",
    )


def _release_metric_model(
    release_model: Callable[[object], VramReleaseRecord],
    model: object,
) -> VramReleaseRecord:
    record = release_model(model)
    if not isinstance(record, VramReleaseRecord):
        raise TypeError("release_model must return VramReleaseRecord")
    return record


def _attach_metric_retry_release(
    primary_error: BaseException,
    *,
    record: VramReleaseRecord | None = None,
    release_error: BaseException | None = None,
) -> None:
    try:
        if record is not None:
            setattr(primary_error, "metric_retry_model_release_record", record)
        if release_error is not None:
            setattr(primary_error, "metric_retry_model_release_error", release_error)
    except BaseException:
        try:
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                if record is not None:
                    add_note("metric retry replacement model was released")
                if release_error is not None:
                    add_note(
                        "metric retry replacement release also failed: "
                        f"{type(release_error).__name__}"
                    )
        except BaseException:
            pass


def run_metric_sky(
    model: object,
    frames: Sequence[DA3Frame],
    processed_camera: PinholeCamera,
    output_dir: Path,
    *,
    initial_batch_size: int,
    retry_batch_size: int,
    release_model: Callable[[object], VramReleaseRecord],
    retry_model_factory: Callable[[], object],
    torch_module: object | None = None,
    promote: Callable[[Path, Path], None] = promote_directory,
) -> MetricSkyResult:
    validated_frames = _validated_frames(frames)
    validated_camera = _validated_processed_camera(processed_camera)
    _prepare_output_directory(output_dir)
    if not callable(release_model) or not callable(retry_model_factory):
        raise TypeError("metric retry boundaries must be callable")
    active_model = model
    retry_model: object | None = None

    def operation(batch_size: int) -> _MetricOperationResult:
        return _run_metric_batches(
            active_model,
            validated_frames,
            validated_camera,
            batch_size,
        )

    def release_failed_model() -> VramReleaseRecord:
        nonlocal active_model, retry_model
        release_record = _release_metric_model(release_model, active_model)
        replacement_model = retry_model_factory()
        if replacement_model is active_model:
            reuse_error = ValueError(
                "retry_model_factory must return a new model object"
            )
            setattr(
                reuse_error,
                "metric_initial_model_release_record",
                release_record,
            )
            raise reuse_error
        retry_model = replacement_model
        active_model = replacement_model
        return release_record

    try:
        retry_result = run_with_smaller_batch_retry(
            "da3_metric_sky",
            initial_batch_size,
            retry_batch_size,
            operation,
            chunk_independent=True,
            release=release_failed_model,
            torch_module=torch_module,
        )
    except Exception as primary_error:
        if retry_model is not None:
            try:
                retry_release_record = _release_metric_model(
                    release_model,
                    retry_model,
                )
            except Exception as release_error:
                _attach_metric_retry_release(
                    primary_error,
                    release_error=release_error,
                )
            else:
                _attach_metric_retry_release(
                    primary_error,
                    record=retry_release_record,
                )
        raise
    retry_model_release_record = (
        None
        if retry_model is None
        else _release_metric_model(release_model, retry_model)
    )
    return _publish_metric_artifacts(
        validated_frames,
        retry_result.value,
        validated_camera,
        retry_result.attempts,
        retry_result.release_record,
        retry_model_release_record,
        output_dir,
        promote,
    )


def _matrix3_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _validated_final_cameras(
    frames: tuple[DA3Frame, ...],
    cameras: Sequence[FinalPoseCamera],
) -> tuple[FinalPoseCamera, ...]:
    if isinstance(cameras, (str, bytes)):
        raise ValueError("camera mapping must contain one record per frame")
    try:
        values = tuple(cameras)
    except TypeError as error:
        raise ValueError("camera mapping must contain one record per frame") from error
    if len(values) != len(frames):
        raise ValueError("camera mapping count must exactly match frames")
    result: list[FinalPoseCamera] = []
    sizes: set[tuple[int, int]] = set()
    for frame, camera in zip(frames, values):
        if not isinstance(camera, FinalPoseCamera):
            raise ValueError("camera mapping must contain FinalPoseCamera records")
        if camera.image_name != frame.image_name or camera.frame_id != frame.frame_id:
            raise ValueError("camera mapping names and ids must exactly match frames")
        try:
            w2c = normalize_w2c(camera.w2c)
        except ValueError as error:
            raise ValueError(f"camera mapping W2C is invalid: {error}") from error
        try:
            intrinsic_array = np.asarray(camera.intrinsics)
            validated_pinhole = robust_shared_pinhole(
                intrinsic_array[None],
                (camera.width, camera.height),
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"camera mapping intrinsics are invalid: {error}") from error
        intrinsic_values = np.array(intrinsic_array, dtype=np.float64, copy=True)
        if validated_pinhole.matrix != _matrix3_tuple(intrinsic_values):
            raise ValueError("camera mapping intrinsics must be canonical PINHOLE")
        sizes.add((camera.width, camera.height))
        result.append(
            FinalPoseCamera(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                width=camera.width,
                height=camera.height,
                w2c=_matrix_tuple(w2c),
                intrinsics=_matrix3_tuple(intrinsic_values),
            )
        )
    if len(sizes) != 1:
        raise ValueError("camera mapping image sizes must agree for one video")
    return tuple(result)


def _final_camera_payload(camera: FinalPoseCamera) -> dict[str, object]:
    return {
        "frame_id": camera.frame_id,
        "height": camera.height,
        "image_name": camera.image_name,
        "intrinsics": camera.intrinsics,
        "w2c": camera.w2c,
        "width": camera.width,
    }


def _matches_pinned_camera_roundtrip(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> bool:
    if np.allclose(predicted, expected, rtol=0.0, atol=1.0e-6):
        return True
    float32_expected = np.asarray(expected, dtype=np.float32).astype(np.float64)
    return np.array_equal(predicted, float32_expected)


def _publish_final_depth_artifacts(
    frames: tuple[DA3Frame, ...],
    cameras: tuple[FinalPoseCamera, ...],
    depth: np.ndarray,
    confidence: np.ndarray | None,
    processed_camera: PinholeCamera,
    chunks: tuple[ChunkProvenance, ...],
    contributions: tuple[FrameContribution, ...],
    attempts: tuple[BatchAttemptRecord, ...],
    output_dir: Path,
    promote: Callable[[Path, Path], None],
) -> PoseConditionedDepthResult:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    artifacts: list[FramePredictionArtifact] = []
    artifact_payload: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        token = _artifact_token(frame)
        depth_name = f"{token}.final_depth.npy"
        confidence_name = (
            None if confidence is None else f"{token}.confidence.npy"
        )
        with (staging / depth_name).open("wb") as handle:
            np.save(handle, depth[index], allow_pickle=False)
        if confidence_name is not None:
            with (staging / confidence_name).open("wb") as handle:
                np.save(handle, confidence[index], allow_pickle=False)
        artifacts.append(
            FramePredictionArtifact(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                source_path=frame.path,
                depth_path=output_dir / depth_name,
                confidence_path=(
                    None if confidence_name is None else output_dir / confidence_name
                ),
                sky_path=None,
            )
        )
        artifact_payload.append(
            {
                "confidence_path": confidence_name,
                "depth_path": depth_name,
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sky_path": None,
                "source_path": str(frame.path),
            }
        )
    contribution_payload = [
        {
            "chunk_indices": contribution.chunk_indices,
            "frame_id": contribution.frame_id,
            "image_name": contribution.image_name,
        }
        for contribution in contributions
    ]
    metadata = {
        "artifacts": artifact_payload,
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
        "chunks": [_chunk_payload(chunk) for chunk in chunks],
        "contributions": contribution_payload,
        "fusion": {
            "confidence": "maximum_contributing_confidence",
            "depth": "confidence_weighted_mean_in_final_colmap_scale",
        },
        "input_cameras": [_final_camera_payload(camera) for camera in cameras],
        "process_resolution": ANCHOR_PROCESS_RESOLUTION,
        "processed_camera": _camera_payload(processed_camera),
        "schema_version": 1,
        "stage": "da3_final_pose_depth",
    }
    (staging / "metadata.json").write_bytes(_strict_json_bytes(metadata))
    promote(staging, output_dir)
    return PoseConditionedDepthResult(
        artifacts=tuple(artifacts),
        cameras=cameras,
        processed_camera=processed_camera,
        chunks=chunks,
        contributions=contributions,
        attempts=attempts,
        metadata_path=output_dir / "metadata.json",
    )


def run_pose_conditioned_depth(
    model: object,
    frames: Sequence[DA3Frame],
    cameras: Sequence[FinalPoseCamera],
    output_dir: Path,
    *,
    torch_module: object | None = None,
    promote: Callable[[Path, Path], None] = promote_directory,
) -> PoseConditionedDepthResult:
    validated_frames = _validated_frames(frames)
    validated_cameras = _validated_final_cameras(validated_frames, cameras)
    _prepare_output_directory(output_dir)
    inference = getattr(model, "inference", None)
    if not callable(inference):
        raise TypeError("model must expose a callable inference method")

    scheduled_chunks = schedule_overlapping_chunks(len(validated_frames))
    depth_contributions: list[list[np.ndarray]] = [
        [] for _ in validated_frames
    ]
    confidence_contributions: list[list[np.ndarray]] = [
        [] for _ in validated_frames
    ]
    chunk_contributions: list[list[int]] = [[] for _ in validated_frames]
    chunk_records: list[ChunkProvenance] = []
    attempts: list[BatchAttemptRecord] = []
    processed_intrinsics: list[np.ndarray] = []
    processed_shape: tuple[int, int] | None = None
    confidence_presence: bool | None = None

    for chunk in scheduled_chunks:
        chunk_frames = validated_frames[chunk.start : chunk.stop]
        chunk_cameras = validated_cameras[chunk.start : chunk.stop]
        input_extrinsics = np.asarray(
            [camera.w2c for camera in chunk_cameras],
            dtype=np.float64,
        )
        input_intrinsics = np.asarray(
            [camera.intrinsics for camera in chunk_cameras],
            dtype=np.float64,
        )
        try:
            raw_prediction = inference(
                image=[str(frame.path) for frame in chunk_frames],
                extrinsics=input_extrinsics,
                intrinsics=input_intrinsics,
                align_to_input_ext_scale=True,
                process_res=ANCHOR_PROCESS_RESOLUTION,
                process_res_method="upper_bound_resize",
            )
        except Exception as error:
            if _is_injected_cuda_oom(error, torch_module):
                raise FinalPoseStageFailure(
                    chunk_index=chunk.index,
                    chunk_start=chunk.start,
                    chunk_stop=chunk.stop,
                    actual_chunk_size=len(chunk_frames),
                ) from error
            raise
        try:
            prediction = _validated_prediction(
                raw_prediction,
                len(chunk_frames),
                require_cameras=True,
            )
        except ValueError as error:
            raise ValueError(f"final prediction is invalid: {error}") from error
        for predicted, expected in zip(prediction.cameras, input_extrinsics):
            if not _matches_pinned_camera_roundtrip(predicted, expected):
                raise ValueError(
                    "final prediction extrinsics drifted from final COLMAP W2C"
                )
        current_shape = prediction.depth.shape[1:]
        if processed_shape is None:
            processed_shape = current_shape
        elif processed_shape != current_shape:
            raise ValueError("final prediction processed H/W changed between chunks")
        has_confidence = prediction.confidence is not None
        if confidence_presence is None:
            confidence_presence = has_confidence
        elif confidence_presence != has_confidence:
            raise ValueError(
                "final prediction confidence presence changed between chunks"
            )
        raw_intrinsics = np.asarray(
            _required_prediction_attribute(raw_prediction, "intrinsics"),
            dtype=np.float64,
        )
        processed_intrinsics.extend(np.array(raw_intrinsics, copy=True))
        for local_index, global_index in enumerate(chunk.frame_indices):
            depth_contributions[global_index].append(prediction.depth[local_index])
            if prediction.confidence is not None:
                confidence_contributions[global_index].append(
                    prediction.confidence[local_index]
                )
            chunk_contributions[global_index].append(chunk.index)
        chunk_records.append(
            ChunkProvenance(
                index=chunk.index,
                start=chunk.start,
                stop=chunk.stop,
                image_names=tuple(frame.image_name for frame in chunk_frames),
                frame_ids=tuple(frame.frame_id for frame in chunk_frames),
                requested_size=MAX_CHUNK_FRAMES,
            )
        )
        attempts.append(
            BatchAttemptRecord(size=len(chunk_frames), outcome="succeeded")
        )

    if processed_shape is None:
        raise AssertionError("final depth scheduler produced no chunks")
    try:
        processed_camera = robust_shared_pinhole(
            np.asarray(processed_intrinsics),
            (processed_shape[1], processed_shape[0]),
        )
    except ValueError as error:
        raise ValueError(f"final prediction processed intrinsics are invalid: {error}") from error

    fused_depth: list[np.ndarray] = []
    fused_confidence: list[np.ndarray] = []
    contributions: list[FrameContribution] = []
    for frame, depths, confidences, chunk_indices in zip(
        validated_frames,
        depth_contributions,
        confidence_contributions,
        chunk_contributions,
    ):
        depth_stack = np.stack(depths).astype(np.float64)
        if confidence_presence:
            confidence_stack = np.stack(confidences).astype(np.float64)
            weight_sum = np.sum(confidence_stack, axis=0)
            combined = np.zeros(processed_shape, dtype=np.float64)
            np.divide(
                np.sum(depth_stack * confidence_stack, axis=0),
                weight_sum,
                out=combined,
                where=weight_sum > 0.0,
            )
            fused_depth.append(
                _finite_float32(
                    combined,
                    "final fused depth",
                    positive_mask=weight_sum > 0.0,
                )
            )
            maximum_confidence = np.max(confidence_stack, axis=0)
            fused_confidence.append(
                _finite_float32(
                    maximum_confidence,
                    "final fused confidence",
                    positive_mask=maximum_confidence > 0.0,
                )
            )
        else:
            fused_depth.append(
                _finite_float32(
                    np.mean(depth_stack, axis=0),
                    "final fused depth",
                    positive_mask=np.ones(processed_shape, dtype=bool),
                )
            )
        contributions.append(
            FrameContribution(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                chunk_indices=tuple(chunk_indices),
            )
        )
    depth_values = np.stack(fused_depth)
    confidence_values = (
        np.stack(fused_confidence) if confidence_presence else None
    )
    valid = (
        np.ones(depth_values.shape, dtype=bool)
        if confidence_values is None
        else confidence_values > 0.0
    )
    depth_output = _finite_float32(
        depth_values,
        "final depth artifacts",
        positive_mask=valid,
    )
    confidence_output = (
        _finite_float32(
            confidence_values,
            "final confidence artifacts",
            positive_mask=confidence_values > 0.0,
        )
        if confidence_values is not None
        else None
    )
    return _publish_final_depth_artifacts(
        validated_frames,
        validated_cameras,
        depth_output,
        confidence_output,
        processed_camera,
        tuple(chunk_records),
        tuple(contributions),
        tuple(attempts),
        output_dir,
        promote,
    )


def schedule_overlapping_chunks(frame_count: int) -> tuple[FrameChunk, ...]:
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive plain integer")
    if frame_count <= MAX_CHUNK_FRAMES:
        starts = (0,)
    else:
        candidates = list(range(0, frame_count - MAX_CHUNK_FRAMES + 1, CHUNK_STRIDE))
        final_start = frame_count - MAX_CHUNK_FRAMES
        if candidates[-1] != final_start:
            candidates.append(final_start)
        starts = tuple(candidates)
    return tuple(
        FrameChunk(
            index=index,
            start=start,
            stop=min(start + MAX_CHUNK_FRAMES, frame_count),
            frame_indices=tuple(range(start, min(start + MAX_CHUNK_FRAMES, frame_count))),
        )
        for index, start in enumerate(starts)
    )


__all__ = [
    "ANCHOR_PROCESS_RESOLUTION",
    "AnchorInferenceResult",
    "AnchorStageFailure",
    "CameraRecord",
    "CHUNK_STRIDE",
    "ChunkProvenance",
    "DA3Frame",
    "FinalPoseCamera",
    "FinalPoseStageFailure",
    "FrameChunk",
    "FrameContribution",
    "FramePredictionArtifact",
    "MAX_CHUNK_FRAMES",
    "MetricSkyResult",
    "PinholeCamera",
    "PoseConditionedDepthResult",
    "normalize_w2c",
    "robust_shared_pinhole",
    "run_anchor_inference",
    "run_metric_sky",
    "run_pose_conditioned_depth",
    "schedule_overlapping_chunks",
    "select_anchor_indices",
]
