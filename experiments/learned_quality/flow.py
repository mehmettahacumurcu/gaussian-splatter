from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import unicodedata
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from backend.static_pipeline.stage_cache import promote_directory

from .contracts import FrameArtifact, ModelRef, StageRecord
from .dependencies import CHECKPOINT_MODEL_REFS
from .lifecycle import (
    BatchAttemptRecord,
    BatchRetryResult,
    VramReleaseRecord,
    run_with_smaller_batch_retry,
)


def _model_ref(repo_id: str) -> ModelRef:
    matches = tuple(model for model in CHECKPOINT_MODEL_REFS if model.repo_id == repo_id)
    if len(matches) != 1:
        raise RuntimeError(f"expected one pinned model ref for {repo_id}")
    return matches[0]


SEA_RAFT_MODEL_REF = _model_ref("MemorySlices/Tartan-C-T-TSKH-spring540x960-M")


@dataclass(frozen=True)
class FlowGatePolicy:
    z_buffer_relative_tolerance: float
    depth_edge_dilation_fraction: float
    cycle_absolute_pixels: float
    cycle_relative_fraction: float
    uncertainty_mad_multiplier: float
    static_track_min_length: int
    static_track_max_reprojection_error: float
    residual_mad_multiplier: float

    def __post_init__(self) -> None:
        _require_finite_number(
            self.z_buffer_relative_tolerance,
            "z_buffer_relative_tolerance",
            minimum=0.0,
            strict_minimum=True,
            maximum=1.0,
        )
        _require_finite_number(
            self.depth_edge_dilation_fraction,
            "depth_edge_dilation_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.cycle_absolute_pixels,
            "cycle_absolute_pixels",
            minimum=0.0,
        )
        _require_finite_number(
            self.cycle_relative_fraction,
            "cycle_relative_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.uncertainty_mad_multiplier,
            "uncertainty_mad_multiplier",
            minimum=0.0,
            strict_minimum=True,
        )
        if (
            type(self.static_track_min_length) is not int
            or self.static_track_min_length < 2
        ):
            raise ValueError("static_track_min_length must be a plain integer >= 2")
        _require_finite_number(
            self.static_track_max_reprojection_error,
            "static_track_max_reprojection_error",
            minimum=0.0,
        )
        _require_finite_number(
            self.residual_mad_multiplier,
            "residual_mad_multiplier",
            minimum=0.0,
            strict_minimum=True,
        )


@dataclass(frozen=True)
class RigidFrameEvidence:
    frame: FrameArtifact
    width: int
    height: int
    registered: bool
    w2c_4x4: tuple[float, ...] | None
    pinhole_fx_fy_cx_cy: tuple[float, float, float, float] | None
    depth_path: Path | None
    depth_sha256: str | None


@dataclass(frozen=True)
class TrackObservation:
    frame_id: str
    x: float
    y: float


@dataclass(frozen=True)
class StaticTrack:
    track_id: int
    xyz: tuple[float, float, float]
    mean_reprojection_error: float
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class RigidSceneEvidence:
    frames: tuple[RigidFrameEvidence, ...]
    static_tracks: tuple[StaticTrack, ...]
    geometry_digest: str
    depth_digest: str


@dataclass(frozen=True)
class MotionFrameEvidence:
    frame: FrameArtifact
    width: int
    height: int
    confirmed_without_semantic_path: Path
    confirmed_without_semantic_sha256: str
    requires_semantic_path: Path
    requires_semantic_sha256: str
    uncertain_path: Path
    uncertain_sha256: str
    strength_path: Path
    strength_sha256: str


@dataclass(frozen=True)
class MotionEvidence:
    policy: FlowGatePolicy
    scene_digest: str
    frames: tuple[MotionFrameEvidence, ...]
    stage_records: tuple[StageRecord, ...]
    pair_manifest_path: Path
    pair_manifest_sha256: str
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class PairGateResult:
    valid: np.ndarray
    motion: np.ndarray
    uncertain: np.ndarray
    strength: np.ndarray
    uncertainty_threshold: float | None


@dataclass(frozen=True)
class FrameMotionMaps:
    confirmed_without_semantic: np.ndarray
    requires_semantic: np.ndarray
    uncertain: np.ndarray
    strength: np.ndarray


@dataclass(frozen=True)
class FlowPairRequest:
    pair_index: int
    source_frame_id: str
    target_frame_id: str
    source_path: Path
    target_path: Path


@dataclass(frozen=True)
class SeaRaftPairPrediction:
    source_frame_id: str
    target_frame_id: str
    forward_flow: np.ndarray
    backward_flow: np.ndarray
    forward_uncertainty: np.ndarray
    backward_uncertainty: np.ndarray


class SeaRaftAdapter(Protocol):
    model_id: str
    revision: str
    code_commit: str | None

    def infer_bidirectional(
        self,
        pairs: tuple[FlowPairRequest, ...],
        *,
        batch_size: int,
    ) -> tuple[SeaRaftPairPrediction, ...]: ...


@dataclass(frozen=True)
class _ModelProvenance:
    model_id: str
    revision: str
    code_commit: str | None


@dataclass(frozen=True)
class _ValidatedPairPrediction:
    request: FlowPairRequest
    forward_flow: np.ndarray
    backward_flow: np.ndarray
    forward_uncertainty: np.ndarray
    backward_uncertainty: np.ndarray


@dataclass(frozen=True)
class _SceneInputs:
    frames: tuple[FrameArtifact, ...]
    rigid_frames: tuple[RigidFrameEvidence, ...]
    depths: tuple[np.ndarray | None, ...]
    input_frame_digest: str
    geometry_digest: str
    depth_digest: str
    scene_digest: str


def _require_finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and (
        result <= minimum if strict_minimum else result < minimum
    ):
        relation = ">" if strict_minimum else ">="
        raise ValueError(f"{label} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _camera_arrays(
    evidence: RigidFrameEvidence,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if not isinstance(evidence, RigidFrameEvidence):
        raise ValueError("camera evidence must be a RigidFrameEvidence record")
    if not evidence.registered:
        raise ValueError("rigid projection requires a registered frame")
    if type(evidence.width) is not int or evidence.width <= 0:
        raise ValueError("rigid frame width must be a positive plain integer")
    if type(evidence.height) is not int or evidence.height <= 0:
        raise ValueError("rigid frame height must be a positive plain integer")
    if evidence.w2c_4x4 is None or len(evidence.w2c_4x4) != 16:
        raise ValueError("registered rigid frame must contain 16 W2C values")
    try:
        w2c = np.asarray(evidence.w2c_4x4, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as error:
        raise ValueError("registered rigid frame must contain numeric W2C values") from error
    if not np.isfinite(w2c).all():
        raise ValueError("W2C values must be finite")
    if not np.array_equal(w2c[3], np.array((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("W2C must have the exact affine bottom row")
    rotation = w2c[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError("W2C rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("W2C rotation must be proper and non-reflecting")
    if evidence.pinhole_fx_fy_cx_cy is None or len(evidence.pinhole_fx_fy_cx_cy) != 4:
        raise ValueError("registered rigid frame must contain PINHOLE intrinsics")
    try:
        fx, fy, cx, cy = (float(value) for value in evidence.pinhole_fx_fy_cx_cy)
    except (TypeError, ValueError) as error:
        raise ValueError("PINHOLE intrinsics must be numeric") from error
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("PINHOLE intrinsics must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("PINHOLE focal lengths must be positive")
    if not (0.0 <= cx < evidence.width and 0.0 <= cy < evidence.height):
        raise ValueError("PINHOLE principal point must be inside the frame")
    return w2c, (fx, fy, cx, cy)


def _depth_array(value: object, width: int, height: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite numeric depth map") from error
    if array.shape != (height, width) or array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must have shape ({height}, {width})")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _narrow_float32(array: np.ndarray, label: str) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = source.astype(np.float32)
    if not np.isfinite(narrowed).all():
        raise ValueError(f"{label} cannot be represented as finite float32")
    if np.any((source != 0.0) & (narrowed == 0.0)):
        raise ValueError(f"{label} has nonzero values that underflow float32")
    return _readonly(narrowed)


def project_rigid_flow(
    source: RigidFrameEvidence,
    target: RigidFrameEvidence,
    source_depth: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project source pixels through W2C cameras into the target image."""

    source_w2c, (source_fx, source_fy, source_cx, source_cy) = _camera_arrays(source)
    target_w2c, (target_fx, target_fy, target_cx, target_cy) = _camera_arrays(target)
    if (source.width, source.height) != (target.width, target.height):
        raise ValueError("rigid flow requires equal source and target dimensions")
    depth = _depth_array(source_depth, source.width, source.height, "source_depth")
    yy, xx = np.indices((source.height, source.width), dtype=np.float64)
    source_points = np.stack(
        (
            (xx - source_cx) * depth / source_fx,
            (yy - source_cy) * depth / source_fy,
            depth,
            np.ones_like(depth),
        ),
        axis=-1,
    )
    source_c2w = np.linalg.inv(source_w2c)
    world_points = source_points @ source_c2w.T
    target_points = world_points @ target_w2c.T
    target_z = target_points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        target_x = target_fx * target_points[..., 0] / target_z + target_cx
        target_y = target_fy * target_points[..., 1] / target_z + target_cy
    flow = np.stack((target_x - xx, target_y - yy), axis=-1)
    positive_in_bounds = (
        (depth > 0.0)
        & (target_z > 0.0)
        & np.isfinite(flow).all(axis=-1)
        & (target_x >= 0.0)
        & (target_x <= target.width - 1)
        & (target_y >= 0.0)
        & (target_y <= target.height - 1)
    )
    flow = np.where(positive_in_bounds[..., None], flow, 0.0)
    return (
        _narrow_float32(flow, "rigid flow"),
        _narrow_float32(target_z, "projected target depth"),
        _readonly(positive_in_bounds),
    )


def _project_track_point(
    xyz: tuple[float, float, float],
    frame: RigidFrameEvidence,
) -> tuple[float, float] | None:
    w2c, (fx, fy, cx, cy) = _camera_arrays(frame)
    try:
        point = np.asarray((*xyz, 1.0), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("static track xyz must contain three numeric values") from error
    if point.shape != (4,) or not np.isfinite(point).all():
        raise ValueError("static track xyz must contain three finite values")
    camera_point = w2c @ point
    z = float(camera_point[2])
    if z <= 0.0:
        return None
    x = fx * float(camera_point[0]) / z + cx
    y = fy * float(camera_point[1]) / z + cy
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _validated_static_tracks_payload(
    scene: RigidSceneEvidence,
) -> tuple[dict[str, object], ...]:
    if type(scene.frames) is not tuple or any(
        not isinstance(frame, RigidFrameEvidence) for frame in scene.frames
    ):
        raise ValueError("scene frames must be an immutable tuple of rigid evidence")
    if type(scene.static_tracks) is not tuple:
        raise ValueError("static_tracks must be an immutable tuple")
    frame_ids = tuple(frame.frame.frame_id for frame in scene.frames)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("scene frame_id values must be unique")
    known_frame_ids = set(frame_ids)
    seen_track_ids: set[int] = set()
    payload: list[dict[str, object]] = []
    for track in scene.static_tracks:
        if not isinstance(track, StaticTrack):
            raise ValueError("static_tracks must contain StaticTrack records")
        if type(track.track_id) is not int or track.track_id < 0:
            raise ValueError("static track_id must be a nonnegative plain integer")
        if track.track_id in seen_track_ids:
            raise ValueError("static track_id values must be unique")
        seen_track_ids.add(track.track_id)
        if type(track.xyz) is not tuple or len(track.xyz) != 3:
            raise ValueError("static track xyz must be an immutable three-value tuple")
        xyz = tuple(
            _require_finite_number(value, "static track xyz") for value in track.xyz
        )
        reprojection_error = _require_finite_number(
            track.mean_reprojection_error,
            "mean_reprojection_error",
            minimum=0.0,
        )
        if type(track.observations) is not tuple:
            raise ValueError("track observations must be an immutable tuple")
        seen_observation_frames: set[str] = set()
        observation_payload: list[dict[str, object]] = []
        for observation in track.observations:
            if not isinstance(observation, TrackObservation):
                raise ValueError("track observations must be TrackObservation records")
            if observation.frame_id not in known_frame_ids:
                raise ValueError("track observation frame_id is missing from the scene")
            if observation.frame_id in seen_observation_frames:
                raise ValueError("a static track cannot observe one frame more than once")
            seen_observation_frames.add(observation.frame_id)
            observation_payload.append(
                {
                    "frame_id": observation.frame_id,
                    "x": _require_finite_number(observation.x, "track observation x"),
                    "y": _require_finite_number(observation.y, "track observation y"),
                }
            )
        payload.append(
            {
                "mean_reprojection_error": reprojection_error,
                "observations": observation_payload,
                "track_id": track.track_id,
                "xyz": list(xyz),
            }
        )
    return tuple(payload)


def calibrate_residual_threshold(
    scene: RigidSceneEvidence,
    policy: FlowGatePolicy,
    directional_residuals: Mapping[tuple[str, str], object] | None = None,
) -> float | None:
    """Calibrate from measured rigid-compensated residuals at static tracks."""

    if not isinstance(scene, RigidSceneEvidence):
        raise ValueError("scene must be a RigidSceneEvidence")
    if not isinstance(policy, FlowGatePolicy):
        raise ValueError("policy must be a FlowGatePolicy")
    if directional_residuals is None:
        _validated_static_tracks_payload(scene)
        return None
    if not isinstance(directional_residuals, Mapping):
        raise ValueError("directional_residuals must be a mapping")
    _validated_static_tracks_payload(scene)
    frame_by_id = {frame.frame.frame_id: frame for frame in scene.frames}
    frame_index = {
        frame.frame.frame_id: index for index, frame in enumerate(scene.frames)
    }
    fields: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, raw_value in directional_residuals.items():
        if (
            type(key) is not tuple
            or len(key) != 2
            or key[0] not in frame_by_id
            or key[1] not in frame_by_id
            or abs(frame_index[key[0]] - frame_index[key[1]]) != 1
        ):
            raise ValueError("directional residual keys must join adjacent scene frames")
        source = frame_by_id[key[0]]
        raw_field = raw_value
        raw_valid: object | None = None
        if type(raw_value) is tuple and len(raw_value) == 2:
            raw_field, raw_valid = raw_value
        field = _scalar_map(
            raw_field,
            source.width,
            source.height,
            f"directional residual {key[0]}->{key[1]}",
        )
        if raw_valid is None:
            valid = np.ones(field.shape, dtype=bool)
        else:
            valid = np.asarray(raw_valid)
            if valid.shape != field.shape or valid.dtype.kind != "b":
                raise ValueError("directional residual validity must be a boolean HxW map")
        fields[key] = (field, valid)

    residual_samples: list[float] = []
    sampled_track_count = 0
    for track in scene.static_tracks:
        if (
            len(track.observations) < policy.static_track_min_length
            or float(track.mean_reprojection_error)
            > policy.static_track_max_reprojection_error
        ):
            continue
        observations = {
            observation.frame_id: observation for observation in track.observations
        }
        track_samples: list[float] = []
        for source, target in zip(scene.frames, scene.frames[1:]):
            source_id = source.frame.frame_id
            target_id = target.frame.frame_id
            source_observation = observations.get(source_id)
            target_observation = observations.get(target_id)
            forward_field = fields.get((source_id, target_id))
            backward_field = fields.get((target_id, source_id))
            if (
                source_observation is None
                or target_observation is None
                or forward_field is None
                or backward_field is None
                or not source.registered
                or not target.registered
            ):
                continue
            pair_samples: list[float] = []
            for observation, (field, valid_field) in (
                (source_observation, forward_field),
                (target_observation, backward_field),
            ):
                x = np.asarray(float(observation.x), dtype=np.float64)
                y = np.asarray(float(observation.y), dtype=np.float64)
                sampled, inside = _bilinear_sample(field, x, y)
                sampled_valid, valid_inside = _bilinear_sample(
                    valid_field.astype(np.float64), x, y
                )
                if not bool(inside) or not bool(valid_inside) or float(sampled_valid) < 1.0:
                    pair_samples = []
                    break
                pair_samples.append(float(sampled))
            if len(pair_samples) == 2:
                track_samples.extend(pair_samples)
        if track_samples:
            sampled_track_count += 1
            residual_samples.extend(track_samples)

    if (
        sampled_track_count == 0
        or len(residual_samples) < policy.static_track_min_length
    ):
        return None
    residual_array = np.asarray(residual_samples, dtype=np.float64)
    median = float(np.median(residual_array))
    mad = float(np.median(np.abs(residual_array - median)))
    return median + policy.residual_mad_multiplier * 1.4826 * mad


def _flow_array(value: object, width: int, height: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite numeric HxWx2 array") from error
    if array.shape != (height, width, 2) or array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must have shape ({height}, {width}, 2)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _scalar_map(value: object, width: int, height: int, label: str) -> np.ndarray:
    result = _depth_array(value, width, height, label)
    if np.any(result < 0.0):
        raise ValueError(f"{label} must be nonnegative")
    return result


def _bilinear_sample(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = array.shape[:2]
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    safe_x = np.clip(np.where(inside, x, 0.0), 0.0, width - 1)
    safe_y = np.clip(np.where(inside, y, 0.0), 0.0, height - 1)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = safe_x - x0
    wy = safe_y - y0
    trailing = (None,) * (array.ndim - 2)
    wx_expanded = wx[(...,) + trailing]
    wy_expanded = wy[(...,) + trailing]
    sampled = (
        array[y0, x0] * (1.0 - wx_expanded) * (1.0 - wy_expanded)
        + array[y0, x1] * wx_expanded * (1.0 - wy_expanded)
        + array[y1, x0] * (1.0 - wx_expanded) * wy_expanded
        + array[y1, x1] * wx_expanded * wy_expanded
    )
    return sampled, inside


def _depth_edges(depth: np.ndarray, relative_tolerance: float) -> np.ndarray:
    valid = depth > 0.0
    edges = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        left = np.take(depth, indices=range(depth.shape[axis] - 1), axis=axis)
        right = np.take(depth, indices=range(1, depth.shape[axis]), axis=axis)
        left_valid = left > 0.0
        right_valid = right > 0.0
        scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
        discontinuity = (left_valid != right_valid) | (
            left_valid & right_valid & (np.abs(left - right) > relative_tolerance * scale)
        )
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        edges[tuple(first)] |= discontinuity
        edges[tuple(second)] |= discontinuity
    return edges | ~valid


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        target_y0 = source_y0 + dy
        target_y1 = source_y1 + dy
        for dx in range(-radius, radius + 1):
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_x0 = source_x0 + dx
            target_x1 = source_x1 + dx
            result[target_y0:target_y1, target_x0:target_x1] |= mask[
                source_y0:source_y1, source_x0:source_x1
            ]
    return result


def _z_buffer_gate(
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_z: np.ndarray,
    eligible: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    height, width = eligible.shape
    ix = np.clip(np.rint(target_x).astype(np.int64), 0, width - 1)
    iy = np.clip(np.rint(target_y).astype(np.int64), 0, height - 1)
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, (iy[eligible], ix[eligible]), target_z[eligible])
    nearest = z_buffer[iy, ix]
    return eligible & (target_z <= nearest * (1.0 + tolerance))


def evaluate_flow_pair(
    source: RigidFrameEvidence,
    target: RigidFrameEvidence,
    source_depth: object,
    target_depth: object,
    forward_flow: object,
    backward_flow: object,
    forward_uncertainty: object,
    backward_uncertainty: object,
    *,
    policy: FlowGatePolicy,
    residual_threshold: float,
) -> PairGateResult:
    """Apply geometry, cycle, uncertainty, and rigid-residual motion gates."""

    if not isinstance(policy, FlowGatePolicy):
        raise ValueError("policy must be a FlowGatePolicy")
    threshold = _require_finite_number(
        residual_threshold,
        "residual_threshold",
        minimum=0.0,
    )
    target_depth_array = _depth_array(target_depth, target.width, target.height, "target_depth")
    source_depth_array = _depth_array(source_depth, source.width, source.height, "source_depth")
    forward = _flow_array(forward_flow, source.width, source.height, "forward_flow")
    backward = _flow_array(backward_flow, target.width, target.height, "backward_flow")
    forward_unc = _scalar_map(
        forward_uncertainty,
        source.width,
        source.height,
        "forward_uncertainty",
    )
    backward_unc = _scalar_map(
        backward_uncertainty,
        target.width,
        target.height,
        "backward_uncertainty",
    )
    rigid_flow, projected_z, positive_in_bounds = project_rigid_flow(
        source,
        target,
        source_depth_array,
    )
    yy, xx = np.indices((source.height, source.width), dtype=np.float64)
    rigid_x = xx + rigid_flow[..., 0]
    rigid_y = yy + rigid_flow[..., 1]
    sampled_target_depth, target_depth_inside = _bilinear_sample(
        target_depth_array,
        rigid_x,
        rigid_y,
    )
    depth_scale = np.maximum(
        np.maximum(np.abs(projected_z), np.abs(sampled_target_depth)),
        1e-12,
    )
    depth_agreement = (
        target_depth_inside
        & (sampled_target_depth > 0.0)
        & (
            np.abs(projected_z - sampled_target_depth)
            <= policy.z_buffer_relative_tolerance * depth_scale
        )
    )
    z_buffer = _z_buffer_gate(
        rigid_x,
        rigid_y,
        projected_z,
        positive_in_bounds,
        policy.z_buffer_relative_tolerance,
    )

    radius = int(
        math.ceil(
            policy.depth_edge_dilation_fraction * min(source.width, source.height)
        )
    )
    source_edges = _dilate(
        _depth_edges(source_depth_array, policy.z_buffer_relative_tolerance),
        radius,
    )
    target_edges = _dilate(
        _depth_edges(target_depth_array, policy.z_buffer_relative_tolerance),
        radius,
    )
    sampled_target_edges, target_edge_inside = _bilinear_sample(
        target_edges.astype(np.float64),
        rigid_x,
        rigid_y,
    )
    edge_clear = (~source_edges) & target_edge_inside & (sampled_target_edges < 0.5)

    measured_x = xx + forward[..., 0]
    measured_y = yy + forward[..., 1]
    sampled_backward, backward_inside = _bilinear_sample(backward, measured_x, measured_y)
    cycle_error = np.linalg.norm(forward + sampled_backward, axis=-1)
    cycle_limit = policy.cycle_absolute_pixels + policy.cycle_relative_fraction * np.maximum(
        np.linalg.norm(forward, axis=-1),
        np.linalg.norm(sampled_backward, axis=-1),
    )
    cycle_consistent = backward_inside & (cycle_error <= cycle_limit)

    sampled_backward_unc, backward_unc_inside = _bilinear_sample(
        backward_unc,
        measured_x,
        measured_y,
    )
    combined_uncertainty = np.maximum(forward_unc, sampled_backward_unc)
    uncertainty_population = combined_uncertainty[
        np.isfinite(combined_uncertainty) & backward_unc_inside
    ]
    if uncertainty_population.size == 0:
        uncertainty_threshold = None
        uncertainty_ok = np.zeros((source.height, source.width), dtype=bool)
    else:
        uncertainty_median = float(np.median(uncertainty_population))
        uncertainty_mad = float(
            np.median(np.abs(uncertainty_population - uncertainty_median))
        )
        uncertainty_threshold = uncertainty_median + (
            policy.uncertainty_mad_multiplier * 1.4826 * uncertainty_mad
        )
        uncertainty_ok = (
            backward_unc_inside & (combined_uncertainty <= uncertainty_threshold)
        )

    valid = (
        positive_in_bounds
        & depth_agreement
        & z_buffer
        & edge_clear
        & cycle_consistent
        & uncertainty_ok
    )
    residual_delta = forward - np.asarray(rigid_flow, dtype=np.float64)
    residual = np.hypot(residual_delta[..., 0], residual_delta[..., 1])
    motion = valid & (residual > threshold)
    uncertain = ~valid
    return PairGateResult(
        valid=_readonly(valid),
        motion=_readonly(motion),
        uncertain=_readonly(uncertain),
        strength=_narrow_float32(residual, "strength map"),
        uncertainty_threshold=uncertainty_threshold,
    )


def _validated_pair_result(value: PairGateResult, label: str) -> PairGateResult:
    if not isinstance(value, PairGateResult):
        raise ValueError(f"{label} must be a PairGateResult")
    valid = np.asarray(value.valid)
    motion = np.asarray(value.motion)
    uncertain = np.asarray(value.uncertain)
    strength = np.asarray(value.strength)
    if valid.ndim != 2 or valid.dtype.kind != "b":
        raise ValueError(f"{label}.valid must be a boolean HxW map")
    if motion.shape != valid.shape or motion.dtype.kind != "b":
        raise ValueError(f"{label}.motion must match the valid map")
    if uncertain.shape != valid.shape or uncertain.dtype.kind != "b":
        raise ValueError(f"{label}.uncertain must match the valid map")
    if strength.shape != valid.shape or strength.dtype.kind not in "iuf":
        raise ValueError(f"{label}.strength must be a finite HxW map")
    if not np.isfinite(strength).all() or np.any(strength < 0.0):
        raise ValueError(f"{label}.strength must be finite and nonnegative")
    _narrow_float32(np.asarray(strength, dtype=np.float64), f"{label}.strength")
    if value.uncertainty_threshold is not None:
        _require_finite_number(
            value.uncertainty_threshold,
            f"{label}.uncertainty_threshold",
            minimum=0.0,
        )
    if np.any(motion & ~valid) or not np.array_equal(uncertain, ~valid):
        raise ValueError(f"{label} contains inconsistent gate masks")
    return value


def classify_temporal_motion(
    incoming: PairGateResult | None,
    outgoing: PairGateResult | None,
    *,
    is_first: bool,
    is_last: bool,
) -> FrameMotionMaps:
    """Partition directional evidence without using semantic evidence."""

    if type(is_first) is not bool or type(is_last) is not bool:
        raise ValueError("endpoint flags must be plain booleans")
    if is_first and incoming is not None:
        raise ValueError("first-frame motion cannot have incoming evidence")
    if is_last and outgoing is not None:
        raise ValueError("last-frame motion cannot have outgoing evidence")
    available = outgoing if incoming is None else incoming
    if available is None:
        raise ValueError("at least one directional result is required")
    available = _validated_pair_result(available, "directional result")
    shape = available.valid.shape

    if is_first or is_last:
        direction = outgoing if is_first else incoming
        if direction is None:
            raise ValueError("endpoint requires its sole chronological direction")
        direction = _validated_pair_result(direction, "endpoint direction")
        confirmed = np.asarray(direction.motion, dtype=bool)
        requires_semantic = np.zeros(shape, dtype=bool)
        uncertain = np.asarray(direction.uncertain, dtype=bool)
        strength = np.where(confirmed, direction.strength, 0.0)
    elif incoming is None or outgoing is None:
        confirmed = np.zeros(shape, dtype=bool)
        requires_semantic = np.zeros(shape, dtype=bool)
        uncertain = np.ones(shape, dtype=bool)
        strength = np.zeros(shape, dtype=np.float32)
    else:
        incoming = _validated_pair_result(incoming, "incoming")
        outgoing = _validated_pair_result(outgoing, "outgoing")
        if incoming.valid.shape != outgoing.valid.shape:
            raise ValueError("incoming and outgoing maps must have equal shape")
        jointly_valid = incoming.valid & outgoing.valid
        confirmed = jointly_valid & incoming.motion & outgoing.motion
        requires_semantic = jointly_valid & np.logical_xor(
            incoming.motion,
            outgoing.motion,
        )
        uncertain = ~jointly_valid
        strength = np.zeros(shape, dtype=np.float32)
        strength[confirmed] = np.minimum(
            incoming.strength[confirmed],
            outgoing.strength[confirmed],
        )
        strength[requires_semantic] = np.maximum(
            incoming.strength[requires_semantic],
            outgoing.strength[requires_semantic],
        )
    return FrameMotionMaps(
        confirmed_without_semantic=_readonly(confirmed),
        requires_semantic=_readonly(requires_semantic),
        uncertain=_readonly(uncertain),
        strength=_readonly(np.asarray(strength, dtype=np.float32)),
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_lower_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _safe_relative_image_name(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("frame image_name must be a safe relative path")
    parts = tuple(value.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("frame image_name must not contain path traversal")
    if Path(value).is_absolute():
        raise ValueError("frame image_name must be relative")
    return parts


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as error:
        raise ValueError(f"frame image is unreadable: {path}") from error
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("frame image dimensions must be positive")
    return width, height


def _validated_input_frames(
    frames: tuple[FrameArtifact, ...],
) -> tuple[tuple[FrameArtifact, ...], tuple[tuple[int, int], ...], str]:
    if type(frames) is not tuple or not frames:
        raise ValueError("frames must be a non-empty tuple of FrameArtifact records")
    frame_ids: set[str] = set()
    image_names: set[str] = set()
    source_paths: set[Path] = set()
    dimensions: list[tuple[int, int]] = []
    frame_payload: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, FrameArtifact):
            raise ValueError("frames must contain only FrameArtifact records")
        parts = _safe_relative_image_name(frame.image_name)
        if (
            not isinstance(frame.frame_id, str)
            or not frame.frame_id
            or any(
                unicodedata.category(character) == "Cc"
                for character in frame.frame_id
            )
        ):
            raise ValueError("frame_id must be non-empty and control-free")
        if not isinstance(frame.path, Path) or not frame.path.is_absolute():
            raise ValueError("frame path must be absolute")
        if (
            not os.path.lexists(frame.path)
            or frame.path.is_symlink()
            or not frame.path.is_file()
        ):
            raise ValueError("frame path must be a regular non-symlink file")
        if tuple(frame.path.parts[-len(parts) :]) != parts:
            raise ValueError("frame image_name must match the source path suffix")
        digest = _require_sha256(frame.sha256, "frame sha256")
        if _sha256_path(frame.path) != digest:
            raise ValueError("frame sha256 does not match the source bytes")
        if frame.frame_id in frame_ids:
            raise ValueError("frame_id values must be unique")
        if frame.image_name in image_names:
            raise ValueError("image_name values must be unique")
        resolved_path = frame.path.resolve(strict=True)
        if resolved_path in source_paths:
            raise ValueError("frame source paths must be unique")
        frame_ids.add(frame.frame_id)
        image_names.add(frame.image_name)
        source_paths.add(resolved_path)
        dimensions.append(_image_dimensions(frame.path))
        frame_payload.append(
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sha256": digest,
                "size_bytes": frame.path.stat().st_size,
            }
        )
    digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": frame_payload,
                "schema": "learned_quality.input_frames.v1",
            }
        )
    ).hexdigest()
    return frames, tuple(dimensions), digest


def _load_depth(evidence: RigidFrameEvidence) -> np.ndarray:
    if evidence.depth_path is None or evidence.depth_sha256 is None:
        raise ValueError("registered rigid frame requires depth path and digest")
    path = evidence.depth_path
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("depth path must be absolute")
    if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
        raise ValueError("depth path must be a regular non-symlink file")
    digest = _require_sha256(evidence.depth_sha256, "depth sha256")
    if _sha256_path(path) != digest:
        raise ValueError("depth sha256 does not match the depth bytes")
    try:
        with path.open("rb") as handle:
            raw = np.load(handle, allow_pickle=False)
    except Exception as error:
        raise ValueError("depth artifact must be a deterministic NPY array") from error
    depth = _depth_array(raw, evidence.width, evidence.height, "depth artifact")
    if np.any(depth < 0.0):
        raise ValueError("depth artifact must be nonnegative")
    return _narrow_float32(depth, "depth artifact")


def _validated_scene_inputs(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
) -> _SceneInputs:
    validated_frames, dimensions, input_frame_digest = _validated_input_frames(frames)
    if len(validated_frames) < 2:
        raise ValueError("motion evidence requires at least two frames")
    if not isinstance(scene, RigidSceneEvidence):
        raise ValueError("scene must be a RigidSceneEvidence")
    geometry_digest = _require_sha256(scene.geometry_digest, "geometry_digest")
    depth_digest = _require_sha256(scene.depth_digest, "depth_digest")
    if type(scene.frames) is not tuple or len(scene.frames) != len(validated_frames):
        raise ValueError("scene must contain the exact frame order")
    if tuple(rigid.frame for rigid in scene.frames) != validated_frames:
        raise ValueError("scene must contain the exact frame order and frame joins")
    depths: list[np.ndarray | None] = []
    rigid_payload: list[dict[str, object]] = []
    for frame, rigid, (width, height) in zip(
        validated_frames,
        scene.frames,
        dimensions,
    ):
        if not isinstance(rigid, RigidFrameEvidence):
            raise ValueError("scene frames must contain RigidFrameEvidence records")
        if rigid.frame != frame:
            raise ValueError("scene must contain the exact frame order and frame joins")
        if type(rigid.registered) is not bool:
            raise ValueError("rigid registered flag must be a plain boolean")
        if (rigid.width, rigid.height) != (width, height):
            raise ValueError("rigid frame dimensions must exactly match source images")
        if rigid.registered:
            _camera_arrays(rigid)
            depths.append(_load_depth(rigid))
        else:
            if any(
                value is not None
                for value in (
                    rigid.w2c_4x4,
                    rigid.pinhole_fx_fy_cx_cy,
                    rigid.depth_path,
                    rigid.depth_sha256,
                )
            ):
                raise ValueError("unregistered rigid frame must not claim geometry")
            depths.append(None)
        rigid_payload.append(
            {
                "depth_sha256": rigid.depth_sha256,
                "depth_size_bytes": (
                    None
                    if rigid.depth_path is None
                    else rigid.depth_path.stat().st_size
                ),
                "frame_id": frame.frame_id,
                "height": rigid.height,
                "image_name": frame.image_name,
                "pinhole_fx_fy_cx_cy": (
                    None
                    if rigid.pinhole_fx_fy_cx_cy is None
                    else [float(value) for value in rigid.pinhole_fx_fy_cx_cy]
                ),
                "registered": rigid.registered,
                "w2c_4x4": (
                    None
                    if rigid.w2c_4x4 is None
                    else [float(value) for value in rigid.w2c_4x4]
                ),
                "width": rigid.width,
            }
        )
    for source, target in zip(scene.frames, scene.frames[1:]):
        if (source.width, source.height) != (target.width, target.height):
            raise ValueError("adjacent flow frames must have equal dimensions")
    static_tracks_payload = _validated_static_tracks_payload(scene)
    scene_digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "depth_digest": depth_digest,
                "frames": rigid_payload,
                "geometry_digest": geometry_digest,
                "input_frame_digest": input_frame_digest,
                "static_tracks": static_tracks_payload,
            }
        )
    ).hexdigest()
    return _SceneInputs(
        frames=validated_frames,
        rigid_frames=scene.frames,
        depths=tuple(depths),
        input_frame_digest=input_frame_digest,
        geometry_digest=geometry_digest,
        depth_digest=depth_digest,
        scene_digest=scene_digest,
    )


def _validated_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    if output_dir.name in {"", ".", ".."} or any(
        unicodedata.category(character) == "Cc" for character in output_dir.name
    ):
        raise ValueError("output_dir must have a safe final component")
    resolved_output = output_dir.resolve(strict=False)
    if output_dir != resolved_output:
        raise ValueError("output_dir must be canonical and contain no symlink or traversal")
    if os.path.lexists(output_dir):
        raise FileExistsError("motion output_dir must be previously absent")
    parent = output_dir.parent
    if not os.path.lexists(parent) or parent.is_symlink() or not parent.is_dir():
        raise ValueError("output_dir parent must be an existing non-symlink directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError("output_dir must be canonical and contain no symlink or traversal")
    return output_dir


def _text_provenance(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(f"SEA-RAFT {label} must be non-empty and control-free")
    return value


def _model_provenance(model: object) -> _ModelProvenance:
    try:
        model_id = _text_provenance(getattr(model, "model_id", None), "model_id")
        revision = _text_provenance(getattr(model, "revision", None), "revision")
        code_commit = _text_provenance(
            getattr(model, "code_commit", None), "code_commit"
        )
    except ValueError as error:
        raise ValueError(
            "SEA-RAFT provenance must match the exact pinned model ref"
        ) from error
    provenance = _ModelProvenance(
        model_id=model_id,
        revision=revision,
        code_commit=code_commit,
    )
    expected = _ModelProvenance(
        model_id=SEA_RAFT_MODEL_REF.repo_id,
        revision=SEA_RAFT_MODEL_REF.revision,
        code_commit=SEA_RAFT_MODEL_REF.code_commit,
    )
    if provenance != expected:
        raise ValueError("SEA-RAFT provenance must match the exact pinned model ref")
    return provenance


def _pair_requests(inputs: _SceneInputs) -> tuple[FlowPairRequest, ...]:
    return tuple(
        FlowPairRequest(
            pair_index=index,
            source_frame_id=source.frame_id,
            target_frame_id=target.frame_id,
            source_path=source.path,
            target_path=target.path,
        )
        for index, (source, target) in enumerate(
            zip(inputs.frames, inputs.frames[1:])
        )
    )


def _validated_pair_predictions(
    raw_predictions: object,
    requests: tuple[FlowPairRequest, ...],
    inputs: _SceneInputs,
) -> tuple[_ValidatedPairPrediction, ...]:
    if type(raw_predictions) is not tuple or len(raw_predictions) != len(requests):
        raise ValueError("SEA-RAFT must return one prediction per exact pair")
    validated: list[_ValidatedPairPrediction] = []
    for raw, request in zip(raw_predictions, requests):
        if not isinstance(raw, SeaRaftPairPrediction):
            raise ValueError("SEA-RAFT predictions must use SeaRaftPairPrediction")
        if (
            raw.source_frame_id != request.source_frame_id
            or raw.target_frame_id != request.target_frame_id
        ):
            raise ValueError("SEA-RAFT predictions must preserve exact pair order and joins")
        source = inputs.rigid_frames[request.pair_index]
        target = inputs.rigid_frames[request.pair_index + 1]
        forward = _narrow_float32(
            _flow_array(
                raw.forward_flow,
                source.width,
                source.height,
                "forward_flow",
            ),
            "forward_flow",
        )
        backward = _narrow_float32(
            _flow_array(
                raw.backward_flow,
                target.width,
                target.height,
                "backward_flow",
            ),
            "backward_flow",
        )
        forward_uncertainty = _narrow_float32(
            _scalar_map(
                raw.forward_uncertainty,
                source.width,
                source.height,
                "forward_uncertainty",
            ),
            "forward_uncertainty",
        )
        backward_uncertainty = _narrow_float32(
            _scalar_map(
                raw.backward_uncertainty,
                target.width,
                target.height,
                "backward_uncertainty",
            ),
            "backward_uncertainty",
        )
        validated.append(
            _ValidatedPairPrediction(
                request=request,
                forward_flow=forward,
                backward_flow=backward,
                forward_uncertainty=forward_uncertainty,
                backward_uncertainty=backward_uncertainty,
            )
        )
    return tuple(validated)


def _unknown_maps(width: int, height: int) -> FrameMotionMaps:
    return FrameMotionMaps(
        confirmed_without_semantic=_readonly(np.zeros((height, width), dtype=bool)),
        requires_semantic=_readonly(np.zeros((height, width), dtype=bool)),
        uncertain=_readonly(np.ones((height, width), dtype=bool)),
        strength=_readonly(np.zeros((height, width), dtype=np.float32)),
    )


def _directional_rigid_residuals(
    inputs: _SceneInputs,
    predictions: tuple[_ValidatedPairPrediction, ...],
) -> dict[tuple[str, str], object]:
    fields: dict[tuple[str, str], object] = {}
    for prediction in predictions:
        index = prediction.request.pair_index
        source = inputs.rigid_frames[index]
        target = inputs.rigid_frames[index + 1]
        source_depth = inputs.depths[index]
        target_depth = inputs.depths[index + 1]
        if (
            not source.registered
            or not target.registered
            or source_depth is None
            or target_depth is None
        ):
            continue
        rigid_forward, _, forward_valid = project_rigid_flow(
            source, target, source_depth
        )
        rigid_backward, _, backward_valid = project_rigid_flow(
            target, source, target_depth
        )
        forward_delta = np.asarray(prediction.forward_flow, dtype=np.float64) - np.asarray(
            rigid_forward, dtype=np.float64
        )
        backward_delta = np.asarray(
            prediction.backward_flow, dtype=np.float64
        ) - np.asarray(rigid_backward, dtype=np.float64)
        forward_residual = np.hypot(
            forward_delta[..., 0], forward_delta[..., 1]
        )
        backward_residual = np.hypot(
            backward_delta[..., 0], backward_delta[..., 1]
        )
        if not np.isfinite(forward_residual).all() or not np.isfinite(
            backward_residual
        ).all():
            raise ValueError("rigid-compensated SEA-RAFT residuals must be finite")
        source_id = prediction.request.source_frame_id
        target_id = prediction.request.target_frame_id
        fields[(source_id, target_id)] = (
            _readonly(forward_residual),
            forward_valid,
        )
        fields[(target_id, source_id)] = (
            _readonly(backward_residual),
            backward_valid,
        )
    return fields


def _revalidate_source_inputs(inputs: _SceneInputs) -> None:
    try:
        _, dimensions, input_frame_digest = _validated_input_frames(inputs.frames)
        expected_dimensions = tuple(
            (rigid.width, rigid.height) for rigid in inputs.rigid_frames
        )
        if (
            input_frame_digest != inputs.input_frame_digest
            or dimensions != expected_dimensions
        ):
            raise ValueError("source image digest or dimensions changed")
        for rigid, expected_depth in zip(inputs.rigid_frames, inputs.depths):
            if not rigid.registered:
                continue
            current_depth = _load_depth(rigid)
            if expected_depth is None or not np.array_equal(
                current_depth, expected_depth
            ):
                raise ValueError("depth content or dimensions changed")
    except Exception as error:
        raise ValueError(
            "source image or depth artifact changed during SEA-RAFT inference"
        ) from error


def _evaluate_predictions(
    inputs: _SceneInputs,
    predictions: tuple[_ValidatedPairPrediction, ...],
    policy: FlowGatePolicy,
    residual_threshold: float | None,
) -> tuple[tuple[FrameMotionMaps, ...], tuple[dict[str, object], ...]]:
    incoming: list[PairGateResult | None] = [None] * len(inputs.frames)
    outgoing: list[PairGateResult | None] = [None] * len(inputs.frames)
    pair_payloads: list[dict[str, object]] = []
    for prediction in predictions:
        index = prediction.request.pair_index
        source = inputs.rigid_frames[index]
        target = inputs.rigid_frames[index + 1]
        source_depth = inputs.depths[index]
        target_depth = inputs.depths[index + 1]
        forward_result: PairGateResult | None = None
        backward_result: PairGateResult | None = None
        status = "calibration_unavailable" if residual_threshold is None else "missing_geometry"
        if (
            residual_threshold is not None
            and source.registered
            and target.registered
            and source_depth is not None
            and target_depth is not None
        ):
            forward_result = evaluate_flow_pair(
                source,
                target,
                source_depth,
                target_depth,
                prediction.forward_flow,
                prediction.backward_flow,
                prediction.forward_uncertainty,
                prediction.backward_uncertainty,
                policy=policy,
                residual_threshold=residual_threshold,
            )
            backward_result = evaluate_flow_pair(
                target,
                source,
                target_depth,
                source_depth,
                prediction.backward_flow,
                prediction.forward_flow,
                prediction.backward_uncertainty,
                prediction.forward_uncertainty,
                policy=policy,
                residual_threshold=residual_threshold,
            )
            outgoing[index] = forward_result
            incoming[index + 1] = backward_result
            status = "evaluated"
        pair_payloads.append(
            {
                "backward_motion_pixels": (
                    0 if backward_result is None else int(np.count_nonzero(backward_result.motion))
                ),
                "backward_uncertainty_threshold": (
                    None if backward_result is None else backward_result.uncertainty_threshold
                ),
                "backward_valid_pixels": (
                    0 if backward_result is None else int(np.count_nonzero(backward_result.valid))
                ),
                "forward_motion_pixels": (
                    0 if forward_result is None else int(np.count_nonzero(forward_result.motion))
                ),
                "forward_uncertainty_threshold": (
                    None if forward_result is None else forward_result.uncertainty_threshold
                ),
                "forward_valid_pixels": (
                    0 if forward_result is None else int(np.count_nonzero(forward_result.valid))
                ),
                "pair_index": index,
                "source_frame_id": prediction.request.source_frame_id,
                "target_frame_id": prediction.request.target_frame_id,
                "status": status,
            }
        )

    maps: list[FrameMotionMaps] = []
    for index, rigid in enumerate(inputs.rigid_frames):
        incoming_result = incoming[index]
        outgoing_result = outgoing[index]
        if incoming_result is None and outgoing_result is None:
            maps.append(_unknown_maps(rigid.width, rigid.height))
            continue
        maps.append(
            classify_temporal_motion(
                incoming_result,
                outgoing_result,
                is_first=index == 0,
                is_last=index == len(inputs.frames) - 1,
            )
        )
    return tuple(maps), tuple(pair_payloads)


def _policy_payload(policy: FlowGatePolicy) -> dict[str, object]:
    return {
        "cycle_absolute_pixels": policy.cycle_absolute_pixels,
        "cycle_relative_fraction": policy.cycle_relative_fraction,
        "depth_edge_dilation_fraction": policy.depth_edge_dilation_fraction,
        "residual_mad_multiplier": policy.residual_mad_multiplier,
        "static_track_max_reprojection_error": policy.static_track_max_reprojection_error,
        "static_track_min_length": policy.static_track_min_length,
        "uncertainty_mad_multiplier": policy.uncertainty_mad_multiplier,
        "z_buffer_relative_tolerance": policy.z_buffer_relative_tolerance,
    }


def _attempt_payload(attempt: BatchAttemptRecord) -> dict[str, object]:
    return {
        "error_message": attempt.error_message,
        "error_type": attempt.error_type,
        "outcome": attempt.outcome,
        "size": attempt.size,
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


def _model_payload(provenance: _ModelProvenance) -> dict[str, object]:
    return {
        "code_commit": provenance.code_commit,
        "model_id": provenance.model_id,
        "revision": provenance.revision,
    }


def _write_bytes_fsync(path: Path, payload: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_npy_fsync(path: Path, array: np.ndarray) -> str:
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_path(path)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _write_binary_png_fsync(path: Path, mask: np.ndarray) -> str:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind != "b":
        raise ValueError("binary evidence map must be a boolean HxW array")
    height, width = array.shape
    pixels = np.where(array, 255, 0).astype(np.uint8)
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return _write_bytes_fsync(path, payload)


def _artifact_token(index: int, frame: FrameArtifact) -> str:
    suffix = hashlib.sha256(
        f"{frame.frame_id}\0{frame.image_name}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{index:06d}-{suffix}"


def _publish_motion_evidence(
    inputs: _SceneInputs,
    policy: FlowGatePolicy,
    maps: tuple[FrameMotionMaps, ...],
    pair_payloads: tuple[dict[str, object], ...],
    residual_threshold: float | None,
    provenance: _ModelProvenance,
    retry_result: BatchRetryResult[tuple[_ValidatedPairPrediction, ...]],
    final_release: VramReleaseRecord | None,
    stage_record: StageRecord,
    output_dir: Path,
) -> MotionEvidence:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    artifacts: list[MotionFrameEvidence] = []
    inventory: list[dict[str, object]] = []
    try:
        for index, (frame, rigid, frame_maps) in enumerate(
            zip(inputs.frames, inputs.rigid_frames, maps)
        ):
            token = _artifact_token(index, frame)
            names = {
                "confirmed_without_semantic": f"{token}.confirmed_without_semantic.png",
                "requires_semantic": f"{token}.requires_semantic.png",
                "uncertain": f"{token}.uncertain.png",
                "strength": f"{token}.strength.npy",
            }
            digests = {
                "confirmed_without_semantic": _write_binary_png_fsync(
                    staging / names["confirmed_without_semantic"],
                    frame_maps.confirmed_without_semantic,
                ),
                "requires_semantic": _write_binary_png_fsync(
                    staging / names["requires_semantic"],
                    frame_maps.requires_semantic,
                ),
                "uncertain": _write_binary_png_fsync(
                    staging / names["uncertain"],
                    frame_maps.uncertain,
                ),
                "strength": _write_npy_fsync(
                    staging / names["strength"],
                    _narrow_float32(
                        np.asarray(frame_maps.strength, dtype=np.float64),
                        "published strength map",
                    ),
                ),
            }
            artifacts.append(
                MotionFrameEvidence(
                    frame=frame,
                    width=rigid.width,
                    height=rigid.height,
                    confirmed_without_semantic_path=output_dir
                    / names["confirmed_without_semantic"],
                    confirmed_without_semantic_sha256=digests[
                        "confirmed_without_semantic"
                    ],
                    requires_semantic_path=output_dir / names["requires_semantic"],
                    requires_semantic_sha256=digests["requires_semantic"],
                    uncertain_path=output_dir / names["uncertain"],
                    uncertain_sha256=digests["uncertain"],
                    strength_path=output_dir / names["strength"],
                    strength_sha256=digests["strength"],
                )
            )
            inventory.append(
                {
                    "files": {
                        label: {
                            "path": names[label],
                            "sha256": digests[label],
                        }
                        for label in sorted(names)
                    },
                    "frame_id": frame.frame_id,
                    "height": rigid.height,
                    "image_name": frame.image_name,
                    "width": rigid.width,
                }
            )

        pair_manifest = {
            "attempts": [_attempt_payload(value) for value in retry_result.attempts],
            "depth_digest": inputs.depth_digest,
            "failed_model_release": _release_payload(retry_result.release_record),
            "final_model_release": _release_payload(final_release),
            "geometry_digest": inputs.geometry_digest,
            "initial_pair_batch_size": retry_result.initial_size,
            "input_frame_digest": inputs.input_frame_digest,
            "model": _model_payload(provenance),
            "pairs": pair_payloads,
            "policy": _policy_payload(policy),
            "residual_threshold_pixels": residual_threshold,
            "retry_pair_batch_size": retry_result.retry_size,
            "schema_version": 1,
            "scene_digest": inputs.scene_digest,
            "stage": "sea_raft_motion",
        }
        pair_manifest_bytes = _strict_json_bytes(pair_manifest)
        pair_manifest_sha256 = _write_bytes_fsync(
            staging / "pairs.json",
            pair_manifest_bytes,
        )
        manifest = {
            "depth_digest": inputs.depth_digest,
            "frames": inventory,
            "geometry_digest": inputs.geometry_digest,
            "input_frame_digest": inputs.input_frame_digest,
            "model": _model_payload(provenance),
            "pair_manifest": {
                "path": "pairs.json",
                "sha256": pair_manifest_sha256,
            },
            "policy": _policy_payload(policy),
            "scene_digest": inputs.scene_digest,
            "schema_version": 1,
            "stage": "sea_raft_motion",
            "stage_status": stage_record.status,
        }
        manifest_bytes = _strict_json_bytes(manifest)
        manifest_sha256 = _write_bytes_fsync(staging / "manifest.json", manifest_bytes)
        promote_directory(staging, output_dir)
    except BaseException:
        if os.path.lexists(staging) and not staging.is_symlink() and staging.is_dir():
            shutil.rmtree(staging)
        raise
    return MotionEvidence(
        policy=policy,
        scene_digest=inputs.scene_digest,
        frames=tuple(artifacts),
        stage_records=(stage_record,),
        pair_manifest_path=output_dir / "pairs.json",
        pair_manifest_sha256=pair_manifest_sha256,
        manifest_path=output_dir / "manifest.json",
        manifest_sha256=manifest_sha256,
    )


def run_motion_evidence(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    output_dir: Path,
    *,
    policy: FlowGatePolicy,
    model_factory: Callable[[], SeaRaftAdapter],
    initial_pair_batch_size: int,
    retry_pair_batch_size: int,
    release_model: Callable[[object], VramReleaseRecord],
) -> MotionEvidence:
    """Run exact bidirectional SEA-RAFT pairs and publish rigid motion evidence."""

    if not isinstance(policy, FlowGatePolicy):
        raise ValueError("policy must be a FlowGatePolicy")
    if not callable(model_factory) or not callable(release_model):
        raise TypeError("model_factory and release_model must be callable")
    inputs = _validated_scene_inputs(frames, scene)
    output_dir = _validated_output_dir(output_dir)
    requests = _pair_requests(inputs)
    active_model: object | None = None
    expected_provenance: _ModelProvenance | None = None
    created_model_ids: set[int] = set()

    def operation(batch_size: int) -> tuple[_ValidatedPairPrediction, ...]:
        nonlocal active_model, expected_provenance
        if not requests:
            if expected_provenance is None:
                expected_provenance = _model_provenance(model_factory)
            return ()
        if active_model is None:
            candidate = model_factory()
            if id(candidate) in created_model_ids:
                raise ValueError("model_factory must recreate a released SEA-RAFT model")
            created_model_ids.add(id(candidate))
            active_model = candidate
            candidate_provenance = _model_provenance(candidate)
            if expected_provenance is None:
                expected_provenance = candidate_provenance
            elif candidate_provenance != expected_provenance:
                raise ValueError("retry SEA-RAFT provenance must exactly match")
        infer = getattr(active_model, "infer_bidirectional", None)
        if not callable(infer):
            raise TypeError("SEA-RAFT adapter must expose infer_bidirectional")
        raw_predictions = infer(requests, batch_size=batch_size)
        return _validated_pair_predictions(raw_predictions, requests, inputs)

    def release_active_model() -> VramReleaseRecord:
        nonlocal active_model
        if active_model is None:
            raise RuntimeError("no active SEA-RAFT model to release")
        model = active_model
        active_model = None
        record = release_model(model)
        if not isinstance(record, VramReleaseRecord):
            raise TypeError("release_model must return VramReleaseRecord")
        return record

    try:
        retry_result = run_with_smaller_batch_retry(
            "sea_raft_pair_batches",
            initial_pair_batch_size,
            retry_pair_batch_size,
            operation,
            chunk_independent=True,
            release=release_active_model,
            torch_module=getattr(model_factory, "torch_module", None),
        )
    except Exception as error:
        traceback = error.__traceback__
        if active_model is not None:
            try:
                release_active_model()
            except Exception as release_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"SEA-RAFT cleanup also failed: {release_error!r}")
        raise error.with_traceback(traceback)
    final_release = (
        release_active_model() if active_model is not None else None
    )
    if expected_provenance is None:
        raise AssertionError("SEA-RAFT provenance was not established")
    residual_threshold = calibrate_residual_threshold(
        scene,
        policy,
        _directional_rigid_residuals(inputs, retry_result.value),
    )
    maps, pair_payloads = _evaluate_predictions(
        inputs,
        retry_result.value,
        policy,
        residual_threshold,
    )
    _revalidate_source_inputs(inputs)
    status = (
        "skipped"
        if residual_threshold is None
        else "fallback"
        if retry_result.retry_size is not None
        else "accepted"
    )
    stage_record = StageRecord(
        stage_id="sea_raft_motion",
        status=status,
        details=MappingProxyType(
            {
                "attempts": tuple(
                    f"{attempt.size}:{attempt.outcome}"
                    for attempt in retry_result.attempts
                ),
                "model_id": expected_provenance.model_id,
                "pair_count": len(requests),
                "residual_threshold_pixels": residual_threshold,
            }
        ),
    )
    return _publish_motion_evidence(
        inputs,
        policy,
        maps,
        pair_payloads,
        residual_threshold,
        expected_provenance,
        retry_result,
        final_release,
        stage_record,
        output_dir,
    )
