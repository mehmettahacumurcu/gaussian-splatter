from __future__ import annotations

import math
import hashlib
import json
import os
import shutil
import struct
import tempfile
import unicodedata
import zlib
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final, Literal

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.optimize import lsq_linear

from backend.static_pipeline.stage_cache import promote_directory

from .contracts import FrameArtifact
from .flow import (
    RigidFrameEvidence,
    RigidSceneEvidence,
    StaticTrack,
    TrackObservation,
)
from .masks import FusedMaskFrame, MaskFusionEvidence, MaskFusionPolicy


GAIN_MIN: Final[float] = 0.75
GAIN_MAX: Final[float] = 1.33
BIAS_MIN: Final[float] = -0.10
BIAS_MAX: Final[float] = 0.10
MAX_ADDITIONAL_CLIP_FRACTION: Final[float] = 0.005


@dataclass(frozen=True)
class PhotometricPolicy:
    robust_loss_delta: float
    temporal_smoothness_weight: float
    heldout_stride: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.robust_loss_delta, bool)
            or not isinstance(self.robust_loss_delta, Real)
            or not math.isfinite(float(self.robust_loss_delta))
            or float(self.robust_loss_delta) <= 0.0
        ):
            raise ValueError(
                "robust_loss_delta must be a finite number greater than zero"
            )
        if (
            isinstance(self.temporal_smoothness_weight, bool)
            or not isinstance(self.temporal_smoothness_weight, Real)
            or not math.isfinite(float(self.temporal_smoothness_weight))
            or float(self.temporal_smoothness_weight) < 0.0
        ):
            raise ValueError(
                "temporal_smoothness_weight must be a finite nonnegative number"
            )
        if type(self.heldout_stride) is not int or self.heldout_stride < 2:
            raise ValueError("heldout_stride must be a plain integer >= 2")


@dataclass(frozen=True)
class RgbAffineTransform:
    frame_id: str
    gain_rgb: tuple[float, float, float]
    bias_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class PhotometricEvidence:
    decision: Literal["accepted", "rejected"]
    reference_frame_id: str
    transforms: tuple[RgbAffineTransform, ...]
    original_frames: tuple[FrameArtifact, ...]
    training_frames: tuple[FrameArtifact, ...]
    original_rgb_digest: str
    training_rgb_digest: str
    heldout_error_before: float
    heldout_error_after: float
    maximum_additional_clip_fraction: float
    rejection_reasons: tuple[str, ...]
    report_path: Path
    report_sha256: str
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class _ValidatedInputs:
    frames: tuple[FrameArtifact, ...]
    images: tuple[np.ndarray, ...]
    hard_exclude: tuple[np.ndarray, ...]
    uncertain: tuple[np.ndarray, ...]
    source_frame_digest: str
    original_rgb_digest: str
    scene_snapshot_digest: str
    mask_snapshot_digest: str


@dataclass(frozen=True)
class _TrackSamples:
    track_id: int
    frame_indices: tuple[int, ...]
    rgb: tuple[tuple[float, float, float], ...]


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_file(path: object, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} path must be an absolute Path")
    if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} path must be a regular non-symlink file")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} path must be canonical")
    return path


def _safe_image_name(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or Path(value).is_absolute()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("frame image_name must be a safe relative path")
    parts = tuple(value.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("frame image_name must not contain path traversal")
    return parts


def _load_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"{label} must use exact RGB pixels")
            values = np.array(image, dtype=np.uint8, copy=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{label} must be a readable RGB image") from error
    if values.ndim != 3 or values.shape[2] != 3 or min(values.shape[:2]) <= 0:
        raise ValueError(f"{label} must have positive RGB dimensions")
    values.setflags(write=False)
    return values


def _load_binary_png(
    path: object,
    digest_value: object,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    artifact_path = _regular_file(path, label)
    digest = _require_sha256(digest_value, f"{label} sha256")
    if _sha256_path(artifact_path) != digest:
        raise ValueError(f"{label} sha256 does not match artifact bytes")
    try:
        with Image.open(artifact_path) as image:
            if image.format != "PNG" or image.mode != "L":
                raise ValueError(f"{label} must be an exact grayscale PNG")
            values = np.array(image, dtype=np.uint8, copy=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{label} must be a readable grayscale PNG") from error
    if values.shape != shape or not np.isin(values, (0, 255)).all():
        raise ValueError(f"{label} must match frame dimensions and contain only 0/255")
    result = values == 255
    result.setflags(write=False)
    return result


def _load_depth(
    evidence: RigidFrameEvidence,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if evidence.registered:
        if evidence.depth_path is None or evidence.depth_sha256 is None:
            raise ValueError("registered rigid frame must provide depth and digest")
        path = _regular_file(evidence.depth_path, "depth")
        digest = _require_sha256(evidence.depth_sha256, "depth sha256")
        if _sha256_path(path) != digest:
            raise ValueError("depth sha256 does not match artifact bytes")
        try:
            with path.open("rb") as stream:
                raw = np.load(stream, allow_pickle=False)
            depth = np.asarray(raw, dtype=np.float64)
        except Exception as error:
            raise ValueError(
                "depth artifact must be a deterministic numeric NPY"
            ) from error
        if (
            depth.shape != (evidence.height, evidence.width)
            or not np.isfinite(depth).all()
            or np.any(depth < 0.0)
        ):
            raise ValueError(
                "depth artifact must be finite, nonnegative, and exact-sized"
            )
        depth.setflags(write=False)
        payload = {"path": str(path), "sha256": digest}
        return depth, payload
    if any(
        value is not None
        for value in (
            evidence.w2c_4x4,
            evidence.pinhole_fx_fy_cx_cy,
            evidence.depth_path,
            evidence.depth_sha256,
        )
    ):
        raise ValueError("unregistered rigid frame must not claim geometry or depth")
    return None, {"path": None, "sha256": None}


def _policy_payload(policy: PhotometricPolicy) -> dict[str, object]:
    return {
        "heldout_stride": policy.heldout_stride,
        "robust_loss_delta": float(policy.robust_loss_delta),
        "temporal_smoothness_weight": float(policy.temporal_smoothness_weight),
    }


def _mask_policy_payload(policy: MaskFusionPolicy) -> dict[str, object]:
    return {
        field_name: getattr(policy, field_name)
        for field_name in policy.__dataclass_fields__
    }


def _validate_inputs(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    masks: MaskFusionEvidence,
) -> _ValidatedInputs:
    if type(frames) is not tuple or not frames:
        raise ValueError("frames must be a non-empty immutable tuple")
    frame_ids: set[str] = set()
    image_names: set[str] = set()
    normalized_names: set[str] = set()
    occupied_paths: set[Path] = set()
    images: list[np.ndarray] = []
    frame_payload: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, FrameArtifact):
            raise ValueError("frames must contain only FrameArtifact records")
        parts = _safe_image_name(frame.image_name)
        if (
            not isinstance(frame.frame_id, str)
            or not frame.frame_id
            or any(
                unicodedata.category(character) == "Cc" for character in frame.frame_id
            )
        ):
            raise ValueError("frame_id must be non-empty and control-free")
        path = _regular_file(frame.path, "source frame")
        if tuple(path.parts[-len(parts) :]) != parts:
            raise ValueError(
                "frame image_name must exactly match its source path suffix"
            )
        digest = _require_sha256(frame.sha256, "source frame sha256")
        if _sha256_path(path) != digest:
            raise ValueError("source frame sha256 does not match source bytes")
        normalized_name = os.path.normcase("/".join(parts))
        resolved = path.resolve(strict=True)
        if (
            frame.frame_id in frame_ids
            or frame.image_name in image_names
            or normalized_name in normalized_names
            or resolved in occupied_paths
        ):
            raise ValueError(
                "source frames contain duplicate identities, names, or paths"
            )
        frame_ids.add(frame.frame_id)
        image_names.add(frame.image_name)
        normalized_names.add(normalized_name)
        occupied_paths.add(resolved)
        image = _load_rgb(path, "source frame")
        images.append(image)
        frame_payload.append(
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sha256": digest,
            }
        )

    if not isinstance(scene, RigidSceneEvidence):
        raise ValueError("scene must be a RigidSceneEvidence")
    geometry_digest = _require_sha256(scene.geometry_digest, "geometry_digest")
    depth_digest = _require_sha256(scene.depth_digest, "depth_digest")
    if (
        type(scene.frames) is not tuple
        or tuple(item.frame for item in scene.frames) != frames
    ):
        raise ValueError(
            "scene must preserve the exact canonical frame joins and order"
        )
    rigid_payload: list[dict[str, object]] = []
    for frame, image, rigid in zip(frames, images, scene.frames, strict=True):
        if not isinstance(rigid, RigidFrameEvidence) or rigid.frame != frame:
            raise ValueError("scene must contain exact RigidFrameEvidence joins")
        height, width = image.shape[:2]
        if type(rigid.width) is not int or type(rigid.height) is not int:
            raise ValueError("rigid frame dimensions must be plain integers")
        if (rigid.width, rigid.height) != (width, height):
            raise ValueError("rigid frame dimensions must match original RGB")
        if type(rigid.registered) is not bool:
            raise ValueError("rigid registered must be a plain boolean")
        if rigid.registered:
            if type(rigid.w2c_4x4) is not tuple or len(rigid.w2c_4x4) != 16:
                raise ValueError("registered geometry must contain 16 W2C values")
            if (
                type(rigid.pinhole_fx_fy_cx_cy) is not tuple
                or len(rigid.pinhole_fx_fy_cx_cy) != 4
            ):
                raise ValueError("registered geometry must contain four PINHOLE values")
            values = tuple(rigid.w2c_4x4) + tuple(rigid.pinhole_fx_fy_cx_cy)
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError("registered geometry values must be finite numbers")
        _, depth_payload = _load_depth(rigid)
        if depth_payload["path"] is not None:
            resolved_depth = Path(str(depth_payload["path"])).resolve(strict=True)
            if resolved_depth in occupied_paths:
                raise ValueError("input artifact paths must not collide")
            occupied_paths.add(resolved_depth)
        rigid_payload.append(
            {
                "depth": {"sha256": depth_payload["sha256"]},
                "frame_id": frame.frame_id,
                "height": height,
                "pinhole_fx_fy_cx_cy": rigid.pinhole_fx_fy_cx_cy,
                "registered": rigid.registered,
                "w2c_4x4": rigid.w2c_4x4,
                "width": width,
            }
        )

    if type(scene.static_tracks) is not tuple:
        raise ValueError("scene static_tracks must be an immutable tuple")
    known_frames = set(frame_ids)
    seen_track_ids: set[int] = set()
    tracks_payload: list[dict[str, object]] = []
    for track in scene.static_tracks:
        if not isinstance(track, StaticTrack):
            raise ValueError("scene static_tracks must contain StaticTrack records")
        if type(track.track_id) is not int or track.track_id < 0:
            raise ValueError("static track_id must be a nonnegative plain integer")
        if track.track_id in seen_track_ids:
            raise ValueError("static track_id values must be unique")
        seen_track_ids.add(track.track_id)
        if (
            type(track.xyz) is not tuple
            or len(track.xyz) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in track.xyz
            )
        ):
            raise ValueError("static track xyz must contain three finite values")
        if (
            isinstance(track.mean_reprojection_error, bool)
            or not isinstance(track.mean_reprojection_error, Real)
            or not math.isfinite(float(track.mean_reprojection_error))
            or float(track.mean_reprojection_error) < 0.0
        ):
            raise ValueError(
                "static track reprojection error must be finite and nonnegative"
            )
        if type(track.observations) is not tuple or len(track.observations) < 3:
            raise ValueError(
                "producer-qualified static tracks require at least 3 observations"
            )
        seen_observations: set[str] = set()
        observation_payload: list[dict[str, object]] = []
        for observation in track.observations:
            if not isinstance(observation, TrackObservation):
                raise ValueError(
                    "static track observations must be TrackObservation records"
                )
            if observation.frame_id not in known_frames:
                raise ValueError(
                    "static track observation frame_id is missing from scene"
                )
            if observation.frame_id in seen_observations:
                raise ValueError("static track observations must use unique frame IDs")
            seen_observations.add(observation.frame_id)
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in (observation.x, observation.y)
            ):
                raise ValueError("static track observation coordinates must be finite")
            observation_payload.append(
                {
                    "frame_id": observation.frame_id,
                    "x": float(observation.x),
                    "y": float(observation.y),
                }
            )
        tracks_payload.append(
            {
                "mean_reprojection_error": float(track.mean_reprojection_error),
                "observations": observation_payload,
                "track_id": track.track_id,
                "xyz": [float(value) for value in track.xyz],
            }
        )

    if not isinstance(masks, MaskFusionEvidence):
        raise ValueError("masks must be a MaskFusionEvidence")
    if not isinstance(masks.policy, MaskFusionPolicy):
        raise ValueError("masks policy must be a MaskFusionPolicy")
    mask_set_digest = _require_sha256(masks.mask_set_digest, "mask_set_digest")
    manifest_path = _regular_file(masks.manifest_path, "mask manifest")
    manifest_digest = _require_sha256(masks.manifest_sha256, "mask manifest sha256")
    if _sha256_path(manifest_path) != manifest_digest:
        raise ValueError("mask manifest sha256 does not match manifest bytes")
    if manifest_path.resolve(strict=True) in occupied_paths:
        raise ValueError("input artifact paths must not collide")
    occupied_paths.add(manifest_path.resolve(strict=True))
    if (
        type(masks.frames) is not tuple
        or tuple(item.frame for item in masks.frames) != frames
    ):
        raise ValueError(
            "masks must preserve the exact canonical frame joins and order"
        )
    hard_exclude: list[np.ndarray] = []
    uncertain_maps: list[np.ndarray] = []
    mask_payload: list[dict[str, object]] = []
    map_fields = (
        "semantic_confirmed",
        "sky_confirmed",
        "motion_confirmed",
        "uncertain",
        "hard_exclude",
        "colmap_keep",
        "training_validity",
    )
    for frame, image, evidence in zip(frames, images, masks.frames, strict=True):
        if not isinstance(evidence, FusedMaskFrame) or evidence.frame != frame:
            raise ValueError("masks must contain exact FusedMaskFrame joins")
        height, width = image.shape[:2]
        if (evidence.width, evidence.height) != (width, height):
            raise ValueError("fused mask dimensions must match original RGB")
        loaded: dict[str, np.ndarray] = {}
        files: dict[str, dict[str, object]] = {}
        for field_name in map_fields:
            path = getattr(evidence, f"{field_name}_path")
            digest_value = getattr(evidence, f"{field_name}_sha256")
            loaded[field_name] = _load_binary_png(
                path,
                digest_value,
                (height, width),
                f"{field_name} mask",
            )
            resolved = Path(path).resolve(strict=True)
            if resolved in occupied_paths:
                raise ValueError("input artifact paths must not collide")
            occupied_paths.add(resolved)
            files[field_name] = {"sha256": digest_value}
        confirmed_union = (
            loaded["semantic_confirmed"]
            | loaded["sky_confirmed"]
            | loaded["motion_confirmed"]
        )
        if not np.array_equal(loaded["hard_exclude"], confirmed_union):
            raise ValueError("hard_exclude must equal the exact confirmed-map union")
        if not np.array_equal(loaded["colmap_keep"], ~loaded["hard_exclude"]):
            raise ValueError("colmap_keep must be the inverse of hard_exclude")
        if not np.array_equal(loaded["training_validity"], ~loaded["hard_exclude"]):
            raise ValueError("training_validity must be the inverse of hard_exclude")
        hard_exclude.append(loaded["hard_exclude"])
        uncertain_maps.append(loaded["uncertain"])
        mask_payload.append(
            {
                "decision": evidence.decision,
                "files": files,
                "frame_id": frame.frame_id,
                "height": height,
                "warnings": list(evidence.warnings),
                "width": width,
            }
        )

    original_rgb_digest = hashlib.sha256(
        _strict_json_bytes(
            [
                {
                    "frame_id": frame.frame_id,
                    "image_name": frame.image_name,
                    "sha256": frame.sha256,
                }
                for frame in frames
            ]
        )
    ).hexdigest()
    source_frame_digest = hashlib.sha256(_strict_json_bytes(frame_payload)).hexdigest()
    scene_snapshot_digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "depth_digest": depth_digest,
                "frames": rigid_payload,
                "geometry_digest": geometry_digest,
                "static_tracks": tracks_payload,
            }
        )
    ).hexdigest()
    mask_snapshot_digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": mask_payload,
                "manifest": {
                    "sha256": manifest_digest,
                },
                "mask_set_digest": mask_set_digest,
                "policy": _mask_policy_payload(masks.policy),
            }
        )
    ).hexdigest()
    return _ValidatedInputs(
        frames=frames,
        images=tuple(images),
        hard_exclude=tuple(hard_exclude),
        uncertain=tuple(uncertain_maps),
        source_frame_digest=source_frame_digest,
        original_rgb_digest=original_rgb_digest,
        scene_snapshot_digest=scene_snapshot_digest,
        mask_snapshot_digest=mask_snapshot_digest,
    )


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    if output_dir.name in {"", ".", ".."} or any(
        unicodedata.category(character) == "Cc" for character in output_dir.name
    ):
        raise ValueError("output_dir must have a safe final component")
    if output_dir.resolve(strict=False) != output_dir:
        raise ValueError("output_dir must be canonical and traversal-free")
    if os.path.lexists(output_dir):
        raise FileExistsError("photometric output_dir must be previously absent")
    parent = output_dir.parent
    if not os.path.lexists(parent) or parent.is_symlink() or not parent.is_dir():
        raise ValueError("output_dir parent must be an existing non-symlink directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError("output_dir parent must be canonical")
    return output_dir


def _sample_tracks(
    inputs: _ValidatedInputs,
    scene: RigidSceneEvidence,
) -> tuple[_TrackSamples, ...]:
    frame_index = {frame.frame_id: index for index, frame in enumerate(inputs.frames)}
    eligible: list[_TrackSamples] = []
    for track in sorted(scene.static_tracks, key=lambda item: item.track_id):
        sampled_indices: list[int] = []
        sampled_rgb: list[tuple[float, float, float]] = []
        for observation in track.observations:
            index = frame_index[observation.frame_id]
            image = inputs.images[index]
            height, width = image.shape[:2]
            x = float(observation.x)
            y = float(observation.y)
            x0 = math.floor(x)
            y0 = math.floor(y)
            if x < 0.0 or y < 0.0 or x0 + 1 >= width or y0 + 1 >= height:
                continue
            footprint = np.s_[y0 : y0 + 2, x0 : x0 + 2]
            if np.any(inputs.hard_exclude[index][footprint]) or np.any(
                inputs.uncertain[index][footprint]
            ):
                continue
            dx = x - x0
            dy = y - y0
            pixels = image[y0 : y0 + 2, x0 : x0 + 2].astype(np.float64) / 255.0
            color = (
                pixels[0, 0] * (1.0 - dx) * (1.0 - dy)
                + pixels[0, 1] * dx * (1.0 - dy)
                + pixels[1, 0] * (1.0 - dx) * dy
                + pixels[1, 1] * dx * dy
            )
            if not np.isfinite(color).all():
                continue
            sampled_indices.append(index)
            sampled_rgb.append(tuple(float(value) for value in color))
        if len(sampled_indices) >= 3:
            eligible.append(
                _TrackSamples(
                    track_id=track.track_id,
                    frame_indices=tuple(sampled_indices),
                    rgb=tuple(sampled_rgb),
                )
            )
    return tuple(eligible)


def _support_connected(
    frame_count: int,
    reference_index: int,
    tracks: tuple[_TrackSamples, ...],
) -> bool:
    adjacency = [set() for _ in range(frame_count)]
    support_counts = [0] * frame_count
    for track in tracks:
        unique = tuple(dict.fromkeys(track.frame_indices))
        for index in unique:
            support_counts[index] += 1
        for source in unique:
            adjacency[source].update(target for target in unique if target != source)
    if any(count < 3 for count in support_counts):
        return False
    visited = {reference_index}
    pending = [reference_index]
    while pending:
        source = pending.pop()
        for target in sorted(adjacency[source]):
            if target not in visited:
                visited.add(target)
                pending.append(target)
    return len(visited) == frame_count


def _solve_channel(
    tracks: tuple[_TrackSamples, ...],
    frame_count: int,
    reference_index: int,
    channel: int,
    policy: PhotometricPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    non_reference = tuple(
        index for index in range(frame_count) if index != reference_index
    )
    gain_column = {
        frame_index: 2 * offset for offset, frame_index in enumerate(non_reference)
    }
    bias_column = {
        frame_index: 2 * offset + 1 for offset, frame_index in enumerate(non_reference)
    }
    latent_offset = 2 * len(non_reference)
    variable_count = latent_offset + len(tracks)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    row = 0
    for track_index, track in enumerate(tracks):
        latent_column = latent_offset + track_index
        for frame_index, rgb in zip(track.frame_indices, track.rgb, strict=True):
            color = float(rgb[channel])
            if frame_index == reference_index:
                row_indices.append(row)
                column_indices.append(latent_column)
                values.append(-1.0)
                targets.append(-color)
            else:
                row_indices.extend((row, row, row))
                column_indices.extend(
                    (gain_column[frame_index], bias_column[frame_index], latent_column)
                )
                values.extend((color, 1.0, -1.0))
                targets.append(0.0)
            row += 1
    data_rows = row
    smoothness = float(policy.temporal_smoothness_weight)
    if smoothness > 0.0:
        scale = math.sqrt(smoothness)
        for previous, current in zip(range(frame_count - 1), range(1, frame_count)):
            constant = 0.0
            if current == reference_index:
                constant += scale
            else:
                row_indices.append(row)
                column_indices.append(gain_column[current])
                values.append(scale)
            if previous == reference_index:
                constant -= scale
            else:
                row_indices.append(row)
                column_indices.append(gain_column[previous])
                values.append(-scale)
            targets.append(-constant)
            row += 1
            if current != reference_index:
                row_indices.append(row)
                column_indices.append(bias_column[current])
                values.append(scale)
            if previous != reference_index:
                row_indices.append(row)
                column_indices.append(bias_column[previous])
                values.append(-scale)
            targets.append(0.0)
            row += 1
    matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(row, variable_count),
        dtype=np.float64,
    )
    target = np.asarray(targets, dtype=np.float64)
    lower = np.full(variable_count, -np.inf, dtype=np.float64)
    upper = np.full(variable_count, np.inf, dtype=np.float64)
    for frame_index in non_reference:
        lower[gain_column[frame_index]] = GAIN_MIN
        upper[gain_column[frame_index]] = GAIN_MAX
        lower[bias_column[frame_index]] = BIAS_MIN
        upper[bias_column[frame_index]] = BIAS_MAX
    estimate = np.zeros(variable_count, dtype=np.float64)
    for frame_index in non_reference:
        estimate[gain_column[frame_index]] = 1.0
    for track_index, track in enumerate(tracks):
        estimate[latent_offset + track_index] = float(
            np.median([rgb[channel] for rgb in track.rgb])
        )
    delta = float(policy.robust_loss_delta)
    for _ in range(300):
        residual = np.asarray(matrix @ estimate - target, dtype=np.float64)
        weights = np.ones(row, dtype=np.float64)
        absolute = np.abs(residual[:data_rows])
        outliers = absolute > delta
        weights[:data_rows][outliers] = delta / absolute[outliers]
        square_root = np.sqrt(weights)
        weighted_matrix = matrix.multiply(square_root[:, None])
        weighted_target = target * square_root
        solved = lsq_linear(
            weighted_matrix,
            weighted_target,
            bounds=(lower, upper),
            method="trf",
            lsq_solver="lsmr",
            tol=1e-12,
            lsmr_tol=1e-12,
            max_iter=500,
            verbose=0,
        )
        if not solved.success or not np.isfinite(solved.x).all():
            raise ArithmeticError("bounded robust photometric solve failed")
        updated = np.asarray(solved.x, dtype=np.float64)
        if np.max(np.abs(updated - estimate)) <= 1e-8:
            estimate = updated
            break
        estimate = updated
    else:
        raise ArithmeticError("bounded robust photometric IRLS did not converge")
    gains = np.ones(frame_count, dtype=np.float64)
    biases = np.zeros(frame_count, dtype=np.float64)
    for frame_index in non_reference:
        gain = float(np.clip(estimate[gain_column[frame_index]], GAIN_MIN, GAIN_MAX))
        bias = float(np.clip(estimate[bias_column[frame_index]], BIAS_MIN, BIAS_MAX))
        if abs(gain - GAIN_MIN) <= 1e-10:
            gain = GAIN_MIN
        elif abs(gain - GAIN_MAX) <= 1e-10:
            gain = GAIN_MAX
        if abs(bias - BIAS_MIN) <= 1e-10:
            bias = BIAS_MIN
        elif abs(bias - BIAS_MAX) <= 1e-10:
            bias = BIAS_MAX
        gains[frame_index] = gain
        biases[frame_index] = bias
    gains[reference_index] = 1.0
    biases[reference_index] = 0.0
    return gains, biases


def _identity_transforms(
    frames: tuple[FrameArtifact, ...],
) -> tuple[RgbAffineTransform, ...]:
    return tuple(
        RgbAffineTransform(
            frame_id=frame.frame_id,
            gain_rgb=(1.0, 1.0, 1.0),
            bias_rgb=(0.0, 0.0, 0.0),
        )
        for frame in frames
    )


def _solve_transforms(
    frames: tuple[FrameArtifact, ...],
    tracks: tuple[_TrackSamples, ...],
    reference_index: int,
    policy: PhotometricPolicy,
) -> tuple[RgbAffineTransform, ...]:
    channel_solutions = tuple(
        _solve_channel(tracks, len(frames), reference_index, channel, policy)
        for channel in range(3)
    )
    transforms: list[RgbAffineTransform] = []
    for frame_index, frame in enumerate(frames):
        if frame_index == reference_index:
            transforms.append(
                RgbAffineTransform(frame.frame_id, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
            )
            continue
        transforms.append(
            RgbAffineTransform(
                frame_id=frame.frame_id,
                gain_rgb=tuple(
                    float(channel_solutions[channel][0][frame_index])
                    for channel in range(3)
                ),
                bias_rgb=tuple(
                    float(channel_solutions[channel][1][frame_index])
                    for channel in range(3)
                ),
            )
        )
    return tuple(transforms)


def _heldout_error(
    tracks: tuple[_TrackSamples, ...],
    transforms: tuple[RgbAffineTransform, ...] | None,
) -> float:
    squared: list[float] = []
    for track in tracks:
        values = np.asarray(track.rgb, dtype=np.float64)
        if transforms is not None:
            corrected = []
            for frame_index, value in zip(track.frame_indices, values, strict=True):
                transform = transforms[frame_index]
                corrected.append(
                    np.asarray(transform.gain_rgb, dtype=np.float64) * value
                    + np.asarray(transform.bias_rgb, dtype=np.float64)
                )
            values = np.asarray(corrected, dtype=np.float64)
        for first in range(len(values)):
            for second in range(first + 1, len(values)):
                squared.extend((values[first] - values[second]) ** 2)
    if not squared:
        return 0.0
    result = float(math.sqrt(float(np.mean(squared))))
    if not math.isfinite(result):
        raise ArithmeticError("heldout cross-frame error is non-finite")
    return result


def _corrected_arrays_and_clipping(
    inputs: _ValidatedInputs,
    transforms: tuple[RgbAffineTransform, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[float, float, float], ...], float]:
    corrected_arrays: list[np.ndarray] = []
    fractions: list[tuple[float, float, float]] = []
    maximum = 0.0
    for image, transform in zip(inputs.images, transforms, strict=True):
        normalized = image.astype(np.float64) / 255.0
        gain = np.asarray(transform.gain_rgb, dtype=np.float64)
        bias = np.asarray(transform.bias_rgb, dtype=np.float64)
        corrected = normalized * gain[None, None, :] + bias[None, None, :]
        original_saturated = (normalized <= 0.0) | (normalized >= 1.0)
        would_clip = (corrected < 0.0) | (corrected > 1.0)
        additional = would_clip & ~original_saturated
        per_channel = tuple(
            float(np.mean(additional[:, :, channel])) for channel in range(3)
        )
        maximum = max(maximum, *per_channel)
        quantized = np.rint(np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)
        quantized.setflags(write=False)
        corrected_arrays.append(quantized)
        fractions.append(per_channel)
    return tuple(corrected_arrays), tuple(fractions), float(maximum)


def _write_bytes_fsync(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_rgb_png_fsync(path: Path, pixels: np.ndarray) -> str:
    values = np.asarray(pixels)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("corrected training RGB must be an HxWx3 uint8 array")
    height, width, _ = values.shape
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(row).tobytes() for row in values
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return _write_bytes_fsync(path, payload)


def _rgb_digest(frames: tuple[FrameArtifact, ...]) -> str:
    return hashlib.sha256(
        _strict_json_bytes(
            [
                {
                    "frame_id": frame.frame_id,
                    "image_name": frame.image_name,
                    "sha256": frame.sha256,
                }
                for frame in frames
            ]
        )
    ).hexdigest()


def _assert_inputs_unchanged(
    expected: _ValidatedInputs,
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    masks: MaskFusionEvidence,
) -> None:
    current = _validate_inputs(frames, scene, masks)
    if (
        current.source_frame_digest != expected.source_frame_digest
        or current.original_rgb_digest != expected.original_rgb_digest
        or current.scene_snapshot_digest != expected.scene_snapshot_digest
        or current.mask_snapshot_digest != expected.mask_snapshot_digest
    ):
        raise ValueError(
            "source, mask, depth, or geometry evidence changed during solve"
        )


def _publish(
    *,
    inputs: _ValidatedInputs,
    scene: RigidSceneEvidence,
    masks: MaskFusionEvidence,
    output_dir: Path,
    policy: PhotometricPolicy,
    reference_frame_id: str,
    transforms: tuple[RgbAffineTransform, ...],
    corrected_arrays: tuple[np.ndarray, ...],
    clip_fractions: tuple[tuple[float, float, float], ...],
    maximum_clip: float,
    training_tracks: tuple[_TrackSamples, ...],
    heldout_tracks: tuple[_TrackSamples, ...],
    heldout_before: float,
    heldout_after: float,
    rejection_reasons: tuple[str, ...],
) -> PhotometricEvidence:
    decision: Literal["accepted", "rejected"] = (
        "accepted" if not rejection_reasons else "rejected"
    )
    reference_index = tuple(frame.frame_id for frame in inputs.frames).index(
        reference_frame_id
    )
    support_counts = tuple(
        sum(frame_index in track.frame_indices for track in training_tracks)
        for frame_index in range(len(inputs.frames))
    )
    support_connected = bool(training_tracks) and _support_connected(
        len(inputs.frames), reference_index, training_tracks
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    training_frames: list[FrameArtifact] = []
    inventory: list[dict[str, object]] = []
    try:
        if decision == "accepted":
            for frame, pixels in zip(inputs.frames, corrected_arrays, strict=True):
                name_parts = _safe_image_name(frame.image_name)
                relative = Path(
                    "training_rgb",
                    *name_parts[:-1],
                    f"{name_parts[-1]}.png",
                )
                digest = _write_rgb_png_fsync(staging / relative, pixels)
                training_frame = FrameArtifact(
                    image_name=frame.image_name,
                    frame_id=frame.frame_id,
                    path=output_dir / relative,
                    sha256=digest,
                )
                training_frames.append(training_frame)
                inventory.append(
                    {
                        "frame_id": frame.frame_id,
                        "image_name": frame.image_name,
                        "path": relative.as_posix(),
                        "sha256": digest,
                    }
                )
            training_tuple = tuple(training_frames)
        else:
            training_tuple = inputs.frames
        training_rgb_digest = (
            _rgb_digest(training_tuple)
            if decision == "accepted"
            else inputs.original_rgb_digest
        )
        transform_payload = [
            {
                "additional_clip_fraction_rgb": list(clip_fraction),
                "bias_rgb": list(transform.bias_rgb),
                "frame_id": transform.frame_id,
                "gain_rgb": list(transform.gain_rgb),
            }
            for transform, clip_fraction in zip(transforms, clip_fractions, strict=True)
        ]
        report = {
            "decision": decision,
            "eligibility": {
                "conservative_sampling_footprint": "bilinear_2x2_all_outside_hard_exclude_and_uncertain",
                "minimum_observations_after_masking": 3,
                "reprojection_validation": "finite_nonnegative_producer_prefiltered_value",
                "scene_static_tracks_are_producer_prefiltered": True,
            },
            "geometry": {
                "depth_digest": scene.depth_digest,
                "geometry_digest": scene.geometry_digest,
                "input_rgb": "original_frames_only",
                "scene_snapshot_digest": inputs.scene_snapshot_digest,
            },
            "heldout": {
                "error_after": heldout_after,
                "error_before": heldout_before,
                "metric": "cross_frame_pairwise_rgb_rmse",
                "track_ids": [track.track_id for track in heldout_tracks],
            },
            "mask_set_digest": masks.mask_set_digest,
            "mask_snapshot_digest": inputs.mask_snapshot_digest,
            "maximum_additional_clip_fraction": maximum_clip,
            "original_rgb_digest": inputs.original_rgb_digest,
            "policy": _policy_payload(policy),
            "reference_frame_id": reference_frame_id,
            "rejection_reasons": list(rejection_reasons),
            "schema_version": 1,
            "solver": {
                "bias_bounds": [BIAS_MIN, BIAS_MAX],
                "formulation": "global_track_latent_rgb_affine",
                "gain_bounds": [GAIN_MIN, GAIN_MAX],
                "reference_gauge": "exact_identity",
                "robust_loss": "huber_irls",
                "robust_loss_delta": float(policy.robust_loss_delta),
                "temporal_edge_count": max(0, len(inputs.frames) - 1),
                "temporal_smoothness_weight": float(policy.temporal_smoothness_weight),
            },
            "source_frame_digest": inputs.source_frame_digest,
            "stage": "deterministic_photometric_validation",
            "support_graph": {
                "connected_to_reference": support_connected,
                "direct_reference_overlap_required": False,
                "training_track_count_by_frame": list(support_counts),
            },
            "training": {
                "corrected_inventory": inventory,
                "track_ids": [track.track_id for track in training_tracks],
            },
            "training_rgb_digest": training_rgb_digest,
            "transforms": transform_payload,
            "warnings": [],
        }
        report_sha256 = _write_bytes_fsync(
            staging / "photometric_report.json", _strict_json_bytes(report)
        )
        manifest = {
            "decision": decision,
            "depth_digest": scene.depth_digest,
            "geometry_digest": scene.geometry_digest,
            "heldout_error_after": heldout_after,
            "heldout_error_before": heldout_before,
            "heldout_track_ids": [track.track_id for track in heldout_tracks],
            "mask_set_digest": masks.mask_set_digest,
            "maximum_additional_clip_fraction": maximum_clip,
            "original_rgb_digest": inputs.original_rgb_digest,
            "policy": _policy_payload(policy),
            "reference_frame_id": reference_frame_id,
            "rejection_reasons": list(rejection_reasons),
            "report": {
                "path": "photometric_report.json",
                "sha256": report_sha256,
            },
            "schema_version": 1,
            "solver": {
                "bias_bounds": [BIAS_MIN, BIAS_MAX],
                "formulation": "global_track_latent_rgb_affine",
                "gain_bounds": [GAIN_MIN, GAIN_MAX],
                "reference_gauge": "exact_identity",
                "robust_loss": "huber_irls",
                "robust_loss_delta": float(policy.robust_loss_delta),
                "temporal_edge_count": max(0, len(inputs.frames) - 1),
                "temporal_smoothness_weight": float(policy.temporal_smoothness_weight),
            },
            "stage": "deterministic_photometric_validation",
            "support_graph": {
                "connected_to_reference": support_connected,
                "direct_reference_overlap_required": False,
                "training_track_count_by_frame": list(support_counts),
            },
            "training_inventory": inventory,
            "training_rgb_digest": training_rgb_digest,
        }
        manifest_sha256 = _write_bytes_fsync(
            staging / "manifest.json", _strict_json_bytes(manifest)
        )
        _assert_inputs_unchanged(inputs, inputs.frames, scene, masks)
        _fsync_directory_tree(staging)
        promote_directory(staging, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        if os.path.lexists(staging) and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return PhotometricEvidence(
        decision=decision,
        reference_frame_id=reference_frame_id,
        transforms=transforms,
        original_frames=inputs.frames,
        training_frames=training_tuple,
        original_rgb_digest=inputs.original_rgb_digest,
        training_rgb_digest=training_rgb_digest,
        heldout_error_before=heldout_before,
        heldout_error_after=heldout_after,
        maximum_additional_clip_fraction=maximum_clip,
        rejection_reasons=rejection_reasons,
        report_path=output_dir / "photometric_report.json",
        report_sha256=report_sha256,
        manifest_path=output_dir / "manifest.json",
        manifest_sha256=manifest_sha256,
    )


def fit_and_validate_photometric_transforms(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    masks: MaskFusionEvidence,
    output_dir: Path,
    *,
    reference_frame_id: str,
    policy: PhotometricPolicy,
) -> PhotometricEvidence:
    if not isinstance(policy, PhotometricPolicy):
        raise ValueError("policy must be a PhotometricPolicy")
    inputs = _validate_inputs(frames, scene, masks)
    output_dir = _validate_output_dir(output_dir)
    if not isinstance(reference_frame_id, str) or reference_frame_id not in {
        frame.frame_id for frame in frames
    }:
        raise ValueError("reference_frame_id must identify exactly one canonical frame")
    reference_index = tuple(frame.frame_id for frame in frames).index(
        reference_frame_id
    )
    eligible = _sample_tracks(inputs, scene)
    heldout = tuple(
        track
        for position, track in enumerate(eligible)
        if position % policy.heldout_stride == 0
    )
    training = tuple(
        track
        for position, track in enumerate(eligible)
        if position % policy.heldout_stride != 0
    )
    rejection_reasons: list[str] = []
    if not eligible:
        rejection_reasons.append("no_static_samples")
    if len(training) < 3 or not heldout:
        rejection_reasons.append("insufficient_static_samples")
    if training and not _support_connected(len(frames), reference_index, training):
        rejection_reasons.append("disconnected_static_support")
    transforms = _identity_transforms(frames)
    heldout_before = _heldout_error(heldout, None)
    heldout_after = heldout_before
    if not rejection_reasons:
        try:
            transforms = _solve_transforms(frames, training, reference_index, policy)
            heldout_after = _heldout_error(heldout, transforms)
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            transforms = _identity_transforms(frames)
            heldout_after = heldout_before
            rejection_reasons.append("numeric_failure")
    corrected_arrays, clip_fractions, maximum_clip = _corrected_arrays_and_clipping(
        inputs, transforms
    )
    if not rejection_reasons and not heldout_after < heldout_before:
        rejection_reasons.append("no_strict_heldout_improvement")
    if not rejection_reasons and maximum_clip > MAX_ADDITIONAL_CLIP_FRACTION:
        rejection_reasons.append("additional_clipping_exceeded")
    _assert_inputs_unchanged(inputs, frames, scene, masks)
    return _publish(
        inputs=inputs,
        scene=scene,
        masks=masks,
        output_dir=output_dir,
        policy=policy,
        reference_frame_id=reference_frame_id,
        transforms=transforms,
        corrected_arrays=corrected_arrays,
        clip_fractions=clip_fractions,
        maximum_clip=maximum_clip,
        training_tracks=training,
        heldout_tracks=heldout,
        heldout_before=heldout_before,
        heldout_after=heldout_after,
        rejection_reasons=tuple(rejection_reasons),
    )
