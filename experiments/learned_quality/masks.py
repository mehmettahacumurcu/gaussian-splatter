from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from io import BytesIO
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Final, Literal

import numpy as np
from PIL import Image
from scipy import ndimage

from backend.static_pipeline.stage_cache import promote_directory

from .contracts import FrameArtifact
from .flow import (
    FlowGatePolicy,
    MotionEvidence,
    MotionFrameEvidence,
    RigidFrameEvidence,
    RigidSceneEvidence,
    StaticTrack,
    TrackObservation,
)
from .segmentation import (
    SemanticEvidence,
    SemanticFrameEvidence,
    SemanticPolicy,
    TRANSIENT_PROMPT,
)


RUNAWAY_TRIM_FRACTION: Final[float] = 0.45
RUNAWAY_REJECT_FRACTION: Final[float] = 0.55


@dataclass(frozen=True)
class MaskFusionPolicy:
    boundary_dilation_fraction: float
    opening_fraction: float
    minimum_component_area_fraction: float
    temporal_area_change_limit: float
    propagation_collapse_limit: float
    sky_far_depth_quantile: float
    sky_track_support_radius_fraction: float
    sky_max_component_support_fraction: float
    sky_frame_min_fraction: float

    def __post_init__(self) -> None:
        for field_name in (
            "boundary_dilation_fraction",
            "opening_fraction",
            "minimum_component_area_fraction",
            "temporal_area_change_limit",
            "propagation_collapse_limit",
            "sky_far_depth_quantile",
            "sky_track_support_radius_fraction",
            "sky_max_component_support_fraction",
            "sky_frame_min_fraction",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > 1.0
            ):
                raise ValueError(f"{field_name} must be a finite number in [0, 1]")
        if float(self.sky_frame_min_fraction) == 0.0:
            raise ValueError("sky_frame_min_fraction must be greater than zero")


@dataclass(frozen=True)
class FusedMaskFrame:
    frame: FrameArtifact
    width: int
    height: int
    semantic_confirmed_path: Path
    semantic_confirmed_sha256: str
    sky_confirmed_path: Path
    sky_confirmed_sha256: str
    motion_confirmed_path: Path
    motion_confirmed_sha256: str
    uncertain_path: Path
    uncertain_sha256: str
    hard_exclude_path: Path
    hard_exclude_sha256: str
    colmap_keep_path: Path
    colmap_keep_sha256: str
    training_validity_path: Path
    training_validity_sha256: str
    exclusion_fraction_before_trim: float
    exclusion_fraction_after_trim: float
    decision: Literal["accepted", "trimmed_motion", "rejected_unmasked"]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MaskFusionEvidence:
    policy: MaskFusionPolicy
    frames: tuple[FusedMaskFrame, ...]
    mask_set_digest: str
    manifest_path: Path
    manifest_sha256: str


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


def _safe_image_name(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("frame image_name must be a safe relative path")
    pure = PurePosixPath(value)
    parts = pure.parts
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(
            "frame image_name must be canonical and must not contain path traversal"
        )
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
    }
    if any(
        part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in reserved
        for part in parts
    ):
        raise ValueError("frame image_name contains an unsafe platform component")
    return parts


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if os.path.lexists(current) and current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


@dataclass
class _ArtifactRegistry:
    path_keys: set[str]
    file_keys: set[tuple[int, int]]
    snapshots: list[tuple[Path, str]]

    @classmethod
    def create(cls) -> _ArtifactRegistry:
        return cls(set(), set(), [])

    def register(self, path: object, digest: object, label: str) -> Path:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{label} path must be an absolute Path")
        if ".." in path.parts:
            raise ValueError(f"{label} path must not contain traversal")
        if (
            not os.path.lexists(path)
            or _has_symlink_component(path)
            or not path.is_file()
        ):
            raise ValueError(f"{label} path must be a regular non-symlink file")
        expected = _require_sha256(digest, f"{label} sha256")
        resolved = path.resolve(strict=True)
        path_key = os.path.normcase(str(resolved))
        stat = resolved.stat()
        file_key = (int(stat.st_dev), int(stat.st_ino))
        if path_key in self.path_keys or file_key in self.file_keys:
            raise ValueError(f"{label} path collides with another input artifact")
        actual = _sha256_path(path)
        if actual != expected:
            raise ValueError(f"{label} sha256 does not match artifact bytes")
        self.path_keys.add(path_key)
        self.file_keys.add(file_key)
        self.snapshots.append((path, expected))
        return path


def _source_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as error:
        raise ValueError(f"source frame is unreadable: {path}") from error
    if width <= 0 or height <= 0:
        raise ValueError("source frame dimensions must be positive")
    return int(width), int(height)


def _validate_binary_png(
    path: Path,
    shape: tuple[int, int],
    label: str,
) -> None:
    try:
        with Image.open(path) as image:
            if image.mode != "L" or image.format != "PNG":
                raise ValueError(f"{label} must be a grayscale PNG")
            array = np.asarray(image, dtype=np.uint8)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{label} must be a readable grayscale PNG") from error
    if array.shape != shape:
        raise ValueError(f"{label} dimensions must exactly match the source frame")
    if not np.all((array == 0) | (array == 255)):
        raise ValueError(f"{label} must contain only binary 0/255 values")


def _validate_float32_npy(
    path: Path,
    shape: tuple[int, int],
    label: str,
) -> None:
    try:
        raw = _load_npy(path)
    except Exception as error:
        raise ValueError(f"{label} must be a deterministic NPY array") from error
    if raw.dtype != np.dtype(np.float32) or raw.shape != shape:
        raise ValueError(f"{label} must be float32 at exact source dimensions")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError(f"{label} must contain finite nonnegative values")


def _validate_bool_npy(path: Path, label: str) -> tuple[int, int]:
    try:
        raw = _load_npy(path)
    except Exception as error:
        raise ValueError(f"{label} must be a deterministic NPY array") from error
    if raw.dtype != np.dtype(np.bool_) or raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional boolean NPY")
    return int(raw.shape[0]), int(raw.shape[1])


def _validate_depth_npy(
    path: Path,
    shape: tuple[int, int],
    label: str,
) -> None:
    try:
        raw = _load_npy(path)
    except Exception as error:
        raise ValueError(f"{label} must be a deterministic NPY array") from error
    if raw.shape != shape or raw.dtype.kind != "f":
        raise ValueError(f"{label} must be floating point at exact source dimensions")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError(f"{label} must contain finite nonnegative depth")


def _frame_set_digest(frames: tuple[FrameArtifact, ...]) -> str:
    return hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "image_name": frame.image_name,
                        "sha256": frame.sha256,
                        "size_bytes": frame.path.stat().st_size,
                    }
                    for frame in frames
                ],
                "schema": "learned_quality.input_frames.v1",
            }
        )
    ).hexdigest()


def _expected_scene_digest(
    frames: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
) -> str:
    rigid_payload = [
        {
            "depth_sha256": rigid.depth_sha256,
            "depth_size_bytes": (
                None if rigid.depth_path is None else rigid.depth_path.stat().st_size
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
        for frame, rigid in zip(frames, scene.frames, strict=True)
    ]
    static_tracks_payload = [
        {
            "mean_reprojection_error": float(track.mean_reprojection_error),
            "observations": [
                {
                    "frame_id": observation.frame_id,
                    "x": float(observation.x),
                    "y": float(observation.y),
                }
                for observation in track.observations
            ],
            "track_id": track.track_id,
            "xyz": [float(value) for value in track.xyz],
        }
        for track in scene.static_tracks
    ]
    return hashlib.sha256(
        _strict_json_bytes(
            {
                "depth_digest": scene.depth_digest,
                "frames": rigid_payload,
                "geometry_digest": scene.geometry_digest,
                "input_frame_digest": _frame_set_digest(frames),
                "static_tracks": static_tracks_payload,
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class _ValidatedInputs:
    dimensions: tuple[tuple[int, int], ...]
    snapshots: tuple[tuple[Path, str], ...]
    sky_processed_shapes: tuple[tuple[int, int], ...]


def _input_binding_payload(
    frames: tuple[FrameArtifact, ...],
    semantic: SemanticEvidence,
    motion: MotionEvidence,
    sky_proposals: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    validated: _ValidatedInputs,
) -> dict[str, object]:
    source_frames = [
        {
            "frame_id": frame.frame_id,
            "height": validated.dimensions[index][1],
            "image_name": frame.image_name,
            "sha256": frame.sha256,
            "size_bytes": frame.path.stat().st_size,
            "width": validated.dimensions[index][0],
        }
        for index, frame in enumerate(frames)
    ]
    semantic_frames = [
        {
            "direct_confirmed_sha256": record.direct_confirmed_sha256,
            "direct_support_sha256": record.direct_support_sha256,
            "frame_id": record.frame.frame_id,
            "height": record.height,
            "propagated_candidate_sha256": record.propagated_candidate_sha256,
            "width": record.width,
        }
        for record in semantic.frames
    ]
    motion_frames = [
        {
            "confirmed_without_semantic_sha256": (
                record.confirmed_without_semantic_sha256
            ),
            "frame_id": record.frame.frame_id,
            "height": record.height,
            "requires_semantic_sha256": record.requires_semantic_sha256,
            "strength_sha256": record.strength_sha256,
            "uncertain_sha256": record.uncertain_sha256,
            "width": record.width,
        }
        for record in motion.frames
    ]
    sky_payload = [
        {
            "frame_id": proposal.frame_id,
            "image_name": proposal.image_name,
            "processed_shape": list(validated.sky_processed_shapes[index]),
            "resized": validated.sky_processed_shapes[index]
            != (
                validated.dimensions[index][1],
                validated.dimensions[index][0],
            ),
            "sha256": proposal.sha256,
            "size_bytes": proposal.path.stat().st_size,
            "source_shape": [
                validated.dimensions[index][1],
                validated.dimensions[index][0],
            ],
        }
        for index, proposal in enumerate(sky_proposals)
    ]
    rigid_frames = [
        {
            "depth_sha256": rigid.depth_sha256,
            "depth_size_bytes": (
                None if rigid.depth_path is None else rigid.depth_path.stat().st_size
            ),
            "frame_id": rigid.frame.frame_id,
            "height": rigid.height,
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
        for rigid in scene.frames
    ]
    static_tracks = [
        {
            "mean_reprojection_error": float(track.mean_reprojection_error),
            "observations": [
                {
                    "frame_id": observation.frame_id,
                    "x": float(observation.x),
                    "y": float(observation.y),
                }
                for observation in track.observations
            ],
            "track_id": track.track_id,
            "xyz": [float(value) for value in track.xyz],
        }
        for track in scene.static_tracks
    ]
    return {
        "frames": source_frames,
        "input_frame_digest": _frame_set_digest(frames),
        "motion": {
            "frames": motion_frames,
            "manifest_sha256": motion.manifest_sha256,
            "pair_manifest_sha256": motion.pair_manifest_sha256,
            "policy": asdict(motion.policy),
            "scene_digest": motion.scene_digest,
        },
        "scene": {
            "depth_digest": scene.depth_digest,
            "frames": rigid_frames,
            "geometry_digest": scene.geometry_digest,
            "static_tracks": static_tracks,
        },
        "semantic": {
            "frames": semantic_frames,
            "manifest_sha256": semantic.manifest_sha256,
            "policy": asdict(semantic.policy),
            "prompt": semantic.prompt,
        },
        "sky_proposals": sky_payload,
    }


def _validate_output_dir(output_dir: object) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    if output_dir.name in {"", ".", ".."} or ".." in output_dir.parts:
        raise ValueError("output_dir must have a safe final component")
    if any(unicodedata.category(character) == "Cc" for character in output_dir.name):
        raise ValueError("output_dir must have a control-free final component")
    if os.path.lexists(output_dir):
        raise FileExistsError("mask fusion output_dir must be previously absent")
    parent = output_dir.parent
    if (
        not os.path.lexists(parent)
        or _has_symlink_component(parent)
        or not parent.is_dir()
    ):
        raise ValueError("output_dir parent must be an existing non-symlink directory")
    return output_dir


def _validate_inputs(
    frames: tuple[FrameArtifact, ...],
    semantic: SemanticEvidence,
    motion: MotionEvidence,
    sky_proposals: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    output_dir: Path,
    policy: MaskFusionPolicy,
) -> _ValidatedInputs:
    _validate_output_dir(output_dir)
    if not isinstance(policy, MaskFusionPolicy):
        raise ValueError("policy must be a MaskFusionPolicy")
    if type(frames) is not tuple or not frames:
        raise ValueError("frames must be a non-empty canonical tuple")
    registry = _ArtifactRegistry.create()
    frame_ids: set[str] = set()
    image_names: set[str] = set()
    output_keys: set[str] = set()
    dimensions: list[tuple[int, int]] = []
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
        if frame.frame_id in frame_ids or frame.image_name in image_names:
            raise ValueError("canonical frames must have unique ids and image names")
        output_key = "/".join(parts).casefold() + ".png"
        if output_key in output_keys:
            raise ValueError("canonical image names create an output path collision")
        source_path = registry.register(frame.path, frame.sha256, "source frame")
        if tuple(source_path.parts[-len(parts) :]) != parts:
            raise ValueError("frame image_name must match the source path suffix")
        frame_ids.add(frame.frame_id)
        image_names.add(frame.image_name)
        output_keys.add(output_key)
        dimensions.append(_source_dimensions(source_path))

    if (
        not isinstance(semantic, SemanticEvidence)
        or not isinstance(semantic.policy, SemanticPolicy)
        or semantic.prompt != TRANSIENT_PROMPT
    ):
        raise ValueError("semantic evidence must preserve the exact transient prompt")
    if type(semantic.frames) is not tuple or len(semantic.frames) != len(frames):
        raise ValueError(
            "semantic evidence must contain the exact canonical frame tuple"
        )
    if tuple(record.frame for record in semantic.frames) != frames:
        raise ValueError("semantic evidence must preserve exact frame order and joins")
    for index, (record, frame, (width, height)) in enumerate(
        zip(semantic.frames, frames, dimensions, strict=True)
    ):
        if not isinstance(record, SemanticFrameEvidence) or record.frame != frame:
            raise ValueError("semantic frame records must preserve exact joins")
        if (
            type(record.width) is not int
            or type(record.height) is not int
            or (record.width, record.height) != (width, height)
        ):
            raise ValueError("semantic dimensions must exactly match source images")
        for field_name in (
            "direct_confirmed",
            "propagated_candidate",
            "direct_support",
        ):
            path = registry.register(
                getattr(record, f"{field_name}_path"),
                getattr(record, f"{field_name}_sha256"),
                f"semantic frame {index} {field_name}",
            )
            _validate_binary_png(path, (height, width), f"semantic {field_name}")
    registry.register(
        semantic.manifest_path,
        semantic.manifest_sha256,
        "semantic manifest",
    )

    if not isinstance(scene, RigidSceneEvidence):
        raise ValueError("scene must be RigidSceneEvidence")
    _require_sha256(scene.geometry_digest, "geometry_digest")
    _require_sha256(scene.depth_digest, "depth_digest")
    if type(scene.frames) is not tuple or len(scene.frames) != len(frames):
        raise ValueError("scene must contain the exact canonical frame tuple")
    if tuple(record.frame for record in scene.frames) != frames:
        raise ValueError("scene must preserve exact frame order and joins")
    for index, (record, frame, (width, height)) in enumerate(
        zip(scene.frames, frames, dimensions, strict=True)
    ):
        if not isinstance(record, RigidFrameEvidence) or record.frame != frame:
            raise ValueError("rigid frame records must preserve exact joins")
        if (
            type(record.width) is not int
            or type(record.height) is not int
            or (record.width, record.height) != (width, height)
        ):
            raise ValueError("rigid dimensions must exactly match source images")
        if type(record.registered) is not bool:
            raise ValueError("rigid registered must be a boolean")
        if record.registered:
            if (
                type(record.w2c_4x4) is not tuple
                or len(record.w2c_4x4) != 16
                or not all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in record.w2c_4x4
                )
            ):
                raise ValueError("registered rigid pose must contain 16 finite values")
            w2c = np.asarray(record.w2c_4x4, dtype=np.float64).reshape(4, 4)
            rotation = w2c[:3, :3]
            if (
                not np.array_equal(
                    w2c[3], np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                )
                or not np.allclose(
                    rotation @ rotation.T,
                    np.eye(3),
                    rtol=0.0,
                    atol=1e-6,
                )
                or not math.isclose(
                    float(np.linalg.det(rotation)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("registered rigid pose must be a proper affine W2C")
            if (
                type(record.pinhole_fx_fy_cx_cy) is not tuple
                or len(record.pinhole_fx_fy_cx_cy) != 4
                or not all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in record.pinhole_fx_fy_cx_cy
                )
                or float(record.pinhole_fx_fy_cx_cy[0]) <= 0.0
                or float(record.pinhole_fx_fy_cx_cy[1]) <= 0.0
                or not (0.0 <= float(record.pinhole_fx_fy_cx_cy[2]) < width)
                or not (0.0 <= float(record.pinhole_fx_fy_cx_cy[3]) < height)
            ):
                raise ValueError(
                    "registered rigid intrinsics must be finite and physical"
                )
            path = registry.register(
                record.depth_path,
                record.depth_sha256,
                f"rigid frame {index} depth",
            )
            _validate_depth_npy(path, (height, width), "rigid depth")
        elif any(
            value is not None
            for value in (
                record.w2c_4x4,
                record.pinhole_fx_fy_cx_cy,
                record.depth_path,
                record.depth_sha256,
            )
        ):
            raise ValueError("unregistered rigid frame must not claim geometry")

    if type(scene.static_tracks) is not tuple:
        raise ValueError("static_tracks must be an immutable tuple")
    track_ids: set[int] = set()
    dimensions_by_id = {
        frame.frame_id: dimensions[index] for index, frame in enumerate(frames)
    }
    for track in scene.static_tracks:
        if (
            not isinstance(track, StaticTrack)
            or type(track.track_id) is not int
            or track.track_id < 0
        ):
            raise ValueError("static tracks must use immutable typed records")
        if track.track_id in track_ids:
            raise ValueError("static track ids must be unique")
        track_ids.add(track.track_id)
        if (
            type(track.xyz) is not tuple
            or len(track.xyz) != 3
            or not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in track.xyz
            )
            or not isinstance(track.mean_reprojection_error, Real)
            or isinstance(track.mean_reprojection_error, bool)
            or not math.isfinite(float(track.mean_reprojection_error))
            or float(track.mean_reprojection_error) < 0.0
            or type(track.observations) is not tuple
        ):
            raise ValueError("static track geometry must be finite and immutable")
        observed_ids: set[str] = set()
        for observation in track.observations:
            if not isinstance(observation, TrackObservation):
                raise ValueError("track observations must use TrackObservation")
            if (
                observation.frame_id not in dimensions_by_id
                or observation.frame_id in observed_ids
            ):
                raise ValueError("track observations must join unique canonical frames")
            width, height = dimensions_by_id[observation.frame_id]
            if (
                not isinstance(observation.x, Real)
                or isinstance(observation.x, bool)
                or not isinstance(observation.y, Real)
                or isinstance(observation.y, bool)
                or not math.isfinite(float(observation.x))
                or not math.isfinite(float(observation.y))
                or not (0.0 <= float(observation.x) < width)
                or not (0.0 <= float(observation.y) < height)
            ):
                raise ValueError(
                    "track observation coordinates must be finite and in bounds"
                )
            observed_ids.add(observation.frame_id)

    if not isinstance(motion, MotionEvidence) or not isinstance(
        motion.policy, FlowGatePolicy
    ):
        raise ValueError("motion must be MotionEvidence")
    if motion.scene_digest != _expected_scene_digest(frames, scene):
        raise ValueError("motion scene_digest does not bind the canonical rigid scene")
    if type(motion.frames) is not tuple or len(motion.frames) != len(frames):
        raise ValueError("motion evidence must contain the exact canonical frame tuple")
    if tuple(record.frame for record in motion.frames) != frames:
        raise ValueError("motion evidence must preserve exact frame order and joins")
    for index, (record, frame, (width, height)) in enumerate(
        zip(motion.frames, frames, dimensions, strict=True)
    ):
        if not isinstance(record, MotionFrameEvidence) or record.frame != frame:
            raise ValueError("motion frame records must preserve exact joins")
        if (
            type(record.width) is not int
            or type(record.height) is not int
            or (record.width, record.height) != (width, height)
        ):
            raise ValueError("motion dimensions must exactly match source images")
        for field_name in (
            "confirmed_without_semantic",
            "requires_semantic",
            "uncertain",
        ):
            path = registry.register(
                getattr(record, f"{field_name}_path"),
                getattr(record, f"{field_name}_sha256"),
                f"motion frame {index} {field_name}",
            )
            _validate_binary_png(path, (height, width), f"motion {field_name}")
        strength_path = registry.register(
            record.strength_path,
            record.strength_sha256,
            f"motion frame {index} strength",
        )
        _validate_float32_npy(strength_path, (height, width), "motion strength")
    registry.register(
        motion.pair_manifest_path,
        motion.pair_manifest_sha256,
        "motion pair manifest",
    )
    registry.register(motion.manifest_path, motion.manifest_sha256, "motion manifest")

    if type(sky_proposals) is not tuple or len(sky_proposals) != len(frames):
        raise ValueError("sky proposals must contain the exact canonical frame tuple")
    sky_joins = tuple((item.frame_id, item.image_name) for item in sky_proposals)
    canonical_joins = tuple((item.frame_id, item.image_name) for item in frames)
    if sky_joins != canonical_joins:
        raise ValueError("sky proposals must preserve exact frame order and joins")
    sky_shapes: list[tuple[int, int]] = []
    for index, proposal in enumerate(sky_proposals):
        if not isinstance(proposal, FrameArtifact):
            raise ValueError("sky proposals must contain FrameArtifact records")
        path = registry.register(
            proposal.path, proposal.sha256, f"sky proposal {index}"
        )
        sky_shapes.append(_validate_bool_npy(path, "sky proposal"))

    return _ValidatedInputs(
        dimensions=tuple(dimensions),
        snapshots=tuple(registry.snapshots),
        sky_processed_shapes=tuple(sky_shapes),
    )


def _load_binary_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) == 255


def _write_binary_png(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    payload = buffer.getvalue()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


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


def _revalidate_snapshots(snapshots: tuple[tuple[Path, str], ...]) -> None:
    for path, expected_digest in snapshots:
        if (
            not os.path.lexists(path)
            or _has_symlink_component(path)
            or not path.is_file()
        ):
            raise ValueError("an upstream/source artifact changed before publication")
        if _sha256_path(path) != expected_digest:
            raise ValueError(
                "an upstream/source artifact digest changed before publication"
            )


def _load_npy(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return np.array(mask, dtype=bool, copy=True)
    height, width = shape
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return (
        np.asarray(
            image.resize((width, height), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        == 255
    )


def _track_support(
    frame_id: str,
    shape: tuple[int, int],
    scene: RigidSceneEvidence,
    radius_fraction: float,
) -> np.ndarray:
    height, width = shape
    radius = int(math.ceil(min(height, width) * float(radius_fraction)))
    support = np.zeros(shape, dtype=bool)
    for track in scene.static_tracks:
        for observation in track.observations:
            if observation.frame_id != frame_id:
                continue
            center_x = int(math.floor(float(observation.x) + 0.5))
            center_y = int(math.floor(float(observation.y) + 0.5))
            if not (0 <= center_x < width and 0 <= center_y < height):
                continue
            if radius == 0:
                support[center_y, center_x] = True
                continue
            y0 = max(0, center_y - radius)
            y1 = min(height, center_y + radius + 1)
            x0 = max(0, center_x - radius)
            x1 = min(width, center_x + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            support[y0:y1, x0:x1] |= (yy - center_y) ** 2 + (
                xx - center_x
            ) ** 2 <= radius**2
    return support


def _classify_sky(
    frame: FrameArtifact,
    proposal_artifact: FrameArtifact,
    rigid_frame: object,
    scene: RigidSceneEvidence,
    shape: tuple[int, int],
    policy: MaskFusionPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    proposal_raw = _load_npy(proposal_artifact.path)
    proposal = _resize_nearest(np.asarray(proposal_raw, dtype=bool), shape)
    registered = bool(getattr(rigid_frame, "registered", False))
    depth_path = getattr(rigid_frame, "depth_path", None)
    if not registered or depth_path is None:
        return np.zeros(shape, dtype=bool), proposal
    depth = np.asarray(_load_npy(depth_path), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if depth.shape != shape or not np.any(valid):
        return np.zeros(shape, dtype=bool), proposal
    threshold = float(
        np.quantile(
            depth[valid],
            float(policy.sky_far_depth_quantile),
            method="linear",
        )
    )
    far = valid & (depth >= threshold)
    support = _track_support(
        frame.frame_id,
        shape,
        scene,
        float(policy.sky_track_support_radius_fraction),
    )
    labels, count = ndimage.label(proposal, structure=np.ones((3, 3), dtype=bool))
    confirmed = np.zeros(shape, dtype=bool)
    for component_id in range(1, count + 1):
        component = labels == component_id
        component_area = int(np.count_nonzero(component))
        support_fraction = float(np.count_nonzero(component & support)) / component_area
        if support_fraction <= float(policy.sky_max_component_support_fraction):
            confirmed |= component & far & ~support
    return confirmed, proposal & ~confirmed


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy**2 + xx**2) <= radius**2


def _morphology_filter(
    candidate: np.ndarray,
    policy: MaskFusionPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = np.array(candidate, dtype=bool, copy=True)
    height, width = filtered.shape
    opening_radius = int(math.ceil(min(height, width) * float(policy.opening_fraction)))
    if opening_radius > 0 and np.any(filtered):
        filtered = ndimage.binary_opening(
            filtered,
            structure=_disk_structure(opening_radius),
        )
    minimum_area = int(
        math.ceil(height * width * float(policy.minimum_component_area_fraction))
    )
    if minimum_area > 1 and np.any(filtered):
        labels, count = ndimage.label(
            filtered,
            structure=np.ones((3, 3), dtype=bool),
        )
        kept = np.zeros_like(filtered)
        for component_id in range(1, count + 1):
            component = labels == component_id
            if int(np.count_nonzero(component)) >= minimum_area:
                kept |= component
        filtered = kept
    return filtered, candidate & ~filtered


def _artifact_token(index: int, frame: FrameArtifact) -> str:
    suffix = hashlib.sha256(
        f"{frame.frame_id}\0{frame.image_name}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{index:06d}-{suffix}"


@dataclass
class _FrameMaps:
    direct: np.ndarray
    semantic_confirmed: np.ndarray
    sky_confirmed: np.ndarray
    motion_confirmed: np.ndarray
    uncertain: np.ndarray
    strength: np.ndarray


def fuse_evidence_masks(
    frames: tuple[FrameArtifact, ...],
    semantic: SemanticEvidence,
    motion: MotionEvidence,
    sky_proposals: tuple[FrameArtifact, ...],
    scene: RigidSceneEvidence,
    output_dir: Path,
    *,
    policy: MaskFusionPolicy,
) -> MaskFusionEvidence:
    validated_inputs = _validate_inputs(
        frames,
        semantic,
        motion,
        sky_proposals,
        scene,
        output_dir,
        policy,
    )
    computed: list[_FrameMaps] = []
    for index, (
        frame,
        semantic_frame,
        motion_frame,
        sky_artifact,
        rigid_frame,
    ) in enumerate(
        zip(
            frames,
            semantic.frames,
            motion.frames,
            sky_proposals,
            scene.frames,
            strict=True,
        )
    ):
        direct = _load_binary_png(semantic_frame.direct_confirmed_path)
        propagated = _load_binary_png(semantic_frame.propagated_candidate_path)
        direct_support = _load_binary_png(semantic_frame.direct_support_path)
        confirmed_without_semantic = _load_binary_png(
            motion_frame.confirmed_without_semantic_path
        )
        requires_semantic = _load_binary_png(motion_frame.requires_semantic_path)
        producer_uncertain = _load_binary_png(motion_frame.uncertain_path)
        motion_confirmed = confirmed_without_semantic | (requires_semantic & direct)
        semantic_confirmed = direct | (propagated & (direct_support | motion_confirmed))
        propagation_candidate = propagated & ~direct
        propagation_collapse_uncertain = np.zeros_like(direct)
        if float(np.mean(propagation_candidate)) > float(
            policy.propagation_collapse_limit
        ):
            propagation_collapse_uncertain = propagation_candidate
            semantic_confirmed &= ~propagation_candidate
            semantic_confirmed |= direct
        unsupported_propagation = propagated & ~(direct_support | motion_confirmed)
        unsupported_one_sided = requires_semantic & ~direct
        sky_confirmed, sky_uncertain = _classify_sky(
            frame,
            sky_artifact,
            rigid_frame,
            scene,
            direct.shape,
            policy,
        )
        semantic_candidate, semantic_morphology_uncertain = _morphology_filter(
            semantic_confirmed & ~direct,
            policy,
        )
        semantic_confirmed = direct | semantic_candidate
        motion_confirmed, motion_morphology_uncertain = _morphology_filter(
            motion_confirmed,
            policy,
        )
        uncertain = (
            producer_uncertain
            | unsupported_propagation
            | unsupported_one_sided
            | sky_uncertain
            | semantic_morphology_uncertain
            | motion_morphology_uncertain
            | propagation_collapse_uncertain
        )
        computed.append(
            _FrameMaps(
                direct=direct,
                semantic_confirmed=semantic_confirmed,
                sky_confirmed=sky_confirmed,
                motion_confirmed=motion_confirmed,
                uncertain=uncertain,
                strength=np.asarray(
                    _load_npy(motion_frame.strength_path), dtype=np.float32
                ),
            )
        )

    initial_areas = tuple(
        float(
            np.mean(
                frame_maps.semantic_confirmed
                | frame_maps.sky_confirmed
                | frame_maps.motion_confirmed
            )
        )
        for frame_maps in computed
    )
    for index, frame_maps in enumerate(computed):
        neighbor_areas = []
        if index > 0:
            neighbor_areas.append(initial_areas[index - 1])
        if index + 1 < len(computed):
            neighbor_areas.append(initial_areas[index + 1])
        if neighbor_areas and initial_areas[index] - max(neighbor_areas) > float(
            policy.temporal_area_change_limit
        ):
            semantic_candidate = frame_maps.semantic_confirmed & ~frame_maps.direct
            motion_candidate = np.array(frame_maps.motion_confirmed, copy=True)
            frame_maps.semantic_confirmed &= ~semantic_candidate
            frame_maps.semantic_confirmed |= frame_maps.direct
            frame_maps.motion_confirmed &= False
            frame_maps.uncertain |= semantic_candidate | motion_candidate

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        fused_frames: list[FusedMaskFrame] = []
        digest_inventory: list[dict[str, object]] = []
        for index, (frame, frame_maps) in enumerate(zip(frames, computed, strict=True)):
            semantic_confirmed = frame_maps.semantic_confirmed
            sky_confirmed = frame_maps.sky_confirmed
            motion_confirmed = frame_maps.motion_confirmed
            uncertain = frame_maps.uncertain
            boundary_radius = int(
                math.ceil(
                    min(semantic_confirmed.shape)
                    * float(policy.boundary_dilation_fraction)
                )
            )
            if boundary_radius > 0:
                structure = _disk_structure(boundary_radius)
                if np.any(semantic_confirmed):
                    semantic_confirmed = ndimage.binary_dilation(
                        semantic_confirmed,
                        structure=structure,
                    )
                if np.any(sky_confirmed):
                    sky_confirmed = ndimage.binary_dilation(
                        sky_confirmed,
                        structure=structure,
                    )
                    sky_confirmed &= ~_track_support(
                        frame.frame_id,
                        sky_confirmed.shape,
                        scene,
                        float(policy.sky_track_support_radius_fraction),
                    )
                if np.any(motion_confirmed):
                    motion_confirmed = ndimage.binary_dilation(
                        motion_confirmed,
                        structure=structure,
                    )
            hard_exclude = semantic_confirmed | sky_confirmed | motion_confirmed
            fraction_before_trim = float(np.mean(hard_exclude))
            decision: Literal["accepted", "trimmed_motion", "rejected_unmasked"] = (
                "accepted"
            )
            warnings: tuple[str, ...] = ()
            is_sky_frame = float(np.mean(sky_confirmed)) >= float(
                policy.sky_frame_min_fraction
            )
            if not is_sky_frame and fraction_before_trim > RUNAWAY_TRIM_FRACTION:
                motion_only = motion_confirmed & ~(semantic_confirmed | sky_confirmed)
                labels, count = ndimage.label(
                    motion_only,
                    structure=np.ones((3, 3), dtype=bool),
                )
                ranked_components: list[tuple[float, int, np.ndarray]] = []
                for component_id in range(1, count + 1):
                    component = labels == component_id
                    flat_indices = np.flatnonzero(component)
                    producer_component = component & frame_maps.motion_confirmed
                    strength_pixels = (
                        producer_component if np.any(producer_component) else component
                    )
                    ranked_components.append(
                        (
                            float(np.mean(frame_maps.strength[strength_pixels])),
                            int(flat_indices[0]),
                            component,
                        )
                    )
                for _mean_strength, _tie_key, component in sorted(
                    ranked_components,
                    key=lambda item: (item[0], item[1]),
                ):
                    if float(np.mean(hard_exclude)) <= RUNAWAY_TRIM_FRACTION:
                        break
                    motion_confirmed &= ~component
                    uncertain |= component
                    hard_exclude = semantic_confirmed | sky_confirmed | motion_confirmed
                    decision = "trimmed_motion"
            fraction_after_trim = float(np.mean(hard_exclude))
            if not is_sky_frame and fraction_after_trim > RUNAWAY_REJECT_FRACTION:
                uncertain |= hard_exclude
                semantic_confirmed = np.zeros_like(semantic_confirmed)
                sky_confirmed = np.zeros_like(sky_confirmed)
                motion_confirmed = np.zeros_like(motion_confirmed)
                hard_exclude = np.zeros_like(hard_exclude)
                decision = "rejected_unmasked"
                warnings = (
                    "learned mask rejected: exclusion fraction "
                    f"{fraction_after_trim:.6f} exceeds "
                    f"{RUNAWAY_REJECT_FRACTION:.2f}",
                )
            colmap_keep = ~hard_exclude
            training_validity = ~hard_exclude
            token = _artifact_token(index, frame)
            relative_paths = {
                "semantic_confirmed": Path("maps") / token / "semantic_confirmed.png",
                "sky_confirmed": Path("maps") / token / "sky_confirmed.png",
                "motion_confirmed": Path("maps") / token / "motion_confirmed.png",
                "uncertain": Path("maps") / token / "uncertain.png",
                "hard_exclude": Path("maps") / token / "hard_exclude.png",
                "colmap_keep": Path("colmap_masks")
                / Path(*frame.image_name.split("/")).with_name(
                    f"{PurePosixPath(frame.image_name).name}.png"
                ),
                "training_validity": Path("training_validity") / f"{token}.png",
            }
            staged_paths = {
                name: staging / relative_path
                for name, relative_path in relative_paths.items()
            }
            final_paths = {
                name: output_dir / relative_path
                for name, relative_path in relative_paths.items()
            }
            maps = {
                "semantic_confirmed": semantic_confirmed,
                "sky_confirmed": sky_confirmed,
                "motion_confirmed": motion_confirmed,
                "uncertain": uncertain,
                "hard_exclude": hard_exclude,
                "colmap_keep": colmap_keep,
                "training_validity": training_validity,
            }
            digests = {
                name: _write_binary_png(staged_paths[name], value)
                for name, value in maps.items()
            }
            height, width = hard_exclude.shape
            fused_frames.append(
                FusedMaskFrame(
                    frame=frame,
                    width=width,
                    height=height,
                    semantic_confirmed_path=final_paths["semantic_confirmed"],
                    semantic_confirmed_sha256=digests["semantic_confirmed"],
                    sky_confirmed_path=final_paths["sky_confirmed"],
                    sky_confirmed_sha256=digests["sky_confirmed"],
                    motion_confirmed_path=final_paths["motion_confirmed"],
                    motion_confirmed_sha256=digests["motion_confirmed"],
                    uncertain_path=final_paths["uncertain"],
                    uncertain_sha256=digests["uncertain"],
                    hard_exclude_path=final_paths["hard_exclude"],
                    hard_exclude_sha256=digests["hard_exclude"],
                    colmap_keep_path=final_paths["colmap_keep"],
                    colmap_keep_sha256=digests["colmap_keep"],
                    training_validity_path=final_paths["training_validity"],
                    training_validity_sha256=digests["training_validity"],
                    exclusion_fraction_before_trim=fraction_before_trim,
                    exclusion_fraction_after_trim=fraction_after_trim,
                    decision=decision,
                    warnings=warnings,
                )
            )
            digest_inventory.append(
                {
                    "decision": decision,
                    "exclusion_fraction_after_trim": fraction_after_trim,
                    "exclusion_fraction_before_trim": fraction_before_trim,
                    "frame_id": frame.frame_id,
                    "image_name": frame.image_name,
                    "outputs": {
                        name: {
                            "path": relative_paths[name].as_posix(),
                            "sha256": digests[name],
                        }
                        for name in relative_paths
                    },
                    "warnings": list(warnings),
                }
            )
        binding_payload = {
            "frames": digest_inventory,
            "inputs": _input_binding_payload(
                frames,
                semantic,
                motion,
                sky_proposals,
                scene,
                validated_inputs,
            ),
            "policy": asdict(policy),
            "schema_version": 1,
        }
        mask_set_digest = hashlib.sha256(
            _strict_json_bytes(binding_payload)
        ).hexdigest()
        manifest_payload = {
            **binding_payload,
            "mask_set_digest": mask_set_digest,
            "stage": "mask_fusion",
        }
        manifest_bytes = _strict_json_bytes(manifest_payload)
        manifest_sha256 = _write_bytes_fsync(
            staging / "manifest.json",
            manifest_bytes,
        )
        _fsync_directory_tree(staging)
        _revalidate_snapshots(validated_inputs.snapshots)
        promote_directory(staging, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = output_dir / "manifest.json"
    return MaskFusionEvidence(
        policy=policy,
        frames=tuple(fused_frames),
        mask_set_digest=mask_set_digest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
