from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
import zipfile
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
from .da3 import (
    FinalPoseCamera,
    FramePredictionArtifact,
    PinholeCamera,
    PoseConditionedDepthResult,
)
from .masks import FusedMaskFrame, MaskFusionEvidence


MAX_DENSE_SEEDS: Final[int] = 1_000_000
_MODEL_FILES: Final[tuple[str, ...]] = ("cameras.txt", "images.txt", "points3D.txt")
DepthMaskMode = Literal["hard_uncertain", "motion_sky"]


@dataclass(frozen=True)
class DenseSeedPolicy:
    minimum_confidence: float
    relative_depth_tolerance: float
    invalid_boundary_fraction: float
    voxel_size_fraction: float
    minimum_neighbor_support: int

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_confidence",
            "relative_depth_tolerance",
            "invalid_boundary_fraction",
            "voxel_size_fraction",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field_name} must be a finite nonnegative number")
        if float(self.relative_depth_tolerance) == 0.0:
            raise ValueError("relative_depth_tolerance must be greater than zero")
        if not 0.0 <= float(self.invalid_boundary_fraction) <= 0.25:
            raise ValueError("invalid_boundary_fraction must be in [0, 0.25]")
        if not 0.0 < float(self.voxel_size_fraction) <= 0.25:
            raise ValueError("voxel_size_fraction must be in (0, 0.25]")
        if (
            type(self.minimum_neighbor_support) is not int
            or self.minimum_neighbor_support < 1
            or self.minimum_neighbor_support > 8
        ):
            raise ValueError(
                "minimum_neighbor_support must be a plain integer in [1, 8]"
            )


@dataclass(frozen=True)
class ValidatedDepthFrame:
    frame: FrameArtifact
    width: int
    height: int
    depth_path: Path
    depth_sha256: str
    valid_fraction: float


@dataclass(frozen=True)
class DenseSeedArtifact:
    npz_path: Path
    npz_sha256: str
    metadata_path: Path
    metadata_sha256: str
    point_count: int


@dataclass(frozen=True)
class DepthValidationResult:
    policy: DenseSeedPolicy
    frames: tuple[ValidatedDepthFrame, ...]
    dense_seeds: DenseSeedArtifact
    source_model_digest: str
    source_mask_digest: str
    source_depth_digest: str
    source_frame_digest: str


@dataclass(frozen=True)
class SupportedDepthCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    confidence: np.ndarray
    view_support: np.ndarray
    source_frame_index: np.ndarray
    source_xy: np.ndarray
    camera_centers: np.ndarray
    source_model_digest: str
    source_mask_digest: str
    source_depth_digest: str
    source_frame_digest: str

    def __post_init__(self) -> None:
        count = len(self.xyz)
        expected = (
            (self.xyz, (count, 3), "xyz"),
            (self.rgb, (count, 3), "rgb"),
            (self.confidence, (count,), "confidence"),
            (self.view_support, (count,), "view_support"),
            (self.source_frame_index, (count,), "source_frame_index"),
            (self.source_xy, (count, 2), "source_xy"),
        )
        for values, shape, label in expected:
            if not isinstance(values, np.ndarray) or values.shape != shape:
                raise ValueError(f"{label} must have shape {shape}")
        if (
            not isinstance(self.camera_centers, np.ndarray)
            or self.camera_centers.ndim != 2
            or self.camera_centers.shape[1:] != (3,)
        ):
            raise ValueError("camera_centers must have shape (N, 3)")
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.confidence).all():
            raise ValueError("supported depth values must be finite")
        if not np.isfinite(self.camera_centers).all():
            raise ValueError("camera centers must be finite")
        if np.any(self.confidence < 0.0) or np.any(self.view_support < 1):
            raise ValueError("supported depth confidence and support must be positive")
        if np.any(self.source_frame_index < 0) or np.any(self.source_xy < 0):
            raise ValueError("supported depth provenance must be nonnegative")
        for digest in (
            self.source_model_digest,
            self.source_mask_digest,
            self.source_depth_digest,
            self.source_frame_digest,
        ):
            if len(digest) != 64 or digest != digest.lower():
                raise ValueError("supported depth digests must be lowercase sha256")


@dataclass(frozen=True)
class _ValidatedInputs:
    frames: tuple[FrameArtifact, ...]
    depths: tuple[np.ndarray, ...]
    confidence: tuple[np.ndarray, ...]
    invalid: tuple[np.ndarray, ...]
    colors: tuple[np.ndarray, ...]
    cameras: tuple[FinalPoseCamera, ...]
    processed_camera: PinholeCamera
    snapshots: tuple[tuple[Path, str], ...]
    source_model_digest: str
    source_mask_digest: str
    source_depth_digest: str
    source_frame_digest: str


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


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_file(path: object, label: str) -> Path:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not os.path.lexists(path)
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise ValueError(f"{label} must be a canonical regular file")
    return path


def _safe_image_name(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("image_name must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("image_name must be canonical and traversal-free")
    return pure.parts


def _load_float32(path: Path, label: str) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            raw = np.load(stream, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{label} must be a deterministic NPY") from error
    if raw.dtype != np.dtype(np.float32) or raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{label} must be a non-empty float32 HxW array")
    values = np.array(raw, dtype=np.float32, copy=True)
    values.setflags(write=False)
    return values


def _load_binary_mask(path: Path, digest: str, label: str) -> np.ndarray:
    _regular_file(path, label)
    if _sha256_path(path) != _require_digest(digest, f"{label} sha256"):
        raise ValueError(f"{label} digest does not match bytes")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L":
                raise ValueError(f"{label} must be a grayscale PNG")
            raw = np.asarray(image, dtype=np.uint8)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{label} must be readable") from error
    if raw.ndim != 2 or not np.isin(raw, (0, 255)).all():
        raise ValueError(f"{label} must contain only binary 0/255 pixels")
    return raw == 255


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
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


def _load_color(path: Path, shape: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError("source frame must preserve exact RGB pixels")
            resized = image
            if image.size != (shape[1], shape[0]):
                resized = image.resize(
                    (shape[1], shape[0]),
                    resample=Image.Resampling.BILINEAR,
                )
            raw = np.array(resized, dtype=np.uint8, copy=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("source frame must be a readable RGB image") from error
    raw.setflags(write=False)
    return raw


def _validate_processed_camera(camera: object, shape: tuple[int, int]) -> PinholeCamera:
    if not isinstance(camera, PinholeCamera) or camera.model != "PINHOLE":
        raise ValueError("processed depth camera must be PINHOLE")
    if (camera.height, camera.width) != shape:
        raise ValueError("processed camera dimensions must match depth arrays")
    values = (camera.fx, camera.fy, camera.cx, camera.cy)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        )
        or camera.fx <= 0.0
        or camera.fy <= 0.0
        or not (0.0 <= camera.cx < camera.width)
        or not (0.0 <= camera.cy < camera.height)
    ):
        raise ValueError("processed PINHOLE calibration is malformed")
    return camera


def _validate_camera(camera: object, frame: FrameArtifact) -> FinalPoseCamera:
    if (
        not isinstance(camera, FinalPoseCamera)
        or camera.frame_id != frame.frame_id
        or camera.image_name != frame.image_name
    ):
        raise ValueError("final pose cameras must preserve exact frame joins")
    matrix = np.asarray(camera.w2c, dtype=np.float64)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or intrinsics.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.isfinite(intrinsics).all()
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
        raise ValueError("final pose camera must contain finite proper W2C and K")
    return camera


def _artifact_digest_payload(rows: list[dict[str, object]], schema: str) -> str:
    return hashlib.sha256(
        _strict_json_bytes({"artifacts": rows, "schema": schema})
    ).hexdigest()


def _validate_inputs(
    frames: tuple[FrameArtifact, ...],
    final_depth: PoseConditionedDepthResult,
    masks: MaskFusionEvidence,
    winner_model_dir: Path,
    policy: DenseSeedPolicy,
    *,
    mask_mode: DepthMaskMode = "hard_uncertain",
) -> _ValidatedInputs:
    if not isinstance(policy, DenseSeedPolicy):
        raise ValueError("policy must be DenseSeedPolicy")
    if mask_mode not in ("hard_uncertain", "motion_sky"):
        raise ValueError("mask_mode must be hard_uncertain or motion_sky")
    if type(frames) is not tuple or len(frames) < policy.minimum_neighbor_support + 1:
        raise ValueError("frames must provide enough neighbors for support validation")
    snapshots: list[tuple[Path, str]] = []
    frame_rows: list[dict[str, object]] = []
    names: set[str] = set()
    frame_ids: set[str] = set()
    stems: set[str] = set()
    for frame in frames:
        if not isinstance(frame, FrameArtifact):
            raise ValueError("frames must contain FrameArtifact records")
        parts = _safe_image_name(frame.image_name)
        path = _regular_file(frame.path, "source frame")
        if tuple(path.parts[-len(parts) :]) != parts:
            raise ValueError("source frame path must match image_name suffix")
        digest = _require_digest(frame.sha256, "source frame sha256")
        if _sha256_path(path) != digest:
            raise ValueError("source frame digest does not match bytes")
        stem = PurePosixPath(frame.image_name).stem.casefold()
        if (
            frame.image_name.casefold() in names
            or frame.frame_id in frame_ids
            or stem in stems
        ):
            raise ValueError("frame names, ids, and output stems must be unique")
        names.add(frame.image_name.casefold())
        frame_ids.add(frame.frame_id)
        stems.add(stem)
        snapshots.append((path, digest))
        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )

    if not isinstance(final_depth, PoseConditionedDepthResult):
        raise ValueError("final_depth must be PoseConditionedDepthResult")
    if (
        type(final_depth.artifacts) is not tuple
        or type(final_depth.cameras) is not tuple
        or len(final_depth.artifacts) != len(frames)
        or len(final_depth.cameras) != len(frames)
    ):
        raise ValueError("final depth artifacts and cameras must join every frame")
    metadata_path = _regular_file(final_depth.metadata_path, "final depth metadata")
    metadata_digest = _sha256_path(metadata_path)
    snapshots.append((metadata_path, metadata_digest))
    depths: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    depth_rows: list[dict[str, object]] = []
    cameras: list[FinalPoseCamera] = []
    shape: tuple[int, int] | None = None
    for frame, artifact, camera in zip(
        frames,
        final_depth.artifacts,
        final_depth.cameras,
        strict=True,
    ):
        if (
            not isinstance(artifact, FramePredictionArtifact)
            or artifact.frame_id != frame.frame_id
            or artifact.image_name != frame.image_name
            or artifact.source_path != frame.path
            or artifact.sky_path is not None
            or artifact.confidence_path is None
        ):
            raise ValueError(
                "final depth artifact must preserve exact joins and confidence"
            )
        depth_path = _regular_file(artifact.depth_path, "final depth")
        confidence_path = _regular_file(artifact.confidence_path, "final confidence")
        depth = _load_float32(depth_path, "final depth")
        confidence_map = _load_float32(confidence_path, "final confidence")
        if confidence_map.shape != depth.shape:
            raise ValueError("confidence dimensions must match final depth")
        if shape is None:
            shape = depth.shape
        elif depth.shape != shape:
            raise ValueError("all final depth arrays must share exact dimensions")
        if (
            not np.isfinite(depth).all()
            or np.any(depth < 0.0)
            or not np.isfinite(confidence_map).all()
            or np.any(confidence_map < 0.0)
        ):
            raise ValueError("depth and confidence must be finite and nonnegative")
        depth_digest = _sha256_path(depth_path)
        confidence_digest = _sha256_path(confidence_path)
        snapshots.extend(
            ((depth_path, depth_digest), (confidence_path, confidence_digest))
        )
        depths.append(depth)
        confidence.append(confidence_map)
        cameras.append(_validate_camera(camera, frame))
        depth_rows.append(
            {
                "confidence_sha256": confidence_digest,
                "depth_sha256": depth_digest,
                "frame_id": frame.frame_id,
            }
        )
    if shape is None:
        raise AssertionError("depth validation received no frames")
    processed_camera = _validate_processed_camera(final_depth.processed_camera, shape)

    if not isinstance(masks, MaskFusionEvidence) or type(masks.frames) is not tuple:
        raise ValueError("masks must be MaskFusionEvidence")
    if tuple(mask.frame for mask in masks.frames) != frames:
        raise ValueError("masks must preserve exact frame joins and order")
    mask_manifest = _regular_file(masks.manifest_path, "mask manifest")
    mask_manifest_digest = _require_digest(
        masks.manifest_sha256, "mask manifest sha256"
    )
    if _sha256_path(mask_manifest) != mask_manifest_digest:
        raise ValueError("mask manifest digest does not match bytes")
    snapshots.append((mask_manifest, mask_manifest_digest))
    invalid: list[np.ndarray] = []
    mask_rows: list[dict[str, object]] = []
    radius = int(math.ceil(min(shape) * float(policy.invalid_boundary_fraction)))
    structure = _disk_structure(radius)
    for mask, depth, confidence_map in zip(
        masks.frames,
        depths,
        confidence,
        strict=True,
    ):
        if not isinstance(mask, FusedMaskFrame):
            raise ValueError("mask frames must contain FusedMaskFrame")
        if mask_mode == "hard_uncertain":
            first_path = Path(mask.hard_exclude_path)
            first_digest = mask.hard_exclude_sha256
            first_label = "hard_exclude"
            second_path = Path(mask.uncertain_path)
            second_digest = mask.uncertain_sha256
            second_label = "uncertain"
        else:
            first_path = Path(mask.motion_confirmed_path)
            first_digest = mask.motion_confirmed_sha256
            first_label = "motion_confirmed"
            second_path = Path(mask.sky_confirmed_path)
            second_digest = mask.sky_confirmed_sha256
            second_label = "sky_confirmed"
        first = _load_binary_mask(first_path, first_digest, first_label)
        second = _load_binary_mask(second_path, second_digest, second_label)
        snapshots.extend(
            (
                (first_path, first_digest),
                (second_path, second_digest),
            )
        )
        first = _resize_mask(first, shape)
        second = _resize_mask(second, shape)
        current_invalid = (
            first
            | second
            | ~np.isfinite(depth)
            | (depth <= 0.0)
            | ~np.isfinite(confidence_map)
            | (confidence_map < float(policy.minimum_confidence))
        )
        if radius > 0 and np.any(current_invalid):
            current_invalid = ndimage.binary_dilation(
                current_invalid,
                structure=structure,
            )
        current_invalid.setflags(write=False)
        invalid.append(current_invalid)
        mask_rows.append(
            {
                "frame_id": mask.frame.frame_id,
                f"{first_label}_sha256": first_digest,
                f"{second_label}_sha256": second_digest,
            }
        )

    model_dir = Path(winner_model_dir)
    if (
        not model_dir.is_absolute()
        or not model_dir.is_dir()
        or model_dir.is_symlink()
        or model_dir.resolve(strict=True) != model_dir
    ):
        raise ValueError("winner_model_dir must be a canonical regular directory")
    model_rows: list[dict[str, object]] = []
    for name in _MODEL_FILES:
        path = _regular_file(model_dir / name, f"winner {name}")
        digest = _sha256_path(path)
        snapshots.append((path, digest))
        model_rows.append(
            {"name": name, "sha256": digest, "size_bytes": path.stat().st_size}
        )

    colors = tuple(_load_color(frame.path, shape) for frame in frames)
    source_frame_digest = _artifact_digest_payload(
        frame_rows, "learned_quality.depth_frames.v1"
    )
    source_depth_digest = _artifact_digest_payload(
        [
            *depth_rows,
            {"metadata_sha256": metadata_digest},
        ],
        "learned_quality.pose_depth.v1",
    )
    source_mask_digest = hashlib.sha256(
        _strict_json_bytes(
            {
                "frames": mask_rows,
                "manifest_sha256": mask_manifest_digest,
                "mask_set_digest": _require_digest(
                    masks.mask_set_digest, "mask_set_digest"
                ),
                "schema": "learned_quality.depth_masks.v1",
            }
        )
    ).hexdigest()
    source_model_digest = _artifact_digest_payload(
        model_rows, "learned_quality.winner_model.v1"
    )
    return _ValidatedInputs(
        frames=frames,
        depths=tuple(depths),
        confidence=tuple(confidence),
        invalid=tuple(invalid),
        colors=colors,
        cameras=tuple(cameras),
        processed_camera=processed_camera,
        snapshots=tuple(snapshots),
        source_model_digest=source_model_digest,
        source_mask_digest=source_mask_digest,
        source_depth_digest=source_depth_digest,
        source_frame_digest=source_frame_digest,
    )


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return yy**2 + xx**2 <= radius**2


def _neighbor_indices(index: int, count: int) -> tuple[int, ...]:
    candidates = sorted(
        (other for other in range(count) if other != index),
        key=lambda other: (abs(other - index), other),
    )
    return tuple(candidates[:8])


def _world_points(
    depth: np.ndarray,
    camera: FinalPoseCamera,
    processed: PinholeCamera,
) -> np.ndarray:
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    x = (xx - float(processed.cx)) * depth / float(processed.fx)
    y = (yy - float(processed.cy)) * depth / float(processed.fy)
    camera_points = np.stack((x, y, depth.astype(np.float64)), axis=-1)
    w2c = np.asarray(camera.w2c, dtype=np.float64)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return (camera_points - translation) @ rotation


def _project_support(
    world: np.ndarray,
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    target_invalid: np.ndarray,
    target_camera: FinalPoseCamera,
    processed: PinholeCamera,
    tolerance: float,
) -> np.ndarray:
    w2c = np.asarray(target_camera.w2c, dtype=np.float64)
    camera_points = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera_points[..., 2]
    safe_z = np.where(np.isfinite(z) & (z > 0.0), z, 1.0)
    projected_x = camera_points[..., 0] * float(processed.fx) / safe_z + float(
        processed.cx
    )
    projected_y = camera_points[..., 1] * float(processed.fy) / safe_z + float(
        processed.cy
    )
    finite = (
        np.isfinite(projected_x) & np.isfinite(projected_y) & np.isfinite(z) & (z > 0.0)
    )
    x = np.rint(np.where(finite, projected_x, 0.0)).astype(np.int64)
    y = np.rint(np.where(finite, projected_y, 0.0)).astype(np.int64)
    height, width = target_depth.shape
    in_bounds = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    observed = target_depth[safe_y, safe_x]
    valid_target = ~target_invalid[safe_y, safe_x]
    denominator = np.maximum(np.maximum(np.abs(observed), np.abs(z)), 1e-6)
    agreement = np.abs(observed - z) / denominator <= tolerance
    return in_bounds & valid_target & agreement & (source_depth > 0.0)


def _supported_depth_rows(
    inputs: _ValidatedInputs,
    policy: DenseSeedPolicy,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
]:
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    support_parts: list[np.ndarray] = []
    frame_index_parts: list[np.ndarray] = []
    source_xy_parts: list[np.ndarray] = []
    validated_depths: list[np.ndarray] = []
    for index, (depth_raw, invalid, color, camera) in enumerate(
        zip(
            inputs.depths,
            inputs.invalid,
            inputs.colors,
            inputs.cameras,
            strict=True,
        )
    ):
        depth = np.where(invalid, 0.0, depth_raw).astype(np.float64)
        world = _world_points(depth, camera, inputs.processed_camera)
        support = np.zeros(depth.shape, dtype=np.uint16)
        for neighbor in _neighbor_indices(index, len(inputs.frames)):
            support += _project_support(
                world,
                depth,
                np.where(inputs.invalid[neighbor], 0.0, inputs.depths[neighbor]),
                inputs.invalid[neighbor],
                inputs.cameras[neighbor],
                inputs.processed_camera,
                float(policy.relative_depth_tolerance),
            ).astype(np.uint16)
        accepted = (~invalid) & (support >= policy.minimum_neighbor_support)
        validated_depths.append(np.where(accepted, depth_raw, 0.0).astype(np.float32))
        if not np.any(accepted):
            continue
        xyz_parts.append(np.asarray(world[accepted], dtype=np.float64))
        rgb_parts.append(np.asarray(color[accepted], dtype=np.uint8))
        confidence_parts.append(
            np.asarray(inputs.confidence[index][accepted], dtype=np.float64)
        )
        support_parts.append(np.asarray(support[accepted] + 1, dtype=np.uint16))
        yy, xx = np.indices(depth.shape, dtype=np.int32)
        frame_index_parts.append(
            np.full(int(np.count_nonzero(accepted)), index, dtype=np.int32)
        )
        source_xy_parts.append(
            np.stack((xx[accepted], yy[accepted]), axis=-1).astype(np.int32)
        )
    if not xyz_parts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.uint16),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            tuple(validated_depths),
        )
    return (
        np.concatenate(xyz_parts),
        np.concatenate(rgb_parts),
        np.concatenate(confidence_parts),
        np.concatenate(support_parts),
        np.concatenate(frame_index_parts),
        np.concatenate(source_xy_parts),
        tuple(validated_depths),
    )


def _candidate_seeds(
    inputs: _ValidatedInputs,
    policy: DenseSeedPolicy,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
]:
    xyz, rgb, confidence, support, _, _, validated_depths = _supported_depth_rows(
        inputs, policy
    )
    return xyz, rgb, confidence, support, validated_depths


def build_supported_depth_cloud(
    frames: tuple[FrameArtifact, ...],
    final_depth: PoseConditionedDepthResult,
    masks: MaskFusionEvidence,
    winner_model_dir: Path,
    *,
    policy: DenseSeedPolicy,
    mask_mode: DepthMaskMode = "hard_uncertain",
) -> SupportedDepthCloud:
    """Return every multi-view-supported depth sample with exact provenance."""

    inputs = _validate_inputs(
        frames,
        final_depth,
        masks,
        winner_model_dir,
        policy,
        mask_mode=mask_mode,
    )
    xyz, rgb, confidence, support, frame_index, source_xy, _ = _supported_depth_rows(
        inputs, policy
    )
    camera_centers = []
    for camera in inputs.cameras:
        w2c = np.asarray(camera.w2c, dtype=np.float64)
        camera_centers.append(-w2c[:3, 3] @ w2c[:3, :3])
    return SupportedDepthCloud(
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.asarray(rgb, dtype=np.uint8),
        confidence=np.asarray(confidence, dtype=np.float64),
        view_support=np.asarray(support, dtype=np.uint16),
        source_frame_index=np.asarray(frame_index, dtype=np.int32),
        source_xy=np.asarray(source_xy, dtype=np.int32),
        camera_centers=np.asarray(camera_centers, dtype=np.float64),
        source_model_digest=inputs.source_model_digest,
        source_mask_digest=inputs.source_mask_digest,
        source_depth_digest=inputs.source_depth_digest,
        source_frame_digest=inputs.source_frame_digest,
    )


def _fuse_and_cap(
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    voxel_size_fraction: float,
    *,
    max_count: int = MAX_DENSE_SEEDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.uint16),
        )
    lower = np.quantile(xyz, 0.05, axis=0, method="linear")
    upper = np.quantile(xyz, 0.95, axis=0, method="linear")
    extent = max(float(np.max(upper - lower)), 1e-6)
    voxel_size = max(extent * float(voxel_size_fraction), 1e-6)
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    xyz = xyz[order]
    rgb = rgb[order]
    confidence = confidence[order]
    support = support[order]
    boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate((np.array((0,)), boundaries))
    stops = np.concatenate((boundaries, np.array((len(keys),))))
    fused_xyz: list[np.ndarray] = []
    fused_rgb: list[np.ndarray] = []
    fused_confidence: list[float] = []
    fused_support: list[int] = []
    for start, stop in zip(starts, stops, strict=True):
        fused_xyz.append(np.median(xyz[start:stop], axis=0))
        fused_rgb.append(np.rint(np.median(rgb[start:stop], axis=0)).astype(np.uint8))
        fused_confidence.append(float(np.median(confidence[start:stop])))
        fused_support.append(int(np.max(support[start:stop])))
    xyz_array = np.asarray(fused_xyz, dtype=np.float64)
    rgb_array = np.asarray(fused_rgb, dtype=np.uint8)
    confidence_array = np.asarray(fused_confidence, dtype=np.float64)
    support_array = np.asarray(fused_support, dtype=np.uint16)
    rank = np.lexsort(
        (
            xyz_array[:, 2],
            xyz_array[:, 1],
            xyz_array[:, 0],
            -confidence_array,
            -support_array.astype(np.int64),
        )
    )[:max_count]
    return (
        np.asarray(xyz_array[rank], dtype=np.float32),
        np.asarray(rgb_array[rank], dtype=np.uint8),
        np.asarray(confidence_array[rank], dtype=np.float32),
        np.asarray(support_array[rank], dtype=np.uint16),
    )


def _write_npy_fsync(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_path(path)


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    with path.open("xb") as stream:
        with zipfile.ZipFile(
            stream, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name, values in arrays:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(values))
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_path(path)


def _write_bytes_fsync(path: Path, payload: bytes) -> str:
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


def _fsync_tree(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _revalidate(snapshots: tuple[tuple[Path, str], ...]) -> None:
    for path, digest in snapshots:
        if _sha256_path(_regular_file(path, "upstream artifact")) != digest:
            raise ValueError("upstream artifact changed before depth publication")


def _validate_output_dir(output_dir: object) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    if output_dir.resolve(strict=False) != output_dir:
        raise ValueError("output_dir must be canonical")
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise ValueError("output_dir parent must be an existing regular directory")
    return output_dir


def validate_depth_and_fuse_seeds(
    frames: tuple[FrameArtifact, ...],
    final_depth: PoseConditionedDepthResult,
    masks: MaskFusionEvidence,
    winner_model_dir: Path,
    output_dir: Path,
    *,
    policy: DenseSeedPolicy,
) -> DepthValidationResult:
    output_dir = _validate_output_dir(output_dir)
    inputs = _validate_inputs(frames, final_depth, masks, winner_model_dir, policy)
    (
        candidate_xyz,
        candidate_rgb,
        candidate_confidence,
        candidate_support,
        validated_depths,
    ) = _candidate_seeds(inputs, policy)
    xyz, rgb, confidence, support = _fuse_and_cap(
        candidate_xyz,
        candidate_rgb,
        candidate_confidence,
        candidate_support,
        float(policy.voxel_size_fraction),
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        depth_frames: list[ValidatedDepthFrame] = []
        depth_inventory: list[dict[str, object]] = []
        for frame, depth in zip(frames, validated_depths, strict=True):
            relative = (
                Path("depth") / f"{PurePosixPath(frame.image_name).stem}_depth.npy"
            )
            digest = _write_npy_fsync(staging / relative, depth)
            depth_frames.append(
                ValidatedDepthFrame(
                    frame=frame,
                    width=depth.shape[1],
                    height=depth.shape[0],
                    depth_path=output_dir / relative,
                    depth_sha256=digest,
                    valid_fraction=float(np.mean(depth > 0.0)),
                )
            )
            depth_inventory.append(
                {
                    "frame_id": frame.frame_id,
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "valid_fraction": float(np.mean(depth > 0.0)),
                }
            )
        npz_relative = Path("dense_seeds.npz")
        npz_digest = _write_deterministic_npz(
            staging / npz_relative,
            (
                ("xyz", xyz),
                ("rgb", rgb),
                ("confidence", confidence),
                ("view_support", support),
            ),
        )
        metadata_payload = {
            "dense_seeds": {
                "dtypes": {
                    "confidence": "float32",
                    "rgb": "uint8",
                    "view_support": "uint16",
                    "xyz": "float32",
                },
                "hard_ceiling": MAX_DENSE_SEEDS,
                "npz_path": npz_relative.as_posix(),
                "npz_sha256": npz_digest,
                "point_count": len(xyz),
            },
            "depth_frames": depth_inventory,
            "policy": asdict(policy),
            "schema_version": 1,
            "sources": {
                "depth_digest": inputs.source_depth_digest,
                "frame_digest": inputs.source_frame_digest,
                "mask_digest": inputs.source_mask_digest,
                "model_digest": inputs.source_model_digest,
            },
            "stage": "validated_depth_and_dense_seed_fusion",
        }
        metadata_relative = Path("dense_seeds.json")
        metadata_digest = _write_bytes_fsync(
            staging / metadata_relative,
            _strict_json_bytes(metadata_payload),
        )
        _revalidate(inputs.snapshots)
        _fsync_tree(staging)
        promote_directory(staging, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    dense_seeds = DenseSeedArtifact(
        npz_path=output_dir / npz_relative,
        npz_sha256=npz_digest,
        metadata_path=output_dir / metadata_relative,
        metadata_sha256=metadata_digest,
        point_count=len(xyz),
    )
    return DepthValidationResult(
        policy=policy,
        frames=tuple(depth_frames),
        dense_seeds=dense_seeds,
        source_model_digest=inputs.source_model_digest,
        source_mask_digest=inputs.source_mask_digest,
        source_depth_digest=inputs.source_depth_digest,
        source_frame_digest=inputs.source_frame_digest,
    )
