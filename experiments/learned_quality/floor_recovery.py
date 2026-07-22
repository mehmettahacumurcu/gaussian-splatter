from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from backend.image_to_scene.orientation import estimate_world_orientation

from .depth import SupportedDepthCloud


FLOOR_RECOVERY_SCHEMA: Final[str] = "learned_quality.floor_recovery.v1"
MAX_FLOOR_SEEDS: Final[int] = 150_000


def _hex_digest(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


@dataclass(frozen=True)
class FloorArtifactLineage:
    source_revision: str
    source_digest: str
    selection_digest: str
    pretraining_fingerprint: str

    def __post_init__(self) -> None:
        _hex_digest(self.source_revision, 40, "source_revision")
        _hex_digest(self.source_digest, 64, "source_digest")
        _hex_digest(self.selection_digest, 64, "selection_digest")
        _hex_digest(
            self.pretraining_fingerprint,
            64,
            "pretraining_fingerprint",
        )


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class FloorRecoveryPolicy:
    plane_inlier_fraction: float = 0.02
    minimum_plane_inliers: int = 100
    minimum_plane_support_fraction: float = 0.005
    maximum_up_angle_degrees: float = 15.0
    minimum_above_below_ratio: float = 4.0
    grid_fraction: float = 0.01
    footprint_erosion_cells: int = 1
    sparse_support_radius_cells: float = 1.5
    minimum_component_cells: int = 4
    minimum_seed_count: int = 1_000
    maximum_seed_count: int = MAX_FLOOR_SEEDS

    def __post_init__(self) -> None:
        for name in (
            "plane_inlier_fraction",
            "minimum_plane_support_fraction",
            "maximum_up_angle_degrees",
            "minimum_above_below_ratio",
            "grid_fraction",
            "sparse_support_radius_cells",
        ):
            if _finite_number(getattr(self, name), name, positive=True) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.plane_inlier_fraction) <= 0.25:
            raise ValueError("plane_inlier_fraction must be in (0, 0.25]")
        if not 0.0 < float(self.minimum_plane_support_fraction) <= 1.0:
            raise ValueError("minimum_plane_support_fraction must be in (0, 1]")
        if not 0.0 < float(self.maximum_up_angle_degrees) < 90.0:
            raise ValueError("maximum_up_angle_degrees must be in (0, 90)")
        if not 0.0 < float(self.grid_fraction) <= 0.1:
            raise ValueError("grid_fraction must be in (0, 0.1]")
        for name in (
            "minimum_plane_inliers",
            "footprint_erosion_cells",
            "minimum_component_cells",
            "minimum_seed_count",
            "maximum_seed_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "footprint_erosion_cells" else 1):
                raise ValueError(f"{name} must be a positive plain integer")
        if self.maximum_seed_count > MAX_FLOOR_SEEDS:
            raise ValueError(f"maximum_seed_count cannot exceed {MAX_FLOOR_SEEDS}")
        if self.minimum_seed_count > self.maximum_seed_count:
            raise ValueError("minimum_seed_count cannot exceed maximum_seed_count")


def _unit_vector(values: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return vector / length


@dataclass(frozen=True)
class FloorPlane:
    normal: np.ndarray
    offset: float
    basis_u: np.ndarray
    basis_v: np.ndarray
    robust_radius: float
    inlier_tolerance: float
    inlier_count: int
    inlier_fraction: float
    above_below_ratio: float
    median_camera_height: float

    def __post_init__(self) -> None:
        normal = _unit_vector(self.normal, "normal")
        basis_u = _unit_vector(self.basis_u, "basis_u")
        basis_v = _unit_vector(self.basis_v, "basis_v")
        if abs(float(normal @ basis_u)) > 1e-6 or abs(float(normal @ basis_v)) > 1e-6:
            raise ValueError("floor basis must be orthogonal to normal")
        if abs(float(basis_u @ basis_v)) > 1e-6:
            raise ValueError("floor basis vectors must be orthogonal")
        for name in (
            "offset",
            "robust_radius",
            "inlier_tolerance",
            "inlier_fraction",
            "above_below_ratio",
            "median_camera_height",
        ):
            _finite_number(
                getattr(self, name),
                name,
                positive=name in {"robust_radius", "inlier_tolerance", "above_below_ratio"},
            )
        if type(self.inlier_count) is not int or self.inlier_count < 1:
            raise ValueError("inlier_count must be a positive plain integer")


@dataclass(frozen=True)
class FloorHoleMap:
    origin_uv: np.ndarray
    cell_width: float
    shape: tuple[int, int]
    observed: np.ndarray
    holes: np.ndarray
    component_ids: np.ndarray
    candidate_indices: np.ndarray
    candidate_cell_ids: np.ndarray
    initial_hole_cells: int

    def __post_init__(self) -> None:
        if np.asarray(self.origin_uv).shape != (2,) or not np.isfinite(self.origin_uv).all():
            raise ValueError("origin_uv must be a finite 2-vector")
        _finite_number(self.cell_width, "cell_width", positive=True)
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise ValueError("shape must contain two positive integers")
        for values, label in (
            (self.observed, "observed"),
            (self.holes, "holes"),
            (self.component_ids, "component_ids"),
        ):
            if not isinstance(values, np.ndarray) or values.shape != self.shape:
                raise ValueError(f"{label} must match the grid shape")
        if self.candidate_indices.shape != self.candidate_cell_ids.shape:
            raise ValueError("candidate index and cell arrays must align")
        if self.candidate_indices.ndim != 1:
            raise ValueError("candidate indices must be one-dimensional")
        if type(self.initial_hole_cells) is not int or self.initial_hole_cells < 0:
            raise ValueError("initial_hole_cells must be a nonnegative integer")


@dataclass(frozen=True)
class FloorSeedArtifact:
    npz_path: Path
    metadata_path: Path
    plane_path: Path
    holes_path: Path
    point_count: int
    content_fingerprint: str


def _validated_points(values: np.ndarray, label: str, *, minimum: int) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < minimum:
        raise ValueError(f"{label} must have shape (N, 3) with at least {minimum} rows")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} must be finite")
    return points


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    basis_u = np.cross(normal, axis)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def estimate_floor_plane(
    sparse_points: np.ndarray,
    camera_centers: np.ndarray,
    *,
    policy: FloorRecoveryPolicy = FloorRecoveryPolicy(),
    seed: int = 0,
) -> FloorPlane:
    points = _validated_points(sparse_points, "sparse_points", minimum=3)
    cameras = _validated_points(camera_centers, "camera_centers", minimum=2)
    if type(seed) is not int:
        raise ValueError("seed must be a plain integer")
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    robust_radius = float(np.quantile(radii, 0.97, method="linear"))
    if not math.isfinite(robust_radius) or robust_radius <= 1e-6:
        raise ValueError("could not find a reliable floor plane")
    tolerance = float(policy.plane_inlier_fraction) * robust_radius
    core = points[radii <= robust_radius]
    if len(core) < 3:
        raise ValueError("could not find a reliable floor plane")
    orientation = estimate_world_orientation(points, seed=seed)
    up = _unit_vector(orientation.up_raw, "estimated up")
    cosine_limit = math.cos(math.radians(float(policy.maximum_up_angle_degrees)))
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, float, np.ndarray, float, float] | None = None
    for _ in range(600):
        sample = core[rng.choice(len(core), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-10:
            continue
        normal /= length
        if abs(float(normal @ up)) < cosine_limit:
            continue
        if float(normal @ up) < 0.0:
            normal = -normal
        offset = -float(normal @ sample[0])
        distances = points @ normal + offset
        inliers = np.abs(distances) <= tolerance
        if int(inliers.sum()) < 3:
            continue
        plane_center = points[inliers].mean(axis=0)
        _, _, vt = np.linalg.svd(points[inliers] - plane_center, full_matrices=False)
        normal = _unit_vector(vt[-1], "refined normal")
        if float(normal @ up) < 0.0:
            normal = -normal
        if abs(float(normal @ up)) < cosine_limit:
            continue
        offset = -float(normal @ plane_center)
        camera_heights = cameras @ normal + offset
        camera_height = float(np.median(camera_heights))
        if not math.isfinite(camera_height) or camera_height <= 4.0 * tolerance:
            # `up` already has a signed content-side convention. Flipping a plane
            # merely because cameras lie below it turns a ceiling into a floor.
            continue
        distances = points @ normal + offset
        inliers = np.abs(distances) <= tolerance
        positive = int(np.count_nonzero(distances > tolerance))
        negative = int(np.count_nonzero(distances < -tolerance))
        asymmetry = (positive + 1.0) / (negative + 1.0)
        score = float(inliers.sum()) * min(asymmetry, 50.0)
        candidate = (
            score,
            normal,
            offset,
            inliers,
            asymmetry,
            camera_height,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("could not find a reliable floor plane")
    _, normal, offset, inliers, asymmetry, camera_height = best
    inlier_count = int(inliers.sum())
    inlier_fraction = inlier_count / len(points)
    if (
        inlier_count < policy.minimum_plane_inliers
        or inlier_fraction < float(policy.minimum_plane_support_fraction)
        or asymmetry < float(policy.minimum_above_below_ratio)
        or camera_height <= 4.0 * tolerance
    ):
        raise ValueError("could not find a reliable floor plane")
    basis_u, basis_v = _basis(normal)
    return FloorPlane(
        normal=np.asarray(normal, dtype=np.float64),
        offset=float(offset),
        basis_u=basis_u,
        basis_v=basis_v,
        robust_radius=robust_radius,
        inlier_tolerance=tolerance,
        inlier_count=inlier_count,
        inlier_fraction=float(inlier_fraction),
        above_below_ratio=float(asymmetry),
        median_camera_height=float(camera_height),
    )


def _project_uv(points: np.ndarray, plane: FloorPlane) -> np.ndarray:
    return np.stack((points @ plane.basis_u, points @ plane.basis_v), axis=-1)


def _grid_indices(
    uv: np.ndarray, origin: np.ndarray, cell_width: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.floor((uv - origin) / cell_width).astype(np.int64)
    x = np.clip(xy[:, 0], 0, shape[1] - 1)
    y = np.clip(xy[:, 1], 0, shape[0] - 1)
    return y, x


def build_floor_hole_map(
    sparse_points: np.ndarray,
    cloud: SupportedDepthCloud,
    plane: FloorPlane,
    *,
    policy: FloorRecoveryPolicy = FloorRecoveryPolicy(),
) -> FloorHoleMap:
    sparse = _validated_points(sparse_points, "sparse_points", minimum=3)
    if not isinstance(cloud, SupportedDepthCloud):
        raise ValueError("cloud must be SupportedDepthCloud")
    distances = cloud.xyz @ plane.normal + plane.offset
    source_centers = cloud.camera_centers[cloud.source_frame_index]
    ranges = np.linalg.norm(cloud.xyz - source_centers, axis=1)
    near_plane = np.abs(distances) <= np.maximum(
        plane.inlier_tolerance, 0.03 * ranges
    )
    supported = near_plane & (cloud.view_support >= 3)
    candidate_indices = np.flatnonzero(supported)
    if len(candidate_indices) == 0:
        raise ValueError("no_recoverable_floor_hole: no supported floor candidates")
    candidate_uv = _project_uv(cloud.xyz[candidate_indices], plane)
    sparse_distance = np.abs(sparse @ plane.normal + plane.offset)
    sparse_floor = sparse[sparse_distance <= plane.inlier_tolerance]
    if len(sparse_floor) == 0:
        raise ValueError("no_recoverable_floor_hole: no sparse floor support")
    sparse_uv = _project_uv(sparse_floor, plane)
    cell_width = max(plane.robust_radius * float(policy.grid_fraction), 1e-6)
    lower = np.floor(candidate_uv.min(axis=0) / cell_width) * cell_width
    upper = np.ceil(candidate_uv.max(axis=0) / cell_width) * cell_width
    width_height = np.ceil((upper - lower) / cell_width).astype(np.int64) + 1
    shape = (int(width_height[1]), int(width_height[0]))
    if shape[0] > 2048 or shape[1] > 2048 or shape[0] * shape[1] > 4_000_000:
        raise ValueError("floor footprint grid exceeds the safety limit")
    observed = np.zeros(shape, dtype=bool)
    cy, cx = _grid_indices(candidate_uv, lower, cell_width, shape)
    observed[cy, cx] = True
    observed = ndimage.binary_closing(observed, structure=np.ones((3, 3), dtype=bool))
    observed = ndimage.binary_fill_holes(observed)
    if policy.footprint_erosion_cells:
        size = 2 * policy.footprint_erosion_cells + 1
        observed = ndimage.binary_erosion(
            observed, structure=np.ones((size, size), dtype=bool)
        )
    sparse_occupied = np.zeros(shape, dtype=bool)
    sy, sx = _grid_indices(sparse_uv, lower, cell_width, shape)
    sparse_occupied[sy, sx] = True
    distance_cells = ndimage.distance_transform_edt(~sparse_occupied)
    holes = observed & (
        distance_cells > float(policy.sparse_support_radius_cells)
    )
    labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.zeros(count + 1, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep[1:] = sizes[1:] >= policy.minimum_component_cells
    holes = keep[labels]
    labels, _ = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    cell_ids = labels[cy, cx].astype(np.int32)
    selected = cell_ids > 0
    return FloorHoleMap(
        origin_uv=np.asarray(lower, dtype=np.float64),
        cell_width=float(cell_width),
        shape=shape,
        observed=np.asarray(observed, dtype=bool),
        holes=np.asarray(holes, dtype=bool),
        component_ids=np.asarray(labels, dtype=np.int32),
        candidate_indices=np.asarray(candidate_indices[selected], dtype=np.int64),
        candidate_cell_ids=np.asarray(cell_ids[selected], dtype=np.int32),
        initial_hole_cells=int(np.count_nonzero(holes)),
    )


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = BytesIO()
    np.lib.format.write_array(stream, np.asarray(values), allow_pickle=False)
    return stream.getvalue()


def _write_npz(path: Path, arrays: tuple[tuple[str, np.ndarray], ...]) -> str:
    with path.open("xb") as stream:
        with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name, values in arrays:
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(values))
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _plane_payload(plane: FloorPlane) -> dict[str, object]:
    return {
        "above_below_ratio": plane.above_below_ratio,
        "basis_u": plane.basis_u.tolist(),
        "basis_v": plane.basis_v.tolist(),
        "inlier_count": plane.inlier_count,
        "inlier_fraction": plane.inlier_fraction,
        "inlier_tolerance": plane.inlier_tolerance,
        "median_camera_height": plane.median_camera_height,
        "normal": plane.normal.tolist(),
        "offset": plane.offset,
        "robust_radius": plane.robust_radius,
    }


def generate_floor_seed_artifact(
    sparse_points: np.ndarray,
    cloud: SupportedDepthCloud,
    plane: FloorPlane,
    holes: FloorHoleMap,
    output_dir: Path,
    *,
    policy: FloorRecoveryPolicy = FloorRecoveryPolicy(),
    lineage: FloorArtifactLineage,
) -> FloorSeedArtifact:
    sparse = _validated_points(sparse_points, "sparse_points", minimum=3)
    target = Path(output_dir)
    if not target.is_absolute() or target.resolve(strict=False) != target:
        raise ValueError("output_dir must be an absolute canonical path")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise ValueError("output_dir parent must exist")
    if len(holes.candidate_indices) == 0:
        raise ValueError("no_recoverable_floor_hole: hole map has no candidates")
    indices = holes.candidate_indices
    xyz = np.asarray(cloud.xyz[indices], dtype=np.float64)
    distances = xyz @ plane.normal + plane.offset
    xyz = xyz - distances[:, None] * plane.normal[None, :]
    uv = _project_uv(xyz, plane)
    voxel_width = 0.5 * holes.cell_width
    keys = np.floor((uv - holes.origin_uv) / voxel_width).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys = keys[order]
    indices = indices[order]
    xyz = xyz[order]
    boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate((np.array((0,)), boundaries))
    stops = np.concatenate((boundaries, np.array((len(keys),))))
    rows: list[tuple[np.ndarray, np.ndarray, float, int, int, int]] = []
    ordered_cell_ids = holes.candidate_cell_ids[order]
    for start, stop in zip(starts, stops, strict=True):
        source = indices[start:stop]
        position = np.median(xyz[start:stop], axis=0)
        position -= (float(position @ plane.normal) + plane.offset) * plane.normal
        rows.append(
            (
                position,
                np.rint(np.median(cloud.rgb[source], axis=0)).astype(np.uint8),
                float(np.median(cloud.confidence[source])),
                int(np.max(cloud.view_support[source])),
                int(np.min(cloud.source_frame_index[source])),
                int(np.min(ordered_cell_ids[start:stop])),
            )
        )
    fused_xyz = np.asarray([row[0] for row in rows], dtype=np.float64)
    fused_rgb = np.asarray([row[1] for row in rows], dtype=np.uint8)
    fused_confidence = np.asarray([row[2] for row in rows], dtype=np.float64)
    fused_support = np.asarray([row[3] for row in rows], dtype=np.uint16)
    fused_frame = np.asarray([row[4] for row in rows], dtype=np.int32)
    fused_cell = np.asarray([row[5] for row in rows], dtype=np.int32)
    sparse_distance = np.abs(sparse @ plane.normal + plane.offset)
    sparse_floor_uv = _project_uv(
        sparse[sparse_distance <= plane.inlier_tolerance], plane
    )
    tree = cKDTree(sparse_floor_uv)
    distance_to_sparse = tree.query(_project_uv(fused_xyz, plane), k=1)[0]
    rank = np.lexsort(
        (
            fused_xyz[:, 2],
            fused_xyz[:, 1],
            fused_xyz[:, 0],
            -distance_to_sparse,
            -fused_confidence,
            -fused_support.astype(np.int64),
        )
    )[: policy.maximum_seed_count]
    fused_xyz = np.asarray(fused_xyz[rank], dtype=np.float32)
    fused_rgb = np.asarray(fused_rgb[rank], dtype=np.uint8)
    fused_confidence = np.asarray(fused_confidence[rank], dtype=np.float32)
    fused_support = np.asarray(fused_support[rank], dtype=np.uint16)
    fused_frame = np.asarray(fused_frame[rank], dtype=np.int32)
    fused_cell = np.asarray(fused_cell[rank], dtype=np.int32)
    if len(fused_xyz) < policy.minimum_seed_count:
        raise ValueError(
            "no_recoverable_floor_hole: insufficient fused floor seeds "
            f"({len(fused_xyz)} < {policy.minimum_seed_count})"
        )
    fingerprint = hashlib.sha256()
    fingerprint.update(FLOOR_RECOVERY_SCHEMA.encode("ascii"))
    fingerprint.update(_json_bytes(asdict(policy)))
    fingerprint.update(_json_bytes(asdict(lineage)))
    fingerprint.update(_json_bytes(_plane_payload(plane)))
    fingerprint.update(
        _json_bytes(
            {
                "cell_width": holes.cell_width,
                "initial_hole_cells": holes.initial_hole_cells,
                "shape": list(holes.shape),
                "source_depth_digest": cloud.source_depth_digest,
                "source_frame_digest": cloud.source_frame_digest,
                "source_mask_digest": cloud.source_mask_digest,
                "source_model_digest": cloud.source_model_digest,
            }
        )
    )
    for values in (
        fused_xyz,
        fused_rgb,
        fused_confidence,
        fused_support,
        fused_frame,
        fused_cell,
    ):
        fingerprint.update(_npy_bytes(values))
    content_fingerprint = fingerprint.hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        npz_name = "floor_seeds.npz"
        npz_digest = _write_npz(
            staging / npz_name,
            (
                ("xyz", fused_xyz),
                ("rgb", fused_rgb),
                ("confidence", fused_confidence),
                ("view_support", fused_support),
                ("source_frame_index", fused_frame),
                ("hole_cell_id", fused_cell),
            ),
        )
        plane_name = "floor_plane.json"
        _write_bytes(
            staging / plane_name,
            _json_bytes({"schema": FLOOR_RECOVERY_SCHEMA, **_plane_payload(plane)}),
        )
        holes_name = "floor_holes.json"
        _write_bytes(
            staging / holes_name,
            _json_bytes(
                {
                    "candidate_count": int(len(holes.candidate_indices)),
                    "cell_width": holes.cell_width,
                    "initial_hole_cells": holes.initial_hole_cells,
                    "origin_uv": holes.origin_uv.tolist(),
                    "schema": FLOOR_RECOVERY_SCHEMA,
                    "shape": list(holes.shape),
                }
            ),
        )
        metadata_name = "floor_seeds.json"
        _write_bytes(
            staging / metadata_name,
            _json_bytes(
                {
                    "content_fingerprint": content_fingerprint,
                    "npz_sha256": npz_digest,
                    "point_count": int(len(fused_xyz)),
                    "policy": asdict(policy),
                    "schema": FLOOR_RECOVERY_SCHEMA,
                    "lineage": asdict(lineage),
                    "source_digests": {
                        "depth": cloud.source_depth_digest,
                        "frames": cloud.source_frame_digest,
                        "masks": cloud.source_mask_digest,
                        "model": cloud.source_model_digest,
                    },
                }
            ),
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return FloorSeedArtifact(
        npz_path=target / npz_name,
        metadata_path=target / metadata_name,
        plane_path=target / plane_name,
        holes_path=target / holes_name,
        point_count=int(len(fused_xyz)),
        content_fingerprint=content_fingerprint,
    )
