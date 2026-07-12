"""Confidence-gated metadata and preview assets for static splat results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from backend.image_to_scene.collider import derive_minimal_collider
from backend.image_to_scene.orientation import (
    estimate_world_orientation,
    rotate_points,
)
from backend.preprocess.parse_colmap import (
    load_points3d_from_model,
    parse_cameras_from_model,
)

from .contracts import ReconstructionBundle
from .polish import validate_static_ply


PreviewRenderBackend = Callable[
    [Path, Mapping[str, object], int, int],
    np.ndarray,
]


def _unit_vector(value: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return vector / length


def estimate_camera_up(cameras: Mapping[str, Mapping[str, object]]) -> np.ndarray:
    """Estimate robust raw-frame up from registered COLMAP cameras."""

    if not cameras:
        raise ValueError("at least one registered camera is required")
    raw_up: list[np.ndarray] = []
    for name in sorted(cameras):
        camera = cameras[name]
        rotation = np.asarray(camera.get("R"), dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(f"camera {name!r} has an invalid rotation")
        raw_up.append(_unit_vector(rotation.T @ [0.0, -1.0, 0.0], "camera up"))

    vectors = np.stack(raw_up)
    reference = np.median(vectors, axis=0)
    if np.linalg.norm(reference) <= 1e-12:
        reference = vectors[0]
    reference = _unit_vector(reference, "camera up reference")
    aligned = np.array(
        [vector if np.dot(vector, reference) >= 0.0 else -vector for vector in vectors]
    )
    robust_reference = _unit_vector(np.median(aligned, axis=0), "camera up median")
    angles = np.degrees(np.arccos(np.clip(aligned @ robust_reference, -1.0, 1.0)))
    inliers = aligned[angles <= 30.0]
    if len(inliers) == 0:
        raise ValueError("registered cameras have no consistent up direction")
    return _unit_vector(np.mean(inliers, axis=0), "camera up mean")


def _camera_center(camera: Mapping[str, object], name: str) -> np.ndarray:
    rotation = np.asarray(camera.get("R"), dtype=np.float64)
    translation = np.asarray(camera.get("t"), dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError(f"camera {name!r} has an invalid pose")
    return -(rotation.T @ translation)


def _json_vector(value: np.ndarray) -> list[float]:
    return [round(float(component), 8) for component in np.asarray(value).ravel()]


def _json_matrix(value: np.ndarray) -> list[list[float]]:
    return [
        [round(float(component), 8) for component in row] for row in np.asarray(value)
    ]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def build_scene_metadata(
    ply_path: str | Path,
    reconstruction: ReconstructionBundle,
) -> dict[str, object]:
    """Derive conservative viewer metadata without changing the selected PLY."""

    if not reconstruction.decision.passed or reconstruction.decision.failures:
        raise ValueError("scene metadata requires a passing reconstruction")
    validated = validate_static_ply(ply_path)
    vertices = validated.vertices
    points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(
        np.float64,
        copy=False,
    )
    weights = _sigmoid(vertices["opacity"])
    sh_c0 = 0.2820948
    colors = np.clip(
        0.5
        + sh_c0 * np.column_stack([vertices[f"f_dc_{index}"] for index in range(3)]),
        0.0,
        1.0,
    )
    plane = estimate_world_orientation(points, weights=weights, colors=colors)

    cameras = parse_cameras_from_model(reconstruction.accepted_model_dir)
    registered_names = reconstruction.decision.dominant.registered_names
    if set(cameras) != set(registered_names):
        raise ValueError("accepted cameras differ from the reconstruction decision")
    camera_up = estimate_camera_up(cameras)
    plane_up = _unit_vector(plane.up_raw, "plane up")
    agreement_deg = float(
        np.degrees(
            np.arccos(np.clip(abs(float(np.dot(camera_up, plane_up))), -1.0, 1.0))
        )
    )
    apply_orientation = plane.plane_inlier_frac >= 0.5 and agreement_deg <= 20.0
    metadata_points = (
        rotate_points(points, plane.quaternion) if apply_orientation else points
    )
    bounds_min, bounds_max = np.percentile(metadata_points, (0.5, 99.5), axis=0)
    if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
        raise ValueError("scene bounds are non-finite")
    collider = derive_minimal_collider(metadata_points)

    ordered_names = tuple(
        record.output_name
        for record in reconstruction.selected_manifest.selected_frames
        if record.output_name in cameras
    )
    if tuple(ordered_names) == () or set(ordered_names) != set(cameras):
        raise ValueError("camera order cannot be derived from the selection manifest")
    raw_centers = np.stack(
        [_camera_center(cameras[name], name) for name in ordered_names]
    )
    metadata_centers = (
        rotate_points(raw_centers, plane.quaternion)
        if apply_orientation
        else raw_centers
    )
    median_center = np.median(metadata_centers, axis=0)
    default_index = int(
        np.argmin(np.linalg.norm(metadata_centers - median_center, axis=1))
    )
    default_name = ordered_names[default_index]
    default_camera = cameras[default_name]

    orientation: dict[str, object] = {
        "applied": apply_orientation,
        "camera_up_raw": _json_vector(camera_up),
        "plane_up_raw": _json_vector(plane_up),
        "plane_inlier_fraction": round(float(plane.plane_inlier_frac), 8),
        "camera_plane_agreement_degrees": round(agreement_deg, 8),
        "above_below_ratio": round(float(plane.above_below_ratio), 8),
        "sky_agrees": plane.sky_agrees,
    }
    if apply_orientation:
        orientation["transform"] = {
            "quaternion_xyzw": _json_vector(plane.quaternion),
            "source_frame": "raw_colmap",
            "target_frame": "viewer_y_up",
        }

    source_cameras = []
    for index, name in enumerate(ordered_names):
        camera = cameras[name]
        source_cameras.append(
            {
                "image_name": name,
                "center_raw": _json_vector(raw_centers[index]),
                "center_metadata": _json_vector(metadata_centers[index]),
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "K": _json_matrix(np.asarray(camera["K"])),
                "w2c": _json_matrix(np.asarray(camera["w2c"])),
            }
        )

    return {
        "schema_version": 1,
        "coordinate_convention": {
            "source": "colmap_world_to_camera_opencv",
            "metadata_frame": "viewer_y_up" if apply_orientation else "raw_colmap",
            "ply_transform_baked": False,
        },
        "bounds": {
            "percentiles": [0.5, 99.5],
            "minimum": _json_vector(bounds_min),
            "maximum": _json_vector(bounds_max),
        },
        "navigation": {
            "ground_y": round(float(collider.ground_y), 8),
            "collider_minimum": _json_vector(collider.bbox_min),
            "collider_maximum": _json_vector(collider.bbox_max),
            "spawn_position": _json_vector(collider.spawn_position),
            "spawn_look_direction": _json_vector(collider.spawn_look_direction),
        },
        "orientation": orientation,
        "default_camera": {
            "image_name": default_name,
            "source": "registered_colmap_camera",
            "K": _json_matrix(np.asarray(default_camera["K"])),
            "w2c": _json_matrix(np.asarray(default_camera["w2c"])),
            "width": int(default_camera["width"]),
            "height": int(default_camera["height"]),
            "center_raw": _json_vector(raw_centers[default_index]),
            "center_metadata": _json_vector(metadata_centers[default_index]),
        },
        "registered_cameras": source_cameras,
        "environment": {
            "sky": "viewer_recommendation_only",
            "background_rgb": [0.0, 0.0, 0.0],
            "lighting": "use_splat_appearance",
        },
        "source": {
            "ply": Path(ply_path).name,
            "accepted_model": reconstruction.accepted_model_dir.name,
            "registered_camera_count": len(ordered_names),
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    return path


def write_scene_metadata(
    path: str | Path,
    metadata: Mapping[str, object],
    *,
    world_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write scene metadata and optional JSON-only viewer hints."""

    metadata_path = _write_json(Path(path), metadata)
    written = {"scene_metadata": metadata_path}
    if world_dir is None:
        return written

    world = Path(world_dir)
    navigation = metadata["navigation"]
    orientation = metadata["orientation"]
    default_camera = metadata["default_camera"]
    registered_cameras = metadata["registered_cameras"]
    if not all(
        isinstance(value, Mapping)
        for value in (navigation, orientation, default_camera)
    ) or not isinstance(registered_cameras, list):
        raise ValueError("scene metadata has invalid world asset sections")

    collider_payload: dict[str, object] = {
        "schema_version": 1,
        "groundPlane": {"y": navigation["ground_y"]},
        "boundingWalls": {
            "xMin": navigation["collider_minimum"][0],
            "xMax": navigation["collider_maximum"][0],
            "zMin": navigation["collider_minimum"][2],
            "zMax": navigation["collider_maximum"][2],
            "yMax": navigation["collider_maximum"][1],
        },
        "spawn": {
            "position": navigation["spawn_position"],
            "lookDirection": navigation["spawn_look_direction"],
        },
    }
    transform = orientation.get("transform")
    if isinstance(transform, Mapping):
        collider_payload["worldRotation"] = {"quaternion": transform["quaternion_xyzw"]}
    trajectory_payload = {
        "schema_version": 1,
        "kind": "registered-camera-trajectory",
        "default_camera": default_camera["image_name"],
        "cameras": registered_cameras,
    }
    request_payload = {
        "schema_version": 1,
        "kind": "static-splat-world-metadata",
        "source_ply": metadata["source"]["ply"],
        "environment": metadata["environment"],
        "viewer_bundle_included": False,
    }
    written.update(
        {
            "collider": _write_json(world / "collider.json", collider_payload),
            "trajectory": _write_json(
                world / "trajectory.json",
                trajectory_payload,
            ),
            "request": _write_json(world / "request.json", request_payload),
        }
    )
    return written


def _preview_size(
    camera: Mapping[str, object],
    max_long_edge: int,
) -> tuple[int, int]:
    width = int(camera["width"])
    height = int(camera["height"])
    if width <= 0 or height <= 0 or max_long_edge <= 0:
        raise ValueError("preview dimensions must be positive")
    scale = min(1.0, max_long_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _default_preview_renderer(
    ply_path: Path,
    camera: Mapping[str, object],
    width: int,
    height: int,
) -> np.ndarray:
    import torch

    from backend.eval.nvs_eval import _scale_K
    from backend.model.renderer import render_view

    from .polish import _renderer_arrays

    validated = validate_static_ply(ply_path)

    def tensor(value: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        return torch.from_numpy(contiguous).to("cuda")

    means, quaternions, scales, opacities, colors = map(
        tensor,
        _renderer_arrays(validated),
    )
    intrinsic = tensor(np.asarray(camera["K"], dtype=np.float32))
    intrinsic = _scale_K(
        intrinsic,
        int(camera["width"]),
        int(camera["height"]),
        width,
        height,
    )
    world_to_camera = tensor(np.asarray(camera["w2c"], dtype=np.float32))
    with torch.no_grad():
        rendered, _, _ = render_view(
            means=means,
            quats=quaternions,
            scales=scales,
            opacities=opacities,
            colors=colors,
            K=intrinsic,
            w2c=world_to_camera,
            width=width,
            height=height,
            sh_degree=validated.sh_degree,
        )
    return rendered.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()


def _sparse_preview(
    reconstruction: ReconstructionBundle,
    camera: Mapping[str, object],
    width: int,
    height: int,
    background_rgb: list[float],
) -> np.ndarray:
    points, colors, _, _ = load_points3d_from_model(reconstruction.accepted_model_dir)
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    intrinsic = np.asarray(camera["K"], dtype=np.float64).copy()
    native_width = int(camera["width"])
    native_height = int(camera["height"])
    intrinsic[0, (0, 2)] *= width / native_width
    intrinsic[1, (1, 2)] *= height / native_height
    world_to_camera = np.asarray(camera["w2c"], dtype=np.float64)
    camera_xyz = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera_xyz[:, 2]
    in_front = depth > 1e-8
    safe_depth = np.where(in_front, depth, 1.0)
    pixel_x = np.rint(
        intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2]
    ).astype(int)
    pixel_y = np.rint(
        intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2]
    ).astype(int)
    visible = (
        in_front
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    background = np.clip(
        np.asarray(background_rgb, dtype=np.float64),
        0.0,
        1.0,
    )
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = np.rint(background * 255.0).astype(np.uint8)
    visible_indices = np.flatnonzero(visible)
    for index in visible_indices[np.argsort(depth[visible_indices])[::-1]]:
        x = pixel_x[index]
        y = pixel_y[index]
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        image[y0:y1, x0:x1] = colors[index]
    return image


def render_preview(
    ply_path: str | Path,
    reconstruction: ReconstructionBundle,
    metadata: Mapping[str, object],
    output_path: str | Path,
    *,
    render_backend: PreviewRenderBackend | None = None,
    max_long_edge: int = 960,
) -> dict[str, object]:
    """Render a required preview, falling back to sparse COLMAP projection."""

    from PIL import Image

    default_camera = metadata.get("default_camera")
    environment = metadata.get("environment")
    if not isinstance(default_camera, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("scene metadata is missing preview camera/environment")
    width, height = _preview_size(default_camera, max_long_edge)
    renderer = render_backend or _default_preview_renderer
    fallback_used = False
    warning: str | None = None
    try:
        image = np.asarray(renderer(Path(ply_path), default_camera, width, height))
        if image.shape != (height, width, 3) or not np.isfinite(image).all():
            raise ValueError("preview renderer returned invalid pixels")
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
            image = np.rint(image * 255.0).astype(np.uint8)
    except Exception:
        background = environment.get("background_rgb", [0.0, 0.0, 0.0])
        if not isinstance(background, list) or len(background) != 3:
            raise ValueError("scene metadata has an invalid preview background")
        image = _sparse_preview(
            reconstruction,
            default_camera,
            width,
            height,
            background,
        )
        fallback_used = True
        warning = "preview_renderer_failed_sparse_fallback"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(destination, format="PNG")
    return {
        "path": destination,
        "fallback_used": fallback_used,
        "warning": warning,
    }
