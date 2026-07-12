"""Parse COLMAP text models: cameras.txt, images.txt, and points3D.txt."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_CAMERA_PARAM_COUNTS = {
    "PINHOLE": 4,
    "SIMPLE_PINHOLE": 3,
    "SIMPLE_RADIAL": 4,
    "OPENCV": 8,
}
_IMAGE_COUNT_RE = re.compile(
    r"^#\s*Number of images:\s*(\d+)(?:\s*,|$)",
)


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _find_sparse_dir(colmap_dir: Path) -> Path:
    """Find the legacy sparse sub-model that has converted camera text."""
    sparse_root = colmap_dir / "sparse"
    if sparse_root.exists() and sparse_root.is_dir():
        candidates = [
            directory
            for directory in sparse_root.iterdir()
            if directory.is_dir() and (directory / "cameras.txt").exists()
        ]
        if candidates:

            def _size(directory: Path) -> int:
                total = 0
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    binary_file = directory / name
                    if binary_file.exists():
                        total += binary_file.stat().st_size
                return total

            candidates.sort(key=_size, reverse=True)
            return candidates[0]

    if (colmap_dir / "cameras.txt").exists():
        return colmap_dir

    raise FileNotFoundError(
        f"cameras.txt not found under {colmap_dir}; tried sparse/* and the root"
    )


def _parse_camera_row(line: str, line_number: int) -> tuple[int, dict[str, object]]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Malformed camera row at line {line_number}")

    model = parts[1]
    if model not in _CAMERA_PARAM_COUNTS:
        raise NotImplementedError(f"Camera model unsupported: {model}")

    expected_params = _CAMERA_PARAM_COUNTS[model]
    if len(parts) != 4 + expected_params:
        raise ValueError(
            f"Malformed camera row at line {line_number}: {model} expects "
            f"{expected_params} parameters"
        )
    try:
        camera_id = int(parts[0])
        width = int(parts[2])
        height = int(parts[3])
        params = tuple(float(value) for value in parts[4:])
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Malformed camera row at line {line_number}") from exc

    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        focal, cx, cy = params[:3]
        fx = fy = focal
    else:
        fx, fy, cx, cy = params[:4]

    intrinsics = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return camera_id, {
        "K": intrinsics,
        "width": width,
        "height": height,
    }


def _parse_cameras_text(model_dir: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    with (model_dir / "cameras.txt").open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            camera_id, camera = _parse_camera_row(line, line_number)
            if camera_id in cameras:
                raise ValueError(f"Duplicate camera ID: {camera_id}")
            cameras[camera_id] = camera
    if not cameras:
        raise RuntimeError("cameras.txt contains no camera records")
    return cameras


def _validate_points2d_row(row: str, line_number: int) -> None:
    if not row:
        return
    parts = row.split()
    if len(parts) % 3 != 0:
        raise ValueError(f"Malformed POINTS2D row at line {line_number}")
    try:
        for offset in range(0, len(parts), 3):
            float(parts[offset])
            float(parts[offset + 1])
            int(parts[offset + 2])
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Malformed POINTS2D row at line {line_number}") from exc


def _image_record_lines(images_file: Path) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    pending_header: tuple[int, str] | None = None
    declared_count: int | None = None
    with images_file.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line.startswith("#"):
                count_match = _IMAGE_COUNT_RE.match(line)
                if count_match is not None:
                    parsed_count = int(count_match.group(1))
                    if declared_count is not None and parsed_count != declared_count:
                        raise ValueError(
                            "images.txt has conflicting declared image counts"
                        )
                    declared_count = parsed_count
                continue
            if pending_header is None:
                if not line:
                    continue
                pending_header = (line_number, line)
                continue

            _validate_points2d_row(line, line_number)
            records.append(pending_header)
            pending_header = None

    if pending_header is not None:
        line_number, _ = pending_header
        raise ValueError(
            f"Missing POINTS2D record after image header at line {line_number}"
        )
    if declared_count is None:
        raise ValueError("images.txt is missing its declared image count")
    if len(records) != declared_count:
        raise ValueError(
            "POINTS2D/image count mismatch: "
            f"declared {declared_count}, parsed {len(records)}"
        )
    return records


def _parse_image_header(
    header: str,
    line_number: int,
) -> tuple[int, tuple[float, ...], tuple[float, ...], int, str]:
    parts = header.split(maxsplit=9)
    if len(parts) != 10 or not parts[9]:
        raise ValueError(f"Malformed image header at line {line_number}")
    try:
        image_id = int(parts[0])
        quaternion = tuple(float(value) for value in parts[1:5])
        translation = tuple(float(value) for value in parts[5:8])
        camera_id = int(parts[8])
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Malformed image header at line {line_number}") from exc

    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    quaternion_norm = float(np.linalg.norm(quaternion_array))
    if not np.all(np.isfinite(quaternion_array)) or not np.isfinite(quaternion_norm):
        raise ValueError(f"Invalid quaternion at image line {line_number}")
    if quaternion_norm <= 1e-12:
        raise ValueError(f"Near-zero quaternion at image line {line_number}")
    return image_id, quaternion, translation, camera_id, parts[9]


def _parse_camera_and_image_text(model_dir: Path) -> dict[str, dict]:
    cameras = _parse_cameras_text(model_dir)
    parsed_images: list[tuple[int, tuple[float, ...], tuple[float, ...], int, str]] = []
    image_ids: set[int] = set()
    image_names: set[str] = set()
    for line_number, header in _image_record_lines(model_dir / "images.txt"):
        parsed = _parse_image_header(header, line_number)
        image_id, _, _, camera_id, name = parsed
        if image_id in image_ids:
            raise ValueError(f"Duplicate image ID: {image_id}")
        if name in image_names:
            raise ValueError(f"Duplicate image name: {name}")
        if camera_id not in cameras:
            raise ValueError(
                f"Image {image_id} references missing camera ID {camera_id}"
            )
        image_ids.add(image_id)
        image_names.add(name)
        parsed_images.append(parsed)

    images: dict[str, dict] = {}
    for image_id, quaternion, translation, camera_id, name in parsed_images:
        rotation = quat_to_rotmat(*quaternion)
        translation_array = np.asarray(translation, dtype=np.float64)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation_array
        camera = cameras[camera_id]
        images[name] = {
            "image_id": image_id,
            "K": camera["K"],
            "R": rotation,
            "t": translation_array,
            "w2c": world_to_camera,
            "width": camera["width"],
            "height": camera["height"],
        }

    if not images:
        raise RuntimeError("images.txt contains no image records")
    return images


def parse_cameras_from_model(model_dir: str | Path) -> dict[str, dict]:
    """Read cameras and registered images from one exact COLMAP text model."""
    model = Path(model_dir)
    cameras_file = model / "cameras.txt"
    images_file = model / "images.txt"
    if (
        cameras_file.is_symlink()
        or images_file.is_symlink()
        or not cameras_file.is_file()
        or not images_file.is_file()
    ):
        raise FileNotFoundError(f"Incomplete COLMAP text model: {model}")
    return _parse_camera_and_image_text(model)


def parse_cameras(colmap_dir: str | Path) -> dict[str, dict]:
    """Read cameras using the legacy sparse-model discovery policy."""
    cameras = parse_cameras_from_model(_find_sparse_dir(Path(colmap_dir)))
    print(f"[ok] {len(cameras)} camera poses loaded")
    return cameras


def _parse_points3d_text(
    points_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz_rows: list[list[float]] = []
    rgb_rows: list[list[int]] = []
    track_lengths: list[int] = []
    reprojection_errors: list[float] = []
    point_ids: set[int] = set()

    with points_file.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Malformed point row at line {line_number}")
            if (len(parts) - 8) % 2 != 0:
                raise ValueError(f"Malformed track data at point line {line_number}")
            try:
                point_id = int(parts[0])
                xyz = [float(value) for value in parts[1:4]]
                rgb = [int(value) for value in parts[4:7]]
                reprojection_error = float(parts[7])
                for offset in range(8, len(parts), 2):
                    int(parts[offset])
                    int(parts[offset + 1])
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"Malformed point row at line {line_number}") from exc

            if point_id in point_ids:
                raise ValueError(f"Duplicate point ID: {point_id}")
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise ValueError(f"Malformed point RGB at line {line_number}")
            point_ids.add(point_id)
            xyz_rows.append(xyz)
            rgb_rows.append(rgb)
            track_lengths.append((len(parts) - 8) // 2)
            reprojection_errors.append(reprojection_error)

    if not xyz_rows:
        raise RuntimeError(f"{points_file.name} is empty")
    return (
        np.asarray(xyz_rows, dtype=np.float32),
        np.asarray(rgb_rows, dtype=np.uint8),
        np.asarray(track_lengths, dtype=np.int32),
        np.asarray(reprojection_errors, dtype=np.float32),
    )


def load_points3d_from_model(
    model_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read points and confidence metrics from one exact COLMAP text model."""
    points_file = Path(model_dir) / "points3D.txt"
    if points_file.is_symlink() or not points_file.is_file():
        raise FileNotFoundError(f"points3D.txt not found: {points_file}")
    return _parse_points3d_text(points_file)


def load_points3d(colmap_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sparse points using the legacy sparse-model discovery policy."""
    xyz, rgb, _, _ = load_points3d_from_model(_find_sparse_dir(Path(colmap_dir)))
    print(f"[ok] {len(xyz)} sparse 3D points loaded")
    return xyz, rgb


def load_points3d_with_confidence(
    colmap_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sparse points plus track lengths and reprojection errors."""
    xyz, rgb, track_lengths, reprojection_errors = load_points3d_from_model(
        _find_sparse_dir(Path(colmap_dir))
    )
    print(
        f"[ok] {len(xyz)} sparse points + confidence loaded "
        f"(track median={np.median(track_lengths):.0f}, "
        f"max={track_lengths.max()}; "
        f"error median={np.median(reprojection_errors):.2f}, "
        f"max={reprojection_errors.max():.2f})"
    )
    return xyz, rgb, track_lengths, reprojection_errors


def scene_extent(xyz: np.ndarray) -> float:
    """Return the radial extent of a point cloud around its centroid."""
    centroid = xyz.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(xyz - centroid, axis=1).max())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse COLMAP text output")
    parser.add_argument("colmap_dir", type=str)
    arguments = parser.parse_args()

    parsed_cameras = parse_cameras(arguments.colmap_dir)
    parsed_xyz, parsed_rgb = load_points3d(arguments.colmap_dir)
    first_camera = next(iter(parsed_cameras.values()))
    print(f"\nExample camera intrinsics:\n{first_camera['K']}")
    print(f"Scene radial extent: {scene_extent(parsed_xyz):.3f}")
