"""Minimal collider derivation — spec §8.

Not a real surface mesh. Just a ground plane + 4-wall bounding box derived from
the splat point cloud, so the user has something to walk on inside the splat.

The full surface-accurate collider is sub-project C's job.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class MinimalCollider:
    ground_y: float
    bbox_min: np.ndarray   # (3,) [x_min, y_min, z_min]
    bbox_max: np.ndarray   # (3,) [x_max, y_max, z_max]
    spawn_position: np.ndarray
    spawn_look_direction: np.ndarray


def derive_minimal_collider(
    points: np.ndarray,
    *,
    eye_height_m: float = 1.7,
    floor_percentile: float = 0.10,
    extent_percentile: float = 0.95,
) -> MinimalCollider:
    """Derive ground plane (lowest 10% Y) + axis-aligned XZ bbox (95th percentile)."""
    if points.shape[0] < 10:
        raise ValueError("Need at least 10 points to derive a collider.")

    ys = points[:, 1]
    ground_y = float(np.quantile(ys, floor_percentile))

    xs = points[:, 0]
    zs = points[:, 2]
    half = (1.0 - extent_percentile) / 2.0
    x_min = float(np.quantile(xs, half))
    x_max = float(np.quantile(xs, 1.0 - half))
    z_min = float(np.quantile(zs, half))
    z_max = float(np.quantile(zs, 1.0 - half))
    y_max = float(np.quantile(ys, extent_percentile))

    bbox_min = np.array([x_min, ground_y, z_min], dtype=np.float32)
    bbox_max = np.array([x_max, y_max, z_max], dtype=np.float32)

    spawn = np.array([0.0, ground_y + eye_height_m, 0.0], dtype=np.float64)
    look = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return MinimalCollider(
        ground_y=ground_y,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        spawn_position=spawn,
        spawn_look_direction=look,
    )


def write_collider_json(path: Path, collider: MinimalCollider) -> None:
    """Serialize to the JSON shape consumed by the frontend WorldCollider loader."""
    payload = {
        "schema_version": 1,
        "groundPlane": {"y": collider.ground_y},
        "boundingWalls": {
            "xMin": float(collider.bbox_min[0]),
            "xMax": float(collider.bbox_max[0]),
            "zMin": float(collider.bbox_min[2]),
            "zMax": float(collider.bbox_max[2]),
            "yMax": float(collider.bbox_max[1]),
        },
        "spawn": {
            "position": [round(float(v), 6) for v in collider.spawn_position],
            "lookDirection": [round(float(v), 6) for v in collider.spawn_look_direction],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
