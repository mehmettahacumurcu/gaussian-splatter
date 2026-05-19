"""Bounded-room camera trajectory synthesis.

The first pose is always at the origin facing +Z (the input image's viewpoint).
Subsequent poses spiral outward in concentric rings, with yaw sweeping the full
range so the user can later look in any direction from any point.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import List


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray   # (3,)
    rotation: np.ndarray   # (3, 3) world-from-camera; columns 0/1/2 = right/up/forward
    fov_horizontal_deg: float

    def as_world_from_camera(self) -> np.ndarray:
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.rotation
        M[:3, 3] = self.position
        return M


def _yaw_rotation(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _pitch_rotation(pitch_rad: float) -> np.ndarray:
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def generate_bounded_room_trajectory(
    n_views: int,
    bubble_radius_m: float = 5.0,
    n_orbit_rings: int = 3,
    pitch_range_deg: float = 30.0,
    yaw_full_360: bool = True,
    fov_horizontal_deg: float = 60.0,
) -> List[CameraPose]:
    """Build a list of camera poses for the outpaint loop.

    Pose 0 is always at the origin facing +Z (matches the input image).
    Remaining poses distribute across n_orbit_rings concentric circles,
    each at a different radius/pitch, with yaw varied per pose.
    """
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    poses: List[CameraPose] = []

    # Pose 0: origin, identity rotation.
    poses.append(CameraPose(
        position=np.zeros(3, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        fov_horizontal_deg=fov_horizontal_deg,
    ))

    remaining = n_views - 1
    if remaining <= 0:
        return poses

    yaw_span = 2 * math.pi if yaw_full_360 else math.pi

    # Distribute yaw globally across all remaining poses so that, regardless
    # of how many rings there are, the full yaw span is always covered.
    for pose_index in range(remaining):
        # Which ring does this pose belong to?
        ring = (pose_index * n_orbit_rings) // remaining
        radius_frac = (ring + 1) / n_orbit_rings
        ring_radius = bubble_radius_m * radius_frac * 0.7
        pitch_deg = pitch_range_deg * (radius_frac - 0.5)
        pitch_rad = math.radians(pitch_deg)

        # Yaw evenly distributed across all remaining poses.
        yaw = yaw_span * pose_index / remaining
        position = np.array([
            math.sin(yaw) * ring_radius,
            0.0,
            math.cos(yaw) * ring_radius - ring_radius,
        ], dtype=np.float64)
        r = float(np.linalg.norm(position))
        if r > bubble_radius_m:
            position *= bubble_radius_m / r

        rotation = _yaw_rotation(yaw) @ _pitch_rotation(pitch_rad)
        poses.append(CameraPose(
            position=position,
            rotation=rotation,
            fov_horizontal_deg=fov_horizontal_deg,
        ))

    return poses[:n_views]
