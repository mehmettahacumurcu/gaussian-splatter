"""Camera intrinsics for sub-project B.

Default path: assume FOV from config (60° H) and compute fx, fy from image
dimensions. Future enhancement: extract FOV from EXIF (Task 2.2).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K


def intrinsics_from_fov(
    width: int,
    height: int,
    fov_horizontal_deg: float = 60.0,
) -> CameraIntrinsics:
    """Build intrinsics assuming square pixels, principal point at image center."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    cx = width / 2.0
    cy = height / 2.0
    fx = cx / math.tan(math.radians(fov_horizontal_deg) / 2.0)
    return CameraIntrinsics(fx=fx, fy=fx, cx=cx, cy=cy, width=width, height=height)
