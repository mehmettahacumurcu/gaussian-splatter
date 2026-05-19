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


from pathlib import Path
from typing import Optional


def fov_from_exif(image_path: Path) -> Optional[float]:
    """Extract horizontal FOV (degrees) from EXIF if possible. Returns None on failure.

    Estimates from FocalLength + assumed 36 mm full-frame sensor.
    Returns None when EXIF is absent or insufficient — caller falls back to config.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return None

    try:
        with Image.open(image_path) as img:
            exif = img._getexif() or {}
    except Exception:
        return None

    if not exif:
        return None

    tag_map = {TAGS.get(k, k): v for k, v in exif.items()}

    focal_length = tag_map.get("FocalLength")
    if focal_length is None:
        return None

    if hasattr(focal_length, "numerator"):
        focal_mm = focal_length.numerator / focal_length.denominator
    else:
        focal_mm = float(focal_length)

    if focal_mm <= 0:
        return None

    sensor_width_mm = 36.0
    fov_deg = math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_mm)))

    if not (15.0 < fov_deg < 130.0):
        return None
    return fov_deg


def intrinsics_for_image(
    image_path: Path,
    fallback_fov_deg: float = 60.0,
) -> CameraIntrinsics:
    """Build intrinsics for a single image, using EXIF FOV if available."""
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size

    fov_deg = fov_from_exif(image_path) or fallback_fov_deg
    return intrinsics_from_fov(width=width, height=height, fov_horizontal_deg=fov_deg)
