"""B-specific tunables. See spec §6 (config profiles) and §3 (algorithm).

Three named profiles balance runtime vs quality. The "default" profile
targets the spec's 90-min budget on a 3060 Ti.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ImageToSceneConfig:
    # Camera trajectory
    n_views: int = 30
    bubble_radius_m: float = 5.0
    n_orbit_rings: int = 3  # spiral layers around capture point
    pitch_range_deg: float = 30.0
    yaw_full_360: bool = True

    # Intrinsics defaults — used if EXIF FOV unavailable
    default_fov_deg: float = 60.0
    image_max_dim: int = 512  # working resolution; downscaled before depth/inpaint

    # Inpaint
    inpaint_strength: float = 0.99
    inpaint_steps: int = 30
    inpaint_guidance: float = 7.5

    # Depth alignment
    align_min_overlap_pixels: int = 500
    align_residual_reject_threshold: float = 0.25

    # Per-view Gaussian budget
    max_new_gaussians_per_view_frac: float = 0.20

    # Final 3DGS training
    train_iterations: int = 3000
    train_learning_rate: float = 1.6e-4

    # VRAM budget
    peak_vram_gb_target: float = 7.5

    # Output options
    export_spz: bool = True
    export_thumbnail: bool = True


_PROFILES = {
    "fast": ImageToSceneConfig(n_views=15, train_iterations=1500),
    "default": ImageToSceneConfig(),
    "quality": ImageToSceneConfig(n_views=50, train_iterations=5000),
}


def profile(name: str) -> ImageToSceneConfig:
    """Return a named profile. Raises ValueError on unknown name."""
    if name not in _PROFILES:
        raise ValueError(f"unknown profile {name!r}; valid: {list(_PROFILES)}")
    import dataclasses
    return dataclasses.replace(_PROFILES[name])
