from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import NotebookQualityProfile, StaticNotebookRunSpec

if TYPE_CHECKING:
    from backend.config import Config


@dataclass(frozen=True)
class ResolvedStaticConfig:
    profile: NotebookQualityProfile
    legacy_preset: str
    foundation: bool
    run_eval: bool
    native_resolution: bool
    image_resolution: tuple[int, int]
    n_iters: int
    max_gaussians: int
    resolution_long_edge_cap: int | None
    lambda_ssim: float
    lambda_lpips: float
    lambda_depth: float
    density_start_iter: int
    density_end_iter: int
    density_interval: int
    densify_grad_threshold: float
    prune_min_opacity: float
    prune_max_scale: float
    opacity_reset_interval: int
    sh_degree: int
    multires_schedule: tuple[tuple[int, int], ...]
    config_digest: str


_ADVANCED_TRAIN_TARGETS = {
    "lambda_ssim": "lambda_ssim",
    "lambda_lpips": "lambda_lpips",
    "lambda_depth": "lambda_depth",
    "density_start_iter": "density_start_iter",
    "density_end_iter": "density_end_iter",
    "density_interval": "density_interval",
    "densify_grad_threshold": "densify_grad_threshold",
    "prune_min_opacity": "prune_min_opacity",
    "prune_max_scale": "prune_max_scale",
    "opacity_reset_interval": "opacity_reset_interval",
    "multires_schedule": "multires_schedule",
}

_STATIC_ZEROED_TRAIN_FIELDS = (
    "lambda_mask_motion",
    "lambda_track",
    "lambda_flow",
    "lambda_smoothness",
    "lambda_rigidity",
    "lambda_deform_reg",
    "lambda_fourier_reg",
    "lambda_multiview_consistency",
)


def _canonical_config_digest(cfg: Config) -> str:
    payload = json.dumps(
        dataclasses.asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_legacy_static_runtime(
    cfg: Config,
    *,
    foundation: bool,
    native_image_size: tuple[int, int] | None,
) -> None:
    """Apply mutations the legacy pipeline performs before static training."""

    for field in _STATIC_ZEROED_TRAIN_FIELDS:
        setattr(cfg.train, field, 0.0)
    cfg.train.auto_static_dynamic = False
    if not foundation:
        cfg.train.lambda_depth = 0.0

    if not cfg.train.native_resolution or native_image_size is None:
        return

    width, height = native_image_size
    cfg.train.image_resolution = (width, height)
    native_long_edge = max(width, height)
    clamped_schedule: list[tuple[int, int]] = []
    for iteration, long_edge in map(tuple, cfg.train.multires_schedule or []):
        clamped_edge = min(int(long_edge), native_long_edge)
        if clamped_schedule and clamped_edge <= clamped_schedule[-1][1]:
            continue
        clamped_schedule.append((int(iteration), clamped_edge))
    cfg.train.multires_schedule = clamped_schedule


def resolve_static_training_config(
    spec: StaticNotebookRunSpec,
    source_long_edge: int | None = None,
    *,
    native_image_size: tuple[int, int] | None = None,
) -> tuple[Config, ResolvedStaticConfig]:
    if source_long_edge is not None and (
        type(source_long_edge) is not int or source_long_edge <= 0
    ):
        raise ValueError("source_long_edge must be a positive integer when set")
    if native_image_size is not None and (
        not isinstance(native_image_size, tuple)
        or len(native_image_size) != 2
        or any(type(value) is not int or value <= 0 for value in native_image_size)
    ):
        raise ValueError(
            "native_image_size must contain two positive integers when set"
        )

    from backend.config import default_config
    from backend.static_presets import apply_static_preset_for_api

    from .presets import get_notebook_profile

    quality = spec.quality
    advanced = quality.advanced
    profile = get_notebook_profile(quality.profile)

    cfg = default_config()
    legacy_preset = apply_static_preset_for_api(cfg, profile.legacy_preset)

    if quality.n_iters is not None:
        cfg.train.n_iters = quality.n_iters
    if quality.max_gaussians is not None:
        cfg.train.max_gaussians = quality.max_gaussians

    for advanced_name, target_name in _ADVANCED_TRAIN_TARGETS.items():
        value = getattr(advanced, advanced_name)
        if value is not None:
            setattr(cfg.train, target_name, value)
    if advanced.sh_degree is not None:
        cfg.model.sh_degree = advanced.sh_degree

    run_eval = (
        advanced.run_eval if advanced.run_eval is not None else profile.run_eval_default
    )
    foundation = (
        advanced.foundation
        if advanced.foundation is not None
        else profile.foundation_default
    )
    native_resolution = profile.id.value == "ultra"

    resolution_long_edge_cap = (
        advanced.resolution_long_edge_cap
        if advanced.resolution_long_edge_cap is not None
        else profile.resolution_long_edge_cap
    )
    if resolution_long_edge_cap is not None and source_long_edge is not None:
        resolution_long_edge_cap = min(resolution_long_edge_cap, source_long_edge)

    cfg.preprocess.resize_long_edge = resolution_long_edge_cap
    cfg.train.static_mode = True
    cfg.train.nvs_eval_enabled = run_eval
    cfg.train.native_resolution = native_resolution
    cfg.export.num_timestamps = 1
    _normalize_legacy_static_runtime(
        cfg,
        foundation=foundation,
        native_image_size=native_image_size,
    )

    resolved = ResolvedStaticConfig(
        profile=profile.id,
        legacy_preset=legacy_preset,
        foundation=foundation,
        run_eval=run_eval,
        native_resolution=native_resolution,
        image_resolution=tuple(cfg.train.image_resolution),
        n_iters=cfg.train.n_iters,
        max_gaussians=cfg.train.max_gaussians,
        resolution_long_edge_cap=cfg.preprocess.resize_long_edge,
        lambda_ssim=cfg.train.lambda_ssim,
        lambda_lpips=cfg.train.lambda_lpips,
        lambda_depth=cfg.train.lambda_depth,
        density_start_iter=cfg.train.density_start_iter,
        density_end_iter=cfg.train.density_end_iter,
        density_interval=cfg.train.density_interval,
        densify_grad_threshold=cfg.train.densify_grad_threshold,
        prune_min_opacity=cfg.train.prune_min_opacity,
        prune_max_scale=cfg.train.prune_max_scale,
        opacity_reset_interval=cfg.train.opacity_reset_interval,
        sh_degree=cfg.model.sh_degree,
        multires_schedule=tuple(map(tuple, cfg.train.multires_schedule)),
        config_digest=_canonical_config_digest(cfg),
    )
    return cfg, resolved
