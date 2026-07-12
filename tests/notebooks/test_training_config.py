from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.notebooks.models import (
    NotebookAdvancedConfig,
    NotebookQualityProfile,
    NotebookQualitySpec,
    StaticNotebookRunSpec,
)
from backend.notebooks.training_config import (
    ResolvedStaticConfig,
    resolve_static_training_config,
)


def _spec(
    *,
    profile: NotebookQualityProfile = NotebookQualityProfile.BALANCED_L4,
    n_iters: int | None = None,
    max_gaussians: int | None = None,
    advanced: NotebookAdvancedConfig | None = None,
) -> StaticNotebookRunSpec:
    return StaticNotebookRunSpec(
        input_folder="captures/room",
        quality=NotebookQualitySpec(
            profile=profile,
            n_iters=n_iters,
            max_gaussians=max_gaussians,
            advanced=advanced or NotebookAdvancedConfig(),
        ),
    )


def _config_digest(cfg: object) -> str:
    payload = json.dumps(
        dataclasses.asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_training_config_module_keeps_backend_config_lazy() -> None:
    repository = Path(__file__).resolve().parents[2]
    command = (
        "import sys; "
        "import backend.notebooks.training_config; "
        "raise SystemExit(any(name in sys.modules for name in "
        "('backend.config', 'backend.static_presets')))"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_balanced_l4_retains_legacy_balanced_training_values() -> None:
    cfg, resolved = resolve_static_training_config(_spec())

    assert resolved.profile is NotebookQualityProfile.BALANCED_L4
    assert resolved.legacy_preset == "balanced"
    assert resolved.foundation is True
    assert resolved.run_eval is False
    assert resolved.native_resolution is False
    assert cfg.train.static_mode is True
    assert cfg.export.num_timestamps == 1
    assert cfg.train.n_iters == resolved.n_iters == 30_000
    assert cfg.train.max_gaussians == resolved.max_gaussians == 250_000
    assert cfg.preprocess.resize_long_edge == resolved.resolution_long_edge_cap == 1280


def test_foundation_false_is_normalized_before_snapshot_and_digest() -> None:
    cfg, resolved = resolve_static_training_config(
        _spec(advanced=NotebookAdvancedConfig(foundation=False))
    )

    assert resolved.foundation is False
    assert cfg.train.lambda_depth == resolved.lambda_depth == 0.0
    assert cfg.train.auto_static_dynamic is False
    for field in (
        "lambda_mask_motion",
        "lambda_track",
        "lambda_flow",
        "lambda_smoothness",
        "lambda_rigidity",
        "lambda_deform_reg",
        "lambda_fourier_reg",
        "lambda_multiview_consistency",
    ):
        assert getattr(cfg.train, field) == 0.0
    assert resolved.config_digest == _config_digest(cfg)


def test_ultra_native_resolution_is_normalized_before_snapshot_and_digest() -> None:
    advanced = NotebookAdvancedConfig(
        multires_schedule=[(0, 640), (20_000, 2560), (40_000, 3840)]
    )

    cfg, resolved = resolve_static_training_config(
        _spec(profile=NotebookQualityProfile.ULTRA, advanced=advanced),
        native_image_size=(1920, 1080),
    )

    assert cfg.train.image_resolution == resolved.image_resolution == (1920, 1080)
    assert cfg.train.multires_schedule == [(0, 640), (20_000, 1920)]
    assert resolved.multires_schedule == ((0, 640), (20_000, 1920))
    assert resolved.config_digest == _config_digest(cfg)


def test_primary_and_advanced_overrides_map_only_to_approved_targets() -> None:
    advanced = NotebookAdvancedConfig(
        run_eval=True,
        foundation=False,
        resolution_long_edge_cap=1440,
        lambda_ssim=0.0,
        lambda_lpips=0.25,
        lambda_depth=0.0,
        density_start_iter=0,
        density_end_iter=30_000,
        density_interval=250,
        densify_grad_threshold=0.0003,
        prune_min_opacity=0.0,
        prune_max_scale=0.03,
        opacity_reset_interval=0,
        sh_degree=2,
        multires_schedule=[(0, 640), (20_000, 1280)],
    )
    cfg, resolved = resolve_static_training_config(
        _spec(n_iters=40_000, max_gaussians=600_000, advanced=advanced)
    )

    assert cfg.train.n_iters == resolved.n_iters == 40_000
    assert cfg.train.max_gaussians == resolved.max_gaussians == 600_000
    assert cfg.preprocess.resize_long_edge == resolved.resolution_long_edge_cap == 1440
    assert cfg.train.lambda_ssim == resolved.lambda_ssim == 0.0
    assert cfg.train.lambda_lpips == resolved.lambda_lpips == 0.25
    assert cfg.train.lambda_depth == resolved.lambda_depth == 0.0
    assert cfg.train.density_start_iter == resolved.density_start_iter == 0
    assert cfg.train.density_end_iter == resolved.density_end_iter == 30_000
    assert cfg.train.density_interval == resolved.density_interval == 250
    assert cfg.train.densify_grad_threshold == resolved.densify_grad_threshold == 0.0003
    assert cfg.train.prune_min_opacity == resolved.prune_min_opacity == 0.0
    assert cfg.train.prune_max_scale == resolved.prune_max_scale == 0.03
    assert cfg.train.opacity_reset_interval == resolved.opacity_reset_interval == 0
    assert cfg.model.sh_degree == resolved.sh_degree == 2
    assert cfg.train.multires_schedule == [(0, 640), (20_000, 1280)]
    assert resolved.multires_schedule == ((0, 640), (20_000, 1280))
    assert cfg.train.nvs_eval_enabled is resolved.run_eval is True
    assert resolved.foundation is False


@pytest.mark.parametrize(
    ("profile", "explicit_cap", "source_long_edge", "expected_cap"),
    [
        (NotebookQualityProfile.BALANCED_L4, None, 900, 900),
        (NotebookQualityProfile.BALANCED_L4, None, 2000, 1280),
        (NotebookQualityProfile.HIGH, 1600, 1000, 1000),
        (NotebookQualityProfile.ULTRA, None, 1920, None),
        (NotebookQualityProfile.ULTRA, 2560, 1920, 1920),
    ],
)
def test_resolution_cap_never_upscales_known_sources_and_ultra_can_remain_native(
    profile: NotebookQualityProfile,
    explicit_cap: int | None,
    source_long_edge: int,
    expected_cap: int | None,
) -> None:
    advanced = NotebookAdvancedConfig(resolution_long_edge_cap=explicit_cap)

    cfg, resolved = resolve_static_training_config(
        _spec(profile=profile, advanced=advanced),
        source_long_edge=source_long_edge,
    )

    assert cfg.preprocess.resize_long_edge == expected_cap
    assert resolved.resolution_long_edge_cap == expected_cap
    assert cfg.train.native_resolution is (profile is NotebookQualityProfile.ULTRA)
    assert resolved.native_resolution is (profile is NotebookQualityProfile.ULTRA)


@pytest.mark.parametrize("source_long_edge", [True, False, 0, -1, 1280.0, "1280"])
def test_source_long_edge_must_be_a_positive_integer(source_long_edge: object) -> None:
    with pytest.raises(ValueError, match="source_long_edge"):
        resolve_static_training_config(_spec(), source_long_edge=source_long_edge)


def test_resolved_config_is_frozen_complete_and_uses_canonical_config_digest() -> None:
    cfg, resolved = resolve_static_training_config(
        _spec(profile=NotebookQualityProfile.PREMIUM)
    )

    assert dataclasses.fields(ResolvedStaticConfig)
    assert resolved.config_digest == _config_digest(cfg)
    assert len(resolved.config_digest) == 64
    assert resolved.lambda_ssim == cfg.train.lambda_ssim
    assert resolved.lambda_lpips == cfg.train.lambda_lpips
    assert resolved.lambda_depth == cfg.train.lambda_depth
    assert resolved.density_start_iter == cfg.train.density_start_iter
    assert resolved.density_end_iter == cfg.train.density_end_iter
    assert resolved.density_interval == cfg.train.density_interval
    assert resolved.densify_grad_threshold == cfg.train.densify_grad_threshold
    assert resolved.prune_min_opacity == cfg.train.prune_min_opacity
    assert resolved.prune_max_scale == cfg.train.prune_max_scale
    assert resolved.opacity_reset_interval == cfg.train.opacity_reset_interval
    assert resolved.sh_degree == cfg.model.sh_degree
    assert resolved.multires_schedule == tuple(map(tuple, cfg.train.multires_schedule))
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.n_iters = 1  # type: ignore[misc]
