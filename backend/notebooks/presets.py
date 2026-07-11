from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scripts.static_3dgs import PRESETS

from .models import NotebookQualityProfile, StrictModel


@dataclass(frozen=True)
class StaticNotebookProfile:
    id: NotebookQualityProfile
    label: str
    description: str
    legacy_preset: str
    n_iters: int
    max_gaussians: int
    selected_frame_budget: int
    resolution_long_edge_cap: int | None
    intended_gpu: str
    minimum_vram_gb: int
    foundation_default: bool
    run_eval_default: bool
    warnings: tuple[str, ...]


class NumericLimit(StrictModel):
    min: int
    max: int


class OverrideLimits(StrictModel):
    fixed_fps: NumericLimit
    n_iters: NumericLimit
    max_gaussians: NumericLimit


class NotebookProfileMetadata(StrictModel):
    id: NotebookQualityProfile
    label: str
    description: str
    n_iters: int
    max_gaussians: int
    selected_frame_budget: int
    resolution_long_edge_cap: int | None
    intended_gpu: str
    minimum_vram_gb: int
    foundation_default: bool
    run_eval_default: bool
    warnings: list[str]


class StaticNotebookPresetManifest(StrictModel):
    schema_version: Literal[1] = 1
    default_profile: NotebookQualityProfile
    profiles: list[NotebookProfileMetadata]
    override_limits: OverrideLimits


PROFILE_ROWS = (
    ("balanced_l4", "Balanced / L4", "Low-mid product run with L4 headroom", "balanced", 30_000, 250_000, 300, 1280, "NVIDIA L4 24 GB", 22, True, False, ()),
    ("high", "High", "Higher-detail run with more memory and time", "high", 50_000, 500_000, 450, 1920, "32 GB class", 30, True, False, ("Use a runtime with at least 30 GB VRAM",)),
    ("premium", "Premium", "Large-memory high-quality run", "premium", 100_000, 1_000_000, 600, 2560, "48 GB class", 46, True, False, ("Long-running 46+ GB profile",)),
    ("ultra", "Ultra", "Maximum-quality native-resolution run", "ultra", 120_000, 3_000_000, 800, None, "A100 80 GB class", 75, True, False, ("A100 80 GB class runtime required",)),
)


def _profile_from_row(row: tuple) -> StaticNotebookProfile:
    (
        profile_id,
        label,
        description,
        legacy_preset,
        approved_n_iters,
        approved_max_gaussians,
        selected_frame_budget,
        resolution_long_edge_cap,
        intended_gpu,
        minimum_vram_gb,
        foundation_default,
        run_eval_default,
        warnings,
    ) = row
    legacy_values = PRESETS[legacy_preset]
    training_values = (legacy_values["n_iters"], legacy_values["max_gaussians"])
    if training_values != (approved_n_iters, approved_max_gaussians):
        raise RuntimeError(f"Notebook profile {profile_id!r} no longer matches preset {legacy_preset!r}")
    return StaticNotebookProfile(
        id=NotebookQualityProfile(profile_id),
        label=label,
        description=description,
        legacy_preset=legacy_preset,
        n_iters=training_values[0],
        max_gaussians=training_values[1],
        selected_frame_budget=selected_frame_budget,
        resolution_long_edge_cap=resolution_long_edge_cap,
        intended_gpu=intended_gpu,
        minimum_vram_gb=minimum_vram_gb,
        foundation_default=foundation_default,
        run_eval_default=run_eval_default,
        warnings=warnings,
    )


_PROFILES = {profile.id: profile for profile in map(_profile_from_row, PROFILE_ROWS)}


def get_notebook_profile(profile: NotebookQualityProfile) -> StaticNotebookProfile:
    return _PROFILES[NotebookQualityProfile(profile)]


def get_static_preset_manifest() -> StaticNotebookPresetManifest:
    profiles = [
        NotebookProfileMetadata(
            id=profile.id,
            label=profile.label,
            description=profile.description,
            n_iters=profile.n_iters,
            max_gaussians=profile.max_gaussians,
            selected_frame_budget=profile.selected_frame_budget,
            resolution_long_edge_cap=profile.resolution_long_edge_cap,
            intended_gpu=profile.intended_gpu,
            minimum_vram_gb=profile.minimum_vram_gb,
            foundation_default=profile.foundation_default,
            run_eval_default=profile.run_eval_default,
            warnings=list(profile.warnings),
        )
        for profile in _PROFILES.values()
    ]
    return StaticNotebookPresetManifest(
        default_profile=NotebookQualityProfile.BALANCED_L4,
        profiles=profiles,
        override_limits=OverrideLimits(
            fixed_fps=NumericLimit(min=1, max=30),
            n_iters=NumericLimit(min=1_000, max=120_000),
            max_gaussians=NumericLimit(min=50_000, max=6_000_000),
        ),
    )
