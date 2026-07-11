from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .drive_paths import normalize_input_folder


class FrameSelectionMode(str, Enum):
    SMART = "smart"
    FIXED_FPS = "fixed_fps"


class NotebookQualityProfile(str, Enum):
    BALANCED_L4 = "balanced_l4"
    HIGH = "high"
    PREMIUM = "premium"
    ULTRA = "ultra"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameSelectionSpec(StrictModel):
    mode: FrameSelectionMode = FrameSelectionMode.SMART
    fixed_fps: Annotated[int, Field(ge=1, le=30)] = 4


class NotebookAdvancedConfig(StrictModel):
    run_eval: bool | None = None
    foundation: bool | None = None
    resolution_long_edge_cap: Annotated[int | None, Field(ge=320, le=3840)] = None
    lambda_ssim: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    lambda_lpips: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    lambda_depth: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    density_start_iter: Annotated[int | None, Field(ge=0, le=120_000)] = None
    density_end_iter: Annotated[int | None, Field(ge=0, le=120_000)] = None
    density_interval: Annotated[int | None, Field(ge=10, le=5_000)] = None
    densify_grad_threshold: Annotated[float | None, Field(gt=0.0, le=0.1)] = None
    prune_min_opacity: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    prune_max_scale: Annotated[float | None, Field(gt=0.0, le=1.0)] = None
    opacity_reset_interval: Annotated[int | None, Field(ge=0, le=120_000)] = None
    sh_degree: Annotated[int | None, Field(ge=0, le=3)] = None
    multires_schedule: list[tuple[int, int]] | None = None

    @field_validator("multires_schedule")
    @classmethod
    def validate_schedule(cls, value: list[tuple[int, int]] | None) -> list[tuple[int, int]] | None:
        if value is None:
            return value
        if not value or value[0][0] != 0:
            raise ValueError("multires_schedule must start at iteration 0")
        starts = [stage[0] for stage in value]
        if starts != sorted(set(starts)) or any(start < 0 or edge < 320 or edge > 3840 for start, edge in value):
            raise ValueError("multires_schedule stages must be unique, increasing, and 320..3840 px")
        return value


class NotebookQualitySpec(StrictModel):
    profile: NotebookQualityProfile = NotebookQualityProfile.BALANCED_L4
    n_iters: Annotated[int | None, Field(ge=1_000, le=120_000)] = None
    max_gaussians: Annotated[int | None, Field(ge=50_000, le=6_000_000)] = None
    advanced: NotebookAdvancedConfig = Field(default_factory=NotebookAdvancedConfig)

    @model_validator(mode="after")
    def validate_iteration_relationships(self) -> "NotebookQualitySpec":
        advanced = self.advanced
        if advanced.density_start_iter is not None and advanced.density_end_iter is not None and advanced.density_start_iter >= advanced.density_end_iter:
            raise ValueError("density_start_iter must be below density_end_iter")
        return self


class PublishSpec(StrictModel):
    replace_owned_result: bool = True


class StaticNotebookRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    frame_selection: FrameSelectionSpec = Field(default_factory=FrameSelectionSpec)
    quality: NotebookQualitySpec = Field(default_factory=NotebookQualitySpec)
    publish: PublishSpec = Field(default_factory=PublishSpec)

    @field_validator("input_folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        return normalize_input_folder(value)

    @model_validator(mode="after")
    def validate_effective_iteration_bounds(self) -> "StaticNotebookRunSpec":
        from .presets import get_notebook_profile

        effective_iters = self.quality.n_iters or get_notebook_profile(self.quality.profile).n_iters
        advanced = self.quality.advanced
        if advanced.density_start_iter is not None and advanced.density_start_iter >= effective_iters:
            raise ValueError("density_start_iter must be below effective n_iters")
        if advanced.density_end_iter is not None and advanced.density_end_iter > effective_iters:
            raise ValueError("density_end_iter cannot exceed effective n_iters")
        if advanced.multires_schedule and advanced.multires_schedule[-1][0] >= effective_iters:
            raise ValueError("multires_schedule stages must start before effective n_iters")
        return self


def parse_run_spec_json(data: str | bytes) -> StaticNotebookRunSpec:
    return StaticNotebookRunSpec.model_validate_json(data)
