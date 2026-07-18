from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from backend.notebooks.drive_paths import normalize_input_folder
from backend.notebooks.models import (
    FrameSelectionMode,
    FrameSelectionSpec,
    NotebookAdvancedConfig,
    NotebookQualityProfile,
    NotebookQualitySpec,
    PublishSpec,
    StaticNotebookRunSpec,
    StrictModel,
)
from backend.static_pipeline.contracts import (
    ColmapAttempt,
    GateDecision,
    ReconstructionBundle,
    SelectionManifest,
)

if TYPE_CHECKING:
    from backend.notebooks.training_config import ResolvedStaticConfig


RESULT_SUFFIX = "_learned_test_result"
DIAGNOSTICS_SUFFIX = "_learned_test_diagnostics"
CACHE_SUFFIX = "_learned_test_cache"
GENERATOR_ID = "4dgs-studio.learned-quality-a100"


class LearnedPublishSpec(StrictModel):
    replace_owned_result: bool = True


class LearnedQualityRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    publish: LearnedPublishSpec = Field(default_factory=LearnedPublishSpec)

    @field_validator("input_folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise ValueError("Input folder cannot contain control characters")
        canonical = normalize_input_folder(value)
        if canonical.endswith((RESULT_SUFFIX, DIAGNOSTICS_SUFFIX, CACHE_SUFFIX)):
            raise ValueError("Choose the input folder, not an experiment output")
        return canonical


def parse_learned_spec_json(data: str | bytes) -> LearnedQualityRunSpec:
    return LearnedQualityRunSpec.model_validate_json(data)


def to_static_run_spec(spec: LearnedQualityRunSpec) -> StaticNotebookRunSpec:
    return StaticNotebookRunSpec(
        input_folder=spec.input_folder,
        frame_selection=FrameSelectionSpec(mode=FrameSelectionMode.SMART),
        quality=NotebookQualitySpec(
            profile=NotebookQualityProfile.ULTRA,
            n_iters=120_000,
            max_gaussians=6_000_000,
            advanced=NotebookAdvancedConfig(
                density_start_iter=500,
                density_end_iter=80_000,
                density_interval=100,
                densify_grad_threshold=1e-4,
            ),
        ),
        publish=PublishSpec(
            replace_owned_result=spec.publish.replace_owned_result,
        ),
    )


def derive_learned_result_path(input_folder: Path) -> Path:
    return input_folder.with_name(f"{input_folder.name}{RESULT_SUFFIX}")


def derive_learned_diagnostics_root(input_folder: Path) -> Path:
    return input_folder.with_name(f"{input_folder.name}{DIAGNOSTICS_SUFFIX}")


def derive_learned_cache_root(input_folder: Path) -> Path:
    return input_folder.with_name(f"{input_folder.name}{CACHE_SUFFIX}")


@dataclass(frozen=True)
class ModelRef:
    repo_id: str
    revision: str
    code_commit: str | None
    license_id: str


@dataclass(frozen=True)
class FrameArtifact:
    image_name: str
    frame_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class StageRecord:
    stage_id: str
    status: Literal["accepted", "rejected", "fallback", "failed", "skipped"]
    vram_before_bytes: int | None = None
    vram_after_bytes: int | None = None
    peak_vram_bytes: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedArtifacts:
    da3: object | None = None
    base_evidence: object | None = None
    semantic: object | None = None
    flow: object | None = None
    masks: object | None = None
    photometric: object | None = None
    depth: object | None = None
    dense_seeds: object | None = None
    model_manifest_path: Path | None = None
    stage_records: tuple[StageRecord, ...] = ()


@dataclass(frozen=True)
class GeometryCandidateReport:
    candidate_id: Literal["classical", "learned_hybrid"]
    attempt: ColmapAttempt
    decision: GateDecision
    model_dir: Path
    selected_manifest: SelectionManifest
    frame_set_digest: str
    covered_endpoint_count: int


@dataclass(frozen=True)
class LearnedReconstructionOutput:
    bundle: ReconstructionBundle
    frames_dir: Path
    artifacts: LearnedArtifacts
    geometry_candidates: tuple[GeometryCandidateReport, ...]

    @property
    def decision(self) -> GateDecision:
        return self.bundle.decision

    @property
    def selected_manifest(self) -> SelectionManifest:
        return self.bundle.selected_manifest

    @property
    def accepted_model_dir(self) -> Path:
        return self.bundle.accepted_model_dir

    @property
    def attempts(self) -> tuple[ColmapAttempt, ...]:
        return self.bundle.attempts

    @property
    def decisions(self) -> tuple[GateDecision, ...]:
        return self.bundle.decisions


@dataclass(frozen=True)
class LearnedTrainingOutput:
    raw_ply_path: Path
    status: Mapping[str, object]
    resolved_config: ResolvedStaticConfig
    run_manifest_path: Path
    density_history_path: Path
    final_gaussian_count: int
    training_rgb_digest: str
    fallbacks: tuple[str, ...] = ()
