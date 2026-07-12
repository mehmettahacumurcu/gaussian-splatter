from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PublishReceipt:
    run_id: str
    final_path: Path
    artifacts: tuple[ArtifactRecord, ...]
    manifest_sha256: str
    replaced_previous: bool


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    schema_version: int
    root: Path
    kind: Literal["video", "photo_set"]
    media_files: tuple[SourceFile, ...]
    all_files: tuple[SourceFile, ...]
    digest: str

    def to_manifest_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value


@dataclass(frozen=True)
class FrameMetrics:
    sharpness: float
    exposure_score: float
    duplicate_similarity: float
    overlap_score: float


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    source_relative_path: str
    source_index: int | None
    source_pts: int | None
    timestamp_s: float | None
    output_name: str
    sha256: str
    selected: bool
    metrics: FrameMetrics | None
    selection_score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    mode: Literal["smart", "fixed_fps"]
    frame_budget: int
    resolution_long_edge_cap: int | None
    fixed_fps: int = 4
    candidate_fps: int = 12
    candidate_long_edge: int = 320
    version: str = "selection-v1"


@dataclass(frozen=True)
class SelectionManifest:
    schema_version: int
    source_digest: str
    effective_mode: Literal["smart", "fixed_fps", "photo_set_all"]
    policy: SelectionPolicy
    frames: tuple[FrameRecord, ...]
    image_set_digest: str
    reconstruction_guardrail: str | None = None

    @property
    def selected_frames(self) -> tuple[FrameRecord, ...]:
        return tuple(frame for frame in self.frames if frame.selected)


@dataclass(frozen=True)
class UncoveredInterval:
    start_s: float
    end_s: float
    missing_frame_ids: tuple[str, ...]
    coverage_unit: Literal["seconds", "photo_order"] = "seconds"
    kind: Literal["start", "interior", "end", "all"] = "interior"
    left_boundary_frame_id: str | None = None
    right_boundary_frame_id: str | None = None


@dataclass(frozen=True)
class ColmapPolicy:
    camera_model: str
    matcher: Literal["sequential", "exhaustive"]
    sequential_overlap: int
    use_gpu: bool
    version: str = "colmap-policy-v1"


@dataclass(frozen=True)
class ColmapAttempt:
    root: Path
    database_path: Path
    model_dirs: tuple[Path, ...]
    colmap_version: str
    fingerprint: str


@dataclass(frozen=True)
class ModelMetrics:
    model_dir: Path
    registered_names: frozenset[str]
    registered_count: int
    registered_ratio: float
    registered_share: float
    temporal_coverage_s: float
    max_interior_gap_s: float
    start_gap_s: float
    end_gap_s: float
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    median_track_length: float
    sparse_point_count: int
    valid_names_intrinsics_and_poses: bool
    coverage_unit: Literal["seconds", "photo_order"] = "seconds"


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    dominant: ModelMetrics
    failures: tuple[str, ...]
    uncovered_intervals: tuple[UncoveredInterval, ...]
    retry_recommended: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconstructionBundle:
    selected_manifest: SelectionManifest
    accepted_model_dir: Path
    decision: GateDecision
    attempts: tuple[ColmapAttempt, ...]
    decisions: tuple[GateDecision, ...] = ()


def _require_finite_real(
    name: str,
    value: Real,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        below = numeric < minimum if minimum_inclusive else numeric <= minimum
        if below:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        above = numeric > maximum if maximum_inclusive else numeric >= maximum
        if above:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}")
    return numeric


@dataclass(frozen=True)
class PolishPolicy:
    max_removed_fraction: float = 0.15
    max_opacity_mass_loss: float = 0.05
    max_mean_psnr_drop_db: float = 0.25
    max_mean_ssim_drop: float = 0.005
    max_single_view_psnr_drop_db: float = 1.0
    min_opacity: float = 0.005
    max_relative_scale: float = 0.03
    max_anisotropy: float = 30.0
    crop_margin_fraction: float = 0.05
    version: str = "polish-v1"

    def __post_init__(self) -> None:
        _require_finite_real(
            "max_removed_fraction",
            self.max_removed_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_real(
            "max_opacity_mass_loss",
            self.max_opacity_mass_loss,
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_real(
            "max_mean_psnr_drop_db",
            self.max_mean_psnr_drop_db,
            minimum=0.0,
        )
        _require_finite_real(
            "max_mean_ssim_drop",
            self.max_mean_ssim_drop,
            minimum=0.0,
            maximum=2.0,
        )
        _require_finite_real(
            "max_single_view_psnr_drop_db",
            self.max_single_view_psnr_drop_db,
            minimum=0.0,
        )
        _require_finite_real(
            "min_opacity",
            self.min_opacity,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        _require_finite_real(
            "max_relative_scale",
            self.max_relative_scale,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite_real(
            "max_anisotropy",
            self.max_anisotropy,
            minimum=1.0,
        )
        _require_finite_real(
            "crop_margin_fraction",
            self.crop_margin_fraction,
            minimum=0.0,
            maximum=0.5,
            maximum_inclusive=False,
        )
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")


@dataclass(frozen=True)
class RenderViewMetric:
    image_name: str
    psnr_db: float
    ssim: float

    def __post_init__(self) -> None:
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")
        if isinstance(self.psnr_db, bool) or not isinstance(self.psnr_db, Real):
            raise ValueError("psnr_db must be a real number")
        if isinstance(self.ssim, bool) or not isinstance(self.ssim, Real):
            raise ValueError("ssim must be a real number")


@dataclass(frozen=True)
class PolishReport:
    accepted: bool
    raw_path: Path
    candidate_path: Path | None
    selected_path: Path
    original_count: int
    kept_count: int
    opacity_mass_loss: float
    render_metrics: dict[str, object]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        if not isinstance(self.raw_path, Path):
            raise TypeError("raw_path must be a Path")
        if self.candidate_path is not None and not isinstance(
            self.candidate_path, Path
        ):
            raise TypeError("candidate_path must be a Path or None")
        if not isinstance(self.selected_path, Path):
            raise TypeError("selected_path must be a Path")
        if isinstance(self.original_count, bool) or not isinstance(
            self.original_count, Integral
        ):
            raise TypeError("original_count must be an integer")
        if isinstance(self.kept_count, bool) or not isinstance(
            self.kept_count, Integral
        ):
            raise TypeError("kept_count must be an integer")
        if self.original_count <= 0:
            raise ValueError("original_count must be positive")
        if not 0 <= self.kept_count <= self.original_count:
            raise ValueError("kept_count must be between zero and original_count")
        _require_finite_real(
            "opacity_mass_loss",
            self.opacity_mass_loss,
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(self.render_metrics, dict):
            raise TypeError("render_metrics must be a dict")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.reasons
        ):
            raise TypeError("reasons must be a tuple of non-empty strings")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must not contain duplicates")
        if self.accepted:
            if self.candidate_path is None or self.selected_path != self.candidate_path:
                raise ValueError("an accepted report must select its candidate")
            if self.reasons:
                raise ValueError("an accepted report cannot contain rejection reasons")
        else:
            if self.selected_path != self.raw_path:
                raise ValueError("a rejected report must select its raw PLY")
            if not self.reasons:
                raise ValueError("a rejected report must contain a reason")
