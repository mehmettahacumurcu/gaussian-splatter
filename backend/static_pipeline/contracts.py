from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


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
