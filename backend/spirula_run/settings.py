"""One description of a Spirula run, resolved the same way for Colab and for this PC.

``SpirulaSettings`` is what the UI edits; ``resolve`` turns it into concrete
command-line values. The notebook generator and the local runner both call
``resolve`` so "notebook üret" and "local başlat" train the same splat.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Presets follow the ROOM2 final-pipeline decision (2026-10-19): Spirula high,
# 3M splats / 50k steps, extreme SfM, appearance on, floater suppression off.
PRESETS: dict[str, dict] = {
    "baseline": dict(label="Hızlı", quality="medium", iterations=30000, cap_max=1_000_000, sfm_quality="high"),
    "quality": dict(label="Önerilen", quality="high", iterations=50000, cap_max=3_000_000, sfm_quality="extreme"),
    "ultra": dict(label="Maksimum detay", quality="ultra", iterations=80000, cap_max=10_000_000, sfm_quality="extreme"),
}

GEOMETRY_MODELS = ("moge2-vits", "moge2-vitb", "moge2-vitl")
GEOMETRY_MAX_SIZE = 1064     # MoGe-2's own pipeline value
GEOMETRY_NUM_TOKENS = 3600   # documented MoGe-2 maximum
CHECKPOINT_INTERVAL = 10000
MIN_REGISTERED_COVERAGE = 0.80
MIN_DOMINANT_COMPONENT_SHARE = 0.80


class SpirulaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["baseline", "quality", "ultra"] = "quality"
    iterations: int | None = Field(default=None, ge=1000, le=200_000)
    cap_max: int | None = Field(default=None, ge=50_000, le=100_000_000)
    fps: float = Field(default=4.0, gt=0, le=30)
    sfm_quality: Literal["high", "extreme"] | None = None
    geometry_model: Literal["moge2-vits", "moge2-vitb", "moge2-vitl"] = "moge2-vitb"
    generate_depth: bool = True
    appearance: bool = True
    floater_suppression: Literal["off", "mild", "strong"] = "off"
    resolution_divisor: Literal[1, 2, 4] = 1
    allow_partial: bool = False


@dataclass(frozen=True)
class ResolvedRun:
    quality: str
    iterations: int
    cap_max: int
    fps: float
    sfm_quality: str
    geometry_model: str
    generate_depth: bool
    appearance: bool
    floater_suppression: str
    resolution_divisor: int
    allow_partial: bool

    def as_dict(self) -> dict:
        return asdict(self)


def resolve(settings: SpirulaSettings) -> ResolvedRun:
    p = PRESETS[settings.preset]
    return ResolvedRun(
        quality=p["quality"],
        iterations=settings.iterations or p["iterations"],
        cap_max=settings.cap_max or p["cap_max"],
        fps=settings.fps,
        sfm_quality=settings.sfm_quality or p["sfm_quality"],
        geometry_model=settings.geometry_model,
        generate_depth=settings.generate_depth,
        appearance=settings.appearance,
        floater_suppression=settings.floater_suppression,
        resolution_divisor=settings.resolution_divisor,
        allow_partial=settings.allow_partial,
    )


def frame_skip(source_fps: float, target_fps: float) -> int:
    return max(1, round(source_fps / target_fps))


def extract_args(binary: str, video: str, out: str, skip: int, device: str = "auto") -> list[str]:
    return [binary, "sam", "extract", video, "--out", out, "--skip", str(skip), "--quality", "101",
            "--adaptive", "--device", device, "--lang", "en"]


def sfm_args(binary: str, images: str, dataset: str, run: ResolvedRun, capture_kind: str,
             device: str = "auto") -> list[str]:
    args = [binary, "sfm", "auto", images, "--output", dataset, "--quality", run.sfm_quality,
            "--data-type", capture_kind, "--pairs", "auto", "--device", device, "--lang", "en"]
    if capture_kind == "video":
        args += ["--sequence", ".", "--prefilter-sequential"]
    return args


def geometry_args(binary: str, dataset: str, run: ResolvedRun, device: str = "auto") -> list[str]:
    args = [binary, "geometry", dataset, "--model", run.geometry_model, "--max-size", str(GEOMETRY_MAX_SIZE),
            "--num-tokens", str(GEOMETRY_NUM_TOKENS), "--normal-format", "png", "--overwrite",
            "--device", device, "--lang", "en"]
    if run.generate_depth:
        args += ["--depth", "--depth-units", "relative"]
    return args


def train_args(binary: str, dataset: str, recon_dir: str, out_prefix: str, run: ResolvedRun,
               device: str = "auto", viewer_port: int | None = None) -> list[str]:
    """``spirula train``; ``viewer_port`` turns on Spirula's web viewer (local live preview)."""
    args = [binary, "train", "3dgs", "--data", dataset, "--colmap-recon-dir", recon_dir,
            "--output-dir-prefix", out_prefix, "--output-dir-name", "training",
            "--quality", run.quality, "--num-iterations", str(run.iterations),
            "--max-steps", str(run.iterations), "--cap-max", str(run.cap_max),
            "--steps-per-save", str(CHECKPOINT_INTERVAL), "--save-full-checkpoint", "true",
            "--save-only-latest-checkpoint", "true",
            "--train-resolution-divisor", str(run.resolution_divisor), "--sh-degree", "3",
            "--cache-images", "disk",
            "--load-normals", "true", "--normal-supervision-weight", "0.01",
            "--load-depths", "true" if run.generate_depth else "false",
            "--depth-supervision-weight", "0.01" if run.generate_depth else "0",
            "--device", device, "--lang", "en"]
    # Preset defaults are left implicit so the final-pipeline command stays unchanged.
    if not run.appearance:  # per-photo colour correction (bilateral grid + PPISP), not exported to the PLY
        args += ["--use-bilateral-grid", "false", "--use-ppisp", "false"]
    if run.floater_suppression != "off":
        args += ["--floater-suppression", run.floater_suppression]
    if viewer_port is None:
        args += ["--disable-viewer", "true", "--keep-viewer-alive", "false"]
    else:
        args += ["--disable-viewer", "false", "--viewer-port", str(viewer_port), "--keep-viewer-alive", "false"]
    return args
