from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from backend.notebooks.presets import get_notebook_profile

if TYPE_CHECKING:
    from backend.notebooks.models import StaticNotebookRunSpec

    from .progress import StageReporter


class RuntimePreflightError(RuntimeError):
    """The selected profile is unsafe on the current notebook runtime."""


@dataclass(frozen=True)
class NotebookRuntimePaths:
    drive_root: Path
    work_root: Path


@dataclass(frozen=True)
class HardwareInfo:
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    disk_free_gb: float
    colmap_gpu_sift: bool | None = None
    colmap_available: bool = True
    ffmpeg_available: bool = True
    host_ram_gb: float | None = None


@dataclass(frozen=True)
class NotebookRunResult:
    run_id: str
    final_path: Path
    local_bundle: Path
    quality_report_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SelectionOutput:
    inventory: object
    manifest: object
    frames_dir: Path
    source_manifest_path: Path


@dataclass(frozen=True)
class MetadataPreviewOutput:
    metadata: Mapping[str, object]
    metadata_path: Path
    preview_path: Path
    world_dir: Path
    preview: Mapping[str, object]


@dataclass(frozen=True)
class ReconstructionOutput:
    bundle: object
    frames_dir: Path

    @property
    def decision(self) -> object:
        return self.bundle.decision

    @property
    def selected_manifest(self) -> object:
        return self.bundle.selected_manifest

    @property
    def accepted_model_dir(self) -> Path:
        return self.bundle.accepted_model_dir

    @property
    def attempts(self) -> object:
        return self.bundle.attempts

    @property
    def decisions(self) -> object:
        return self.bundle.decisions


@dataclass(frozen=True)
class PretrainingRestore:
    selection: object
    reconstruction: object


@dataclass(frozen=True)
class RunnerServices:
    discover_source: Callable[..., object]
    copy_input: Callable[..., object]
    select_frames: Callable[..., object]
    reconstruct: Callable[..., object]
    train: Callable[..., object]
    polish: Callable[..., object]
    build_metadata_preview: Callable[..., object]
    validate_bundle: Callable[..., object]
    publish_result: Callable[..., object]
    publish_diagnostics: Callable[..., object]
    inspect_hardware: Callable[..., HardwareInfo]
    preflight: Callable[..., object]
    assemble_reports: Callable[..., Path]
    resolve_input: Callable[..., Path | None] | None = None
    restore_pretraining: Callable[..., PretrainingRestore | None] | None = None
    save_pretraining: Callable[..., object] | None = None
    restore_selection: Callable[..., object | None] | None = None
    save_selection: Callable[..., object] | None = None
    validate_reconstruction: Callable[[object], None] | None = None


def runtime_paths_from_env() -> NotebookRuntimePaths:
    return NotebookRuntimePaths(
        drive_root=Path(
            os.environ.get("STATIC_NOTEBOOK_DRIVE_ROOT", "/content/drive/MyDrive")
        ).resolve(),
        work_root=Path(
            os.environ.get("STATIC_NOTEBOOK_WORK_ROOT", "/content/4dgs-runs")
        ).resolve(),
    )


def preflight_runtime(
    spec: StaticNotebookRunSpec,
    hardware: HardwareInfo,
    *,
    input_size_gb: float,
) -> None:
    if (
        not isinstance(input_size_gb, (int, float))
        or isinstance(input_size_gb, bool)
        or not math.isfinite(float(input_size_gb))
        or input_size_gb < 0
    ):
        raise ValueError("input_size_gb must be a finite non-negative number")
    profile = get_notebook_profile(spec.quality.profile)
    if not hardware.cuda_available:
        raise RuntimePreflightError(
            f"{profile.label} requires a CUDA-capable NVIDIA GPU"
        )
    if not hardware.colmap_available:
        raise RuntimePreflightError("COLMAP is not installed in this runtime")
    if not hardware.ffmpeg_available:
        raise RuntimePreflightError("FFmpeg is not installed in this runtime")
    if (
        not math.isfinite(hardware.vram_gb)
        or hardware.vram_gb < profile.minimum_vram_gb
    ):
        raise RuntimePreflightError(
            f"{profile.label} requires at least {profile.minimum_vram_gb} GB VRAM; "
            f"detected {hardware.vram_gb:.1f} GB on {hardware.gpu_name}"
        )
    required_disk_gb = max(25.0, float(input_size_gb) * 4.0 + 15.0)
    if (
        not math.isfinite(hardware.disk_free_gb)
        or hardware.disk_free_gb < required_disk_gb
    ):
        raise RuntimePreflightError(
            f"Insufficient working disk for {profile.label}: "
            f"need {required_disk_gb:.1f} GB, detected {hardware.disk_free_gb:.1f} GB"
        )
    summary = {
        "event": "static_notebook_preflight",
        "gpu": hardware.gpu_name,
        "vram_gb": hardware.vram_gb,
        "disk_free_gb": hardware.disk_free_gb,
        "profile": profile.id.value,
        "iterations": spec.quality.n_iters or profile.n_iters,
        "max_gaussians": spec.quality.max_gaussians or profile.max_gaussians,
        "frame_budget": profile.selected_frame_budget,
        "resolution_long_edge_cap": (
            spec.quality.advanced.resolution_long_edge_cap
            if spec.quality.advanced.resolution_long_edge_cap is not None
            else profile.resolution_long_edge_cap
        ),
        "colmap_gpu_sift": hardware.colmap_gpu_sift,
        "colmap_matcher_mode": (
            "gpu_exhaustive" if hardware.colmap_gpu_sift else "cpu_bounded"
        ),
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


def _inspect_hardware(paths: NotebookRuntimePaths) -> HardwareInfo:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        first_line = completed.stdout.splitlines()[0]
        gpu_name, memory_mib = (part.strip() for part in first_line.rsplit(",", 1))
        vram_gb = float(memory_mib) / 1024.0
        cuda_available = True
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        gpu_name = "No NVIDIA GPU detected"
        vram_gb = 0.0
        cuda_available = False
    paths.work_root.mkdir(parents=True, exist_ok=True)
    colmap_gpu_sift: bool | None = None
    colmap_executable = shutil.which("colmap")
    ffmpeg_executable = shutil.which("ffmpeg")
    if cuda_available and colmap_executable is not None:
        from .colmap import probe_colmap_gpu_support

        colmap_gpu_sift = probe_colmap_gpu_support(
            colmap_executable,
            paths.work_root / f".colmap-gpu-probe-{uuid.uuid4().hex}",
        )
    disk_free_gb = shutil.disk_usage(paths.work_root).free / (1024**3)
    host_ram_gb: float | None = None
    try:
        host_ram_gb = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return HardwareInfo(
        gpu_name,
        vram_gb,
        cuda_available,
        disk_free_gb,
        colmap_gpu_sift,
        colmap_executable is not None,
        ffmpeg_executable is not None,
        host_ram_gb,
    )


def _resolve_drive_input(
    spec: StaticNotebookRunSpec,
    runtime_paths: NotebookRuntimePaths,
    resolver: Callable[..., Path | None] | None,
) -> Path:
    drive_root = Path(runtime_paths.drive_root).resolve(strict=False)
    if resolver is None:
        candidate = drive_root.joinpath(*spec.input_folder.split("/"))
    else:
        resolved_by_service = resolver(spec=spec, runtime_paths=runtime_paths)
        candidate = (
            Path(resolved_by_service)
            if resolved_by_service is not None
            else drive_root.joinpath(*spec.input_folder.split("/"))
        )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(drive_root)
    except ValueError as exc:
        raise ValueError(
            "resolved input folder escapes the mounted Drive root"
        ) from exc
    if resolver is None and not resolved.is_dir():
        raise ValueError(f"Drive input folder does not exist: {resolved}")
    return resolved


def _inventory_signature(inventory: object) -> tuple[object, tuple[object, ...]]:
    digest = getattr(inventory, "digest", None)
    files = tuple(
        (
            getattr(item, "relative_path", None),
            getattr(item, "size_bytes", None),
            getattr(item, "sha256", None),
        )
        for item in getattr(inventory, "all_files", ())
    )
    return digest, files


def _input_size_gb(inventory: object) -> float:
    total = sum(int(getattr(item, "size_bytes")) for item in inventory.all_files)
    return total / (1024**3)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return _jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return ((completed.stdout or completed.stderr).splitlines() or ["unknown"])[0]
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _repository_commit() -> str:
    return _command_version(["git", "rev-parse", "HEAD"])


def _model_metrics_payload(metrics: object) -> dict[str, object]:
    payload = dict(_jsonable(metrics))
    model_dir = getattr(metrics, "model_dir", None)
    if model_dir is not None:
        payload["model_dir"] = Path(model_dir).name
    return payload


def _decision_payload(decision: object) -> dict[str, object]:
    return {
        "passed": decision.passed,
        "dominant": _model_metrics_payload(decision.dominant),
        "failures": decision.failures,
        "uncovered_intervals": decision.uncovered_intervals,
        "retry_recommended": getattr(decision, "retry_recommended", False),
        "warnings": getattr(decision, "warnings", ()),
    }


def _selection_adapter(
    inventory: object,
    *,
    spec: StaticNotebookRunSpec,
    output_root: Path,
) -> SelectionOutput:
    from .contracts import SelectionPolicy
    from .selection import bound_selection_for_reconstruction, select_frames

    profile = get_notebook_profile(spec.quality.profile)
    policy = SelectionPolicy(
        mode=spec.frame_selection.mode.value,
        frame_budget=profile.selected_frame_budget,
        resolution_long_edge_cap=(
            spec.quality.advanced.resolution_long_edge_cap
            if spec.quality.advanced.resolution_long_edge_cap is not None
            else profile.resolution_long_edge_cap
        ),
        fixed_fps=spec.frame_selection.fixed_fps,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    selected_dir = output_root / "selected"
    original = select_frames(inventory, selected_dir, policy)
    manifest = original
    frames_dir = selected_dir
    if len(original.selected_frames) > min(profile.selected_frame_budget, 800):
        frames_dir = output_root / "bounded"
        manifest = bound_selection_for_reconstruction(
            inventory,
            original,
            frames_dir,
            max_frames=min(profile.selected_frame_budget, 800),
        )
    return SelectionOutput(
        inventory=inventory,
        manifest=manifest,
        frames_dir=frames_dir,
        source_manifest_path=selected_dir / "selection_manifest.json",
    )


def _reconstruction_adapter(
    selection: SelectionOutput,
    *,
    spec: StaticNotebookRunSpec,
    hardware: HardwareInfo,
    output_root: Path,
) -> ReconstructionOutput:
    del spec
    from .colmap import run_colmap_attempt
    from .reconstruction import reconstruct_with_gate
    from .selection import plan_backfill

    frames_dir = selection.frames_dir

    def run_attempt(manifest, attempt_dir, policy):
        del manifest
        return run_colmap_attempt(frames_dir, attempt_dir, policy)

    def materialize_backfill(manifest, uncovered):
        nonlocal frames_dir
        frames_dir = output_root / "backfilled"
        return plan_backfill(
            selection.inventory,
            manifest,
            uncovered,
            frames_dir,
        )

    bundle = reconstruct_with_gate(
        selection.manifest,
        use_gpu=hardware.colmap_gpu_sift is True,
        attempt_root=output_root,
        run_attempt=run_attempt,
        materialize_backfill=materialize_backfill,
    )
    return ReconstructionOutput(bundle=bundle, frames_dir=frames_dir)


def _training_adapter(
    reconstruction: ReconstructionOutput,
    *,
    spec: StaticNotebookRunSpec,
    selection: SelectionOutput,
    run_id: str,
    output_root: Path,
) -> object:
    from .training import PreparedTrainingInput, run_validated_training

    prepared = PreparedTrainingInput(
        run_id=run_id,
        data_root=output_root,
        scene_name="scene",
        frames_dir=reconstruction.frames_dir,
        reconstruction=reconstruction.bundle,
        source_digest=selection.inventory.digest,
        selection_digest=reconstruction.bundle.selected_manifest.image_set_digest,
    )
    return run_validated_training(prepared, spec)


def _polish_adapter(
    training: object,
    *,
    reconstruction: ReconstructionOutput,
    spec: StaticNotebookRunSpec,
    selection: SelectionOutput,
    output_root: Path,
) -> object:
    del spec, selection
    from backend.preprocess.frame_alignment import join_registered_frames
    from backend.preprocess.parse_colmap import parse_cameras_from_model

    from .polish import polish_static_ply

    output_root.mkdir(parents=True, exist_ok=False)
    cameras = parse_cameras_from_model(reconstruction.accepted_model_dir)
    registered = join_registered_frames(
        reconstruction.frames_dir,
        cameras,
        manifest=reconstruction.selected_manifest,
    )
    return polish_static_ply(
        training.raw_ply_path,
        output_root / "candidate.ply",
        reconstruction.bundle,
        registered,
    )


def _metadata_adapter(
    polish: object,
    *,
    reconstruction: ReconstructionOutput,
    spec: StaticNotebookRunSpec,
    output_root: Path,
) -> MetadataPreviewOutput:
    del spec
    from .scene_metadata import (
        build_scene_metadata,
        render_preview,
        write_scene_metadata,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    metadata = build_scene_metadata(polish.selected_path, reconstruction.bundle)
    metadata_path = output_root / "scene_metadata.json"
    world_dir = output_root / "world"
    write_scene_metadata(metadata_path, metadata, world_dir=world_dir)
    preview_path = output_root / "preview.png"
    preview = render_preview(
        polish.selected_path,
        reconstruction.bundle,
        metadata,
        preview_path,
    )
    return MetadataPreviewOutput(
        metadata=metadata,
        metadata_path=metadata_path,
        preview_path=preview_path,
        world_dir=world_dir,
        preview=preview,
    )


def _assemble_reports_adapter(
    *,
    spec: StaticNotebookRunSpec,
    run_id: str,
    source_inventory: object,
    selection: SelectionOutput,
    reconstruction: ReconstructionOutput,
    training: object,
    polish: object,
    metadata_preview: MetadataPreviewOutput,
    hardware: HardwareInfo,
    timings: Mapping[str, float],
    bundle_root: Path,
) -> Path:
    from .publish import GENERATOR_ID, MANIFEST_SCHEMA_VERSION, inventory_bundle

    bundle_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(polish.selected_path, bundle_root / "splat.ply")
    shutil.copy2(metadata_preview.metadata_path, bundle_root / "scene_metadata.json")
    shutil.copy2(metadata_preview.preview_path, bundle_root / "preview.png")
    final_selection_manifest = reconstruction.frames_dir / "selection_manifest.json"
    if not final_selection_manifest.is_file():
        raise FileNotFoundError("final selection manifest is missing")
    shutil.copy2(final_selection_manifest, bundle_root / "selection_manifest.json")
    if selection.source_manifest_path.resolve() != final_selection_manifest.resolve():
        shutil.copy2(
            selection.source_manifest_path,
            bundle_root / "source_selection_manifest.json",
        )
    if metadata_preview.world_dir.is_dir():
        shutil.copytree(metadata_preview.world_dir, bundle_root / "world")
    logs = bundle_root / "logs"
    logs.mkdir()
    (logs / "pipeline.log").write_text(
        f"run_id={run_id}\nstatus=success\n",
        encoding="utf-8",
    )

    manifest = reconstruction.selected_manifest
    decisions = tuple(reconstruction.decisions) or (reconstruction.decision,)
    try:
        from .reconstruction import measure_models

        component_metrics = [
            [
                _model_metrics_payload(model)
                for model in measure_models(attempt.model_dirs, manifest)
            ]
            for attempt in reconstruction.attempts
        ]
    except (AttributeError, OSError, ValueError):
        component_metrics = [
            [_model_metrics_payload(decision.dominant)] for decision in decisions
        ]
    registration_warning = (
        "registered_ratio_below_95_percent"
        if reconstruction.decision.dominant.registered_ratio < 0.95
        else None
    )
    rejected_frames = [frame for frame in manifest.frames if not frame.selected]
    quality_report = {
        "schema_version": 1,
        "status": "passed",
        "selection": {
            "mode": manifest.effective_mode,
            "image_set_digest": manifest.image_set_digest,
            "selected_count": len(manifest.selected_frames),
            "rejected_count": len(rejected_frames),
            "frames": manifest.frames,
            "reconstruction_guardrail": manifest.reconstruction_guardrail,
        },
        "colmap": {
            "components_by_attempt": component_metrics,
            "decisions": [_decision_payload(decision) for decision in decisions],
            "accepted_model": reconstruction.accepted_model_dir.name,
            "registration_target_warning": registration_warning,
        },
        "gate": _decision_payload(reconstruction.decision),
        "evaluation": {
            "label": (
                "holdout_evaluation"
                if training.resolved_config.run_eval
                else "training_view_checks"
            ),
            "metrics": polish.render_metrics,
        },
        "training": training.status,
        "polish": {
            "accepted": polish.accepted,
            "original_count": getattr(polish, "original_count", None),
            "kept_count": getattr(polish, "kept_count", None),
            "opacity_mass_loss": getattr(polish, "opacity_mass_loss", None),
            "render_metrics": polish.render_metrics,
            "reasons": getattr(polish, "reasons", ()),
            "selected_artifact": "splat.ply",
        },
        "orientation": metadata_preview.metadata.get("orientation"),
        "preview": {
            "artifact": "preview.png",
            "fallback_used": metadata_preview.preview.get("fallback_used"),
            "warning": metadata_preview.preview.get("warning"),
        },
        "orbit": {"created": False, "warning": None},
        "warnings": [
            warning
            for warning in (
                registration_warning,
                metadata_preview.preview.get("warning"),
            )
            if warning
        ],
        "capture_limitations": [
            "Unobserved surfaces cannot be recovered from the supplied capture."
        ],
    }
    _write_json(bundle_root / "quality_report.json", quality_report)

    artifacts = inventory_bundle(bundle_root)
    tool_versions = {
        "python": os.sys.version.split()[0],
        "cuda": _command_version(
            [
                os.sys.executable,
                "-c",
                "import torch; print(torch.version.cuda or 'unavailable')",
            ]
        ),
        "colmap": _command_version(["colmap", "--version"]),
        "ffmpeg": _command_version(["ffmpeg", "-version"]),
        "pytorch": _command_version(
            [os.sys.executable, "-c", "import torch; print(torch.__version__)"]
        ),
        "gsplat": _command_version(
            [
                os.sys.executable,
                "-c",
                "import gsplat; print(getattr(gsplat, '__version__', 'unknown'))",
            ]
        ),
    }
    run_manifest = {
        "generator_id": GENERATOR_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "success",
        "run_spec": spec.model_dump(mode="json"),
        "resolved_config": training.resolved_config,
        "selected_profile": spec.quality.profile.value,
        "source": {
            "digest": source_inventory.digest,
            "kind": getattr(source_inventory, "kind", None),
            "files": getattr(source_inventory, "all_files", ()),
            "input_folder": spec.input_folder,
        },
        "selection_digest": manifest.image_set_digest,
        "accepted_model": reconstruction.accepted_model_dir.name,
        "repository_commit": _repository_commit(),
        "tool_versions": tool_versions,
        "timing": dict(timings),
        "hardware": hardware,
        "artifacts": artifacts,
    }
    _write_json(bundle_root / "run_manifest.json", run_manifest)
    return bundle_root


def _failure_diagnostic_files(
    run_root: Path,
    error: Exception,
    selection: object | None,
) -> dict[str, Path]:
    diagnostic_root = run_root / "diagnostics"
    logs = diagnostic_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pipeline.log"
    log_path.write_text(
        f"error_type={type(error).__name__}\nerror={error}\n",
        encoding="utf-8",
    )
    decision = getattr(error, "decision", None)
    quality_path = _write_json(
        diagnostic_root / "quality_report.json",
        {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "gate": decision,
            "components": getattr(error, "decisions", ()),
            "durable_milestone_kind": getattr(error, "durable_milestone_kind", None),
            "durable_milestone_fingerprint": getattr(
                error, "durable_milestone_fingerprint", None
            ),
            "next_stage_id": getattr(error, "next_stage_id", None),
        },
    )
    uncovered_path = _write_json(
        diagnostic_root / "uncovered_intervals.json",
        {
            "uncovered_intervals": (
                getattr(decision, "uncovered_intervals", ())
                if decision is not None
                else ()
            )
        },
    )
    files = {
        "logs/pipeline.log": log_path,
        "quality_report.json": quality_path,
        "uncovered_intervals.json": uncovered_path,
    }
    progress_path = run_root / "logs" / "progress.jsonl"
    if progress_path.is_file():
        files["logs/progress.jsonl"] = progress_path

    frames_dir = getattr(selection, "frames_dir", None)
    manifest = getattr(selection, "manifest", None)
    if frames_dir is None or manifest is None:
        return files
    try:
        from PIL import Image, ImageOps

        image_paths = [
            Path(frames_dir) / frame.output_name
            for frame in manifest.selected_frames[:24]
        ]
        thumbs: list[Image.Image] = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                thumbs.append(ImageOps.fit(image.convert("RGB"), (240, 160)).copy())
        if thumbs:
            columns = 4
            rows = math.ceil(len(thumbs) / columns)
            sheet = Image.new("RGB", (columns * 240, rows * 160), "#111827")
            for index, thumb in enumerate(thumbs):
                sheet.paste(thumb, ((index % columns) * 240, (index // columns) * 160))
            contact_sheet = diagnostic_root / "contact_sheet.png"
            sheet.save(contact_sheet, format="PNG")
            files["contact_sheet.png"] = contact_sheet
    except (OSError, ValueError):
        pass
    return files


def _production_services() -> RunnerServices:
    # Imports stay deferred so CLI validation and --help never load training/Torch.
    from .publish import publish_diagnostics, publish_result, validate_bundle
    from .sources import copy_input_read_only, discover_source

    return RunnerServices(
        discover_source=discover_source,
        copy_input=copy_input_read_only,
        select_frames=_selection_adapter,
        reconstruct=_reconstruction_adapter,
        train=_training_adapter,
        polish=_polish_adapter,
        build_metadata_preview=_metadata_adapter,
        validate_bundle=validate_bundle,
        publish_result=publish_result,
        publish_diagnostics=publish_diagnostics,
        inspect_hardware=_inspect_hardware,
        preflight=preflight_runtime,
        assemble_reports=_assemble_reports_adapter,
    )


@contextmanager
def _reported_stage(
    reporter: StageReporter | None,
    stage_id: str,
    *,
    summary: Callable[[], str | None] | None = None,
):
    if reporter is None:
        yield
        return
    with reporter.stage(stage_id, summary=summary):
        yield


def run_static_notebook(
    spec: StaticNotebookRunSpec,
    *,
    runtime_paths: NotebookRuntimePaths | None = None,
    services: RunnerServices | None = None,
    reporter: StageReporter | None = None,
) -> NotebookRunResult:
    from .progress import use_stage_reporter

    with use_stage_reporter(reporter):
        return _run_static_notebook_with_context(
            spec,
            runtime_paths=runtime_paths,
            services=services,
            reporter=reporter,
        )


def _run_static_notebook_with_context(
    spec: StaticNotebookRunSpec,
    *,
    runtime_paths: NotebookRuntimePaths | None,
    services: RunnerServices | None,
    reporter: StageReporter | None,
) -> NotebookRunResult:
    paths = runtime_paths or runtime_paths_from_env()
    boundaries = services or _production_services()
    run_id = uuid.uuid4().hex
    run_root = Path(paths.work_root) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    if reporter is not None:
        reporter.bind_log(run_root / "logs" / "progress.jsonl", run_id=run_id)
    timings: dict[str, float] = {}
    run_started = time.perf_counter()
    publish_input_path = (
        Path(paths.drive_root)
        .resolve(strict=False)
        .joinpath(*spec.input_folder.split("/"))
    )

    stage_started = time.perf_counter()
    with _reported_stage(reporter, "input_discovery"):
        input_path = _resolve_drive_input(spec, paths, boundaries.resolve_input)
        timings["resolve_input_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        try:
            source_inventory = boundaries.discover_source(input_path)
        except Exception as exc:
            if reporter is not None:
                reporter.annotate_failure(exc)
            files = _failure_diagnostic_files(run_root, exc, None)
            try:
                diagnostics = boundaries.publish_diagnostics(
                    publish_input_path, run_id, files
                )
                setattr(exc, "diagnostics_path", diagnostics)
            except Exception as diagnostics_exc:
                exc.add_note(f"diagnostics publication failed: {diagnostics_exc}")
            raise
        timings["discover_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "runtime_preflight"):
        hardware = boundaries.inspect_hardware(paths)
        boundaries.preflight(
            spec,
            hardware,
            input_size_gb=_input_size_gb(source_inventory),
        )
    timings["preflight_seconds"] = time.perf_counter() - stage_started

    local_input = run_root / "input"
    selection: object | None = None
    reconstruction: object | None = None
    try:
        restored: PretrainingRestore | None = None
        if boundaries.restore_pretraining is not None:
            stage_started = time.perf_counter()
            with _reported_stage(reporter, "cache_restore"):
                restored = boundaries.restore_pretraining(
                    source_inventory=source_inventory,
                    spec=spec,
                    hardware=hardware,
                    run_root=run_root,
                )
            timings["restore_pretraining_seconds"] = time.perf_counter() - stage_started
        if restored is not None:
            selection = restored.selection
            reconstruction = restored.reconstruction
        else:
            if boundaries.restore_selection is not None:
                stage_started = time.perf_counter()
                selection = boundaries.restore_selection(
                    source_inventory=source_inventory,
                    spec=spec,
                    hardware=hardware,
                    run_root=run_root,
                )
                timings["restore_selection_seconds"] = (
                    time.perf_counter() - stage_started
                )
            if selection is not None:
                if reporter is not None:
                    reporter.skip("source_copy", "restored selected frames")
                    reporter.skip("frame_selection", "restored selection milestone")
            else:
                stage_started = time.perf_counter()
                with _reported_stage(reporter, "source_copy"):
                    copied_inventory = boundaries.copy_input(input_path, local_input)
                timings["copy_input_seconds"] = time.perf_counter() - stage_started
                if _inventory_signature(copied_inventory) != _inventory_signature(
                    source_inventory
                ):
                    raise ValueError(
                        "copied input inventory differs from the Drive source"
                    )

                stage_started = time.perf_counter()
                with _reported_stage(reporter, "frame_selection"):
                    selection = boundaries.select_frames(
                        copied_inventory,
                        spec=spec,
                        output_root=run_root / "selection",
                    )
                timings["selection_seconds"] = time.perf_counter() - stage_started
                if boundaries.save_selection is not None:
                    stage_started = time.perf_counter()
                    boundaries.save_selection(
                        source_inventory=source_inventory,
                        selection=selection,
                        spec=spec,
                        hardware=hardware,
                        run_root=run_root,
                    )
                    timings["save_selection_seconds"] = (
                        time.perf_counter() - stage_started
                    )
            stage_started = time.perf_counter()
            with _reported_stage(reporter, "reconstruction"):
                reconstruction = boundaries.reconstruct(
                    selection,
                    spec=spec,
                    hardware=hardware,
                    output_root=run_root / "reconstruction",
                )
            timings["reconstruction_seconds"] = time.perf_counter() - stage_started
        if boundaries.validate_reconstruction is None:
            decision = getattr(reconstruction, "decision", None)
            if decision is None or not decision.passed or decision.failures:
                raise RuntimeError(
                    "reconstruction service returned a non-passing decision"
                )
        else:
            boundaries.validate_reconstruction(reconstruction)
        if restored is None and boundaries.save_pretraining is not None:
            stage_started = time.perf_counter()
            with _reported_stage(reporter, "pretraining_cache_save"):
                boundaries.save_pretraining(
                    source_inventory=source_inventory,
                    selection=selection,
                    reconstruction=reconstruction,
                    spec=spec,
                    hardware=hardware,
                    run_root=run_root,
                )
            timings["save_pretraining_seconds"] = time.perf_counter() - stage_started
    except Exception as exc:
        if reporter is not None:
            reporter.annotate_failure(exc)
        files = _failure_diagnostic_files(run_root, exc, selection)
        try:
            diagnostics = boundaries.publish_diagnostics(
                publish_input_path,
                run_id,
                files,
            )
            setattr(exc, "diagnostics_path", diagnostics)
        except Exception as diagnostics_exc:
            exc.add_note(f"diagnostics publication failed: {diagnostics_exc}")
        raise
    os.environ["FOURDGS_DATA_ROOT"] = str(run_root / "training-data")
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "gaussian_training"):
        training = boundaries.train(
            reconstruction,
            spec=spec,
            selection=selection,
            run_id=run_id,
            output_root=run_root / "training",
        )
    timings["training_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "polish"):
        polish = boundaries.polish(
            training,
            reconstruction=reconstruction,
            spec=spec,
            selection=selection,
            output_root=run_root / "polish",
        )
    timings["polish_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "metadata_preview"):
        metadata_preview = boundaries.build_metadata_preview(
            polish,
            reconstruction=reconstruction,
            spec=spec,
            output_root=run_root / "metadata",
        )
    timings["metadata_preview_seconds"] = time.perf_counter() - stage_started
    bundle_root = run_root / "bundle"
    timings["total_before_reports_seconds"] = time.perf_counter() - run_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "report_assembly"):
        bundle = Path(
            boundaries.assemble_reports(
                spec=spec,
                run_id=run_id,
                source_inventory=source_inventory,
                selection=selection,
                reconstruction=reconstruction,
                training=training,
                polish=polish,
                metadata_preview=metadata_preview,
                hardware=hardware,
                timings=timings,
                bundle_root=bundle_root,
            )
        )
    timings["report_assembly_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "bundle_validation"):
        boundaries.validate_bundle(bundle, run_id=run_id)
    timings["bundle_validation_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    with _reported_stage(reporter, "result_publication"):
        receipt = boundaries.publish_result(
            bundle,
            publish_input_path,
            run_id=run_id,
            replace_owned_result=spec.publish.replace_owned_result,
        )
    timings["publish_seconds"] = time.perf_counter() - stage_started
    return NotebookRunResult(
        run_id=run_id,
        final_path=Path(receipt.final_path),
        local_bundle=bundle,
        quality_report_path=bundle / "quality_report.json",
        manifest_path=bundle / "run_manifest.json",
    )
