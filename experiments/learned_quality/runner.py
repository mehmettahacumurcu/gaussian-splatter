from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.static_pipeline.runner import (
    HardwareInfo,
    NotebookRuntimePaths,
    NotebookRunResult,
    RunnerServices,
    RuntimePreflightError,
    run_static_notebook,
    runtime_paths_from_env,
)

from .contracts import (
    LearnedQualityRunSpec,
    LearnedReconstructionOutput,
    to_static_run_spec,
)
from .publish import publish_learned_diagnostics, publish_learned_result
from .reports import finalize_learned_bundle, validate_learned_bundle


LearnedReconstruct = Callable[..., LearnedReconstructionOutput]
ContactSheetBuilder = Callable[..., Mapping[str, Path]]


@dataclass(frozen=True)
class LearnedQualityContext:
    reconstruct: LearnedReconstruct
    model_manifest_path: Path
    contact_sheet_builder: ContactSheetBuilder | None = None


def preflight_learned_runtime(
    spec: LearnedQualityRunSpec | object,
    hardware: HardwareInfo,
    *,
    input_size_gb: float,
) -> None:
    if not isinstance(spec, LearnedQualityRunSpec):
        LearnedQualityRunSpec(
            input_folder=getattr(spec, "input_folder"),
            publish={
                "replace_owned_result": getattr(
                    getattr(spec, "publish"), "replace_owned_result"
                )
            },
        )
    if (
        not isinstance(input_size_gb, (int, float))
        or isinstance(input_size_gb, bool)
        or not math.isfinite(float(input_size_gb))
        or input_size_gb < 0
    ):
        raise ValueError("input_size_gb must be a finite non-negative number")
    if not hardware.cuda_available:
        raise RuntimePreflightError("The learned-quality experiment requires CUDA")
    if not hardware.colmap_available:
        raise RuntimePreflightError("COLMAP is not installed in this runtime")
    if not hardware.ffmpeg_available:
        raise RuntimePreflightError("FFmpeg is not installed in this runtime")
    if "A100" not in hardware.gpu_name.upper():
        raise RuntimePreflightError(
            "The learned-quality experiment requires an NVIDIA A100 runtime"
        )
    if not math.isfinite(hardware.vram_gb) or hardware.vram_gb < 39.0:
        raise RuntimePreflightError(
            "The learned-quality experiment requires at least 39 GiB of A100 VRAM"
        )
    required_disk_gb = max(35.0, float(input_size_gb) * 5.0 + 20.0)
    if (
        not math.isfinite(hardware.disk_free_gb)
        or hardware.disk_free_gb < required_disk_gb
    ):
        raise RuntimePreflightError(
            "Insufficient working disk for the learned-quality experiment: "
            f"need {required_disk_gb:.1f} GB, detected {hardware.disk_free_gb:.1f} GB"
        )
    print(
        json.dumps(
            {
                "event": "learned_quality_preflight",
                "gpu": hardware.gpu_name,
                "vram_gb": hardware.vram_gb,
                "disk_free_gb": hardware.disk_free_gb,
                "profile": "ultra",
                "iterations": 120_000,
                "gaussian_ceiling": 6_000_000,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _training_adapter(
    reconstruction: LearnedReconstructionOutput,
    *,
    spec: object,
    selection: object,
    run_id: str,
    output_root: Path,
) -> object:
    from backend.static_pipeline.training import PreparedTrainingInput

    from .training import run_learned_training

    prepared = PreparedTrainingInput(
        run_id=run_id,
        data_root=output_root,
        scene_name="scene",
        frames_dir=reconstruction.frames_dir,
        reconstruction=reconstruction.bundle,
        source_digest=selection.inventory.digest,
        selection_digest=reconstruction.selected_manifest.image_set_digest,
    )
    return run_learned_training(prepared, spec, reconstruction)


def _sheet_from_images(paths: tuple[Path, ...], destination: Path) -> Path:
    from PIL import Image, ImageOps

    if not paths:
        raise ValueError(f"no images available for {destination.name}")
    cells = []
    for path in paths[:24]:
        with Image.open(path) as opened:
            cells.append(ImageOps.fit(opened.convert("RGB"), (240, 160)).copy())
    columns = 4
    rows = math.ceil(len(cells) / columns)
    sheet = Image.new("RGB", (columns * 240, rows * 160), "#111827")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 240, (index // columns) * 160))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return destination


def _sheet_from_depth(paths: tuple[Path, ...], destination: Path) -> Path:
    import numpy as np
    from PIL import Image

    rendered: list[Path] = []
    scratch = destination.parent / ".depth-previews"
    scratch.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths[:24]):
        values = np.load(path, allow_pickle=False)
        valid = np.isfinite(values) & (values > 0.0)
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(values[valid], (2.0, 98.0))
            if high > low:
                scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
                normalized[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
        preview = scratch / f"{index:06d}.png"
        Image.fromarray(normalized, mode="L").save(preview, format="PNG")
        rendered.append(preview)
    result = _sheet_from_images(tuple(rendered), destination)
    for path in rendered:
        path.unlink()
    scratch.rmdir()
    return result


def _default_contact_sheets(
    *,
    reconstruction: LearnedReconstructionOutput,
    metadata_preview: object,
    output_root: Path,
    **_: object,
) -> Mapping[str, Path]:
    artifacts = reconstruction.artifacts
    masks = artifacts.masks
    depth = artifacts.depth
    if masks is None or depth is None:
        raise ValueError("learned masks and validated depth are required for reports")
    mask_paths = tuple(Path(frame.hard_exclude_path) for frame in masks.frames)
    depth_paths = tuple(Path(frame.depth_path) for frame in depth.frames)
    geometry_paths = tuple(
        Path(frame.path)
        for frame in getattr(artifacts.photometric, "original_frames", ())
    )
    if not geometry_paths:
        geometry_paths = tuple(
            reconstruction.frames_dir / frame.output_name
            for frame in reconstruction.selected_manifest.selected_frames
        )
    return {
        "masks_contact_sheet.png": _sheet_from_images(
            mask_paths, output_root / "masks_contact_sheet.png"
        ),
        "depth_contact_sheet.png": _sheet_from_depth(
            depth_paths, output_root / "depth_contact_sheet.png"
        ),
        "geometry_contact_sheet.png": _sheet_from_images(
            geometry_paths, output_root / "geometry_contact_sheet.png"
        ),
        "final_render_contact_sheet.png": _sheet_from_images(
            (Path(metadata_preview.preview_path),),
            output_root / "final_render_contact_sheet.png",
        ),
    }


def _reported_winner(reconstruction: object) -> object:
    decision = getattr(reconstruction, "decision")
    for candidate in getattr(reconstruction, "geometry_candidates"):
        if candidate.decision is decision:
            return candidate
    raise ValueError("geometry winner is not present in the candidate report")


def make_learned_quality_services(context: LearnedQualityContext) -> RunnerServices:
    if not isinstance(context, LearnedQualityContext):
        raise TypeError("context must be a LearnedQualityContext")
    from backend.static_pipeline.runner import _production_services

    production = _production_services()
    sheet_builder = context.contact_sheet_builder or _default_contact_sheets

    def preflight(
        spec: object, hardware: HardwareInfo, *, input_size_gb: float
    ) -> None:
        preflight_learned_runtime(spec, hardware, input_size_gb=input_size_gb)

    def assemble_reports(**kwargs: object) -> Path:
        bundle = Path(production.assemble_reports(**kwargs))
        reconstruction = kwargs["reconstruction"]
        training = kwargs["training"]
        metadata_preview = kwargs["metadata_preview"]
        report_root = Path(kwargs["bundle_root"]).parent / "learned-reports"
        sheets = sheet_builder(
            reconstruction=reconstruction,
            training=training,
            metadata_preview=metadata_preview,
            output_root=report_root,
        )
        artifacts = reconstruction.artifacts
        photometric = artifacts.photometric
        if photometric is None:
            raise ValueError("photometric evidence is required for learned reports")
        winner = _reported_winner(reconstruction)
        experiment_report = {
            "status": "passed",
            "winner": winner.candidate_id,
            "final_gaussian_count": training.final_gaussian_count,
            "training_rgb_digest": training.training_rgb_digest,
            "fallbacks": training.fallbacks,
            "stages": artifacts.stage_records,
        }
        return finalize_learned_bundle(
            bundle,
            run_id=str(kwargs["run_id"]),
            experiment_report=experiment_report,
            geometry_candidates=reconstruction.geometry_candidates,
            model_manifest_path=context.model_manifest_path,
            contact_sheets=sheets,
            density_history_path=training.density_history_path,
            photometric_report_path=photometric.report_path,
        )

    return RunnerServices(
        discover_source=production.discover_source,
        copy_input=production.copy_input,
        select_frames=production.select_frames,
        reconstruct=context.reconstruct,
        train=_training_adapter,
        polish=production.polish,
        build_metadata_preview=production.build_metadata_preview,
        validate_bundle=validate_learned_bundle,
        publish_result=publish_learned_result,
        publish_diagnostics=publish_learned_diagnostics,
        inspect_hardware=production.inspect_hardware,
        preflight=preflight,
        assemble_reports=assemble_reports,
        resolve_input=production.resolve_input,
    )


def _directory_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def _late_failure_files(run_root: Path, error: Exception) -> Mapping[str, Path]:
    logs = run_root / "diagnostics" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pipeline.log"
    log_path.write_text(
        f"error_type={type(error).__name__}\nerror={error}\n", encoding="utf-8"
    )
    report = run_root / "diagnostics" / "experiment_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"logs/pipeline.log": log_path, "experiment_report.json": report}


def _default_context() -> LearnedQualityContext:
    from .runtime import run_learned_reconstruction

    manifest = Path(
        os.environ.get(
            "LEARNED_MODEL_MANIFEST", "/content/learned-env/model_manifest.json"
        )
    ).resolve()
    return LearnedQualityContext(
        reconstruct=run_learned_reconstruction,
        model_manifest_path=manifest,
    )


def run_learned_quality_notebook(
    spec: LearnedQualityRunSpec,
    *,
    runtime_paths: NotebookRuntimePaths | None = None,
    context: LearnedQualityContext | None = None,
) -> NotebookRunResult:
    if not isinstance(spec, LearnedQualityRunSpec):
        raise TypeError("spec must be a LearnedQualityRunSpec")
    paths = runtime_paths or runtime_paths_from_env()
    before = _directory_names(paths.work_root)
    active_context = context or _default_context()
    try:
        return run_static_notebook(
            to_static_run_spec(spec),
            runtime_paths=paths,
            services=make_learned_quality_services(active_context),
        )
    except Exception as exc:
        if getattr(exc, "diagnostics_path", None) is not None:
            raise
        created = sorted(_directory_names(paths.work_root) - before)
        run_id = created[-1] if created else "preflight-failure"
        run_root = paths.work_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        input_folder = paths.drive_root.joinpath(*spec.input_folder.split("/"))
        try:
            diagnostics = publish_learned_diagnostics(
                input_folder,
                run_id,
                _late_failure_files(run_root, exc),
            )
            setattr(exc, "diagnostics_path", diagnostics)
        except Exception as diagnostics_error:
            exc.add_note(f"learned diagnostics publication failed: {diagnostics_error}")
        raise
