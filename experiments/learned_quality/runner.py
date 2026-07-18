from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.static_pipeline.progress import (
    StageDefinition,
    StageReporter,
    current_stage_reporter,
)
from backend.static_pipeline.runner import (
    HardwareInfo,
    NotebookRuntimePaths,
    NotebookRunResult,
    RunnerServices,
    RuntimePreflightError,
    SelectionOutput,
    run_static_notebook,
    runtime_paths_from_env,
)

from .cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCheckpointStore,
    checkpoint_fingerprint,
    producer_code_digest,
)
from .contracts import (
    LearnedQualityRunSpec,
    LearnedReconstructionOutput,
    derive_learned_cache_root,
    to_static_run_spec,
)
from .milestones import (
    MilestoneInputs,
    MilestoneRef,
    MilestoneSession,
    MilestoneState,
    milestone_fingerprint,
)
from .publish import publish_learned_diagnostics, publish_learned_result
from .reports import finalize_learned_bundle, validate_learned_bundle


LearnedReconstruct = Callable[..., LearnedReconstructionOutput]
ContactSheetBuilder = Callable[..., Mapping[str, Path]]
ModelPreflight = Callable[[], None]


LEARNED_STAGE_DEFINITIONS = tuple(
    StageDefinition(stage_id, label)
    for stage_id, label in (
        ("input_discovery", "Input discovery and verification"),
        ("runtime_preflight", "A100 runtime preflight"),
        ("cache_restore", "Complete pre-training cache restore"),
        ("learned_model_preflight", "Learned model loading preflight"),
        ("source_copy", "Source copy to local runtime"),
        ("frame_selection", "Smart frame selection"),
        ("reconstruction", "Learned reconstruction pipeline"),
        ("da3_anchor", "DA3 anchor inference"),
        ("classical_colmap", "Classical COLMAP prepass"),
        ("da3_metric_sky", "DA3 metric depth and sky"),
        ("semantic_masks", "Semantic transient masks"),
        ("optical_flow", "Optical-flow motion evidence"),
        ("mask_fusion", "Quality-mask fusion"),
        ("geometry_comparison", "Classical and hybrid geometry comparison"),
        ("photometric_validation", "Photometric normalization validation"),
        ("final_pose_depth", "Final-pose conditioned depth"),
        ("dense_seed_fusion", "Validated depth and dense-seed fusion"),
        ("pretraining_cache_save", "Complete pre-training cache publication"),
        ("gaussian_training", "Gaussian training with live iterations"),
        ("polish", "Splat quality polish"),
        ("metadata_preview", "Metadata and preview generation"),
        ("report_assembly", "Quality report assembly"),
        ("bundle_validation", "Final bundle validation"),
        ("result_publication", "Drive result publication"),
    )
)
_PRETRAINING_PRODUCER_PATHS = (
    "backend/static_pipeline/contracts.py",
    "backend/static_pipeline/sources.py",
    "backend/static_pipeline/selection.py",
    "backend/static_pipeline/colmap.py",
    "backend/static_pipeline/reconstruction.py",
    "experiments/learned_quality/cache.py",
    "experiments/learned_quality/contracts.py",
    "experiments/learned_quality/da3.py",
    "experiments/learned_quality/depth.py",
    "experiments/learned_quality/flow.py",
    "experiments/learned_quality/geometry.py",
    "experiments/learned_quality/lifecycle.py",
    "experiments/learned_quality/masks.py",
    "experiments/learned_quality/model_adapters.py",
    "experiments/learned_quality/photometric.py",
    "experiments/learned_quality/runtime.py",
    "experiments/learned_quality/segmentation.py",
)
_SELECTION_PRODUCER_PATHS = (
    "backend/static_pipeline/contracts.py",
    "backend/static_pipeline/runner.py",
    "backend/static_pipeline/selection.py",
    "backend/static_pipeline/sources.py",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PRETRAINING_SKIPPED_STAGES = (
    "learned_model_preflight",
    "source_copy",
    "frame_selection",
    "reconstruction",
    "da3_anchor",
    "classical_colmap",
    "da3_metric_sky",
    "semantic_masks",
    "optical_flow",
    "mask_fusion",
    "geometry_comparison",
    "photometric_validation",
    "final_pose_depth",
    "dense_seed_fusion",
    "pretraining_cache_save",
)


@dataclass(frozen=True)
class LearnedQualityContext:
    reconstruct: LearnedReconstruct
    model_manifest_path: Path
    contact_sheet_builder: ContactSheetBuilder | None = None
    model_preflight: ModelPreflight | None = None


class _LearnedCacheSession:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self._store: LearnedCheckpointStore | None = None

    def store_for(self, source_inventory: object) -> LearnedCheckpointStore:
        digest = getattr(source_inventory, "digest", None)
        if self._store is None:
            self._store = LearnedCheckpointStore(
                self.cache_root,
                input_identity=digest,
            )
        elif self._store.input_identity != digest:
            raise ValueError("learned cache session received a different input")
        return self._store


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pretraining_settings(spec: object, hardware: HardwareInfo) -> Mapping[str, object]:
    frame_selection = getattr(spec, "frame_selection")
    quality = getattr(spec, "quality")
    advanced = getattr(quality, "advanced")
    return {
        "frame_selection": frame_selection.model_dump(mode="json"),
        "quality_profile": getattr(quality.profile, "value", quality.profile),
        "resolution_long_edge_cap": advanced.resolution_long_edge_cap,
        "colmap_gpu_sift": hardware.colmap_gpu_sift is True,
    }


def _pretraining_cache_fingerprint(
    source_inventory: object,
    selection: object,
    spec: object,
    hardware: HardwareInfo,
    model_manifest_path: Path,
) -> str:
    from .runtime import _colmap_cache_fingerprint, _probe_colmap_version

    source_digest = getattr(source_inventory, "digest", None)
    if getattr(selection.inventory, "digest", None) != source_digest:
        raise ValueError("selection inventory differs from the current source")
    colmap_version = _probe_colmap_version()
    upstream = _colmap_cache_fingerprint(
        selection,
        hardware,
        model_manifest_path,
        version_probe=lambda: colmap_version,
    )
    return checkpoint_fingerprint(
        CheckpointKind.PRETRAINING,
        CheckpointInputs(
            source_digest=source_digest,
            settings=_pretraining_settings(spec, hardware),
            model_manifest_sha256=_sha256_path(model_manifest_path),
            tool_versions={"colmap": colmap_version},
            producer_code_sha256=producer_code_digest(
                _REPOSITORY_ROOT,
                _PRETRAINING_PRODUCER_PATHS,
            ),
            upstream_fingerprint=upstream,
        ),
    )


def _selection_milestone_ref(
    source_inventory: object,
    spec: object,
    hardware: HardwareInfo,
    model_manifest_path: Path,
) -> MilestoneRef:
    del model_manifest_path
    source_digest = getattr(source_inventory, "digest", None)
    return MilestoneRef(
        CheckpointKind.SELECTION,
        milestone_fingerprint(
            CheckpointKind.SELECTION,
            MilestoneInputs(
                source_digest=source_digest,
                selection_digest=source_digest,
                settings=_pretraining_settings(spec, hardware)["frame_selection"],
                # Frame selection has no learned-model input. A neutral digest lets
                # the CPU audit and A100 runtime address the exact same selection.
                model_manifest_sha256="0" * 64,
                tool_versions={"python": platform.python_version()},
                producer_code_sha256=producer_code_digest(
                    _REPOSITORY_ROOT,
                    _SELECTION_PRODUCER_PATHS,
                ),
                upstream={},
            ),
        ),
    )


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


def make_learned_quality_services(
    context: LearnedQualityContext,
    *,
    cache_root: Path | None = None,
) -> RunnerServices:
    if not isinstance(context, LearnedQualityContext):
        raise TypeError("context must be a LearnedQualityContext")
    from backend.static_pipeline.runner import _production_services

    production = _production_services()
    sheet_builder = context.contact_sheet_builder or _default_contact_sheets
    cache_session = _LearnedCacheSession(cache_root) if cache_root is not None else None

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

    reconstruct = context.reconstruct
    restore_pretraining = None
    save_pretraining = None
    restore_selection = None
    save_selection = None
    if cache_session is not None:

        def restore_selection_impl(**kwargs: object) -> SelectionOutput | None:
            source_inventory = kwargs["source_inventory"]
            spec = kwargs["spec"]
            hardware = kwargs["hardware"]
            run_root = Path(kwargs["run_root"])
            store = cache_session.store_for(source_inventory)
            ref = _selection_milestone_ref(
                source_inventory,
                spec,
                hardware,
                context.model_manifest_path,
            )
            restored = store.restore_milestone(
                ref,
                destination=run_root / "selection-restored",
                source_inventory=source_inventory,
            )
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.cache_event(
                    "hit" if restored is not None else "miss",
                    "Selection milestone",
                    str(cache_session.cache_root),
                )
            if restored is None:
                raise RuntimeError(
                    "A verified CPU cache audit selection is required; run the CPU "
                    "cache audit notebook before starting an A100 session"
                )
            if not isinstance(restored.value, SelectionOutput):
                raise ValueError("selection milestone restored the wrong state type")
            selection = restored.value
            if selection.inventory is not source_inventory:
                raise ValueError("selection milestone inventory is not current")
            from .audit import make_audit_inputs, require_audit_receipt
            from .runtime import _colmap_cache_fingerprints
            from .tracks import TrackQualificationPolicy

            audit_error: RuntimeError | None = None
            for colmap_fingerprint in _colmap_cache_fingerprints(
                selection,
                hardware,
                context.model_manifest_path,
            ):
                expected = make_audit_inputs(
                    input_digest=source_inventory.digest,
                    selection_digest=selection.manifest.image_set_digest,
                    colmap_fingerprint=colmap_fingerprint,
                    policy=TrackQualificationPolicy(),
                    repository_root=_REPOSITORY_ROOT,
                )
                try:
                    require_audit_receipt(cache_session.cache_root, expected=expected)
                    audit_error = None
                    break
                except RuntimeError as error:
                    audit_error = error
            if audit_error is not None:
                raise audit_error
            if context.model_preflight is not None:
                if reporter is None:
                    context.model_preflight()
                else:
                    with reporter.stage("learned_model_preflight"):
                        context.model_preflight()
            return selection

        def save_selection_impl(**kwargs: object) -> Path:
            source_inventory = kwargs["source_inventory"]
            selection = kwargs["selection"]
            spec = kwargs["spec"]
            hardware = kwargs["hardware"]
            run_root = Path(kwargs["run_root"])
            if not isinstance(selection, SelectionOutput):
                raise TypeError("selection must be a SelectionOutput")
            store = cache_session.store_for(source_inventory)
            ref = _selection_milestone_ref(
                source_inventory,
                spec,
                hardware,
                context.model_manifest_path,
            )
            generation = store.publish_milestone(
                MilestoneState(
                    ref=ref,
                    upstream={},
                    value=selection,
                    artifact_roots={"selection": selection.frames_dir},
                ),
                run_id=f"{run_root.name}-selection",
            )
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.cache_event("save", "Selection milestone", str(generation))
            return generation

        def cached_reconstruct(
            selection: object,
            **kwargs: object,
        ) -> LearnedReconstructionOutput:
            store = cache_session.store_for(selection.inventory)
            milestone_session = None
            if isinstance(selection, SelectionOutput):
                spec = kwargs["spec"]
                hardware = kwargs["hardware"]
                output_root = Path(kwargs["output_root"])
                selection_ref = _selection_milestone_ref(
                    selection.inventory,
                    spec,
                    hardware,
                    context.model_manifest_path,
                )
                milestone_session = MilestoneSession(
                    store=store,
                    source_inventory=selection.inventory,
                    source_digest=selection.inventory.digest,
                    selection_digest=selection.manifest.image_set_digest,
                    model_manifest_sha256=_sha256_path(context.model_manifest_path),
                    tool_versions={"python": platform.python_version()},
                    selection_ref=selection_ref,
                    repository_root=_REPOSITORY_ROOT,
                    run_id=output_root.parent.name,
                    external_roots={"selection": selection.frames_dir},
                )
            return context.reconstruct(
                selection,
                **kwargs,
                checkpoint_store=store,
                milestone_session=milestone_session,
            )

        def restore_pretraining_impl(**kwargs: object) -> object | None:
            source_inventory = kwargs["source_inventory"]
            spec = kwargs["spec"]
            hardware = kwargs["hardware"]
            run_root = Path(kwargs["run_root"])
            store = cache_session.store_for(source_inventory)
            store.probe_drive_publication(run_id=run_root.name)
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.cache_event(
                    "probe",
                    "Drive cache publication",
                    str(cache_session.cache_root),
                )
            restored = store.restore_pretraining(
                source_inventory=source_inventory,
                destination=run_root / "pretraining-restored",
                expected_fingerprint=lambda selection: _pretraining_cache_fingerprint(
                    source_inventory,
                    selection,
                    spec,
                    hardware,
                    context.model_manifest_path,
                ),
            )
            if reporter is not None:
                reporter.cache_event(
                    "hit" if restored is not None else "miss",
                    "Complete pre-training checkpoint",
                    str(cache_session.cache_root),
                )
                if restored is not None:
                    for stage_id in _PRETRAINING_SKIPPED_STAGES:
                        reporter.skip(stage_id, "restored from verified Drive cache")
            return restored

        def save_pretraining_impl(**kwargs: object) -> Path:
            source_inventory = kwargs["source_inventory"]
            selection = kwargs["selection"]
            reconstruction = kwargs["reconstruction"]
            spec = kwargs["spec"]
            hardware = kwargs["hardware"]
            run_root = Path(kwargs["run_root"])
            store = cache_session.store_for(source_inventory)
            fingerprint = _pretraining_cache_fingerprint(
                source_inventory,
                selection,
                spec,
                hardware,
                context.model_manifest_path,
            )
            generation = store.publish_pretraining(
                source_inventory=source_inventory,
                selection=selection,
                reconstruction=reconstruction,
                fingerprint=fingerprint,
                run_id=run_root.name,
                run_root=run_root,
            )
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.cache_event(
                    "save",
                    "Complete pre-training checkpoint",
                    str(generation),
                )
            return generation

        reconstruct = cached_reconstruct
        restore_pretraining = restore_pretraining_impl
        save_pretraining = save_pretraining_impl
        restore_selection = restore_selection_impl
        save_selection = save_selection_impl

    return RunnerServices(
        discover_source=production.discover_source,
        copy_input=production.copy_input,
        select_frames=production.select_frames,
        reconstruct=reconstruct,
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
        restore_pretraining=restore_pretraining,
        save_pretraining=save_pretraining,
        restore_selection=restore_selection,
        save_selection=save_selection,
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
    files = {"logs/pipeline.log": log_path, "experiment_report.json": report}
    progress = run_root / "logs" / "progress.jsonl"
    if progress.is_file():
        files["logs/progress.jsonl"] = progress
    return files


def _preflight_sea_raft_model() -> None:
    from .lifecycle import release_cuda_model
    from .model_adapters import SeaRaftTorchAdapter

    model = SeaRaftTorchAdapter(
        Path("/content/learned-sources/sea-raft"),
        Path(
            "/content/learned-checkpoints/"
            "MemorySlices--Tartan-C-T-TSKH-spring540x960-M"
        ),
    )
    release_cuda_model(model)


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
        model_preflight=_preflight_sea_raft_model,
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
    use_default_cache = context is None
    active_context = context or _default_context()
    input_folder = paths.drive_root.joinpath(*spec.input_folder.split("/"))
    cache_root = derive_learned_cache_root(input_folder) if use_default_cache else None
    reporter = StageReporter(LEARNED_STAGE_DEFINITIONS)
    try:
        return run_static_notebook(
            to_static_run_spec(spec),
            runtime_paths=paths,
            services=make_learned_quality_services(
                active_context,
                cache_root=cache_root,
            ),
            reporter=reporter,
        )
    except Exception as exc:
        if getattr(exc, "diagnostics_path", None) is not None:
            raise
        created = sorted(_directory_names(paths.work_root) - before)
        run_id = created[-1] if created else "preflight-failure"
        run_root = paths.work_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
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
