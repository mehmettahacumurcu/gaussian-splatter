from __future__ import annotations

import os
import random
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import field_validator

from backend.notebooks.drive_paths import normalize_input_folder
from backend.notebooks.models import StaticNotebookRunSpec, StrictModel
from backend.static_pipeline.training import PreparedTrainingInput, TrainingResult

from .ablation import AblationVariant
from .ablation_staging import StagedAblationInputs
from .ablation_training import make_ablation_static_spec, prepare_ablation_scene
from .contracts import LearnedReconstructionOutput
from .legacy_control_export import select_legacy_control_variant


RUNTIME_PROFILE = "a100_legacy_control_native_1080p_30k_local"
CHECKPOINT_ITERATIONS = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000)
NATIVE_IMAGE_SIZE = (1_920, 1_080)
MINIMUM_STAGE_FREE_BYTES = 100 * 1024**3
MINIMUM_SNAPSHOT_FREE_BYTES = 8 * 1024**3


class LegacyControlLongRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    runtime_profile: Literal[
        "a100_legacy_control_native_1080p_30k_local"
    ] = RUNTIME_PROFILE

    @field_validator("input_folder", mode="before")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("input_folder must be a string")
        canonical = normalize_input_folder(value.strip())
        forbidden = (
            "_legacy_control_5k_result",
            "_learned_test_result",
            "_training_ablation",
            "_floor_recovery_diagnostic",
            "_learned_test_diagnostics",
            "_learned_test_cache",
        )
        if canonical.endswith(forbidden):
            raise ValueError("Choose the input folder, not an experiment output")
        return canonical


@dataclass(frozen=True)
class LegacyControlLongRunResult:
    run_id: str
    local_root: Path
    ply_paths: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...]


def validate_legacy_control_long_hardware(hardware: object) -> None:
    gpu_name = str(getattr(hardware, "gpu_name", ""))
    vram_gb = float(getattr(hardware, "vram_gb", 0.0))
    disk_free_gb = float(getattr(hardware, "disk_free_gb", 0.0))
    raw_host_ram_gb = getattr(hardware, "host_ram_gb", None)
    host_ram_gb = float(raw_host_ram_gb) if raw_host_ram_gb is not None else 0.0
    if "A100" not in gpu_name.upper() or vram_gb < 75.0:
        raise RuntimeError(
            "native-1080p legacy-control training requires an A100 with "
            f"75+ GiB VRAM; got {gpu_name} ({vram_gb:g} GiB)"
        )
    if disk_free_gb < 120.0:
        raise RuntimeError(
            "native-1080p legacy-control training requires at least 120 GiB "
            f"free local disk; got {disk_free_gb:g} GiB"
        )
    if host_ram_gb < 100.0:
        raise RuntimeError(
            "native-1080p legacy-control training requires a High-RAM runtime "
            f"with at least 100 GiB host RAM; got {host_ram_gb:g} GiB"
        )


def long_legacy_control_variant() -> AblationVariant:
    control = select_legacy_control_variant()
    return AblationVariant(
        experiment_id="legacy_control",
        features=control.features,
        n_iterations=30_000,
        checkpoints=CHECKPOINT_ITERATIONS,
        density_events=control.density_events,
    )


def make_legacy_control_long_spec(
    base_spec: StaticNotebookRunSpec,
) -> StaticNotebookRunSpec:
    """Extend the passing 5K control without changing its objective."""

    short = make_ablation_static_spec(base_spec, select_legacy_control_variant())
    advanced = short.quality.advanced.model_copy(
        update={
            "resolution_long_edge_cap": 1_920,
            "multires_schedule": [],
        }
    )
    quality = short.quality.model_copy(
        update={
            "n_iters": 30_000,
            "max_gaussians": 6_000_000,
            "advanced": advanced,
        }
    )
    return short.model_copy(update={"quality": quality})


def require_native_1080p(
    cameras: Mapping[str, Mapping[str, object]],
) -> tuple[int, int]:
    if not cameras:
        raise ValueError("native 1920x1080 training requires registered cameras")
    observed: set[tuple[int, int]] = set()
    for camera in cameras.values():
        try:
            size = (int(camera["width"]), int(camera["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "native 1920x1080 training requires camera dimensions"
            ) from exc
        observed.add(size)
    if observed != {NATIVE_IMAGE_SIZE}:
        raise ValueError(
            "native 1920x1080 training requires one exact 1920x1080 camera "
            f"set; observed {sorted(observed)}"
        )
    return NATIVE_IMAGE_SIZE


def build_resume_checkpoint(
    trainer: object,
    *,
    iteration: int,
    n_iters: int,
    camera_generator: torch.Generator,
) -> dict[str, object]:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None or not hasattr(optimizer, "state_dict"):
        raise RuntimeError("trainer optimizer state is unavailable")
    scheduler = getattr(trainer, "scheduler", None)
    scheduler_state = (
        scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )
    cuda_rng_state: Sequence[torch.Tensor] = ()
    if torch.cuda.is_available():
        cuda_rng_state = tuple(state.cpu() for state in torch.cuda.get_rng_state_all())
    return {
        "schema_version": 1,
        "iter": int(iteration),
        "n_iters": int(n_iters),
        "gs": trainer.gs.state_for_save(),
        "deform": trainer.deform.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "camera_generator_state": camera_generator.get_state().cpu(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": tuple(cuda_rng_state),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "scene_extent": float(trainer.scene_extent),
        "sh_degree": int(trainer.gs.sh_degree),
    }


def _regular_nonempty_file(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} was not created: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise RuntimeError(f"{label} is not a non-empty regular file: {path}")


class LocalSnapshotWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        camera_generator: torch.Generator,
        exporter: Callable[..., Sequence[Path]] | None = None,
        checkpoint_saver: Callable[[object, Path], None] | None = None,
        free_bytes: Callable[[Path], int] | None = None,
        minimum_free_bytes: int = MINIMUM_SNAPSHOT_FREE_BYTES,
    ) -> None:
        if exporter is None:
            from backend.export.to_splat import export_to_ply

            exporter = export_to_ply
        self.output_root = Path(output_root).resolve(strict=False)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.camera_generator = camera_generator
        self.exporter = exporter
        self.checkpoint_saver = checkpoint_saver or torch.save
        self.free_bytes = free_bytes or (
            lambda path: int(shutil.disk_usage(path).free)
        )
        self.minimum_free_bytes = int(minimum_free_bytes)

    @property
    def ply_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for iteration in CHECKPOINT_ITERATIONS
            if (
                path := self.output_root
                / "ply"
                / f"legacy_control_{iteration:06d}.ply"
            ).is_file()
        )

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for iteration in CHECKPOINT_ITERATIONS
            if (
                path := self.output_root
                / "checkpoints"
                / f"legacy_control_{iteration:06d}.pt"
            ).is_file()
        )

    def __call__(
        self,
        trainer: object,
        iteration: int,
        resolution: tuple[int, int],
        sh_degree: int,
    ) -> None:
        if iteration not in CHECKPOINT_ITERATIONS:
            raise ValueError(f"unsupported snapshot iteration: {iteration}")
        if tuple(resolution) != NATIVE_IMAGE_SIZE:
            raise ValueError(
                "snapshot export requires exact 1920x1080 training resolution"
            )
        if int(sh_degree) != int(trainer.gs.sh_degree):
            raise ValueError("snapshot SH degree differs from the trainer")
        available = int(self.free_bytes(self.output_root))
        if available < self.minimum_free_bytes:
            raise RuntimeError(
                "insufficient local disk for the next snapshot: "
                f"{available} < {self.minimum_free_bytes}"
            )

        ply_root = self.output_root / "ply"
        checkpoint_root = self.output_root / "checkpoints"
        ply_root.mkdir(exist_ok=True)
        checkpoint_root.mkdir(exist_ok=True)
        ply_target = ply_root / f"legacy_control_{iteration:06d}.ply"
        checkpoint_target = (
            checkpoint_root / f"legacy_control_{iteration:06d}.pt"
        )
        if os.path.lexists(ply_target) or os.path.lexists(checkpoint_target):
            raise FileExistsError(f"snapshot boundary already exists: {iteration}")

        temporary_ply_root = ply_root / f".ply-{iteration:06d}-{uuid.uuid4().hex}"
        try:
            written = tuple(
                Path(path)
                for path in self.exporter(
                    trainer.gs,
                    None,
                    temporary_ply_root,
                    num_timestamps=1,
                    scene_extent=float(trainer.scene_extent),
                    device=str(getattr(trainer, "device", "cuda")),
                )
            )
            if len(written) != 1:
                raise RuntimeError("static snapshot export must produce one PLY")
            _regular_nonempty_file(written[0], label="temporary PLY snapshot")
            os.replace(written[0], ply_target)
            _regular_nonempty_file(ply_target, label="PLY snapshot")
        finally:
            shutil.rmtree(temporary_ply_root, ignore_errors=True)

        payload = build_resume_checkpoint(
            trainer,
            iteration=iteration,
            n_iters=30_000,
            camera_generator=self.camera_generator,
        )
        temporary_checkpoint = checkpoint_root / (
            f".{checkpoint_target.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            self.checkpoint_saver(payload, temporary_checkpoint)
            _regular_nonempty_file(
                temporary_checkpoint,
                label="temporary training checkpoint",
            )
            os.replace(temporary_checkpoint, checkpoint_target)
            _regular_nonempty_file(
                checkpoint_target,
                label="training checkpoint",
            )
        finally:
            if os.path.lexists(temporary_checkpoint):
                temporary_checkpoint.unlink()
        print(
            f"SNAPSHOT {iteration:06d}: {ply_target} | {checkpoint_target}",
            flush=True,
        )


class LegacyControlLongPipelineRunner:
    def __init__(
        self,
        reconstruction: LearnedReconstructionOutput,
        *,
        pipeline_runner: Callable[..., Mapping[str, object]],
        variant: AblationVariant,
        snapshot_writer: LocalSnapshotWriter,
        camera_generator: torch.Generator,
        prepare_scene: Callable[..., object] = prepare_ablation_scene,
    ) -> None:
        self.reconstruction = reconstruction
        self.pipeline_runner = pipeline_runner
        self.variant = variant
        self.snapshot_writer = snapshot_writer
        self.camera_generator = camera_generator
        self.prepare_scene = prepare_scene
        self.invocations = 0

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        if "video_path" not in kwargs:
            raise ValueError("legacy-control pipeline requires video_path")
        if "trainer_customizer" in kwargs or "trainer_train_kwargs" in kwargs:
            raise ValueError("legacy-control trainer injection cannot be overridden")
        self.invocations += 1
        if self.invocations != 1:
            raise RuntimeError("legacy-control long run must use one pipeline invocation")

        scene_dir = Path(kwargs["video_path"]).parent
        self.prepare_scene(
            self.reconstruction.artifacts,
            accepted_model_dir=self.reconstruction.accepted_model_dir,
            scene_dir=scene_dir,
            variant=self.variant,
        )

        def verify_trainer(trainer: object) -> None:
            expected = {
                "density_start_iter": 500,
                "density_end_iter": 4_500,
                "density_interval": 100,
                "opacity_reset_interval": 3_000,
                "max_gaussians": 6_000_000,
            }
            observed = {
                name: int(getattr(trainer, name))
                for name in expected
            }
            if observed != expected:
                raise RuntimeError(
                    f"legacy-control trainer contract changed: {observed}"
                )

        forwarded = dict(kwargs)
        forwarded["skip_foundation"] = True
        forwarded["trainer_customizer"] = verify_trainer
        forwarded["trainer_train_kwargs"] = {
            "camera_generator": self.camera_generator,
            "diagnostic_iterations": CHECKPOINT_ITERATIONS,
            "diagnostic_callback": self.snapshot_writer,
        }
        status = self.pipeline_runner(**forwarded)
        if not isinstance(status, Mapping):
            raise TypeError("legacy-control pipeline must return a status mapping")
        return dict(status)


def _validate_result(
    result: LegacyControlLongRunResult,
    *,
    expected_root: Path,
) -> LegacyControlLongRunResult:
    root = Path(result.local_root).resolve(strict=True)
    if root != Path(expected_root).resolve(strict=True):
        raise RuntimeError("long-run result root differs from the active local run")
    expected_plys = tuple(
        root / "ply" / f"legacy_control_{iteration:06d}.ply"
        for iteration in CHECKPOINT_ITERATIONS
    )
    expected_checkpoints = tuple(
        root / "checkpoints" / f"legacy_control_{iteration:06d}.pt"
        for iteration in CHECKPOINT_ITERATIONS
    )
    if result.ply_paths != expected_plys:
        raise RuntimeError("long-run PLY snapshot set is incomplete")
    if result.checkpoint_paths != expected_checkpoints:
        raise RuntimeError("long-run checkpoint set is incomplete")
    for path in (*expected_plys, *expected_checkpoints):
        _regular_nonempty_file(path, label="long-run output")
    return result


def run_legacy_control_long_from_staged(
    staged: StagedAblationInputs,
    *,
    base_spec: StaticNotebookRunSpec,
    local_root: Path,
    run_id: str,
    camera_parser: Callable[[Path], Mapping[str, Mapping[str, object]]] | None = None,
    pipeline_runner: Callable[..., Mapping[str, object]] | None = None,
    validated_training_runner: Callable[..., TrainingResult] | None = None,
    snapshot_writer_factory: Callable[..., LocalSnapshotWriter] = LocalSnapshotWriter,
) -> LegacyControlLongRunResult:
    root = Path(local_root).resolve(strict=True)
    reconstruction = staged.reconstruction
    if camera_parser is None:
        from backend.preprocess.parse_colmap import parse_cameras_from_model

        camera_parser = parse_cameras_from_model
    require_native_1080p(camera_parser(reconstruction.accepted_model_dir))
    if pipeline_runner is None:
        from backend.pipeline import run_pipeline

        pipeline_runner = run_pipeline
    if validated_training_runner is None:
        from backend.static_pipeline.training import run_validated_training

        validated_training_runner = run_validated_training

    training_root = root / "training"
    prepared = PreparedTrainingInput(
        run_id="legacy_control_1080p_30k",
        data_root=training_root,
        scene_name="scene",
        frames_dir=reconstruction.frames_dir,
        reconstruction=reconstruction.bundle,
        source_digest=staged.source_digest,
        selection_digest=reconstruction.selected_manifest.image_set_digest,
    )
    variant = long_legacy_control_variant()
    generator = torch.Generator(device="cpu").manual_seed(1701)
    snapshots = snapshot_writer_factory(
        output_root=root,
        camera_generator=generator,
    )
    adapter = LegacyControlLongPipelineRunner(
        reconstruction,
        pipeline_runner=pipeline_runner,
        variant=variant,
        snapshot_writer=snapshots,
        camera_generator=generator,
    )
    training_kwargs: dict[str, object] = {
        "source_long_edge": 1_920,
        "pipeline_runner": adapter,
    }
    acceptance = getattr(reconstruction, "acceptance", None)
    if acceptance is not None and acceptance.mode == "best_effort":
        from .training import make_output_first_reconstruction_validator

        training_kwargs["reconstruction_validator"] = (
            make_output_first_reconstruction_validator(reconstruction)
        )
    validated_training_runner(
        prepared,
        make_legacy_control_long_spec(base_spec),
        **training_kwargs,
    )
    result = LegacyControlLongRunResult(
        run_id=run_id,
        local_root=root,
        ply_paths=snapshots.ply_paths,
        checkpoint_paths=snapshots.checkpoint_paths,
    )
    return _validate_result(result, expected_root=root)


def _git_revision(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_legacy_control_long(
    spec: LegacyControlLongRunSpec,
    *,
    model_manifest_path: Path,
    expected_source_revision: str,
    actual_source_revision: str | None = None,
    drive_root: Path = Path("/content/drive/MyDrive"),
    work_root: Path = Path("/content/legacy_control_1080p_30k"),
    discover_source: Callable[[Path], object] | None = None,
    inspect_hardware: Callable[[object], object] | None = None,
    store_factory: Callable[[Path, str], object] | None = None,
    stage_inputs: Callable[..., StagedAblationInputs] | None = None,
    exact_audit_validator: Callable[[Path, object, object], bool] | None = None,
    pretraining_fingerprint_resolver: Callable[..., str] | None = None,
    execute_staged: Callable[..., LegacyControlLongRunResult] = (
        run_legacy_control_long_from_staged
    ),
) -> LegacyControlLongRunResult:
    """Restore verified inputs once and run one local continuous trainer."""

    from backend.static_pipeline.runner import NotebookRuntimePaths

    from .ablation_runner import _default_exact_audit_validator
    from .ablation_staging import (
        restore_output_first_pretraining,
        stage_ablation_inputs,
    )
    from .contracts import (
        LearnedQualityRunSpec,
        derive_learned_cache_root,
        to_static_run_spec,
    )
    from .floor_recovery_runner import verified_legacy_reference_fingerprint
    from .runner import (
        _CPU_AUDIT_PYTHON_VERSION,
        _CPU_AUDIT_SELECTION_PRODUCER_SHA256,
        _REPOSITORY_ROOT,
        _selection_milestone_ref,
    )

    source_root = Path(__file__).resolve().parents[2]
    revision = actual_source_revision or _git_revision(source_root)
    if revision != expected_source_revision:
        raise RuntimeError(
            "checked-out source differs from the notebook pin: "
            f"{revision} != {expected_source_revision}"
        )
    manifest_path = Path(model_manifest_path).resolve(strict=True)
    mounted_drive = Path(drive_root).resolve(strict=True)
    input_path = mounted_drive.joinpath(*Path(spec.input_folder).parts).resolve(
        strict=True
    )
    if input_path.parent == input_path:
        raise ValueError("input folder cannot be the Drive root")
    if discover_source is None:
        from backend.static_pipeline.sources import discover_source as discover

        discover_source = discover
    inventory = discover_source(input_path)

    local_work_root = Path(work_root).resolve(strict=False)
    local_work_root.mkdir(parents=True, exist_ok=True)
    runtime_paths = NotebookRuntimePaths(
        drive_root=mounted_drive,
        work_root=local_work_root,
    )
    if inspect_hardware is None:
        from backend.static_pipeline.runner import _inspect_hardware

        inspect_hardware = _inspect_hardware
    hardware = inspect_hardware(runtime_paths)
    validate_legacy_control_long_hardware(hardware)

    reference_source = input_path.with_name(f"{input_path.name}_training_ablation")
    fingerprint_resolver = (
        pretraining_fingerprint_resolver
        or verified_legacy_reference_fingerprint
    )
    pretraining_fingerprint = fingerprint_resolver(
        reference_source,
        source_digest=str(getattr(inventory, "digest")),
    )
    learned_spec = LearnedQualityRunSpec(
        input_folder=spec.input_folder,
        recovery_mode="round0_output_first_v1",
    )
    base_spec = to_static_run_spec(learned_spec)
    cache_root = derive_learned_cache_root(input_path)
    if store_factory is None:
        from .cache import LearnedCheckpointStore

        def default_store_factory(root: Path, digest: str) -> object:
            return LearnedCheckpointStore(root, input_identity=digest)

        store_factory = default_store_factory
    store = store_factory(cache_root, str(getattr(inventory, "digest")))

    run_id = uuid.uuid4().hex
    local_root = local_work_root / run_id
    local_root.mkdir(parents=True, exist_ok=False)
    audit_exact = exact_audit_validator or _default_exact_audit_validator

    def require_any_audit(active_inventory: object) -> None:
        audits = cache_root / "audits"
        if not audits.is_dir() or not any(audits.glob("*/_SUCCESS.json")):
            raise RuntimeError(
                "a verified CPU cache audit is required before legacy-control "
                "native-1080p training"
            )
        if getattr(active_inventory, "digest", None) != getattr(
            inventory,
            "digest",
            None,
        ):
            raise ValueError("audit inventory differs from the active input")

    def validate_selection(selection: object) -> None:
        if not audit_exact(cache_root, selection, inventory):
            raise RuntimeError(
                "a matching verified CPU track-audit receipt is required before "
                "legacy-control native-1080p training"
            )

    selection_refs = [
        _selection_milestone_ref(
            inventory,
            base_spec,
            hardware,
            manifest_path,
        ),
        _selection_milestone_ref(
            inventory,
            base_spec,
            hardware,
            manifest_path,
            producer_code_sha256=_CPU_AUDIT_SELECTION_PRODUCER_SHA256,
            python_version=_CPU_AUDIT_PYTHON_VERSION,
        ),
    ]
    selection_refs = list(dict.fromkeys(selection_refs))

    def restore_graph(destination: Path) -> object | None:
        return restore_output_first_pretraining(
            store=store,
            source_inventory=inventory,
            destination=destination,
            selection_refs=tuple(selection_refs),
            hardware=hardware,
            model_manifest_path=manifest_path,
            repository_root=_REPOSITORY_ROOT,
            run_id=run_id,
            selection_validator=validate_selection,
            final_pretraining_fingerprint=pretraining_fingerprint,
        )

    stage = stage_inputs or stage_ablation_inputs
    staged = stage(
        store=store,
        source_inventory=inventory,
        destination=local_root / "inputs",
        drive_root=mounted_drive,
        expected_fingerprint=lambda _selection: pretraining_fingerprint,
        expected_source_revision=expected_source_revision,
        actual_source_revision=revision,
        audit_validator=require_any_audit,
        restored_validator=lambda selection, _reconstruction: validate_selection(
            selection
        ),
        minimum_free_bytes=MINIMUM_STAGE_FREE_BYTES,
        run_id=run_id,
        freeze=True,
        restore_pretraining=restore_graph,
    )
    result = execute_staged(
        staged,
        base_spec=base_spec,
        local_root=local_root,
        run_id=run_id,
    )
    return _validate_result(result, expected_root=local_root)
