from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from backend.notebooks.drive_paths import normalize_input_folder
from backend.notebooks.models import StrictModel

from .ablation import AblationCheckpoint
from .ablation_staging import StagedAblationInputs
from .floor_recovery_training import FloorCheckpointMetrics, FloorTrainingResult


GENERATOR_ID = "4dgs-studio.floor-recovery-diagnostic"
RESULT_SUFFIX = "_floor_recovery_diagnostic"
A100_LEGACY_REFERENCE_REVISION = "ce836c9d82bc0b17bad1cff558f99cca41e4ae28"


class FloorRecoveryPublishSpec(StrictModel):
    replace_owned_result: bool = True


class FloorRecoveryRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    runtime_profile: Literal["a100_floor_recovery"] = "a100_floor_recovery"
    publish: FloorRecoveryPublishSpec = Field(
        default_factory=FloorRecoveryPublishSpec
    )

    @field_validator("input_folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        canonical = normalize_input_folder(value)
        forbidden = (
            RESULT_SUFFIX,
            "_learned_test_result",
            "_training_ablation",
            "_learned_test_diagnostics",
        )
        if canonical.endswith(forbidden):
            raise ValueError("choose the input folder, not an experiment output")
        return canonical


@dataclass(frozen=True)
class FloorRecoveryDecision:
    passed: bool
    reason: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class FloorRecoveryRunResult:
    run_id: str
    status: Literal["success", "rejected", "failed"]
    final_path: Path | None
    local_root: Path
    decision: FloorRecoveryDecision | None


def decide_floor_recovery(
    *,
    legacy: AblationCheckpoint,
    candidate: AblationCheckpoint,
    initial_floor: FloorCheckpointMetrics,
    final_floor: FloorCheckpointMetrics,
) -> FloorRecoveryDecision:
    """Apply the approved floor and global 5K gates without hidden fallbacks."""

    if legacy.iteration != 5_000 or candidate.iteration != 5_000:
        raise ValueError("floor recovery comparison requires exact 5K checkpoints")
    if initial_floor.iteration != 0 or final_floor.iteration != 5_000:
        raise ValueError("floor recovery requires iteration-zero and 5K floor metrics")
    values = (
        legacy.psnr_unmasked,
        candidate.psnr_unmasked,
        initial_floor.floor_alpha_coverage,
        initial_floor.residual_hole_fraction,
        final_floor.floor_alpha_coverage,
        final_floor.residual_hole_fraction,
        final_floor.perturbed_depth_disagreement_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return FloorRecoveryDecision(False, "nonfinite_metrics", ("nonfinite_metrics",))
    initial_holes = float(initial_floor.residual_hole_fraction)
    hole_reduction = (
        (initial_holes - float(final_floor.residual_hole_fraction))
        / max(initial_holes, 1e-9)
    )
    alpha_gain = float(final_floor.floor_alpha_coverage) - float(
        initial_floor.floor_alpha_coverage
    )
    structural = candidate.structural
    failures: list[str] = []
    checks = (
        ("hole_reduction", hole_reduction >= 0.25),
        ("floor_alpha_gain", alpha_gain >= 0.10),
        ("psnr_deficit", legacy.psnr_unmasked - candidate.psnr_unmasked <= 1.0),
        ("visible_white_fraction", structural.visible_white_fraction <= 0.015),
        ("oversized_fraction", structural.oversized_fraction <= 0.01),
        ("out_of_bounds_fraction", structural.out_of_bounds_fraction <= 0.001),
        ("nonfinite_values", structural.nonfinite_count == 0),
        (
            "perturbed_floor_depth",
            final_floor.perturbed_depth_disagreement_ratio <= 1.10,
        ),
    )
    failures.extend(name for name, passed in checks if not passed)
    return FloorRecoveryDecision(
        passed=not failures,
        reason="passed" if not failures else "quality_gates_failed",
        failures=tuple(failures),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def validate_result_target(path: Path) -> Path:
    target = Path(path).resolve(strict=False)
    if not target.name.endswith(RESULT_SUFFIX):
        raise ValueError("result target must be a floor recovery diagnostic folder")
    if target.name == RESULT_SUFFIX or target.parent == target:
        raise ValueError("result target must be a floor recovery diagnostic folder")
    return target


def publish_floor_recovery_report(
    *,
    local_report_root: Path,
    destination: Path,
    run_id: str,
    decision: FloorRecoveryDecision,
    replace_owned_result: bool = True,
) -> Path:
    """Atomically publish a bounded diagnostic and write completion last."""

    source = Path(local_report_root).resolve(strict=True)
    target = validate_result_target(destination)
    if target.resolve(strict=False) == source or target.is_relative_to(source):
        raise ValueError("diagnostic publication cannot target its local source")
    if os.path.lexists(target):
        if not replace_owned_result:
            raise FileExistsError(f"floor recovery diagnostic exists: {target}")
        try:
            owner = json.loads(
                (target / "_OWNERSHIP.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "refusing to replace an unowned floor recovery diagnostic"
            ) from exc
        if owner.get("generator_id") != GENERATOR_ID:
            raise RuntimeError(
                "refusing to replace an unowned floor recovery diagnostic"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        for child in source.iterdir():
            destination_child = staging / child.name
            if child.is_dir():
                shutil.copytree(child, destination_child)
            else:
                shutil.copy2(child, destination_child)
        _write_json(
            staging / "_OWNERSHIP.json",
            {"schema_version": 1, "generator_id": GENERATOR_ID, "run_id": run_id},
        )
        _write_json(staging / "decision.json", asdict(decision))
        _write_json(
            staging / "_SUCCESS.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "success" if decision.passed else "rejected",
                "meaning": "floor_recovery_diagnostic_completed",
            },
        )
        if os.path.lexists(target):
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _git_revision(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_floor_recovery_hardware(hardware: object) -> None:
    gpu_name = str(getattr(hardware, "gpu_name", ""))
    vram_gb = float(getattr(hardware, "vram_gb", 0.0))
    disk_free_gb = float(getattr(hardware, "disk_free_gb", 0.0))
    if "A100" not in gpu_name.upper() or vram_gb < 75.0:
        raise RuntimeError(
            "floor recovery requires an A100 with 75+ GiB VRAM; "
            f"got {gpu_name} ({vram_gb:g} GiB)"
        )
    if disk_free_gb < 40.0:
        raise RuntimeError(
            "floor recovery requires at least 40 GiB local disk; "
            f"got {disk_free_gb:g} GiB"
        )


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain an object")
    return payload


def stage_legacy_reference(
    source: Path,
    destination: Path,
    *,
    staged: StagedAblationInputs,
) -> tuple[AblationCheckpoint, ...]:
    """Stage and verify the exact A100 legacy-control checkpoint once."""

    from .ablation_runner import _record_from_payload

    root = Path(source).resolve(strict=True)
    if not (root / "_SUCCESS.json").is_file():
        raise RuntimeError("verified training ablation result is missing")
    matrix = _load_json(root / "diagnostic_matrix.json", "diagnostic matrix")
    staging = matrix.get("staging")
    if not isinstance(staging, dict):
        raise RuntimeError("diagnostic matrix staging metadata is missing")
    if staging.get("source_revision") != A100_LEGACY_REFERENCE_REVISION:
        raise RuntimeError(
            "legacy reference producer revision differs from the verified A100 matrix"
        )
    if (
        staging.get("source_digest") != staged.source_digest
        or staging.get("pretraining_fingerprint")
        != staged.pretraining_fingerprint
    ):
        raise RuntimeError("legacy reference differs from staged pretraining")
    environment = _load_json(root / "environment.json", "ablation environment")
    if environment.get("runtime_profile") != "a100_reference":
        raise RuntimeError("legacy reference must use the A100 reference profile")
    receipt_path = root / "experiments" / "legacy_control" / "receipt.json"
    record = _record_from_payload(_load_json(receipt_path, "legacy receipt"))
    if not record.passed or record.stopped_early:
        raise RuntimeError("legacy reference did not pass")
    observed = tuple(checkpoint.iteration for checkpoint in record.checkpoints)
    expected = (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
    if observed != expected:
        raise RuntimeError(
            f"legacy reference checkpoint schedule mismatch: {observed}"
        )
    target = Path(destination).resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(target)
    target.mkdir(parents=True)
    try:
        for relative in (
            Path("diagnostic_matrix.json"),
            Path("environment.json"),
            Path("experiments") / "legacy_control" / "receipt.json",
            Path("experiments") / "legacy_control" / "contact_005000.png",
            Path("experiments") / "legacy_control" / "contact_005000_fixed.png",
            Path("experiments") / "legacy_control" / "contact_005000_perturbed.png",
        ):
            active_source = root / relative
            if active_source.is_file():
                active_target = target / relative
                active_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(active_source, active_target)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return record.checkpoints


def _prepare_report(
    report_root: Path,
    *,
    staged: StagedAblationInputs,
    legacy: AblationCheckpoint,
    training: FloorTrainingResult | None,
    decision: FloorRecoveryDecision,
    floor_artifact_root: Path | None,
    reference_root: Path,
) -> None:
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    if floor_artifact_root is not None:
        for child in Path(floor_artifact_root).iterdir():
            shutil.copy2(child, root / child.name)
    if reference_root.is_dir():
        shutil.copytree(reference_root, root / "legacy_reference")
    comparison: dict[str, object] = {
        "schema_version": 1,
        "legacy": asdict(legacy),
        "decision": asdict(decision),
        "staging": {
            "source_digest": staged.source_digest,
            "pretraining_fingerprint": staged.pretraining_fingerprint,
            "source_revision": staged.source_revision,
        },
    }
    if training is not None:
        comparison["candidate"] = asdict(training.checkpoints[-1])
        comparison["floor_initial"] = asdict(training.floor_checkpoints[0])
        comparison["floor_final"] = asdict(training.floor_checkpoints[-1])
        shutil.copy2(training.raw_ply_path, root / "candidate.ply")
        if training.metrics_path is not None:
            shutil.copy2(training.metrics_path, root / "metrics.jsonl")
        shutil.copy2(training.run_manifest_path, root / "run_manifest.json")
    _write_json(root / "comparison.json", comparison)
    _write_json(root / "decision.json", asdict(decision))
    (root / "diagnostic_summary.md").write_text(
        "# Automatic floor-hole recovery\n\n"
        f"Decision: **{decision.reason}**\n\n"
        f"Failures: {', '.join(decision.failures) if decision.failures else 'none'}\n",
        encoding="utf-8",
        newline="\n",
    )


def run_floor_recovery_from_staged(
    staged: StagedAblationInputs,
    *,
    base_spec: object,
    local_root: Path,
    reference_root: Path,
    legacy_checkpoints: tuple[AblationCheckpoint, ...],
    destination: Path,
    run_id: str,
    replace_owned_result: bool = True,
    build_cloud: Callable[..., object] | None = None,
    fit_plane: Callable[..., object] | None = None,
    map_holes: Callable[..., object] | None = None,
    make_seeds: Callable[..., object] | None = None,
    train: Callable[..., FloorTrainingResult] | None = None,
    publish: Callable[..., Path] = publish_floor_recovery_report,
) -> FloorRecoveryRunResult:
    """Execute geometry and one bounded arm using only the staged local graph."""

    from backend.preprocess.parse_colmap import load_points3d_from_model
    from backend.static_pipeline.training import PreparedTrainingInput

    from .floor_recovery import (
        build_floor_hole_map,
        estimate_floor_plane,
        generate_floor_seed_artifact,
    )
    from .floor_recovery_training import run_floor_recovery_training

    root = Path(local_root).resolve(strict=True)
    report_root = root / "report"
    report_root.mkdir(exist_ok=False)
    checkpoints_root = report_root / "checkpoints"
    checkpoints_root.mkdir()
    reconstruction = staged.reconstruction
    artifacts = reconstruction.artifacts
    accepted_model = Path(reconstruction.accepted_model_dir)
    sparse_xyz, _, _, _ = load_points3d_from_model(accepted_model)
    if build_cloud is None:
        from .depth import build_supported_depth_cloud_from_validated_depth

        build_cloud = build_supported_depth_cloud_from_validated_depth
    if fit_plane is None:
        fit_plane = estimate_floor_plane
    if map_holes is None:
        map_holes = build_floor_hole_map
    if make_seeds is None:
        make_seeds = generate_floor_seed_artifact
    if train is None:
        train = run_floor_recovery_training
    legacy = next(
        (checkpoint for checkpoint in legacy_checkpoints if checkpoint.iteration == 5_000),
        None,
    )
    if legacy is None:
        raise RuntimeError("legacy reference is missing its 5K checkpoint")
    floor_artifact = None
    try:
        frames = tuple(artifacts.photometric.original_frames)
        cloud = build_cloud(
            frames,
            artifacts.depth,
            artifacts.masks,
            accepted_model,
            policy=artifacts.depth.policy,
            mask_mode="motion_sky",
        )
        plane = fit_plane(sparse_xyz, cloud.camera_centers, seed=1701)
        holes = map_holes(sparse_xyz, cloud, plane)
        floor_artifact = make_seeds(
            sparse_xyz,
            cloud,
            plane,
            holes,
            root / "floor-artifact",
        )
    except ValueError as exc:
        message = str(exc)
        if "floor plane" in message or "reliable floor" in message:
            reason = "unreliable_floor_plane"
        elif "no_recoverable_floor_hole" in message:
            reason = "no_recoverable_floor_hole"
        else:
            raise
        decision = FloorRecoveryDecision(False, reason, (reason,))
        _prepare_report(
            report_root,
            staged=staged,
            legacy=legacy,
            training=None,
            decision=decision,
            floor_artifact_root=None,
            reference_root=reference_root,
        )
        published = publish(
            local_report_root=report_root,
            destination=destination,
            run_id=run_id,
            decision=decision,
            replace_owned_result=replace_owned_result,
        )
        return FloorRecoveryRunResult(
            run_id=run_id,
            status="rejected",
            final_path=Path(published),
            local_root=root,
            decision=decision,
        )

    from .ablation_staging import materialize_experiment_workspace

    workspace = materialize_experiment_workspace(
        staged,
        experiments_root=root / "experiments",
        experiment_id="floor_recovery",
    )
    prepared = PreparedTrainingInput(
        run_id="floor_recovery",
        data_root=workspace.scene_root,
        scene_name="scene",
        frames_dir=reconstruction.frames_dir,
        reconstruction=reconstruction.bundle,
        source_digest=staged.source_digest,
        selection_digest=reconstruction.selected_manifest.image_set_digest,
    )
    training = train(
        prepared,
        base_spec,
        reconstruction,
        floor_artifact,
        diagnostic_root=checkpoints_root,
        seed=1701,
        control_checkpoints={row.iteration: row for row in legacy_checkpoints},
    )
    candidate = training.checkpoints[-1]
    initial_floor = training.floor_checkpoints[0]
    final_floor = training.floor_checkpoints[-1]
    decision = decide_floor_recovery(
        legacy=legacy,
        candidate=candidate,
        initial_floor=initial_floor,
        final_floor=final_floor,
    )
    _prepare_report(
        report_root,
        staged=staged,
        legacy=legacy,
        training=training,
        decision=decision,
        floor_artifact_root=Path(floor_artifact.npz_path).parent,
        reference_root=reference_root,
    )
    published = publish(
        local_report_root=report_root,
        destination=destination,
        run_id=run_id,
        decision=decision,
        replace_owned_result=replace_owned_result,
    )
    return FloorRecoveryRunResult(
        run_id=run_id,
        status="success" if decision.passed else "rejected",
        final_path=Path(published),
        local_root=root,
        decision=decision,
    )


def run_floor_recovery_diagnostic(
    spec: FloorRecoveryRunSpec,
    *,
    model_manifest_path: Path,
    expected_source_revision: str,
    actual_source_revision: str | None = None,
    drive_root: Path = Path("/content/drive/MyDrive"),
    work_root: Path = Path("/content/4dgs-floor-recovery"),
    discover_source: Callable[[Path], object] | None = None,
    inspect_hardware: Callable[[object], object] | None = None,
    store_factory: Callable[[Path, str], object] | None = None,
    stage_inputs: Callable[..., StagedAblationInputs] | None = None,
    exact_audit_validator: Callable[[Path, object, object], bool] | None = None,
    stage_reference: Callable[..., tuple[AblationCheckpoint, ...]] = (
        stage_legacy_reference
    ),
    execute_staged: Callable[..., FloorRecoveryRunResult] = (
        run_floor_recovery_from_staged
    ),
) -> FloorRecoveryRunResult:
    """Restore once, compare one 5K floor arm, and never start a full run."""

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
    from .runner import (
        _CPU_AUDIT_PYTHON_VERSION,
        _CPU_AUDIT_SELECTION_PRODUCER_SHA256,
        _REPOSITORY_ROOT,
        _pretraining_cache_fingerprint,
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

    runtime_paths = NotebookRuntimePaths(
        drive_root=mounted_drive,
        work_root=Path(work_root).resolve(strict=False),
    )
    if inspect_hardware is None:
        from backend.static_pipeline.runner import _inspect_hardware

        inspect_hardware = _inspect_hardware
    hardware = inspect_hardware(runtime_paths)
    validate_floor_recovery_hardware(hardware)

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
    local_root = Path(work_root).resolve(strict=False) / run_id
    local_root.mkdir(parents=True, exist_ok=False)
    audit_exact = exact_audit_validator or _default_exact_audit_validator

    def require_any_audit(active_inventory: object) -> None:
        audits = cache_root / "audits"
        if not audits.is_dir() or not any(audits.glob("*/_SUCCESS.json")):
            raise RuntimeError(
                "a verified CPU cache audit is required before floor recovery"
            )
        if getattr(active_inventory, "digest", None) != getattr(
            inventory, "digest", None
        ):
            raise ValueError("audit inventory differs from the active input")

    def validate_selection(selection: object) -> None:
        if not audit_exact(cache_root, selection, inventory):
            raise RuntimeError(
                "a matching verified CPU track-audit receipt is required before "
                "floor recovery"
            )

    def validate_restored(selection: object, _reconstruction: object) -> None:
        validate_selection(selection)

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
        )

    stage = stage_inputs or stage_ablation_inputs
    staged = stage(
        store=store,
        source_inventory=inventory,
        destination=local_root / "inputs",
        drive_root=mounted_drive,
        expected_fingerprint=lambda selection: _pretraining_cache_fingerprint(
            inventory,
            selection,
            base_spec,
            hardware,
            manifest_path,
        ),
        expected_source_revision=expected_source_revision,
        actual_source_revision=revision,
        audit_validator=require_any_audit,
        restored_validator=validate_restored,
        minimum_free_bytes=35 * 1024**3,
        run_id=run_id,
        freeze=True,
        restore_pretraining=restore_graph,
    )
    reference_root = local_root / "legacy-reference"
    legacy_checkpoints = stage_reference(
        input_path.with_name(f"{input_path.name}_training_ablation"),
        reference_root,
        staged=staged,
    )
    destination = validate_result_target(
        input_path.with_name(f"{input_path.name}{RESULT_SUFFIX}")
    )
    return execute_staged(
        staged,
        base_spec=base_spec,
        local_root=local_root,
        reference_root=reference_root,
        legacy_checkpoints=legacy_checkpoints,
        destination=destination,
        run_id=run_id,
        replace_owned_result=spec.publish.replace_owned_result,
    )
