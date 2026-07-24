from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from backend.notebooks.drive_paths import normalize_input_folder
from backend.notebooks.models import StrictModel
from backend.static_pipeline.contracts import PolishReport, ReconstructionBundle

from .ablation import PRIMARY_CHECKPOINTS, AblationCheckpoint, AblationVariant, primary_variants
from .ablation_staging import StagedAblationInputs
from .ablation_training import AblationExperimentResult
from .contracts import LearnedReconstructionOutput


RESULT_SUFFIX = "_legacy_control_5k_result"
GENERATOR_ID = "4dgs-studio.legacy-control-5k-dual-ply-v1"
RAW_PLY_NAME = "raw_legacy_control_5k.ply"
POLISHED_PLY_NAME = "polished_legacy_control_5k.ply"
REQUIRED_RESULT_FILES = (
    RAW_PLY_NAME,
    POLISHED_PLY_NAME,
    "polish_report.json",
    "receipt.json",
    "metrics.jsonl",
    "run_manifest.json",
    "contact_005000.png",
    "contact_005000_fixed.png",
    "contact_005000_perturbed.png",
    "provenance.json",
)


class LegacyControlExportPublishSpec(StrictModel):
    replace_owned_result: bool = True


class LegacyControlExportRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    runtime_profile: Literal["a100_legacy_control_5k"] = "a100_legacy_control_5k"
    publish: LegacyControlExportPublishSpec = Field(
        default_factory=LegacyControlExportPublishSpec
    )

    @field_validator("input_folder", mode="before")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("input_folder must be a string")
        canonical = normalize_input_folder(value.strip())
        forbidden = (
            RESULT_SUFFIX,
            "_learned_test_result",
            "_training_ablation",
            "_floor_recovery_diagnostic",
            "_learned_test_diagnostics",
        )
        if canonical.endswith(forbidden):
            raise ValueError("choose the input folder, not an experiment output")
        return canonical


@dataclass(frozen=True)
class LegacyControlExportResult:
    run_id: str
    final_path: Path
    local_root: Path
    raw_ply_path: Path
    polished_ply_path: Path
    polish_accepted: bool
    polish_reasons: tuple[str, ...]


def select_legacy_control_variant() -> AblationVariant:
    matches = tuple(
        variant
        for variant in primary_variants()
        if variant.experiment_id == "legacy_control"
    )
    if len(matches) != 1:
        raise RuntimeError("exactly one legacy_control variant is required")
    variant = matches[0]
    if (
        variant.features
        or variant.n_iterations != 5_000
        or not variant.density_events
    ):
        raise RuntimeError("legacy_control contract changed")
    return variant


def validate_legacy_control_hardware(hardware: object) -> None:
    gpu_name = str(getattr(hardware, "gpu_name", ""))
    vram_gb = float(getattr(hardware, "vram_gb", 0.0))
    disk_free_gb = float(getattr(hardware, "disk_free_gb", 0.0))
    raw_host_ram_gb = getattr(hardware, "host_ram_gb", None)
    host_ram_gb = float(raw_host_ram_gb) if raw_host_ram_gb is not None else 0.0
    if "A100" not in gpu_name.upper() or vram_gb < 75.0:
        raise RuntimeError(
            "legacy-control export requires an A100 with 75+ GiB VRAM; "
            f"got {gpu_name} ({vram_gb:g} GiB)"
        )
    if disk_free_gb < 40.0:
        raise RuntimeError(
            "legacy-control export requires at least 40 GiB local disk; "
            f"got {disk_free_gb:g} GiB"
        )
    if host_ram_gb < 100.0:
        raise RuntimeError(
            "legacy-control export requires a High-RAM runtime with at least "
            f"100 GiB host RAM; got {host_ram_gb:g} GiB"
        )


def diagnostic_polish_bundle(
    reconstruction: LearnedReconstructionOutput,
) -> ReconstructionBundle:
    original = reconstruction.bundle
    if original.decision.passed or not original.decision.failures:
        raise ValueError("diagnostic proxy requires rejected best-effort geometry")
    decision = replace(
        original.decision,
        passed=True,
        failures=(),
        retry_recommended=False,
    )
    return replace(original, decision=decision)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_result_target(path: Path) -> Path:
    target = Path(path).resolve(strict=False)
    if (
        not target.name.endswith(RESULT_SUFFIX)
        or target.name == RESULT_SUFFIX
        or target.parent == target
    ):
        raise ValueError("result target must be a legacy-control 5K result folder")
    return target


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def run_diagnostic_polish(
    training: AblationExperimentResult,
    reconstruction: LearnedReconstructionOutput,
    *,
    output_root: Path,
    camera_parser: Callable[[Path], Mapping[str, Mapping[str, object]]] | None = None,
    frame_joiner: Callable[..., Sequence[object]] | None = None,
    polisher: Callable[..., PolishReport] | None = None,
) -> PolishReport:
    """Run the production filter through an isolated best-effort geometry proxy."""

    if not training.passed or training.raw_ply_path is None:
        raise RuntimeError("legacy-control training must pass before polish")
    if camera_parser is None:
        from backend.preprocess.parse_colmap import parse_cameras_from_model

        camera_parser = parse_cameras_from_model
    if frame_joiner is None:
        from backend.preprocess.frame_alignment import join_registered_frames

        frame_joiner = join_registered_frames
    if polisher is None:
        from backend.static_pipeline.polish import polish_static_ply

        polisher = polish_static_ply

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    cameras = camera_parser(reconstruction.accepted_model_dir)
    registered = tuple(
        frame_joiner(
            reconstruction.frames_dir,
            cameras,
            manifest=reconstruction.selected_manifest,
        )
    )
    report = polisher(
        Path(training.raw_ply_path),
        root / "candidate.ply",
        diagnostic_polish_bundle(reconstruction),
        registered,
    )
    if report.candidate_path is None or not Path(report.candidate_path).is_file():
        raise RuntimeError("production polish did not produce a comparison candidate")
    return report


def _validate_checkpoints(
    checkpoints: Sequence[AblationCheckpoint],
    *,
    label: str,
) -> tuple[AblationCheckpoint, ...]:
    normalized = tuple(checkpoints)
    observed = tuple(row.iteration for row in normalized)
    if observed != PRIMARY_CHECKPOINTS:
        raise RuntimeError(
            f"{label} checkpoints differ from the 5K contract: "
            f"expected {PRIMARY_CHECKPOINTS}, observed {observed}"
        )
    if any(row.experiment_id != "legacy_control" for row in normalized):
        raise RuntimeError(f"{label} checkpoints are not legacy_control")
    return normalized


def prepare_legacy_control_report(
    report_root: Path,
    *,
    staged: object,
    training: AblationExperimentResult,
    polish: PolishReport,
    diagnostic_root: Path,
    historical_checkpoints: Sequence[AblationCheckpoint],
    ply_validator: Callable[[Path], object] | None = None,
) -> Path:
    """Build the complete local report without choosing between its two PLYs."""

    if not training.passed or training.stopped_early:
        raise RuntimeError("legacy-control training did not pass")
    if training.raw_ply_path is None:
        raise RuntimeError("legacy-control training did not return its raw PLY")
    if training.metrics_path is None or training.run_manifest_path is None:
        raise RuntimeError("legacy-control training evidence is incomplete")
    reproduced = _validate_checkpoints(training.checkpoints, label="reproduced")
    historical = _validate_checkpoints(
        historical_checkpoints,
        label="historical",
    )
    if polish.raw_path.resolve(strict=True) != Path(training.raw_ply_path).resolve(
        strict=True
    ):
        raise ValueError("polish report does not reference the training raw PLY")
    if polish.candidate_path is None:
        raise RuntimeError("polish report has no comparison candidate")
    candidate = Path(polish.candidate_path)
    if ply_validator is None:
        from backend.static_pipeline.polish import validate_static_ply

        ply_validator = validate_static_ply
    raw_validated = ply_validator(Path(training.raw_ply_path))
    candidate_validated = ply_validator(candidate)
    raw_count = int(getattr(raw_validated, "count"))
    candidate_count = int(getattr(candidate_validated, "count"))
    if raw_count != polish.original_count or candidate_count != polish.kept_count:
        raise RuntimeError("polish report counts differ from the validated PLYs")
    if _sha256_file(Path(training.raw_ply_path)) == _sha256_file(candidate):
        raise ValueError("polished candidate must differ from the raw PLY")

    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(training.raw_ply_path, root / RAW_PLY_NAME)
    shutil.copy2(candidate, root / POLISHED_PLY_NAME)
    shutil.copy2(training.metrics_path, root / "metrics.jsonl")
    shutil.copy2(training.run_manifest_path, root / "run_manifest.json")
    contacts = (
        "contact_005000.png",
        "contact_005000_fixed.png",
        "contact_005000_perturbed.png",
    )
    for name in contacts:
        source = Path(diagnostic_root) / name
        if not source.is_file():
            raise RuntimeError(f"legacy-control diagnostic is missing {name}")
        shutil.copy2(source, root / name)

    reconstruction = getattr(staged, "reconstruction")
    original_decision = reconstruction.bundle.decision
    acceptance = getattr(reconstruction, "acceptance", None)
    polish_payload = {
        "schema_version": 1,
        "algorithm": "production_polish_static_ply",
        "diagnostic_acceptance_proxy": True,
        "candidate_published_for_manual_comparison": True,
        "accepted": polish.accepted,
        "original_count": polish.original_count,
        "kept_count": polish.kept_count,
        "removed_count": polish.original_count - polish.kept_count,
        "removed_fraction": (
            (polish.original_count - polish.kept_count) / polish.original_count
        ),
        "opacity_mass_loss": polish.opacity_mass_loss,
        "reasons": list(polish.reasons),
        "render_metrics": _jsonable(polish.render_metrics),
        "original_geometry_passed": original_decision.passed,
        "original_geometry_failures": list(original_decision.failures),
        "original_geometry_warnings": list(original_decision.warnings),
        "geometry_acceptance": _jsonable(acceptance),
        "raw_ply": RAW_PLY_NAME,
        "polished_ply": POLISHED_PLY_NAME,
    }
    _write_json(root / "polish_report.json", polish_payload)
    _write_json(
        root / "receipt.json",
        {
            "schema_version": 1,
            "status": "success",
            "experiment_id": "legacy_control",
            "iterations": 5_000,
            "polish_accepted": polish.accepted,
            "polish_reasons": list(polish.reasons),
            "raw_ply": RAW_PLY_NAME,
            "polished_ply": POLISHED_PLY_NAME,
            "reproduced_checkpoints": _jsonable(reproduced),
            "historical_checkpoints": _jsonable(historical),
        },
    )
    _write_json(
        root / "provenance.json",
        {
            "schema_version": 1,
            "source_digest": getattr(staged, "source_digest"),
            "pretraining_fingerprint": getattr(staged, "pretraining_fingerprint"),
            "source_revision": getattr(staged, "source_revision"),
            "training_seed": 1701,
            "fixed_long_edge": 720,
            "diagnostic_acceptance_proxy": True,
            "original_geometry_failures": list(original_decision.failures),
        },
    )
    return root


def run_legacy_control_export_from_staged(
    staged: StagedAblationInputs,
    *,
    base_spec: object,
    local_root: Path,
    historical_checkpoints: Sequence[AblationCheckpoint],
    destination: Path,
    run_id: str,
    replace_owned_result: bool = True,
    materialize_workspace: Callable[..., object] | None = None,
    execute_experiment: Callable[..., AblationExperimentResult] | None = None,
    polish_runner: Callable[..., PolishReport] = run_diagnostic_polish,
    prepare_report: Callable[..., Path] = prepare_legacy_control_report,
    publish: Callable[..., Path] | None = None,
) -> LegacyControlExportResult:
    """Run exactly one 5K legacy control and retain both polish inputs."""

    from backend.static_pipeline.training import PreparedTrainingInput

    if materialize_workspace is None:
        from .ablation_staging import materialize_experiment_workspace

        materialize_workspace = materialize_experiment_workspace
    if execute_experiment is None:
        from .ablation_training import run_ablation_experiment

        execute_experiment = run_ablation_experiment
    if publish is None:
        publish = publish_legacy_control_export

    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    variant = select_legacy_control_variant()
    historical = _validate_checkpoints(
        historical_checkpoints,
        label="historical",
    )
    workspace = materialize_workspace(
        staged,
        experiments_root=root / "experiments",
        experiment_id=variant.experiment_id,
    )
    reconstruction = staged.reconstruction
    prepared = PreparedTrainingInput(
        run_id=variant.experiment_id,
        data_root=Path(getattr(workspace, "scene_root")),
        scene_name="scene",
        frames_dir=reconstruction.frames_dir,
        reconstruction=reconstruction.bundle,
        source_digest=staged.source_digest,
        selection_digest=reconstruction.selected_manifest.image_set_digest,
    )
    diagnostic_root = root / "diagnostics"
    training = execute_experiment(
        prepared,
        base_spec,
        reconstruction,
        variant,
        diagnostic_root=diagnostic_root,
        seed=1701,
        source_long_edge=720,
        control_checkpoints={row.iteration: row for row in historical},
    )
    if not training.passed or training.stopped_early:
        raise RuntimeError("legacy-control 5K training did not pass")
    _validate_checkpoints(training.checkpoints, label="reproduced")
    polish = polish_runner(
        training,
        reconstruction,
        output_root=root / "polish",
    )
    report_root = prepare_report(
        root / "report",
        staged=staged,
        training=training,
        polish=polish,
        diagnostic_root=diagnostic_root,
        historical_checkpoints=historical,
    )
    final_path = Path(
        publish(
            local_report_root=report_root,
            destination=destination,
            run_id=run_id,
            replace_owned_result=replace_owned_result,
        )
    )
    return LegacyControlExportResult(
        run_id=run_id,
        final_path=final_path,
        local_root=root,
        raw_ply_path=final_path / RAW_PLY_NAME,
        polished_ply_path=final_path / POLISHED_PLY_NAME,
        polish_accepted=polish.accepted,
        polish_reasons=polish.reasons,
    )


def publish_legacy_control_export(
    *,
    local_report_root: Path,
    destination: Path,
    run_id: str,
    replace_owned_result: bool = True,
) -> Path:
    """Atomically publish an isolated raw-versus-polished 5K comparison."""

    source = Path(local_report_root).resolve(strict=True)
    missing = tuple(name for name in REQUIRED_RESULT_FILES if not (source / name).is_file())
    if missing:
        raise RuntimeError(f"legacy-control export is incomplete: {missing}")
    raw = source / RAW_PLY_NAME
    polished = source / POLISHED_PLY_NAME
    if raw.stat().st_size <= 0 or polished.stat().st_size <= 0:
        raise ValueError("raw and polished PLYs must be non-empty")
    if _sha256_file(raw) == _sha256_file(polished):
        raise ValueError("polished candidate must differ from the raw PLY")

    target = _validate_result_target(destination)
    if target == source or target.is_relative_to(source):
        raise ValueError("legacy-control publication cannot target its local source")
    if os.path.lexists(target):
        if not replace_owned_result:
            raise FileExistsError(f"legacy-control export exists: {target}")
        try:
            owner = json.loads(
                (target / "_OWNERSHIP.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "refusing to replace an unowned legacy-control export"
            ) from exc
        if owner.get("generator_id") != GENERATOR_ID:
            raise RuntimeError("refusing to replace an unowned legacy-control export")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    backup: Path | None = None
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
        _write_json(
            staging / "_SUCCESS.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "success",
                "meaning": "legacy_control_5k_dual_ply_published",
                "raw_ply": RAW_PLY_NAME,
                "polished_ply": POLISHED_PLY_NAME,
            },
        )
        if os.path.lexists(target):
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            os.replace(target, backup)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if (
            backup is not None
            and os.path.lexists(backup)
            and not os.path.lexists(target)
        ):
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
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


def run_legacy_control_export(
    spec: LegacyControlExportRunSpec,
    *,
    model_manifest_path: Path,
    expected_source_revision: str,
    actual_source_revision: str | None = None,
    drive_root: Path = Path("/content/drive/MyDrive"),
    work_root: Path = Path("/content/4dgs-legacy-control-export"),
    discover_source: Callable[[Path], object] | None = None,
    inspect_hardware: Callable[[object], object] | None = None,
    store_factory: Callable[[Path, str], object] | None = None,
    stage_inputs: Callable[..., StagedAblationInputs] | None = None,
    exact_audit_validator: Callable[[Path, object, object], bool] | None = None,
    reference_fingerprint_resolver: Callable[..., str] | None = None,
    stage_reference: Callable[..., tuple[AblationCheckpoint, ...]] | None = None,
    execute_staged: Callable[..., LegacyControlExportResult] = (
        run_legacy_control_export_from_staged
    ),
) -> LegacyControlExportResult:
    """Restore once, reproduce one 5K control, then publish two PLYs."""

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
    from .floor_recovery_runner import (
        stage_legacy_reference,
        verified_legacy_reference_fingerprint,
    )
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

    runtime_paths = NotebookRuntimePaths(
        drive_root=mounted_drive,
        work_root=Path(work_root).resolve(strict=False),
    )
    if inspect_hardware is None:
        from backend.static_pipeline.runner import _inspect_hardware

        inspect_hardware = _inspect_hardware
    hardware = inspect_hardware(runtime_paths)
    validate_legacy_control_hardware(hardware)

    reference_source = input_path.with_name(f"{input_path.name}_training_ablation")
    fingerprint_resolver = (
        reference_fingerprint_resolver or verified_legacy_reference_fingerprint
    )
    reference_fingerprint = fingerprint_resolver(
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
    local_root = Path(work_root).resolve(strict=False) / run_id
    local_root.mkdir(parents=True, exist_ok=False)
    audit_exact = exact_audit_validator or _default_exact_audit_validator

    def require_any_audit(active_inventory: object) -> None:
        audits = cache_root / "audits"
        if not audits.is_dir() or not any(audits.glob("*/_SUCCESS.json")):
            raise RuntimeError(
                "a verified CPU cache audit is required before legacy-control export"
            )
        if getattr(active_inventory, "digest", None) != getattr(
            inventory, "digest", None
        ):
            raise ValueError("audit inventory differs from the active input")

    def validate_selection(selection: object) -> None:
        if not audit_exact(cache_root, selection, inventory):
            raise RuntimeError(
                "a matching verified CPU track-audit receipt is required before "
                "legacy-control export"
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
            final_pretraining_fingerprint=reference_fingerprint,
        )

    stage = stage_inputs or stage_ablation_inputs
    staged = stage(
        store=store,
        source_inventory=inventory,
        destination=local_root / "inputs",
        drive_root=mounted_drive,
        expected_fingerprint=lambda _selection: reference_fingerprint,
        expected_source_revision=expected_source_revision,
        actual_source_revision=revision,
        audit_validator=require_any_audit,
        restored_validator=lambda selection, _reconstruction: validate_selection(
            selection
        ),
        minimum_free_bytes=35 * 1024**3,
        run_id=run_id,
        freeze=True,
        restore_pretraining=restore_graph,
    )
    reference_root = local_root / "legacy-reference"
    reference_stager = stage_reference or stage_legacy_reference
    historical_checkpoints = reference_stager(
        reference_source,
        reference_root,
        staged=staged,
    )
    destination = _validate_result_target(
        input_path.with_name(f"{input_path.name}{RESULT_SUFFIX}")
    )
    return execute_staged(
        staged,
        base_spec=base_spec,
        local_root=local_root,
        historical_checkpoints=historical_checkpoints,
        destination=destination,
        run_id=run_id,
        replace_owned_result=spec.publish.replace_owned_result,
    )
