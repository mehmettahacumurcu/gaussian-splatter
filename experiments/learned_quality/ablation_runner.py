from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from .ablation import (
    AblationCheckpoint,
    AblationDiagnosis,
    AblationVariant,
    ExperimentOutcome,
    StructuralMetrics,
    classify_experiments,
    pairwise_variants,
    primary_variants,
)
from .ablation_staging import (
    ExperimentWorkspace,
    StagedAblationInputs,
    materialize_experiment_workspace,
)

if TYPE_CHECKING:
    from backend.notebooks.models import StaticNotebookRunSpec


GENERATOR_ID = "4dgs-studio.learned-training-ablation"


@dataclass(frozen=True)
class AblationExperimentRecord:
    experiment_id: str
    passed: bool
    stopped_early: bool
    stop_reasons: tuple[str, ...]
    checkpoints: tuple[AblationCheckpoint, ...]
    status: Mapping[str, object]


@dataclass(frozen=True)
class AblationMatrixResult:
    results: Mapping[str, object]
    errors: Mapping[str, str]
    diagnosis: AblationDiagnosis
    required_experiment_ids: tuple[str, ...]

    @classmethod
    def from_results(
        cls,
        results: Mapping[str, object],
        *,
        errors: Mapping[str, str] | None = None,
        required_experiment_ids: tuple[str, ...] | None = None,
    ) -> "AblationMatrixResult":
        normalized_errors = dict(errors or {})
        outcomes = {
            experiment_id: ExperimentOutcome(
                experiment_id=experiment_id,
                passed=bool(getattr(result, "passed")),
            )
            for experiment_id, result in results.items()
            if experiment_id not in normalized_errors
        }
        required = required_experiment_ids or tuple(
            variant.experiment_id for variant in primary_variants()
        )
        return cls(
            results=dict(results),
            errors=normalized_errors,
            diagnosis=classify_experiments(outcomes),
            required_experiment_ids=required,
        )

    @property
    def complete(self) -> bool:
        required = set(self.required_experiment_ids)
        return required.issubset(self.results) and not required.intersection(
            self.errors
        )


ExperimentExecutor = Callable[
    [
        StagedAblationInputs,
        AblationVariant,
        ExperimentWorkspace,
        "StaticNotebookRunSpec",
        Mapping[int, AblationCheckpoint],
    ],
    object,
]


def _progress_payload(result: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": str(getattr(result, "experiment_id")),
        "passed": bool(getattr(result, "passed")),
        "stopped_early": bool(getattr(result, "stopped_early")),
        "stop_reasons": list(getattr(result, "stop_reasons")),
    }


def run_ablation_matrix(
    staged: StagedAblationInputs,
    *,
    base_spec: StaticNotebookRunSpec,
    experiments_root: Path,
    execute_experiment: ExperimentExecutor,
    publish_progress: Callable[[dict[str, object]], None] | None = None,
) -> AblationMatrixResult:
    """Run a deterministic matrix while treating infrastructure errors separately."""

    results: dict[str, object] = {}
    errors: dict[str, str] = {}
    required = [variant.experiment_id for variant in primary_variants()]
    control_checkpoints: dict[int, AblationCheckpoint] = {}

    def run_variant(variant: AblationVariant) -> None:
        print(
            json.dumps(
                {
                    "event": "ablation_experiment_start",
                    "experiment_id": variant.experiment_id,
                    "iterations": variant.n_iterations,
                    "features": sorted(variant.features),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        workspace = materialize_experiment_workspace(
            staged,
            experiments_root=experiments_root,
            experiment_id=variant.experiment_id,
        )
        try:
            result = execute_experiment(
                staged,
                variant,
                workspace,
                base_spec,
                control_checkpoints,
            )
        except BaseException as error:
            errors[variant.experiment_id] = f"{type(error).__name__}: {error}"
            payload = {
                "schema_version": 1,
                "experiment_id": variant.experiment_id,
                "error": errors[variant.experiment_id],
            }
            if publish_progress is not None:
                publish_progress(payload)
            print(
                json.dumps({"event": "ablation_experiment_error", **payload}),
                flush=True,
            )
            return
        if str(getattr(result, "experiment_id")) != variant.experiment_id:
            raise ValueError("experiment result identifier differs from its variant")
        results[variant.experiment_id] = result
        if variant.experiment_id == "legacy_control":
            control_checkpoints.update(
                {
                    checkpoint.iteration: checkpoint
                    for checkpoint in getattr(result, "checkpoints")
                }
            )
        payload = _progress_payload(result)
        if publish_progress is not None:
            publish_progress(payload)
        print(
            json.dumps({"event": "ablation_experiment_done", **payload}),
            flush=True,
        )

    for variant in primary_variants():
        run_variant(variant)

    primary_matrix = AblationMatrixResult.from_results(
        results,
        errors=errors,
        required_experiment_ids=tuple(required),
    )
    if primary_matrix.diagnosis.requires_pairwise and not errors:
        pairs = pairwise_variants()
        required.extend(variant.experiment_id for variant in pairs)
        for variant in pairs:
            run_variant(variant)

    return AblationMatrixResult.from_results(
        results,
        errors=errors,
        required_experiment_ids=tuple(required),
    )


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
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
    os.replace(temporary, path)
    return path


def _experiment_payload(result: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": str(getattr(result, "experiment_id")),
        "passed": bool(getattr(result, "passed")),
        "stopped_early": bool(getattr(result, "stopped_early")),
        "stop_reasons": list(getattr(result, "stop_reasons")),
        "checkpoints": _jsonable(tuple(getattr(result, "checkpoints"))),
        "status": _jsonable(getattr(result, "status")),
    }


def _checkpoint_from_payload(payload: Mapping[str, object]) -> AblationCheckpoint:
    structural = StructuralMetrics(**dict(payload["structural"]))
    return AblationCheckpoint(
        experiment_id=str(payload["experiment_id"]),
        iteration=int(payload["iteration"]),
        psnr_unmasked=float(payload["psnr_unmasked"]),
        psnr_masked=float(payload["psnr_masked"]),
        ssim_unmasked=float(payload["ssim_unmasked"]),
        ssim_masked=float(payload["ssim_masked"]),
        l1_unmasked=float(payload["l1_unmasked"]),
        l1_masked=float(payload["l1_masked"]),
        gaussian_count=int(payload["gaussian_count"]),
        structural=structural,
    )


def _record_from_payload(payload: Mapping[str, object]) -> AblationExperimentRecord:
    return AblationExperimentRecord(
        experiment_id=str(payload["experiment_id"]),
        passed=bool(payload["passed"]),
        stopped_early=bool(payload["stopped_early"]),
        stop_reasons=tuple(map(str, payload.get("stop_reasons", ()))),
        checkpoints=tuple(
            _checkpoint_from_payload(row)
            for row in payload.get("checkpoints", ())
        ),
        status=dict(payload.get("status", {})),
    )


def _child_experiment_entry(
    staged: StagedAblationInputs,
    variant: AblationVariant,
    workspace: ExperimentWorkspace,
    base_spec: object,
    control_checkpoints: Mapping[int, AblationCheckpoint],
    artifact_root: Path,
    seed: int,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=False)
    receipt_path = artifact_root / "receipt.json"
    try:
        from backend.static_pipeline.training import PreparedTrainingInput

        from .ablation_training import run_ablation_experiment

        reconstruction = staged.reconstruction
        prepared = PreparedTrainingInput(
            run_id=variant.experiment_id,
            data_root=workspace.scene_root,
            scene_name="scene",
            frames_dir=reconstruction.frames_dir,
            reconstruction=reconstruction.bundle,
            source_digest=staged.source_digest,
            selection_digest=reconstruction.selected_manifest.image_set_digest,
        )
        result = run_ablation_experiment(
            prepared,
            base_spec,
            reconstruction,
            variant,
            diagnostic_root=artifact_root,
            seed=seed,
            control_checkpoints=control_checkpoints,
        )
        if result.metrics_path is not None and result.metrics_path.is_file():
            shutil.copy2(result.metrics_path, artifact_root / "metrics.jsonl")
        if result.run_manifest_path is not None and result.run_manifest_path.is_file():
            shutil.copy2(result.run_manifest_path, artifact_root / "run_manifest.json")
        payload = _experiment_payload(result)
        _write_json(receipt_path, payload)
        if result.raw_ply_path is not None:
            Path(result.raw_ply_path).unlink(missing_ok=True)
    except BaseException as error:
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "experiment_id": variant.experiment_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


class ForkedExperimentExecutor:
    """Launch every CUDA experiment in a clean forked child process."""

    def __init__(self, *, report_root: Path, seed: int = 1701) -> None:
        self.report_root = Path(report_root)
        self.seed = int(seed)

    def __call__(
        self,
        staged: StagedAblationInputs,
        variant: AblationVariant,
        workspace: ExperimentWorkspace,
        base_spec: StaticNotebookRunSpec,
        control_checkpoints: Mapping[int, AblationCheckpoint],
    ) -> AblationExperimentRecord:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("training ablation requires a Linux fork runtime")
        artifact_root = self.report_root / "experiments" / variant.experiment_id
        context = multiprocessing.get_context("fork")
        process = context.Process(
            target=_child_experiment_entry,
            args=(
                staged,
                variant,
                workspace,
                base_spec,
                dict(control_checkpoints),
                artifact_root,
                self.seed,
            ),
            daemon=False,
        )
        process.start()
        process.join()
        receipt_path = artifact_root / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(
                f"experiment child ended without a receipt (exit={process.exitcode})"
            )
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        shutil.rmtree(workspace.root, ignore_errors=True)
        if process.exitcode != 0 or "error_type" in payload:
            raise RuntimeError(
                f"{payload.get('error_type', 'ChildProcessError')}: "
                f"{payload.get('error_message', 'experiment child failed')}"
            )
        record = _record_from_payload(payload)
        return record


def _write_metrics_csv(path: Path, matrix: AblationMatrixResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "experiment_id",
                "iteration",
                "passed",
                "psnr_unmasked",
                "psnr_masked",
                "ssim_unmasked",
                "ssim_masked",
                "l1_unmasked",
                "l1_masked",
                "gaussian_count",
                "visible_white_fraction",
                "high_anisotropy_fraction",
                "oversized_fraction",
                "nonfinite_count",
            )
        )
        for experiment_id, result in matrix.results.items():
            for checkpoint in getattr(result, "checkpoints"):
                structural = checkpoint.structural
                writer.writerow(
                    (
                        experiment_id,
                        checkpoint.iteration,
                        bool(getattr(result, "passed")),
                        checkpoint.psnr_unmasked,
                        checkpoint.psnr_masked,
                        checkpoint.ssim_unmasked,
                        checkpoint.ssim_masked,
                        checkpoint.l1_unmasked,
                        checkpoint.l1_masked,
                        checkpoint.gaussian_count,
                        structural.visible_white_fraction,
                        structural.high_anisotropy_fraction,
                        structural.oversized_fraction,
                        structural.nonfinite_count,
                    )
                )


def _write_psnr_plot(path: Path, matrix: AblationMatrixResult) -> None:
    width, height = 1_200, 720
    margin = 70
    image = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#64748b")
    colors = (
        "#22c55e",
        "#38bdf8",
        "#f59e0b",
        "#e879f9",
        "#f43f5e",
        "#a3e635",
        "#fb7185",
        "#c084fc",
        "#2dd4bf",
        "#facc15",
        "#60a5fa",
        "#fb923c",
    )
    all_checkpoints = [
        checkpoint
        for result in matrix.results.values()
        for checkpoint in getattr(result, "checkpoints")
    ]
    max_iteration = max((row.iteration for row in all_checkpoints), default=5_000)
    max_psnr = max((row.psnr_unmasked for row in all_checkpoints), default=30.0)
    max_psnr = max(20.0, max_psnr)
    for index, (experiment_id, result) in enumerate(matrix.results.items()):
        rows = tuple(getattr(result, "checkpoints"))
        points = [
            (
                margin + row.iteration / max_iteration * (width - 2 * margin),
                height
                - margin
                - row.psnr_unmasked / max_psnr * (height - 2 * margin),
            )
            for row in rows
        ]
        color = colors[index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((margin + 10, margin + 20 * index), experiment_id, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_summary(path: Path, matrix: AblationMatrixResult) -> None:
    diagnosis = matrix.diagnosis
    lines = [
        "# Learned training ablation",
        "",
        f"Diagnosis: `{diagnosis.kind}`",
        f"Complete required matrix: `{str(matrix.complete).lower()}`",
        "",
        "| Experiment | Passed | Stop reason | Final PSNR | Gaussians |",
        "|---|---:|---|---:|---:|",
    ]
    for experiment_id in matrix.required_experiment_ids:
        result = matrix.results.get(experiment_id)
        if result is None:
            lines.append(
                f"| {experiment_id} | error | {matrix.errors.get(experiment_id, 'missing')} | - | - |"
            )
            continue
        checkpoints = tuple(getattr(result, "checkpoints"))
        final = checkpoints[-1] if checkpoints else None
        reasons = ", ".join(getattr(result, "stop_reasons")) or "-"
        lines.append(
            f"| {experiment_id} | {getattr(result, 'passed')} | {reasons} | "
            f"{final.psnr_unmasked if final else '-'} | "
            f"{final.gaussian_count if final else '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _copy_report_tree(source: Path, destination: Path) -> None:
    forbidden = {".ply", ".pt", ".pth", ".ckpt"}
    for path in sorted(source.rglob("*")):
        if path.is_dir() or path.suffix.lower() in forbidden:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _prepare_report(
    matrix: AblationMatrixResult,
    *,
    staged: StagedAblationInputs,
    local_report_root: Path,
    run_id: str,
    environment: Mapping[str, object],
) -> None:
    root = Path(local_report_root)
    root.mkdir(parents=True, exist_ok=True)
    for experiment_id, result in matrix.results.items():
        receipt = root / "experiments" / experiment_id / "receipt.json"
        _write_json(receipt, _experiment_payload(result))
    _write_json(
        root / "ablation_report.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "complete": matrix.complete,
            "diagnosis": matrix.diagnosis,
            "required_experiment_ids": matrix.required_experiment_ids,
            "errors": matrix.errors,
            "staging": {
                "source_digest": staged.source_digest,
                "pretraining_fingerprint": staged.pretraining_fingerprint,
                "source_revision": staged.source_revision,
            },
        },
    )
    _write_json(root / "environment.json", dict(environment))
    shutil.copy2(staged.manifest_path, root / "staging_manifest.json")
    _write_metrics_csv(root / "metrics.csv", matrix)
    _write_summary(root / "ablation_summary.md", matrix)
    _write_psnr_plot(root / "psnr_plot.png", matrix)


def publish_ablation_report(
    matrix: AblationMatrixResult,
    *,
    staged: StagedAblationInputs,
    local_report_root: Path,
    destination: Path,
    run_id: str,
    environment: Mapping[str, object],
    replace_owned_result: bool = True,
) -> Path:
    """Publish compact diagnostics and make the completion marker the last write."""

    _prepare_report(
        matrix,
        staged=staged,
        local_report_root=local_report_root,
        run_id=run_id,
        environment=environment,
    )
    target = Path(destination)
    ownership = target / "_OWNERSHIP.json"
    if os.path.lexists(target):
        if not replace_owned_result:
            raise FileExistsError(f"ablation output already exists: {target}")
        try:
            owner = json.loads(ownership.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("refusing to replace an unowned ablation output") from error
        if owner.get("generator_id") != GENERATOR_ID:
            raise RuntimeError("refusing to replace an unowned ablation output")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    _write_json(
        ownership,
        {"schema_version": 1, "generator_id": GENERATOR_ID, "run_id": run_id},
    )
    _copy_report_tree(Path(local_report_root), target)
    marker_name = "_SUCCESS.json" if matrix.complete else "_PARTIAL.json"
    marker = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "success" if matrix.complete else "partial",
        "meaning": "diagnostic_matrix_published",
        "diagnosis": matrix.diagnosis.kind,
    }
    _write_json(target / marker_name, marker)
    return target
