from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.learned_quality.ablation import (
    AblationCheckpoint,
    StructuralMetrics,
    pairwise_variants,
    primary_variants,
)
from experiments.learned_quality.ablation_runner import (
    AblationMatrixResult,
    AblationRunSpec,
    publish_ablation_report,
    run_ablation_matrix,
    run_training_ablation,
)
from experiments.learned_quality.ablation_staging import StagedAblationInputs
from experiments.learned_quality.ablation_training import AblationExperimentResult
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    to_static_run_spec,
)


def _staged(tmp_path: Path) -> StagedAblationInputs:
    root = tmp_path / "inputs"
    root.mkdir()
    manifest = root / "ablation_staging_manifest.json"
    manifest.write_text('{"schema_version":1}', encoding="utf-8")
    return StagedAblationInputs(
        root=root,
        selection=SimpleNamespace(),
        reconstruction=SimpleNamespace(),
        source_digest="a" * 64,
        pretraining_fingerprint="b" * 64,
        source_revision="c" * 40,
        manifest_path=manifest,
    )


def _result(experiment_id: str, *, passed: bool) -> AblationExperimentResult:
    checkpoint = AblationCheckpoint(
        experiment_id=experiment_id,
        iteration=5_000,
        psnr_unmasked=20.0 if passed else 5.0,
        psnr_masked=21.0 if passed else 6.0,
        ssim_unmasked=0.8,
        ssim_masked=0.9,
        l1_unmasked=0.1,
        l1_masked=0.05,
        gaussian_count=100,
        structural=StructuralMetrics(0.0, 0.0, 0.0, 0),
    )
    return AblationExperimentResult(
        experiment_id=experiment_id,
        passed=passed,
        stopped_early=not passed,
        stop_reasons=() if passed else ("psnr_deficit",),
        checkpoints=(checkpoint,),
        raw_ply_path=None,
        metrics_path=None,
        run_manifest_path=None,
        status={"test": True},
    )


def test_matrix_runs_primary_sequentially_and_reports_tiny_progress(
    tmp_path: Path,
) -> None:
    staged = _staged(tmp_path)
    calls: list[tuple[str, Path]] = []
    progress: list[dict[str, object]] = []

    def execute(_staged, variant, workspace, _base_spec, _control):
        assert _staged is staged
        calls.append((variant.experiment_id, workspace.inputs_root))
        return _result(variant.experiment_id, passed=True)

    matrix = run_ablation_matrix(
        staged,
        base_spec=to_static_run_spec(
            LearnedQualityRunSpec(input_folder="myroom_test")
        ),
        experiments_root=tmp_path / "experiments",
        execute_experiment=execute,
        publish_progress=progress.append,
    )

    assert [name for name, _root in calls] == [
        variant.experiment_id for variant in primary_variants()
    ]
    assert all(root == staged.root for _name, root in calls)
    assert len(progress) == len(primary_variants())
    assert all("checkpoints" not in row for row in progress)
    assert matrix.errors == {}
    assert matrix.diagnosis.kind == "inconclusive"


def test_pairwise_runs_only_when_full_failure_has_no_isolated_cause(
    tmp_path: Path,
) -> None:
    staged = _staged(tmp_path)
    calls: list[str] = []
    failing_pair = pairwise_variants()[0].experiment_id

    def execute(_staged, variant, _workspace, _base_spec, _control):
        calls.append(variant.experiment_id)
        passed = variant.experiment_id not in {"full_learned", failing_pair}
        return _result(variant.experiment_id, passed=passed)

    matrix = run_ablation_matrix(
        staged,
        base_spec=to_static_run_spec(
            LearnedQualityRunSpec(input_folder="myroom_test")
        ),
        experiments_root=tmp_path / "experiments",
        execute_experiment=execute,
    )

    assert calls == [
        *(variant.experiment_id for variant in primary_variants()),
        *(variant.experiment_id for variant in pairwise_variants()),
    ]
    assert matrix.diagnosis.kind == "interaction_cause"
    assert matrix.diagnosis.causes == (
        tuple(pairwise_variants()[0].features),
    ) or set(matrix.diagnosis.causes[0]) == set(pairwise_variants()[0].features)


def test_isolated_failure_skips_pairwise_matrix(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    calls: list[str] = []

    def execute(_staged, variant, _workspace, _base_spec, _control):
        calls.append(variant.experiment_id)
        return _result(
            variant.experiment_id,
            passed=variant.experiment_id not in {"masks_only", "full_learned"},
        )

    matrix = run_ablation_matrix(
        staged,
        base_spec=to_static_run_spec(
            LearnedQualityRunSpec(input_folder="myroom_test")
        ),
        experiments_root=tmp_path / "experiments",
        execute_experiment=execute,
    )

    assert len(calls) == len(primary_variants())
    assert matrix.diagnosis.kind == "isolated_cause"
    assert matrix.diagnosis.causes == (("masks",),)


def test_report_publication_writes_success_last_and_never_copies_ply(
    tmp_path: Path,
) -> None:
    staged = _staged(tmp_path)
    report_root = tmp_path / "local-report"
    contact = report_root / "experiments" / "legacy_control" / "contact_005000.png"
    contact.parent.mkdir(parents=True)
    contact.write_bytes(b"png")
    (report_root / "do-not-publish.ply").write_bytes(b"large")
    results = {
        variant.experiment_id: _result(variant.experiment_id, passed=True)
        for variant in primary_variants()
    }
    matrix = AblationMatrixResult.from_results(results)
    destination = tmp_path / "drive" / "myroom_test_training_ablation"

    published = publish_ablation_report(
        matrix,
        staged=staged,
        local_report_root=report_root,
        destination=destination,
        run_id="run-1",
        environment={"gpu": "A100"},
    )

    assert published == destination
    for relative in (
        "ablation_report.json",
        "ablation_summary.md",
        "metrics.csv",
        "environment.json",
        "staging_manifest.json",
        "psnr_plot.png",
        "experiments/legacy_control/receipt.json",
        "experiments/legacy_control/contact_005000.png",
        "_OWNERSHIP.json",
        "_SUCCESS.json",
    ):
        assert (destination / relative).is_file(), relative
    assert not tuple(destination.rglob("*.ply"))
    success = json.loads((destination / "_SUCCESS.json").read_text())
    assert success["meaning"] == "diagnostic_matrix_published"


def test_partial_matrix_publishes_receipts_without_success_marker(
    tmp_path: Path,
) -> None:
    staged = _staged(tmp_path)
    matrix = AblationMatrixResult(
        results={"legacy_control": _result("legacy_control", passed=True)},
        errors={"dense_seeds_only": "child failed"},
        diagnosis=SimpleNamespace(kind="inconclusive", causes=()),
        required_experiment_ids=tuple(
            variant.experiment_id for variant in primary_variants()
        ),
    )
    destination = tmp_path / "drive" / "myroom_test_training_ablation"

    publish_ablation_report(
        matrix,
        staged=staged,
        local_report_root=tmp_path / "local-report",
        destination=destination,
        run_id="run-2",
        environment={},
    )

    assert (destination / "_PARTIAL.json").is_file()
    assert not (destination / "_SUCCESS.json").exists()


def test_run_training_ablation_stages_once_and_uses_the_dedicated_output(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    input_path = drive_root / "myroom_test"
    input_path.mkdir(parents=True)
    audit = (
        drive_root
        / "myroom_test_learned_test_cache"
        / "audits"
        / ("e" * 64)
        / "_SUCCESS.json"
    )
    audit.parent.mkdir(parents=True)
    audit.write_text("{}", encoding="utf-8")
    work_root = tmp_path / "work"
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")
    inventory = SimpleNamespace(digest="a" * 64)
    hardware = SimpleNamespace(
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        disk_free_gb=200.0,
        colmap_gpu_sift=True,
    )
    staged = _staged(tmp_path)
    stage_calls: list[dict[str, object]] = []
    publish_calls: list[dict[str, object]] = []

    def stage_once(**kwargs):
        stage_calls.append(kwargs)
        kwargs["audit_validator"](inventory)
        kwargs["restored_validator"](
            SimpleNamespace(manifest=SimpleNamespace(image_set_digest="d" * 64)),
            SimpleNamespace(),
        )
        return staged

    def execute(_staged, variant, _workspace, _base_spec, _control):
        return _result(variant.experiment_id, passed=True)

    def publish(matrix, **kwargs):
        publish_calls.append({"matrix": matrix, **kwargs})
        kwargs["destination"].mkdir(parents=True)
        return kwargs["destination"]

    result = run_training_ablation(
        AblationRunSpec(input_folder="myroom_test"),
        model_manifest_path=model_manifest,
        expected_source_revision="c" * 40,
        actual_source_revision="c" * 40,
        drive_root=drive_root,
        work_root=work_root,
        discover_source=lambda path: inventory if path == input_path else None,
        inspect_hardware=lambda _paths: hardware,
        store_factory=lambda _root, _digest: SimpleNamespace(),
        stage_inputs=stage_once,
        exact_audit_validator=lambda _cache, selection, _inventory: (
            selection.manifest.image_set_digest == "d" * 64
        ),
        execute_experiment=execute,
        publish_report=publish,
    )

    assert len(stage_calls) == 1
    assert stage_calls[0]["destination"].name == "inputs"
    assert callable(stage_calls[0]["restore_pretraining"])
    assert len(publish_calls) == 1
    assert publish_calls[0]["destination"] == drive_root / (
        "myroom_test_training_ablation"
    )
    assert result.complete is True
    assert result.final_path == drive_root / "myroom_test_training_ablation"
