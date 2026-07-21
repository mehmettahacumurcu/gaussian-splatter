from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.learned_quality.ablation import (
    AblationCheckpoint,
    StructuralMetrics,
    primary_variants,
)
from experiments.learned_quality.ablation_runner import (
    AblationMatrixResult,
    AblationRunSpec,
    derive_structural_findings,
    publish_ablation_report,
    run_ablation_matrix,
    run_training_ablation,
    validate_ablation_hardware,
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


def _checkpoint(
    experiment_id: str,
    iteration: int,
    *,
    psnr: float = 20.0,
    edge_correlation: float = 0.8,
    alpha_coverage: float = 0.9,
    depth_coverage: float = 0.9,
    perturbed_alpha: float = 0.85,
    perturbed_depth: float = 0.85,
    gaussian_count: int = 100,
) -> AblationCheckpoint:
    return AblationCheckpoint(
        experiment_id=experiment_id,
        iteration=iteration,
        psnr_unmasked=psnr,
        psnr_masked=psnr,
        ssim_unmasked=0.8,
        ssim_masked=0.8,
        l1_unmasked=0.1,
        l1_masked=0.1,
        gaussian_count=gaussian_count,
        structural=StructuralMetrics(0.0, 0.0, 0.0, 0),
        edge_correlation=edge_correlation,
        alpha_coverage=alpha_coverage,
        depth_finite_fraction=depth_coverage,
        perturbed_alpha_coverage=perturbed_alpha,
        perturbed_depth_finite_fraction=perturbed_depth,
    )


def _record(
    experiment_id: str,
    checkpoints: tuple[AblationCheckpoint, ...],
) -> AblationExperimentResult:
    return AblationExperimentResult(
        experiment_id=experiment_id,
        passed=True,
        stopped_early=False,
        stop_reasons=(),
        checkpoints=checkpoints,
        raw_ply_path=None,
        metrics_path=None,
        run_manifest_path=None,
        status={"test": True},
    )


def test_structural_findings_separate_input_density_and_off_camera_failures() -> None:
    fixed = _record(
        "fixed_topology_control",
        (
            _checkpoint(
                "fixed_topology_control",
                5_000,
                psnr=8.0,
                edge_correlation=0.2,
            ),
        ),
    )
    legacy = _record(
        "legacy_control",
        (
            _checkpoint("legacy_control", 499, psnr=20.0, gaussian_count=100),
            _checkpoint(
                "legacy_control",
                600,
                psnr=14.0,
                edge_correlation=0.5,
                gaussian_count=250,
            ),
            _checkpoint(
                "legacy_control",
                5_000,
                alpha_coverage=0.9,
                depth_coverage=0.9,
                perturbed_alpha=0.4,
                perturbed_depth=0.5,
            ),
        ),
    )
    matrix = AblationMatrixResult.from_results(
        {
            "legacy_control": legacy,
            "fixed_topology_control": fixed,
        },
        required_experiment_ids=("legacy_control", "fixed_topology_control"),
    )

    findings = derive_structural_findings(matrix)

    assert [finding.kind for finding in findings] == [
        "input_geometry_failure",
        "density_transition_failure",
        "weak_3d_consistency",
    ]


@pytest.mark.parametrize(
    ("profile", "gpu_name", "vram_gb", "accepted"),
    [
        ("l4_diagnostic", "NVIDIA L4", 24.0, True),
        ("l4_diagnostic", "NVIDIA A100-SXM4-80GB", 80.0, True),
        ("l4_diagnostic", "NVIDIA T4", 16.0, False),
        ("l4_diagnostic", "NVIDIA L40S", 48.0, False),
        ("a100_reference", "NVIDIA L4", 24.0, False),
        ("a100_reference", "NVIDIA A100-SXM4-80GB", 80.0, True),
    ],
)
def test_runtime_profile_hardware_admission(
    profile: str,
    gpu_name: str,
    vram_gb: float,
    accepted: bool,
) -> None:
    hardware = SimpleNamespace(gpu_name=gpu_name, vram_gb=vram_gb)

    if accepted:
        validate_ablation_hardware(profile, hardware)
    else:
        with pytest.raises(RuntimeError):
            validate_ablation_hardware(profile, hardware)


def test_runtime_profile_is_strictly_parsed() -> None:
    spec = AblationRunSpec.model_validate(
        {"input_folder": "myroom_test", "runtime_profile": "a100_reference"}
    )

    assert spec.runtime_profile == "a100_reference"
    with pytest.raises(ValueError):
        AblationRunSpec.model_validate(
            {"input_folder": "myroom_test", "runtime_profile": "t4_diagnostic"}
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


def test_matrix_stops_after_primary_rows_even_when_pairwise_would_be_informative(
    tmp_path: Path,
) -> None:
    staged = _staged(tmp_path)
    calls: list[str] = []

    def execute(_staged, variant, _workspace, _base_spec, _control):
        calls.append(variant.experiment_id)
        passed = variant.experiment_id != "full_learned"
        return _result(variant.experiment_id, passed=passed)

    matrix = run_ablation_matrix(
        staged,
        base_spec=to_static_run_spec(
            LearnedQualityRunSpec(input_folder="myroom_test")
        ),
        experiments_root=tmp_path / "experiments",
        execute_experiment=execute,
    )

    assert calls == [variant.experiment_id for variant in primary_variants()]
    assert matrix.diagnosis.kind == "inconclusive"
    assert matrix.diagnosis.requires_pairwise is True


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
        "diagnostic_matrix.json",
        "diagnostic_summary.md",
        "historical_120k.json",
        "ablation_report.json",
        "ablation_summary.md",
        "metrics.csv",
        "environment.json",
        "staging_manifest.json",
        "psnr_plot.png",
        "plots/fixed_view_quality.png",
        "plots/structural_fidelity.png",
        "plots/gaussian_count.png",
        "plots/density_events.png",
        "experiments/legacy_control/receipt.json",
        "experiments/legacy_control/contact_005000.png",
        "_OWNERSHIP.json",
        "_SUCCESS.json",
    ):
        assert (destination / relative).is_file(), relative
    assert not tuple(destination.rglob("*.ply"))
    success = json.loads((destination / "_SUCCESS.json").read_text())
    assert success["meaning"] == "structural_diagnostic_matrix_published"
    diagnostic = json.loads((destination / "diagnostic_matrix.json").read_text())
    assert diagnostic["full_120k_training_started"] is False
    assert "structural_findings" in diagnostic


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


def test_run_training_ablation_stages_once_and_uses_the_dedicated_output_with_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_root = tmp_path / "drive"
    input_path = drive_root / "myroom_test"
    input_path.mkdir(parents=True)
    historical = drive_root / "myroom_test_learned_test_result"
    historical.mkdir()
    (historical / "run_manifest.json").write_text("{}", encoding="utf-8")
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
    audit_calls: list[tuple[Path, object, object]] = []
    restored_selection = SimpleNamespace(
        manifest=SimpleNamespace(image_set_digest="d" * 64)
    )

    def restore_graph(**kwargs):
        kwargs["selection_validator"](restored_selection)
        return SimpleNamespace(restored=True)

    monkeypatch.setattr(
        "experiments.learned_quality.ablation_runner.restore_output_first_pretraining",
        restore_graph,
    )

    def stage_once(**kwargs):
        stage_calls.append(kwargs)
        kwargs["audit_validator"](inventory)
        assert kwargs["restore_pretraining"](tmp_path / "candidate").restored
        kwargs["restored_validator"](
            restored_selection,
            SimpleNamespace(),
        )
        return staged

    def execute(_staged, variant, _workspace, _base_spec, _control):
        return _result(variant.experiment_id, passed=True)

    def publish(matrix, **kwargs):
        publish_calls.append({"matrix": matrix, **kwargs})
        kwargs["destination"].mkdir(parents=True)
        return kwargs["destination"]

    def validate_exact(cache_root, selection, active_inventory):
        audit_calls.append((cache_root, selection, active_inventory))
        return True

    result = run_training_ablation(
        AblationRunSpec(
            input_folder="myroom_test",
            runtime_profile="a100_reference",
        ),
        model_manifest_path=model_manifest,
        expected_source_revision="c" * 40,
        actual_source_revision="c" * 40,
        drive_root=drive_root,
        work_root=work_root,
        discover_source=lambda path: inventory if path == input_path else None,
        inspect_hardware=lambda _paths: hardware,
        store_factory=lambda _root, _digest: SimpleNamespace(),
        stage_inputs=stage_once,
        exact_audit_validator=validate_exact,
        execute_experiment=execute,
        publish_report=publish,
    )

    assert len(stage_calls) == 1
    assert stage_calls[0]["destination"].name == "inputs"
    assert callable(stage_calls[0]["restore_pretraining"])
    assert [call[1:] for call in audit_calls] == [
        (restored_selection, inventory),
        (restored_selection, inventory),
    ]
    assert len(publish_calls) == 1
    assert publish_calls[0]["destination"] == drive_root / (
        "myroom_test_training_ablation"
    )
    assert publish_calls[0]["environment"]["runtime_profile"] == "a100_reference"
    assert publish_calls[0]["staged"].historical_root is not None
    assert (
        publish_calls[0]["staged"].historical_root / "run_manifest.json"
    ).is_file()
    assert result.complete is True
    assert result.final_path == drive_root / "myroom_test_training_ablation"


def test_run_training_ablation_rejects_false_audit_before_large_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_root = tmp_path / "drive"
    input_path = drive_root / "myroom_test"
    input_path.mkdir(parents=True)
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")
    inventory = SimpleNamespace(digest="a" * 64)
    hardware = SimpleNamespace(
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        disk_free_gb=200.0,
        colmap_gpu_sift=True,
    )
    selection = SimpleNamespace(
        manifest=SimpleNamespace(image_set_digest="d" * 64)
    )
    large_restore_started = False

    def restore_graph(**kwargs):
        nonlocal large_restore_started
        kwargs["selection_validator"](selection)
        large_restore_started = True
        raise AssertionError("large restoration started after a failed audit")

    monkeypatch.setattr(
        "experiments.learned_quality.ablation_runner.restore_output_first_pretraining",
        restore_graph,
    )

    def stage_once(**kwargs):
        return kwargs["restore_pretraining"](tmp_path / "candidate")

    with pytest.raises(
        RuntimeError,
        match="matching verified CPU track-audit receipt",
    ):
        run_training_ablation(
            AblationRunSpec(input_folder="myroom_test"),
            model_manifest_path=model_manifest,
            expected_source_revision="c" * 40,
            actual_source_revision="c" * 40,
            drive_root=drive_root,
            work_root=tmp_path / "work",
            discover_source=lambda path: inventory if path == input_path else None,
            inspect_hardware=lambda _paths: hardware,
            store_factory=lambda _root, _digest: SimpleNamespace(),
            stage_inputs=stage_once,
            exact_audit_validator=lambda _cache, _selection, _inventory: False,
        )

    assert large_restore_started is False
