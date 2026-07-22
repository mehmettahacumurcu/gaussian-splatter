from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.learned_quality.ablation import AblationCheckpoint, StructuralMetrics
from experiments.learned_quality.floor_recovery_runner import (
    A100_LEGACY_REFERENCE_REVISION,
    FloorRecoveryDecision,
    FloorRecoveryRunResult,
    FloorRecoveryRunSpec,
    decide_floor_recovery,
    publish_floor_recovery_report,
    run_floor_recovery_diagnostic,
    run_floor_recovery_from_staged,
    stage_legacy_reference,
    validate_floor_recovery_hardware,
    validate_result_target,
)
from experiments.learned_quality.ablation_staging import StagedAblationInputs
from tests.static_pipeline.fixtures import write_colmap_text_model
from experiments.learned_quality.floor_recovery_training import FloorCheckpointMetrics


def _global(psnr: float = 27.5, *, white: float = 0.01) -> AblationCheckpoint:
    return AblationCheckpoint(
        experiment_id="floor_recovery",
        iteration=5_000,
        psnr_unmasked=psnr,
        psnr_masked=psnr,
        ssim_unmasked=0.8,
        ssim_masked=0.8,
        l1_unmasked=0.1,
        l1_masked=0.1,
        gaussian_count=100,
        structural=StructuralMetrics(
            visible_white_fraction=white,
            high_anisotropy_fraction=0.0,
            oversized_fraction=0.004,
            nonfinite_count=0,
            out_of_bounds_fraction=0.0,
        ),
    )


def _floor(iteration: int, coverage: float) -> FloorCheckpointMetrics:
    return FloorCheckpointMetrics(
        iteration=iteration,
        floor_alpha_coverage=coverage,
        residual_hole_fraction=1.0 - coverage,
        plane_depth_relative_error=0.1,
        perturbed_depth_disagreement_ratio=1.05,
        occupied_hole_fraction=max(coverage, 0.5),
    )


def test_floor_decision_requires_coverage_and_structure() -> None:
    decision = decide_floor_recovery(
        legacy=_global(psnr=28.11, white=0.008),
        candidate=_global(psnr=27.4, white=0.010),
        initial_floor=_floor(0, 0.02),
        final_floor=_floor(5_000, 0.32),
    )

    assert decision.passed is True
    assert decision.failures == ()


@pytest.mark.parametrize(
    ("candidate", "final_floor", "failure"),
    (
        (_global(psnr=26.5), _floor(5_000, 0.32), "psnr_deficit"),
        (_global(white=0.02), _floor(5_000, 0.32), "visible_white_fraction"),
        (_global(), _floor(5_000, 0.10), "hole_reduction"),
    ),
)
def test_floor_decision_fails_closed(candidate, final_floor, failure: str) -> None:
    decision = decide_floor_recovery(
        legacy=_global(psnr=28.11),
        candidate=candidate,
        initial_floor=_floor(0, 0.02),
        final_floor=final_floor,
    )

    assert decision.passed is False
    assert failure in decision.failures


def test_publication_cannot_target_production_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="floor recovery diagnostic"):
        validate_result_target(tmp_path / "myroom_learned_test_result")


def test_publication_is_owned_atomic_and_keeps_candidate_ply(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    (report / "decision.json").write_text("{}", encoding="utf-8")
    (report / "splat.ply").write_bytes(b"ply")
    destination = tmp_path / "myroom_floor_recovery_diagnostic"
    decision = FloorRecoveryDecision(True, "passed", ())

    published = publish_floor_recovery_report(
        local_report_root=report,
        destination=destination,
        run_id="run-1",
        decision=decision,
    )

    assert published == destination
    assert (destination / "splat.ply").read_bytes() == b"ply"
    assert (destination / "_SUCCESS.json").is_file()
    assert json.loads((destination / "_OWNERSHIP.json").read_text())[
        "generator_id"
    ] == "4dgs-studio.floor-recovery-diagnostic"


def test_run_spec_rejects_output_folder() -> None:
    with pytest.raises(ValueError, match="input folder"):
        FloorRecoveryRunSpec(input_folder="myroom_floor_recovery_diagnostic")


def test_staged_runner_rejects_weak_plane_without_training(tmp_path: Path) -> None:
    model = write_colmap_text_model(
        tmp_path / "model",
        ("frame.png",),
        (0.2, 0.3, 0.4),
        (2, 2, 2),
    )
    bundle = SimpleNamespace(
        accepted_model_dir=model,
        selected_manifest=SimpleNamespace(image_set_digest="d" * 64),
    )
    reconstruction = SimpleNamespace(
        artifacts=SimpleNamespace(
            photometric=SimpleNamespace(original_frames=()),
            depth=SimpleNamespace(policy=object()),
            masks=object(),
        ),
        accepted_model_dir=model,
        frames_dir=tmp_path / "frames",
        bundle=bundle,
        selected_manifest=bundle.selected_manifest,
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = inputs / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    staged = StagedAblationInputs(
        root=inputs,
        selection=object(),
        reconstruction=reconstruction,
        source_digest="a" * 64,
        pretraining_fingerprint="b" * 64,
        source_revision="c" * 40,
        manifest_path=manifest,
    )
    local = tmp_path / "local"
    local.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    training_calls = 0

    def train(*_args, **_kwargs):
        nonlocal training_calls
        training_calls += 1
        raise AssertionError("training must not start")

    result = run_floor_recovery_from_staged(
        staged,
        base_spec=object(),
        local_root=local,
        reference_root=reference,
        legacy_checkpoints=(_global(psnr=28.11),),
        destination=tmp_path / "myroom_floor_recovery_diagnostic",
        run_id="run-reject",
        build_cloud=lambda *_args, **_kwargs: SimpleNamespace(
            camera_centers=np.zeros((3, 3))
        ),
        fit_plane=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("could not find a reliable floor plane")
        ),
        train=train,
    )

    assert result.status == "rejected"
    assert result.decision is not None
    assert result.decision.reason == "unreliable_floor_plane"
    assert training_calls == 0


def test_floor_recovery_hardware_requires_large_a100() -> None:
    validate_floor_recovery_hardware(
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=100.0,
        )
    )
    with pytest.raises(RuntimeError, match="A100"):
        validate_floor_recovery_hardware(
            SimpleNamespace(gpu_name="NVIDIA L4", vram_gb=24.0, disk_free_gb=100.0)
        )


def test_run_floor_recovery_stages_once_and_uses_diagnostic_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    reference_source = drive_root / "myroom_test_training_ablation"
    reference_source.mkdir()
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")
    inventory = SimpleNamespace(digest="a" * 64)
    hardware = SimpleNamespace(
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        disk_free_gb=200.0,
        colmap_gpu_sift=True,
    )
    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    staged_manifest = staged_root / "manifest.json"
    staged_manifest.write_text("{}", encoding="utf-8")
    staged = StagedAblationInputs(
        root=staged_root,
        selection=object(),
        reconstruction=object(),
        source_digest="a" * 64,
        pretraining_fingerprint="b" * 64,
        source_revision="c" * 40,
        manifest_path=staged_manifest,
    )
    restored_selection = SimpleNamespace(
        manifest=SimpleNamespace(image_set_digest="d" * 64)
    )
    stage_calls: list[dict[str, object]] = []
    reference_calls: list[tuple[Path, Path, StagedAblationInputs]] = []
    execute_calls: list[dict[str, object]] = []
    audit_calls: list[tuple[Path, object, object]] = []

    def restored_graph(**kwargs):
        kwargs["selection_validator"](restored_selection)
        return SimpleNamespace(restored=True)

    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.restore_output_first_pretraining",
        restored_graph,
    )

    def stage_once(**kwargs):
        stage_calls.append(kwargs)
        kwargs["audit_validator"](inventory)
        assert kwargs["restore_pretraining"](tmp_path / "candidate").restored
        kwargs["restored_validator"](restored_selection, object())
        return staged

    def stage_reference(source, destination, *, staged):
        reference_calls.append((source, destination, staged))
        destination.mkdir(parents=True)
        return (_global(psnr=28.11),)

    def execute(staged_input, **kwargs):
        execute_calls.append({"staged": staged_input, **kwargs})
        return FloorRecoveryRunResult(
            run_id=kwargs["run_id"],
            status="rejected",
            final_path=kwargs["destination"],
            local_root=kwargs["local_root"],
            decision=FloorRecoveryDecision(
                False, "quality_gates_failed", ("floor_alpha_gain",)
            ),
        )

    def validate_exact(cache_root, selection, active_inventory):
        audit_calls.append((cache_root, selection, active_inventory))
        return True

    result = run_floor_recovery_diagnostic(
        FloorRecoveryRunSpec(input_folder="myroom_test"),
        model_manifest_path=model_manifest,
        expected_source_revision="c" * 40,
        actual_source_revision="c" * 40,
        drive_root=drive_root,
        work_root=tmp_path / "work",
        discover_source=lambda path: inventory if path == input_path else None,
        inspect_hardware=lambda _paths: hardware,
        store_factory=lambda _root, _digest: SimpleNamespace(),
        stage_inputs=stage_once,
        exact_audit_validator=validate_exact,
        stage_reference=stage_reference,
        execute_staged=execute,
    )

    assert len(stage_calls) == 1
    assert stage_calls[0]["destination"].name == "inputs"
    assert stage_calls[0]["freeze"] is True
    assert len(reference_calls) == 1
    assert reference_calls[0][0] == reference_source
    assert reference_calls[0][2] is staged
    assert len(execute_calls) == 1
    assert execute_calls[0]["destination"] == (
        drive_root / "myroom_test_floor_recovery_diagnostic"
    )
    assert [call[1:] for call in audit_calls] == [
        (restored_selection, inventory),
        (restored_selection, inventory),
    ]
    assert result.status == "rejected"


def test_run_floor_recovery_rejects_false_audit_before_large_restore(
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
        raise AssertionError("large restoration started after failed audit")

    monkeypatch.setattr(
        "experiments.learned_quality.ablation_staging.restore_output_first_pretraining",
        restore_graph,
    )

    def stage_once(**kwargs):
        return kwargs["restore_pretraining"](tmp_path / "candidate")

    with pytest.raises(RuntimeError, match="matching verified CPU track-audit"):
        run_floor_recovery_diagnostic(
            FloorRecoveryRunSpec(input_folder="myroom_test"),
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


def test_legacy_reference_must_match_staged_a100_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "myroom_training_ablation"
    receipt = source / "experiments" / "legacy_control" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    (source / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    manifest = staged_root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    staged = StagedAblationInputs(
        root=staged_root,
        selection=object(),
        reconstruction=object(),
        source_digest="a" * 64,
        pretraining_fingerprint="b" * 64,
        source_revision="d" * 40,
        manifest_path=manifest,
    )
    matrix = {
        "staging": {
            "source_digest": staged.source_digest,
            "pretraining_fingerprint": staged.pretraining_fingerprint,
            "source_revision": A100_LEGACY_REFERENCE_REVISION,
        }
    }
    (source / "diagnostic_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8"
    )
    (source / "environment.json").write_text(
        json.dumps({"runtime_profile": "a100_reference"}), encoding="utf-8"
    )
    schedule = (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
    checkpoints = tuple(
        dataclasses.replace(_global(psnr=28.11), iteration=iteration)
        for iteration in schedule
    )
    monkeypatch.setattr(
        "experiments.learned_quality.ablation_runner._record_from_payload",
        lambda _payload: SimpleNamespace(
            passed=True,
            stopped_early=False,
            checkpoints=checkpoints,
        ),
    )

    restored = stage_legacy_reference(
        source,
        tmp_path / "reference",
        staged=staged,
    )

    assert tuple(row.iteration for row in restored) == schedule
    assert (tmp_path / "reference" / "diagnostic_matrix.json").is_file()
    assert (
        tmp_path / "reference" / "experiments" / "legacy_control" / "receipt.json"
    ).is_file()

    matrix["staging"]["source_revision"] = "e" * 40
    (source / "diagnostic_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="producer revision"):
        stage_legacy_reference(
            source,
            tmp_path / "wrong-reference-producer",
            staged=staged,
        )

    matrix["staging"]["source_revision"] = A100_LEGACY_REFERENCE_REVISION
    matrix["staging"]["source_digest"] = "f" * 64
    (source / "diagnostic_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="differs from staged pretraining"):
        stage_legacy_reference(
            source,
            tmp_path / "mismatched-reference",
            staged=staged,
        )
