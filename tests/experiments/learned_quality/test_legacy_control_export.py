from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.static_pipeline.contracts import (
    GateDecision,
    ModelMetrics,
    PolishReport,
    ReconstructionBundle,
    SelectionManifest,
    SelectionPolicy,
)
from experiments.learned_quality import legacy_control_export as export_module
from experiments.learned_quality.ablation import (
    AblationCheckpoint,
    StructuralMetrics,
)
from experiments.learned_quality.ablation_training import AblationExperimentResult
from experiments.learned_quality.contracts import GeometryAcceptance
from experiments.learned_quality.legacy_control_export import (
    GENERATOR_ID,
    LegacyControlExportRunSpec,
    LegacyControlExportResult,
    diagnostic_polish_bundle,
    prepare_legacy_control_report,
    publish_legacy_control_export,
    run_diagnostic_polish,
    run_legacy_control_export,
    run_legacy_control_export_from_staged,
    select_legacy_control_variant,
    validate_legacy_control_hardware,
)


def _rejected_reconstruction(tmp_path: Path) -> SimpleNamespace:
    model = tmp_path / "model"
    model.mkdir()
    manifest = SelectionManifest(
        schema_version=1,
        source_digest="a" * 64,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=800,
            resolution_long_edge_cap=1280,
        ),
        frames=(),
        image_set_digest="b" * 64,
    )
    metrics = ModelMetrics(
        model_dir=model,
        registered_names=frozenset(),
        registered_count=651,
        registered_ratio=0.81375,
        registered_share=0.963,
        temporal_coverage_s=100.0,
        max_interior_gap_s=2.585,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=1.194,
        p95_reprojection_error_px=2.183,
        median_track_length=5.0,
        sparse_point_count=61_137,
        valid_names_intrinsics_and_poses=True,
    )
    decision = GateDecision(
        passed=False,
        dominant=metrics,
        failures=("registered_ratio", "interior_gap"),
        uncovered_intervals=(),
        retry_recommended=True,
        warnings=("output_first",),
    )
    bundle = ReconstructionBundle(
        selected_manifest=manifest,
        accepted_model_dir=model,
        decision=decision,
        attempts=(),
        decisions=(decision,),
    )
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest=manifest.image_set_digest,
        model_hashes={
            "cameras.txt": "c" * 64,
            "images.txt": "d" * 64,
            "points3D.txt": "e" * 64,
        },
        strict_failures=decision.failures,
        metrics=metrics,
        checks={
            "registered_ratio": False,
            "valid_model": True,
        },
        colmap_fingerprint="f" * 64,
    )
    return SimpleNamespace(
        bundle=bundle,
        accepted_model_dir=model,
        selected_manifest=manifest,
        acceptance=acceptance,
    )


def test_legacy_export_selects_only_exact_5k_control() -> None:
    variant = select_legacy_control_variant()

    assert variant.experiment_id == "legacy_control"
    assert variant.features == frozenset()
    assert variant.n_iterations == 5_000
    assert variant.density_events is True


def test_diagnostic_polish_proxy_does_not_mutate_original(tmp_path: Path) -> None:
    reconstruction = _rejected_reconstruction(tmp_path)
    original = reconstruction.bundle

    proxy = diagnostic_polish_bundle(reconstruction)

    assert proxy is not original
    assert proxy.decision is not original.decision
    assert proxy.decision.passed is True
    assert proxy.decision.failures == ()
    assert proxy.decision.retry_recommended is False
    assert proxy.decision.dominant == original.decision.dominant
    assert proxy.accepted_model_dir == original.accepted_model_dir
    assert original.decision.passed is False
    assert original.decision.failures == ("registered_ratio", "interior_gap")
    assert original.decision.retry_recommended is True


def _complete_local_report(root: Path) -> tuple[bytes, bytes]:
    root.mkdir()
    raw = b"raw legacy-control 5k ply\n"
    polished = b"filtered legacy-control 5k ply\n"
    files: dict[str, bytes] = {
        "raw_legacy_control_5k.ply": raw,
        "polished_legacy_control_5k.ply": polished,
        "polish_report.json": b'{"accepted":false}\n',
        "receipt.json": b'{"status":"success"}\n',
        "metrics.jsonl": b'{"iter":5000}\n',
        "run_manifest.json": b'{"n_iterations":5000}\n',
        "contact_005000.png": b"contact",
        "contact_005000_fixed.png": b"fixed",
        "contact_005000_perturbed.png": b"perturbed",
        "provenance.json": b'{"experiment_id":"legacy_control"}\n',
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    return raw, polished


def test_publication_preserves_both_plys_and_writes_completion_last(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report"
    raw, polished = _complete_local_report(report)
    destination = tmp_path / "myroom_test_legacy_control_5k_result"

    published = publish_legacy_control_export(
        local_report_root=report,
        destination=destination,
        run_id="legacy-run",
    )

    assert published == destination
    assert (published / "raw_legacy_control_5k.ply").read_bytes() == raw
    assert (published / "polished_legacy_control_5k.ply").read_bytes() == polished
    assert json.loads((published / "_OWNERSHIP.json").read_text()) == {
        "generator_id": GENERATOR_ID,
        "run_id": "legacy-run",
        "schema_version": 1,
    }
    success = json.loads((published / "_SUCCESS.json").read_text())
    assert success["status"] == "success"
    assert success["raw_ply"] == "raw_legacy_control_5k.ply"
    assert success["polished_ply"] == "polished_legacy_control_5k.ply"


def test_ply_rescue_is_durable_before_report_publication(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    polished = tmp_path / "polished.ply"
    raw.write_bytes(b"raw")
    polished.write_bytes(b"polished")
    destination = tmp_path / "myroom_test_legacy_control_5k_result"

    rescued = export_module.publish_legacy_control_ply_rescue(
        raw_ply_path=raw,
        polished_ply_path=polished,
        destination=destination,
        run_id="legacy-run",
        polish_accepted=False,
        polish_reasons=("mean_psnr_drop",),
        failure=TypeError("cannot pickle mappingproxy"),
    )

    assert rescued == destination
    assert (rescued / export_module.RAW_PLY_NAME).read_bytes() == b"raw"
    assert (rescued / export_module.POLISHED_PLY_NAME).read_bytes() == b"polished"
    assert not (rescued / "_SUCCESS.json").exists()
    partial = json.loads((rescued / "_PARTIAL.json").read_text())
    assert partial["status"] == "ply_rescue"
    assert partial["error_type"] == "TypeError"
    assert partial["polish_accepted"] is False
    assert partial["polish_reasons"] == ["mean_psnr_drop"]


def test_publication_refuses_to_replace_unowned_result(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _complete_local_report(report)
    destination = tmp_path / "myroom_test_legacy_control_5k_result"
    destination.mkdir()
    (destination / "keep.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="refusing to replace an unowned legacy-control export",
    ):
        publish_legacy_control_export(
            local_report_root=report,
            destination=destination,
            run_id="legacy-run",
        )

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_publication_requires_a_distinct_polish_candidate(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _complete_local_report(report)
    (report / "polished_legacy_control_5k.ply").write_bytes(
        (report / "raw_legacy_control_5k.ply").read_bytes()
    )

    with pytest.raises(
        ValueError,
        match="polished candidate must differ from the raw PLY",
    ):
        publish_legacy_control_export(
            local_report_root=report,
            destination=tmp_path / "myroom_test_legacy_control_5k_result",
            run_id="legacy-run",
        )


def _checkpoint(iteration: int) -> AblationCheckpoint:
    return AblationCheckpoint(
        experiment_id="legacy_control",
        iteration=iteration,
        psnr_unmasked=28.0,
        psnr_masked=28.0,
        ssim_unmasked=0.9,
        ssim_masked=0.9,
        l1_unmasked=0.05,
        l1_masked=0.05,
        gaussian_count=100,
        structural=StructuralMetrics(
            visible_white_fraction=0.01,
            high_anisotropy_fraction=0.0,
            oversized_fraction=0.002,
            nonfinite_count=0,
        ),
    )


def _training_result(root: Path) -> AblationExperimentResult:
    raw = root / "training" / "point_cloud" / "iteration_5000" / "point_cloud.ply"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw-ply")
    metrics = root / "training" / "logs" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"iter":5000}\n', encoding="utf-8")
    manifest = root / "training" / "run_manifest.json"
    manifest.write_text('{"n_iterations":5000}\n', encoding="utf-8")
    return AblationExperimentResult(
        experiment_id="legacy_control",
        passed=True,
        stopped_early=False,
        stop_reasons=(),
        checkpoints=tuple(
            _checkpoint(iteration)
            for iteration in (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
        ),
        raw_ply_path=raw,
        metrics_path=metrics,
        run_manifest_path=manifest,
        status={"ok": True},
    )


def test_diagnostic_polish_runs_production_filter_with_isolated_proxy(
    tmp_path: Path,
) -> None:
    reconstruction = _rejected_reconstruction(tmp_path)
    reconstruction.frames_dir = tmp_path / "frames"
    training = _training_result(tmp_path)
    candidate = tmp_path / "polish" / "candidate.ply"
    captured: dict[str, object] = {}

    def polisher(
        raw_path: Path,
        candidate_path: Path,
        bundle: ReconstructionBundle,
        registered: tuple[object, ...],
    ) -> PolishReport:
        captured.update(
            raw_path=raw_path,
            bundle=bundle,
            registered=registered,
        )
        candidate_path.write_bytes(b"polished-ply")
        return PolishReport(
            accepted=False,
            raw_path=raw_path,
            candidate_path=candidate_path,
            selected_path=raw_path,
            original_count=100,
            kept_count=90,
            opacity_mass_loss=0.02,
            render_metrics={"mean_psnr_drop_db": 0.5},
            reasons=("mean_psnr_drop",),
        )

    report = run_diagnostic_polish(
        training,
        reconstruction,
        output_root=candidate.parent,
        camera_parser=lambda _model: {"frame.png": {}},
        frame_joiner=lambda *_args, **_kwargs: (object(),),
        polisher=polisher,
    )

    assert report.accepted is False
    assert report.candidate_path == candidate
    assert candidate.read_bytes() == b"polished-ply"
    assert captured["raw_path"] == training.raw_ply_path
    proxy = captured["bundle"]
    assert isinstance(proxy, ReconstructionBundle)
    assert proxy.decision.passed is True
    assert proxy.decision.failures == ()
    assert reconstruction.bundle.decision.passed is False
    assert reconstruction.bundle.decision.failures == (
        "registered_ratio",
        "interior_gap",
    )


def test_report_preserves_rejected_candidate_and_explains_geometry_override(
    tmp_path: Path,
) -> None:
    reconstruction = _rejected_reconstruction(tmp_path)
    training = _training_result(tmp_path)
    diagnostic_root = tmp_path / "diagnostics"
    diagnostic_root.mkdir()
    for name in (
        "contact_005000.png",
        "contact_005000_fixed.png",
        "contact_005000_perturbed.png",
    ):
        (diagnostic_root / name).write_bytes(name.encode())
    candidate = tmp_path / "polish" / "candidate.ply"
    candidate.parent.mkdir()
    candidate.write_bytes(b"polished-ply")
    polish = PolishReport(
        accepted=False,
        raw_path=training.raw_ply_path,
        candidate_path=candidate,
        selected_path=training.raw_ply_path,
        original_count=100,
        kept_count=90,
        opacity_mass_loss=0.02,
        render_metrics={
            "mean_psnr_drop_db": 0.5,
            "mean_ssim_drop": 0.001,
        },
        reasons=("mean_psnr_drop",),
    )
    staged = SimpleNamespace(
        source_digest="c" * 64,
        pretraining_fingerprint="d" * 64,
        source_revision="source-revision",
        reconstruction=reconstruction,
    )
    report_root = tmp_path / "report"
    historical = training.checkpoints

    prepare_legacy_control_report(
        report_root,
        staged=staged,
        training=training,
        polish=polish,
        diagnostic_root=diagnostic_root,
        historical_checkpoints=historical,
        ply_validator=lambda path: SimpleNamespace(
            path=Path(path),
            count=90 if Path(path) == candidate else 100,
        ),
    )

    assert (report_root / "raw_legacy_control_5k.ply").read_bytes() == b"raw-ply"
    assert (report_root / "polished_legacy_control_5k.ply").read_bytes() == (
        b"polished-ply"
    )
    payload = json.loads((report_root / "polish_report.json").read_text())
    assert payload["accepted"] is False
    assert payload["candidate_published_for_manual_comparison"] is True
    assert payload["diagnostic_acceptance_proxy"] is True
    assert payload["original_geometry_failures"] == [
        "registered_ratio",
        "interior_gap",
    ]
    assert payload["original_count"] == 100
    assert payload["kept_count"] == 90
    assert payload["removed_count"] == 10
    assert payload["opacity_mass_loss"] == 0.02
    assert payload["reasons"] == ["mean_psnr_drop"]
    assert payload["render_metrics"]["mean_psnr_drop_db"] == 0.5
    assert payload["geometry_acceptance"]["policy_version"] == "output-first-v1"
    assert payload["geometry_acceptance"]["mode"] == "best_effort"
    assert payload["geometry_acceptance"]["model_hashes"] == {
        "cameras.txt": "c" * 64,
        "images.txt": "d" * 64,
        "points3D.txt": "e" * 64,
    }
    assert payload["geometry_acceptance"]["checks"] == {
        "registered_ratio": False,
        "valid_model": True,
    }
    receipt = json.loads((report_root / "receipt.json").read_text())
    assert receipt["status"] == "success"
    assert receipt["iterations"] == 5_000
    assert receipt["experiment_id"] == "legacy_control"
    assert len(receipt["reproduced_checkpoints"]) == 8
    assert len(receipt["historical_checkpoints"]) == 8


def test_run_spec_normalizes_input_and_rejects_result_folders() -> None:
    assert LegacyControlExportRunSpec(
        input_folder="  myroom_test  "
    ).input_folder == "myroom_test"

    with pytest.raises(ValueError, match="[Cc]hoose the input folder"):
        LegacyControlExportRunSpec(
            input_folder="myroom_test_legacy_control_5k_result"
        )


def test_hardware_requires_a100_high_ram_and_local_disk() -> None:
    good = SimpleNamespace(
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        disk_free_gb=120.0,
        host_ram_gb=167.0,
        colmap_gpu_sift=True,
    )
    validate_legacy_control_hardware(good)

    for bad in (
        SimpleNamespace(
            gpu_name="NVIDIA L4",
            vram_gb=23.0,
            disk_free_gb=120.0,
            host_ram_gb=167.0,
        ),
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=20.0,
            host_ram_gb=167.0,
        ),
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=120.0,
            host_ram_gb=83.0,
        ),
    ):
        with pytest.raises(RuntimeError):
            validate_legacy_control_hardware(bad)


def test_staged_runner_executes_only_one_legacy_control_and_publishes_both(
    tmp_path: Path,
) -> None:
    reconstruction = _rejected_reconstruction(tmp_path)
    reconstruction.frames_dir = tmp_path / "frames"
    staged = SimpleNamespace(
        root=tmp_path / "inputs",
        reconstruction=reconstruction,
        source_digest="c" * 64,
        pretraining_fingerprint="d" * 64,
        source_revision="source-revision",
    )
    staged.root.mkdir()
    training = _training_result(tmp_path)
    candidate = tmp_path / "polish-source" / "candidate.ply"
    candidate.parent.mkdir()
    candidate.write_bytes(b"polished-ply")
    polish = PolishReport(
        accepted=False,
        raw_path=training.raw_ply_path,
        candidate_path=candidate,
        selected_path=training.raw_ply_path,
        original_count=100,
        kept_count=90,
        opacity_mass_loss=0.02,
        render_metrics={"mean_psnr_drop_db": 0.5},
        reasons=("mean_psnr_drop",),
    )
    calls: dict[str, object] = {}

    def materialize(_staged: object, **kwargs: object) -> SimpleNamespace:
        calls["materialize"] = kwargs
        root = tmp_path / "experiment"
        root.mkdir()
        scene = root / "scene"
        output = root / "output"
        scene.mkdir()
        output.mkdir()
        return SimpleNamespace(root=root, scene_root=scene, output_root=output)

    def execute(
        _prepared: object,
        _base_spec: object,
        _reconstruction: object,
        variant: object,
        **kwargs: object,
    ) -> AblationExperimentResult:
        calls["variant"] = variant
        calls["execute"] = kwargs
        return training

    def prepare(root: Path, **kwargs: object) -> Path:
        calls["prepare"] = kwargs
        _complete_local_report(root)
        return root

    destination = tmp_path / "myroom_test_legacy_control_5k_result"
    result = run_legacy_control_export_from_staged(
        staged,
        base_spec=SimpleNamespace(),
        local_root=tmp_path / "run",
        historical_checkpoints=training.checkpoints,
        destination=destination,
        run_id="legacy-run",
        materialize_workspace=materialize,
        execute_experiment=execute,
        polish_runner=lambda *_args, **_kwargs: polish,
        prepare_report=prepare,
    )

    variant = calls["variant"]
    assert variant.experiment_id == "legacy_control"
    assert variant.n_iterations == 5_000
    assert variant.features == frozenset()
    assert result.final_path == destination
    assert result.raw_ply_path == destination / "raw_legacy_control_5k.ply"
    assert result.polished_ply_path == (
        destination / "polished_legacy_control_5k.ply"
    )
    assert result.polish_accepted is False
    assert result.polish_reasons == ("mean_psnr_drop",)


def test_staged_runner_rescues_both_plys_when_report_generation_fails(
    tmp_path: Path,
) -> None:
    reconstruction = _rejected_reconstruction(tmp_path)
    reconstruction.frames_dir = tmp_path / "frames"
    staged = SimpleNamespace(
        root=tmp_path / "inputs",
        reconstruction=reconstruction,
        source_digest="c" * 64,
        pretraining_fingerprint="d" * 64,
        source_revision="source-revision",
    )
    staged.root.mkdir()
    training = _training_result(tmp_path)
    candidate = tmp_path / "polish-source" / "candidate.ply"
    candidate.parent.mkdir()
    candidate.write_bytes(b"polished-ply")
    polish = PolishReport(
        accepted=False,
        raw_path=training.raw_ply_path,
        candidate_path=candidate,
        selected_path=training.raw_ply_path,
        original_count=100,
        kept_count=90,
        opacity_mass_loss=0.02,
        render_metrics={"mean_psnr_drop_db": 0.5},
        reasons=("mean_psnr_drop",),
    )

    def materialize(_staged: object, **_kwargs: object) -> SimpleNamespace:
        root = tmp_path / "experiment"
        scene = root / "scene"
        output = root / "output"
        scene.mkdir(parents=True)
        output.mkdir()
        return SimpleNamespace(root=root, scene_root=scene, output_root=output)

    def report_failure(*_args: object, **_kwargs: object) -> Path:
        raise TypeError("cannot pickle mappingproxy")

    destination = tmp_path / "myroom_test_legacy_control_5k_result"
    with pytest.raises(TypeError, match="cannot pickle mappingproxy"):
        run_legacy_control_export_from_staged(
            staged,
            base_spec=SimpleNamespace(),
            local_root=tmp_path / "run",
            historical_checkpoints=training.checkpoints,
            destination=destination,
            run_id="legacy-run",
            materialize_workspace=materialize,
            execute_experiment=lambda *_args, **_kwargs: training,
            polish_runner=lambda *_args, **_kwargs: polish,
            prepare_report=report_failure,
        )

    assert (destination / export_module.RAW_PLY_NAME).read_bytes() == b"raw-ply"
    assert (destination / export_module.POLISHED_PLY_NAME).read_bytes() == (
        b"polished-ply"
    )
    assert json.loads((destination / "_PARTIAL.json").read_text())["status"] == (
        "ply_rescue"
    )


def test_outer_runner_stages_verified_lineage_once_and_uses_isolated_result(
    tmp_path: Path,
) -> None:
    drive = tmp_path / "drive"
    input_path = drive / "myroom_test"
    input_path.mkdir(parents=True)
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    work = tmp_path / "work"
    inventory = SimpleNamespace(digest="a" * 64)
    hardware = SimpleNamespace(
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        disk_free_gb=120.0,
        host_ram_gb=167.0,
        colmap_gpu_sift=True,
    )
    reconstruction = _rejected_reconstruction(tmp_path)
    reconstruction.frames_dir = tmp_path / "frames"
    stage_calls: list[dict[str, object]] = []
    reference_calls: list[tuple[Path, Path]] = []
    execute_calls: list[dict[str, object]] = []

    def stage(**kwargs: object) -> SimpleNamespace:
        stage_calls.append(kwargs)
        destination = Path(kwargs["destination"])
        destination.mkdir(parents=True)
        return SimpleNamespace(
            root=destination,
            reconstruction=reconstruction,
            selection=SimpleNamespace(),
            source_digest=inventory.digest,
            pretraining_fingerprint="b" * 64,
            source_revision="source-revision",
        )

    checkpoints = tuple(
        _checkpoint(iteration)
        for iteration in (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
    )

    def stage_reference(
        source: Path,
        destination: Path,
        **_kwargs: object,
    ) -> tuple[AblationCheckpoint, ...]:
        reference_calls.append((source, destination))
        destination.mkdir()
        return checkpoints

    def execute(staged: object, **kwargs: object) -> LegacyControlExportResult:
        execute_calls.append({"staged": staged, **kwargs})
        destination = Path(kwargs["destination"])
        destination.mkdir()
        raw = destination / "raw_legacy_control_5k.ply"
        polished = destination / "polished_legacy_control_5k.ply"
        raw.write_bytes(b"raw")
        polished.write_bytes(b"polished")
        return LegacyControlExportResult(
            run_id=str(kwargs["run_id"]),
            final_path=destination,
            local_root=Path(kwargs["local_root"]),
            raw_ply_path=raw,
            polished_ply_path=polished,
            polish_accepted=False,
            polish_reasons=("mean_psnr_drop",),
        )

    result = run_legacy_control_export(
        LegacyControlExportRunSpec(input_folder="myroom_test"),
        model_manifest_path=manifest,
        expected_source_revision="source-revision",
        actual_source_revision="source-revision",
        drive_root=drive,
        work_root=work,
        discover_source=lambda _path: inventory,
        inspect_hardware=lambda _paths: hardware,
        store_factory=lambda _root, _digest: object(),
        stage_inputs=stage,
        exact_audit_validator=lambda *_args: True,
        reference_fingerprint_resolver=lambda *_args, **_kwargs: "b" * 64,
        stage_reference=stage_reference,
        execute_staged=execute,
    )

    assert len(stage_calls) == 1
    assert len(reference_calls) == 1
    assert len(execute_calls) == 1
    assert reference_calls[0][0] == drive / "myroom_test_training_ablation"
    assert stage_calls[0]["minimum_free_bytes"] == 35 * 1024**3
    assert not Path(stage_calls[0]["destination"]).is_relative_to(drive)
    assert result.final_path == drive / "myroom_test_legacy_control_5k_result"
