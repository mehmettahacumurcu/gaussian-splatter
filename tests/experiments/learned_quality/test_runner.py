from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.static_pipeline.runner import (
    HardwareInfo,
    NotebookRuntimePaths,
    SelectionOutput,
)
from backend.static_pipeline.progress import StageReporter
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    to_static_run_spec,
)
from experiments.learned_quality.runner import (
    LEARNED_STAGE_DEFINITIONS,
    LearnedQualityContext,
    _late_failure_files,
    _pretraining_cache_fingerprint,
    _reported_winner,
    _selection_milestone_ref,
    make_learned_quality_services,
    preflight_learned_runtime,
    run_learned_quality_notebook,
)


def test_preflight_requires_an_a100_with_at_least_39_gib() -> None:
    spec = LearnedQualityRunSpec(input_folder="captures/room")
    preflight_learned_runtime(
        spec,
        HardwareInfo("NVIDIA A100-SXM4-40GB", 39.5, True, 120.0),
        input_size_gb=1.0,
    )
    with pytest.raises(RuntimeError, match="A100"):
        preflight_learned_runtime(
            spec,
            HardwareInfo("NVIDIA L4", 24.0, True, 120.0),
            input_size_gb=1.0,
        )
    with pytest.raises(RuntimeError, match="39"):
        preflight_learned_runtime(
            spec,
            HardwareInfo("NVIDIA A100", 38.9, True, 120.0),
            input_size_gb=1.0,
        )


def test_service_composition_keeps_production_boundaries_and_replaces_learned_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.static_pipeline import runner as static_runner

    sentinel = SimpleNamespace(
        discover_source=object(),
        copy_input=object(),
        select_frames=object(),
        reconstruct=object(),
        train=object(),
        polish=object(),
        build_metadata_preview=object(),
        validate_bundle=object(),
        publish_result=object(),
        publish_diagnostics=object(),
        inspect_hardware=object(),
        preflight=object(),
        assemble_reports=object(),
        resolve_input=object(),
    )
    monkeypatch.setattr(static_runner, "_production_services", lambda: sentinel)
    context = LearnedQualityContext(
        reconstruct=lambda *args, **kwargs: None,
        model_manifest_path=tmp_path / "model_manifest.json",
    )

    services = make_learned_quality_services(context)

    assert services.discover_source is sentinel.discover_source
    assert services.copy_input is sentinel.copy_input
    assert services.select_frames is sentinel.select_frames
    assert services.polish is sentinel.polish
    assert services.build_metadata_preview is sentinel.build_metadata_preview
    assert services.inspect_hardware is sentinel.inspect_hardware
    assert services.resolve_input is sentinel.resolve_input
    assert services.reconstruct is context.reconstruct
    assert services.train is not sentinel.train
    assert services.validate_bundle is not sentinel.validate_bundle
    assert services.publish_result is not sentinel.publish_result
    assert services.publish_diagnostics is not sentinel.publish_diagnostics
    assert services.restore_pretraining is None
    assert services.save_pretraining is None


def test_cached_service_composition_uses_one_store_for_restore_reconstruct_and_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.static_pipeline import runner as static_runner

    production = SimpleNamespace(
        discover_source=object(),
        copy_input=object(),
        select_frames=object(),
        reconstruct=object(),
        train=object(),
        polish=object(),
        build_metadata_preview=object(),
        validate_bundle=object(),
        publish_result=object(),
        publish_diagnostics=object(),
        inspect_hardware=object(),
        preflight=object(),
        assemble_reports=object(),
        resolve_input=object(),
    )
    monkeypatch.setattr(static_runner, "_production_services", lambda: production)
    calls: list[tuple[str, object]] = []

    class FakeStore:
        def __init__(self, cache_root: Path, *, input_identity: str) -> None:
            self.input_identity = input_identity
            calls.append(("store", (cache_root, input_identity)))

        def probe_drive_publication(self, *, run_id: str) -> None:
            calls.append(("probe", run_id))

        def restore_pretraining(self, **kwargs: object) -> None:
            calls.append(("restore", kwargs["destination"]))
            return None

        def publish_pretraining(self, **kwargs: object) -> None:
            calls.append(("save", kwargs["run_root"]))

    monkeypatch.setattr(
        "experiments.learned_quality.runner.LearnedCheckpointStore",
        FakeStore,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.runner._pretraining_cache_fingerprint",
        lambda *args, **kwargs: "f" * 64,
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")

    def reconstruct(*args: object, **kwargs: object) -> object:
        del args
        calls.append(("reconstruct", kwargs["checkpoint_store"]))
        return SimpleNamespace()

    context = LearnedQualityContext(
        reconstruct,
        model_manifest,
        model_preflight=lambda: calls.append(("model_preflight", None)),
    )
    cache_root = tmp_path / "drive" / "room_learned_test_cache"
    services = make_learned_quality_services(context, cache_root=cache_root)
    source_inventory = SimpleNamespace(digest="a" * 64)
    spec = SimpleNamespace()
    hardware = HardwareInfo("NVIDIA A100", 80.0, True, 120.0)
    run_root = tmp_path / "work" / "run"

    assert services.restore_pretraining is not None
    assert services.save_pretraining is not None
    assert services.validate_reconstruction is None
    assert (
        services.restore_pretraining(
            source_inventory=source_inventory,
            spec=spec,
            hardware=hardware,
            run_root=run_root,
        )
        is None
    )
    services.reconstruct(
        SimpleNamespace(inventory=source_inventory),
        spec=spec,
        hardware=hardware,
        output_root=run_root / "reconstruction",
    )
    services.save_pretraining(
        source_inventory=source_inventory,
        selection=SimpleNamespace(),
        reconstruction=SimpleNamespace(),
        spec=spec,
        hardware=hardware,
        run_root=run_root,
    )

    store = calls[0][1]
    assert store == (cache_root, "a" * 64)
    assert calls[1] == ("probe", "run")
    assert calls[2] == ("restore", run_root / "pretraining-restored")
    assert calls[3][0] == "reconstruct"
    assert isinstance(calls[3][1], FakeStore)
    assert calls[4] == ("save", run_root)


def test_selection_restore_requires_cpu_audit_before_learned_model_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.static_pipeline import runner as static_runner

    production = SimpleNamespace(
        discover_source=object(),
        copy_input=object(),
        select_frames=object(),
        reconstruct=object(),
        train=object(),
        polish=object(),
        build_metadata_preview=object(),
        validate_bundle=object(),
        publish_result=object(),
        publish_diagnostics=object(),
        inspect_hardware=object(),
        preflight=object(),
        assemble_reports=object(),
        resolve_input=object(),
    )
    monkeypatch.setattr(static_runner, "_production_services", lambda: production)
    source = SimpleNamespace(digest="a" * 64)
    selection = SelectionOutput(
        source,
        SimpleNamespace(image_set_digest="b" * 64),
        tmp_path,
        tmp_path / "selection_manifest.json",
    )
    calls: list[str] = []

    class FakeStore:
        def __init__(self, _cache_root: Path, *, input_identity: str) -> None:
            self.input_identity = input_identity

        def restore_milestone(self, *_args: object, **_kwargs: object) -> object:
            calls.append("selection")
            return SimpleNamespace(value=selection)

    monkeypatch.setattr(
        "experiments.learned_quality.runner.LearnedCheckpointStore", FakeStore
    )
    monkeypatch.setattr(
        "experiments.learned_quality.runtime._colmap_cache_fingerprints",
        lambda *_args, **_kwargs: ("c" * 64,),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.require_audit_receipt",
        lambda *_args, **_kwargs: calls.append("audit") or object(),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.make_audit_inputs",
        lambda **_kwargs: object(),
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    context = LearnedQualityContext(
        lambda *_args, **_kwargs: object(),
        model_manifest,
        model_preflight=lambda: calls.append("model_preflight"),
    )
    services = make_learned_quality_services(
        context,
        cache_root=tmp_path / "cache",
    )

    restored = services.restore_selection(
        source_inventory=source,
        spec=to_static_run_spec(LearnedQualityRunSpec(input_folder="room")),
        hardware=HardwareInfo("NVIDIA A100", 80.0, True, 120.0, colmap_gpu_sift=True),
        run_root=tmp_path / "run",
    )

    assert restored is selection
    assert calls == ["selection", "audit", "model_preflight"]

    calls.clear()

    def reject_audit(*_args: object, **_kwargs: object) -> object:
        calls.append("audit")
        raise RuntimeError("CPU cache audit required")

    monkeypatch.setattr(
        "experiments.learned_quality.audit.require_audit_receipt",
        reject_audit,
    )
    with pytest.raises(RuntimeError, match="CPU cache audit"):
        services.restore_selection(
            source_inventory=source,
            spec=to_static_run_spec(LearnedQualityRunSpec(input_folder="room")),
            hardware=HardwareInfo(
                "NVIDIA A100", 80.0, True, 120.0, colmap_gpu_sift=True
            ),
            run_root=tmp_path / "run-again",
        )
    assert calls == ["selection", "audit"]


def test_round0_recovery_restores_audited_selection_from_cpu_notebook_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.static_pipeline import runner as static_runner
    from experiments.learned_quality import runner as learned_runner
    from experiments.learned_quality.cache import CheckpointKind
    from experiments.learned_quality.milestones import (
        MilestoneInputs,
        milestone_fingerprint,
    )

    production = SimpleNamespace(
        discover_source=object(),
        copy_input=object(),
        select_frames=object(),
        reconstruct=object(),
        train=object(),
        polish=object(),
        build_metadata_preview=object(),
        validate_bundle=object(),
        publish_result=object(),
        publish_diagnostics=object(),
        inspect_hardware=object(),
        preflight=object(),
        assemble_reports=object(),
        resolve_input=object(),
    )
    monkeypatch.setattr(static_runner, "_production_services", lambda: production)
    source = SimpleNamespace(digest="a" * 64)
    frames = tmp_path / "frames"
    frames.mkdir()
    selection = SelectionOutput(
        source,
        SimpleNamespace(image_set_digest="b" * 64),
        frames,
        tmp_path / "selection_manifest.json",
    )
    spec = to_static_run_spec(LearnedQualityRunSpec(input_folder="room"))
    hardware = HardwareInfo("NVIDIA A100", 80.0, True, 120.0, colmap_gpu_sift=True)
    legacy_producer = "d" * 64
    legacy_python = "3.12.13"
    monkeypatch.setattr(learned_runner, "producer_code_digest", lambda *_args: "c" * 64)
    monkeypatch.setattr(
        learned_runner,
        "_CPU_AUDIT_SELECTION_PRODUCER_SHA256",
        legacy_producer,
        raising=False,
    )
    monkeypatch.setattr(
        learned_runner,
        "_CPU_AUDIT_PYTHON_VERSION",
        legacy_python,
        raising=False,
    )
    legacy_fingerprint = milestone_fingerprint(
        CheckpointKind.SELECTION,
        MilestoneInputs(
            source_digest=source.digest,
            selection_digest=source.digest,
            settings=spec.frame_selection.model_dump(mode="json"),
            model_manifest_sha256="0" * 64,
            tool_versions={"python": legacy_python},
            producer_code_sha256=legacy_producer,
            upstream={},
        ),
    )
    restored_refs: list[object] = []
    reconstruct_kwargs: dict[str, object] = {}

    class FakeStore:
        def __init__(self, _cache_root: Path, *, input_identity: str) -> None:
            self.input_identity = input_identity

        def restore_milestone(self, ref: object, **_kwargs: object) -> object | None:
            restored_refs.append(ref)
            if getattr(ref, "fingerprint", None) == legacy_fingerprint:
                return SimpleNamespace(value=selection)
            return None

    class FakeMilestoneSession:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(learned_runner, "LearnedCheckpointStore", FakeStore)
    monkeypatch.setattr(learned_runner, "MilestoneSession", FakeMilestoneSession)
    monkeypatch.setattr(
        "experiments.learned_quality.runtime._colmap_cache_fingerprints",
        lambda *_args, **_kwargs: ("e" * 64,),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.require_audit_receipt",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.make_audit_inputs",
        lambda **_kwargs: object(),
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")

    def reconstruct(*_args: object, **kwargs: object) -> object:
        reconstruct_kwargs.update(kwargs)
        return object()

    services = make_learned_quality_services(
        LearnedQualityContext(reconstruct, model_manifest),
        cache_root=tmp_path / "cache",
        recovery_mode="round0_output_first_v1",
    )

    restored = services.restore_selection(
        source_inventory=source,
        spec=spec,
        hardware=hardware,
        run_root=tmp_path / "run",
    )
    services.reconstruct(
        restored,
        spec=spec,
        hardware=hardware,
        output_root=tmp_path / "run" / "reconstruction",
    )

    assert restored is selection
    assert len(restored_refs) == 2
    assert restored_refs[0].fingerprint != legacy_fingerprint
    assert restored_refs[1].fingerprint == legacy_fingerprint
    assert (
        reconstruct_kwargs["milestone_session"].selection_ref.fingerprint
        == legacy_fingerprint
    )


def test_selection_restore_migrates_exact_cpu_audited_colmap_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.static_pipeline import runner as static_runner

    known = "684800f437e38a019e9d96b6928b6a5d3995b93cc6c925eccd620a926756ec82"
    current = "c" * 64
    production = SimpleNamespace(
        discover_source=object(),
        copy_input=object(),
        select_frames=object(),
        reconstruct=object(),
        train=object(),
        polish=object(),
        build_metadata_preview=object(),
        validate_bundle=object(),
        publish_result=object(),
        publish_diagnostics=object(),
        inspect_hardware=object(),
        preflight=object(),
        assemble_reports=object(),
        resolve_input=object(),
    )
    monkeypatch.setattr(static_runner, "_production_services", lambda: production)
    source = SimpleNamespace(digest="a" * 64)
    selection = SelectionOutput(
        source,
        SimpleNamespace(image_set_digest="b" * 64),
        tmp_path,
        tmp_path / "selection_manifest.json",
    )
    calls: list[object] = []
    restored_attempt = SimpleNamespace(root=tmp_path / "restored-colmap")

    class FakeStore:
        def __init__(self, cache_root: Path, *, input_identity: str) -> None:
            self.cache_root = Path(cache_root)
            self.input_identity = input_identity

        def restore_milestone(self, *_args: object, **_kwargs: object) -> object:
            calls.append("selection")
            return SimpleNamespace(value=selection)

        def restore_colmap(self, fingerprint: str, *, destination: Path) -> object:
            calls.append(("restore_colmap", fingerprint, destination))
            return restored_attempt

        def publish_colmap(
            self,
            attempt: object,
            *,
            fingerprint: str,
            run_id: str,
        ) -> Path:
            calls.append(("publish_colmap", attempt, fingerprint, run_id))
            return self.cache_root / "colmap" / fingerprint

    monkeypatch.setattr(
        "experiments.learned_quality.runner.LearnedCheckpointStore", FakeStore
    )
    monkeypatch.setattr(
        "experiments.learned_quality.runtime._colmap_cache_fingerprints",
        lambda *_args, **_kwargs: (current,),
    )

    def reject_current(*_args: object, **_kwargs: object) -> object:
        calls.append("audit-current-miss")
        raise RuntimeError("CPU cache audit required")

    monkeypatch.setattr(
        "experiments.learned_quality.audit.require_audit_receipt",
        reject_current,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.find_compatible_audit_receipts",
        lambda *_args, **_kwargs: (SimpleNamespace(colmap_fingerprint=known),),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.audit.make_audit_inputs",
        lambda **_kwargs: object(),
    )
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    services = make_learned_quality_services(
        LearnedQualityContext(
            lambda *_args, **_kwargs: object(),
            model_manifest,
            model_preflight=lambda: calls.append("model_preflight"),
        ),
        cache_root=tmp_path / "cache",
    )

    restored = services.restore_selection(
        source_inventory=source,
        spec=to_static_run_spec(LearnedQualityRunSpec(input_folder="room")),
        hardware=HardwareInfo("NVIDIA A100", 80.0, True, 120.0, colmap_gpu_sift=True),
        run_root=tmp_path / "run",
    )

    assert restored is selection
    assert (
        "restore_colmap",
        known,
        tmp_path / "run" / "audited-colmap-migration",
    ) in calls
    assert ("publish_colmap", restored_attempt, current, "audit-migration") in calls
    assert calls[-1] == "model_preflight"


def test_notebook_run_enables_full_stage_reporter_and_default_drive_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    reconstruct_kwargs: dict[str, object] = {}
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text("{}", encoding="utf-8")

    def reconstruct(*_args: object, **kwargs: object) -> None:
        reconstruct_kwargs.update(kwargs)

    context = LearnedQualityContext(reconstruct, model_manifest)
    sentinel = SimpleNamespace(final_path=tmp_path / "result")

    def run(*args: object, **kwargs: object) -> object:
        del args
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "experiments.learned_quality.runner._default_context",
        lambda: context,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.runner.run_static_notebook",
        run,
    )
    drive = tmp_path / "drive"
    work = tmp_path / "work"

    result = run_learned_quality_notebook(
        LearnedQualityRunSpec(
            input_folder="myroom_test",
            recovery_mode="round0_output_first_v1",
        ),
        runtime_paths=NotebookRuntimePaths(drive, work),
    )

    assert result is sentinel
    assert isinstance(captured["reporter"], StageReporter)
    assert tuple(row.stage_id for row in LEARNED_STAGE_DEFINITIONS) == (
        "input_discovery",
        "runtime_preflight",
        "cache_restore",
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
        "gaussian_training",
        "polish",
        "metadata_preview",
        "report_assembly",
        "bundle_validation",
        "result_publication",
    )
    services = captured["services"]
    assert services.restore_pretraining is None
    assert services.save_pretraining is None
    assert services.validate_reconstruction is not None
    (tmp_path / "frames").mkdir()
    selection = SelectionOutput(
        inventory=SimpleNamespace(digest="a" * 64),
        manifest=SimpleNamespace(image_set_digest="b" * 64),
        frames_dir=tmp_path / "frames",
        source_manifest_path=tmp_path / "selection_manifest.json",
    )
    services.reconstruct(
        selection,
        spec=to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test")),
        hardware=HardwareInfo("NVIDIA A100", 80.0, True, 120.0),
        output_root=tmp_path / "run" / "reconstruction",
    )
    assert reconstruct_kwargs["recovery_mode"] == "round0_output_first_v1"


def test_complete_cache_fingerprint_tracks_selected_geometry_but_not_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.learned_quality import runtime

    monkeypatch.setattr(
        "experiments.learned_quality.runner.producer_code_digest",
        lambda *args, **kwargs: "c" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "producer_code_digest",
        lambda *args, **kwargs: "d" * 64,
    )
    monkeypatch.setattr(runtime, "_probe_colmap_version", lambda: "COLMAP 3.11.1")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    source = SimpleNamespace(digest="a" * 64)
    selection = SimpleNamespace(
        inventory=source,
        manifest=SimpleNamespace(image_set_digest="b" * 64),
    )
    hardware = HardwareInfo(
        "NVIDIA A100",
        80.0,
        True,
        120.0,
        colmap_gpu_sift=True,
    )
    spec = to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test"))
    training_only_change = spec.model_copy(
        update={
            "quality": spec.quality.model_copy(update={"n_iters": 1}),
        }
    )

    original = _pretraining_cache_fingerprint(
        source,
        selection,
        spec,
        hardware,
        manifest,
    )
    training_only = _pretraining_cache_fingerprint(
        source,
        selection,
        training_only_change,
        hardware,
        manifest,
    )
    selected_change = _pretraining_cache_fingerprint(
        source,
        SimpleNamespace(
            inventory=source,
            manifest=SimpleNamespace(image_set_digest="e" * 64),
        ),
        spec,
        hardware,
        manifest,
    )

    assert training_only == original
    assert selected_change != original


def test_selection_milestone_does_not_depend_on_learned_model_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.learned_quality.runner.producer_code_digest",
        lambda *args, **kwargs: "c" * 64,
    )
    source = SimpleNamespace(digest="a" * 64)
    spec = to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test"))
    hardware = HardwareInfo("CPU audit", 0.0, False, 120.0, colmap_gpu_sift=True)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"model":"one"}\n', encoding="utf-8")
    second.write_text('{"model":"two"}\n', encoding="utf-8")

    assert _selection_milestone_ref(source, spec, hardware, first) == (
        _selection_milestone_ref(source, spec, hardware, second)
    )


def test_late_failure_gets_learned_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "drive"
    source = drive / "captures" / "room"
    source.mkdir(parents=True)
    work = tmp_path / "work"
    published: list[tuple[Path, str]] = []

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        run_root = work / "fixed-run"
        run_root.mkdir(parents=True)
        raise RuntimeError("late training failure")

    monkeypatch.setattr(
        "experiments.learned_quality.runner.run_static_notebook",
        fail,
    )
    monkeypatch.setattr(
        "experiments.learned_quality.runner.publish_learned_diagnostics",
        lambda input_folder, run_id, files: published.append((input_folder, run_id))
        or tmp_path / "diagnostics" / run_id,
    )
    spec = LearnedQualityRunSpec(input_folder="captures/room")
    context = LearnedQualityContext(
        reconstruct=lambda *args, **kwargs: None,
        model_manifest_path=tmp_path / "model_manifest.json",
    )

    with pytest.raises(RuntimeError, match="late training failure") as caught:
        run_learned_quality_notebook(
            spec,
            runtime_paths=NotebookRuntimePaths(drive, work),
            context=context,
        )

    assert published == [(source, "fixed-run")]
    assert caught.value.diagnostics_path == tmp_path / "diagnostics" / "fixed-run"


def test_late_failure_diagnostics_include_bound_progress_log(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    progress = run_root / "logs" / "progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text('{"status":"fail"}\n', encoding="utf-8")

    files = _late_failure_files(run_root, RuntimeError("training failed"))

    assert files["logs/progress.jsonl"] == progress


def test_reported_winner_is_the_exact_geometry_decision() -> None:
    classical_decision = SimpleNamespace(passed=True)
    learned_decision = SimpleNamespace(passed=True)
    candidates = (
        SimpleNamespace(candidate_id="classical", decision=classical_decision),
        SimpleNamespace(candidate_id="learned_hybrid", decision=learned_decision),
    )
    reconstruction = SimpleNamespace(
        decision=learned_decision,
        geometry_candidates=candidates,
    )

    assert _reported_winner(reconstruction).candidate_id == "learned_hybrid"
