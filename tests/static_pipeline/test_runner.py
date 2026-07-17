from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.notebooks.models import (
    FrameSelectionMode,
    StaticNotebookRunSpec,
)
from backend.static_pipeline.reconstruction import ReconstructionGateError
from backend.static_pipeline.runner import (
    HardwareInfo,
    MetadataPreviewOutput,
    NotebookRuntimePaths,
    PretrainingRestore,
    RunnerServices,
    RuntimePreflightError,
    SelectionOutput,
    _assemble_reports_adapter,
    preflight_runtime,
    run_static_notebook,
)
from backend.static_pipeline.publish import validate_bundle
from backend.static_pipeline.progress import StageDefinition, StageReporter


RUNNER_STAGE_DEFINITIONS = tuple(
    StageDefinition(stage_id, label)
    for stage_id, label in (
        ("input_discovery", "Input discovery"),
        ("runtime_preflight", "Runtime preflight"),
        ("cache_restore", "Pre-training cache restore"),
        ("source_copy", "Source copy"),
        ("frame_selection", "Frame selection"),
        ("reconstruction", "Reconstruction"),
        ("pretraining_cache_save", "Pre-training cache save"),
        ("gaussian_training", "Gaussian training"),
        ("polish", "Splat polish"),
        ("metadata_preview", "Metadata and preview"),
        ("report_assembly", "Report assembly"),
        ("bundle_validation", "Bundle validation"),
        ("result_publication", "Drive result publication"),
    )
)


def test_balanced_l4_accepts_24gb_and_rejects_clearly_unsafe_vram() -> None:
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    preflight_runtime(
        spec,
        HardwareInfo(
            gpu_name="NVIDIA L4",
            vram_gb=24.0,
            cuda_available=True,
            disk_free_gb=120.0,
        ),
        input_size_gb=2.0,
    )
    with pytest.raises(RuntimePreflightError, match="Balanced / L4"):
        preflight_runtime(
            spec,
            HardwareInfo(
                gpu_name="T4",
                vram_gb=16.0,
                cuda_available=True,
                disk_free_gb=120.0,
            ),
            input_size_gb=2.0,
        )


def test_preflight_requires_cuda_and_sufficient_working_disk() -> None:
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    with pytest.raises(RuntimePreflightError, match="CUDA"):
        preflight_runtime(
            spec,
            HardwareInfo("CPU", 0.0, False, 120.0),
            input_size_gb=2.0,
        )
    with pytest.raises(RuntimePreflightError, match="disk"):
        preflight_runtime(
            spec,
            HardwareInfo("NVIDIA L4", 24.0, True, 22.9),
            input_size_gb=2.0,
        )


def _services(
    calls: list[str],
    *,
    gate_failure: bool = False,
) -> RunnerServices:
    inventory = SimpleNamespace(
        digest="a" * 64,
        all_files=(SimpleNamespace(size_bytes=2_000_000_000),),
    )
    selection = SimpleNamespace(image_set_digest="b" * 64)
    reconstruction = SimpleNamespace(
        decision=SimpleNamespace(passed=True, failures=()),
    )
    training = SimpleNamespace(raw_ply_path=Path("raw.ply"))
    polish = SimpleNamespace(selected_path=Path("splat.ply"))

    def record(name: str, value: object):
        def call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            return value

        return call

    def reconstruct(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append("colmap_gate")
        if gate_failure:
            decision = SimpleNamespace(failures=("registered_ratio",))
            raise ReconstructionGateError(decision, ())
        return reconstruction

    def assemble(*args: object, **kwargs: object) -> Path:
        del args
        calls.append("assemble_reports")
        bundle = Path(kwargs["bundle_root"])
        bundle.mkdir()
        (bundle / "quality_report.json").write_text("{}", encoding="utf-8")
        (bundle / "run_manifest.json").write_text("{}", encoding="utf-8")
        return bundle

    return RunnerServices(
        resolve_input=record("resolve_input", None),
        discover_source=record("discover", inventory),
        inspect_hardware=lambda _: HardwareInfo("NVIDIA L4", 24.0, True, 120.0),
        preflight=record("preflight", None),
        copy_input=record("copy_input", inventory),
        select_frames=record("select", selection),
        reconstruct=reconstruct,
        train=record("train", training),
        polish=record("polish", polish),
        build_metadata_preview=record("metadata_preview", {"warning": None}),
        assemble_reports=assemble,
        validate_bundle=record("validate_bundle", ()),
        publish_result=record(
            "publish",
            SimpleNamespace(final_path=Path("/drive/captures/room_result")),
        ),
        publish_diagnostics=record("publish_diagnostics", Path("diagnostics")),
    )


def test_runner_never_trains_before_gate_and_publishes_after_validation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    paths = NotebookRuntimePaths(tmp_path / "drive", tmp_path / "work")
    result = run_static_notebook(spec, runtime_paths=paths, services=_services(calls))

    assert calls == [
        "resolve_input",
        "discover",
        "preflight",
        "copy_input",
        "select",
        "colmap_gate",
        "train",
        "polish",
        "metadata_preview",
        "assemble_reports",
        "validate_bundle",
        "publish",
    ]
    assert result.final_path.name == "room_result"


def test_gate_failure_publishes_diagnostics_but_never_trains(tmp_path: Path) -> None:
    calls: list[str] = []
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    paths = NotebookRuntimePaths(tmp_path / "drive", tmp_path / "work")

    with pytest.raises(ReconstructionGateError):
        run_static_notebook(
            spec,
            runtime_paths=paths,
            services=_services(calls, gate_failure=True),
        )

    assert "train" not in calls
    assert "publish" not in calls
    assert calls[-1] == "publish_diagnostics"


def test_complete_pretraining_restore_skips_copy_selection_and_reconstruction(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    services = _services(calls)
    restored_selection = SimpleNamespace(
        inventory=SimpleNamespace(digest="a" * 64),
        image_set_digest="b" * 64,
    )
    restored_reconstruction = SimpleNamespace(
        decision=SimpleNamespace(passed=True, failures=()),
    )

    def restore_pretraining(**kwargs: object) -> PretrainingRestore:
        assert Path(kwargs["run_root"]).parent == tmp_path / "work"
        calls.append("restore_pretraining")
        return PretrainingRestore(restored_selection, restored_reconstruction)

    services = RunnerServices(
        **{
            **services.__dict__,
            "restore_pretraining": restore_pretraining,
        }
    )

    run_static_notebook(
        StaticNotebookRunSpec(input_folder="captures/room"),
        runtime_paths=NotebookRuntimePaths(tmp_path / "drive", tmp_path / "work"),
        services=services,
    )

    assert "restore_pretraining" in calls
    assert "copy_input" not in calls
    assert "select" not in calls
    assert "colmap_gate" not in calls
    assert calls.index("restore_pretraining") < calls.index("train")


def test_verified_pretraining_is_saved_before_training(tmp_path: Path) -> None:
    calls: list[str] = []
    services = _services(calls)

    def save_pretraining(**kwargs: object) -> None:
        assert kwargs["selection"] is not None
        assert kwargs["reconstruction"] is not None
        calls.append("save_pretraining")

    services = RunnerServices(
        **{
            **services.__dict__,
            "save_pretraining": save_pretraining,
        }
    )

    run_static_notebook(
        StaticNotebookRunSpec(input_folder="captures/room"),
        runtime_paths=NotebookRuntimePaths(tmp_path / "drive", tmp_path / "work"),
        services=services,
    )

    assert calls.index("colmap_gate") < calls.index("save_pretraining")
    assert calls.index("save_pretraining") < calls.index("train")


def test_pretraining_save_failure_publishes_diagnostics_and_prevents_training(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    services = _services(calls)

    def fail_save(**_kwargs: object) -> None:
        calls.append("save_pretraining")
        raise RuntimeError("Drive cache publication failed")

    services = RunnerServices(
        **{
            **services.__dict__,
            "save_pretraining": fail_save,
        }
    )

    with pytest.raises(RuntimeError, match="Drive cache publication failed"):
        run_static_notebook(
            StaticNotebookRunSpec(input_folder="captures/room"),
            runtime_paths=NotebookRuntimePaths(
                tmp_path / "drive",
                tmp_path / "work",
            ),
            services=services,
        )

    assert "train" not in calls
    assert calls[-1] == "publish_diagnostics"


def test_training_failure_keeps_pretraining_for_a_fresh_run_resume(
    tmp_path: Path,
) -> None:
    stored: dict[str, object] = {}
    first_calls: list[str] = []
    first = _services(first_calls)

    def save_pretraining(**kwargs: object) -> None:
        first_calls.append("save_pretraining")
        stored["selection"] = kwargs["selection"]
        stored["reconstruction"] = kwargs["reconstruction"]

    def fail_training(*args: object, **kwargs: object) -> object:
        del args, kwargs
        first_calls.append("train")
        raise RuntimeError("training failed")

    first = RunnerServices(
        **{
            **first.__dict__,
            "save_pretraining": save_pretraining,
            "train": fail_training,
        }
    )
    spec = StaticNotebookRunSpec(input_folder="captures/room")

    with pytest.raises(RuntimeError, match="training failed"):
        run_static_notebook(
            spec,
            runtime_paths=NotebookRuntimePaths(
                tmp_path / "drive",
                tmp_path / "first-work",
            ),
            services=first,
        )

    assert first_calls.index("save_pretraining") < first_calls.index("train")

    second_calls: list[str] = []
    second = _services(second_calls)

    def restore_pretraining(**kwargs: object) -> PretrainingRestore:
        assert Path(kwargs["run_root"]).parent == tmp_path / "second-work"
        second_calls.append("restore_pretraining")
        return PretrainingRestore(
            stored["selection"],
            stored["reconstruction"],
        )

    second = RunnerServices(
        **{
            **second.__dict__,
            "restore_pretraining": restore_pretraining,
        }
    )
    run_static_notebook(
        spec,
        runtime_paths=NotebookRuntimePaths(
            tmp_path / "drive",
            tmp_path / "second-work",
        ),
        services=second,
    )

    assert "restore_pretraining" in second_calls
    assert "copy_input" not in second_calls
    assert "select" not in second_calls
    assert "colmap_gate" not in second_calls
    assert second_calls.index("restore_pretraining") < second_calls.index("train")


def test_runner_prints_live_top_level_stages_and_writes_jsonl(tmp_path: Path) -> None:
    output = StringIO()
    reporter = StageReporter(RUNNER_STAGE_DEFINITIONS, stream=output)
    work = tmp_path / "work"

    run_static_notebook(
        StaticNotebookRunSpec(input_folder="captures/room"),
        runtime_paths=NotebookRuntimePaths(tmp_path / "drive", work),
        services=_services([]),
        reporter=reporter,
    )

    lines = output.getvalue().splitlines()
    assert any("START Input discovery" in line for line in lines)
    assert any("DONE  Reconstruction" in line for line in lines)
    assert any("START Gaussian training" in line for line in lines)
    assert any("DONE  Drive result publication" in line for line in lines)
    progress_logs = tuple(work.glob("*/logs/progress.jsonl"))
    assert len(progress_logs) == 1
    rows = [
        json.loads(line)
        for line in progress_logs[0].read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["stage_id"] == "input_discovery"
    assert rows[-1]["stage_id"] == "result_publication"


def test_runner_publishes_progress_log_with_failure_diagnostics(
    tmp_path: Path,
) -> None:
    captured: dict[str, Path] = {}
    services = _services([], gate_failure=True)

    def publish_diagnostics(
        _input_path: Path,
        _run_id: str,
        files: dict[str, Path],
    ) -> Path:
        captured.update(files)
        return tmp_path / "diagnostics"

    services = RunnerServices(
        **{
            **services.__dict__,
            "publish_diagnostics": publish_diagnostics,
        }
    )

    with pytest.raises(ReconstructionGateError):
        run_static_notebook(
            StaticNotebookRunSpec(input_folder="captures/room"),
            runtime_paths=NotebookRuntimePaths(
                tmp_path / "drive",
                tmp_path / "work",
            ),
            services=services,
            reporter=StageReporter(RUNNER_STAGE_DEFINITIONS, stream=StringIO()),
        )

    assert "logs/progress.jsonl" in captured
    rows = [
        json.loads(line)
        for line in captured["logs/progress.jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[-1]["stage_id"] == "reconstruction"
    assert rows[-1]["status"] == "fail"


def test_smart_and_fixed_share_identical_downstream_spec(tmp_path: Path) -> None:
    downstream: list[tuple[str, object, object]] = []

    def capture_services(mode: str) -> RunnerServices:
        services = _services([])
        original_train = services.train
        original_polish = services.polish
        original_publish = services.publish_result

        def train(*args: object, **kwargs: object) -> object:
            spec = kwargs["spec"]
            downstream.append((mode, spec.quality, spec.publish))
            return original_train(*args, **kwargs)

        return RunnerServices(
            **{
                **services.__dict__,
                "train": train,
                "polish": original_polish,
                "publish_result": original_publish,
            }
        )

    for mode in (FrameSelectionMode.SMART, FrameSelectionMode.FIXED_FPS):
        spec = StaticNotebookRunSpec(
            input_folder="captures/room",
            frame_selection={"mode": mode, "fixed_fps": 4},
        )
        run_static_notebook(
            spec,
            runtime_paths=NotebookRuntimePaths(
                tmp_path / f"drive-{mode.value}",
                tmp_path / f"work-{mode.value}",
            ),
            services=capture_services(mode.value),
        )

    assert downstream[0][1:] == downstream[1][1:]


def test_runner_rejects_resolved_input_outside_drive(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    outside = tmp_path / "outside"
    drive.mkdir()
    outside.mkdir()
    source = drive / "room"
    try:
        source.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    services = _services([])
    services = RunnerServices(**{**services.__dict__, "resolve_input": None})

    with pytest.raises(ValueError, match="Drive root"):
        run_static_notebook(
            StaticNotebookRunSpec(input_folder="room"),
            runtime_paths=NotebookRuntimePaths(drive, tmp_path / "work"),
            services=services,
        )


def test_symlinked_input_is_read_from_target_but_published_beside_user_path(
    tmp_path: Path,
) -> None:
    drive = tmp_path / "drive"
    actual = drive / "actual"
    actual.mkdir(parents=True)
    alias = drive / "room"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    published_from: list[Path] = []
    services = _services([])

    def publish(_bundle, input_folder, **_kwargs):
        published_from.append(Path(input_folder))
        return SimpleNamespace(final_path=drive / "room_result")

    services = RunnerServices(
        **{
            **services.__dict__,
            "resolve_input": None,
            "publish_result": publish,
        }
    )
    run_static_notebook(
        StaticNotebookRunSpec(input_folder="room"),
        runtime_paths=NotebookRuntimePaths(drive, tmp_path / "work"),
        services=services,
    )

    assert published_from == [alias]


def test_production_report_assembly_builds_a_valid_portable_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.static_pipeline.runner._command_version",
        lambda _command: "test-version",
    )
    monkeypatch.setattr(
        "backend.static_pipeline.runner._repository_commit",
        lambda: "6480e13",
    )
    selected_ply = tmp_path / "selected.ply"
    selected_ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    metadata_path = tmp_path / "scene_metadata.json"
    metadata_path.write_text('{"schema_version":1}', encoding="utf-8")
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"\x89PNG\r\n\x1a\npreview")
    world = tmp_path / "world"
    world.mkdir()
    (world / "collider.json").write_text('{"schema_version":1}', encoding="utf-8")
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "selection_manifest.json").write_text(
        '{"schema_version":1}', encoding="utf-8"
    )

    frame = SimpleNamespace(selected=True, reasons=(), metrics=None)
    manifest = SimpleNamespace(
        effective_mode="smart",
        image_set_digest="b" * 64,
        selected_frames=(frame,),
        frames=(frame,),
        reconstruction_guardrail=None,
    )
    dominant = SimpleNamespace(registered_ratio=0.96)
    decision = SimpleNamespace(
        passed=True,
        failures=(),
        dominant=dominant,
        uncovered_intervals=(),
    )
    reconstruction = SimpleNamespace(
        selected_manifest=manifest,
        decision=decision,
        decisions=(decision,),
        accepted_model_dir=tmp_path / "validated",
        frames_dir=frames,
    )
    selection = SelectionOutput(
        inventory=SimpleNamespace(digest="a" * 64),
        manifest=manifest,
        frames_dir=frames,
        source_manifest_path=frames / "selection_manifest.json",
    )
    training = SimpleNamespace(
        status={"status": "ok"},
        resolved_config=SimpleNamespace(run_eval=False, profile="balanced_l4"),
    )
    polish = SimpleNamespace(
        selected_path=selected_ply,
        accepted=True,
        render_metrics={"mean_psnr_db": 30.0},
    )
    metadata = MetadataPreviewOutput(
        metadata={"orientation": {"applied": True}},
        metadata_path=metadata_path,
        preview_path=preview_path,
        world_dir=world,
        preview={"path": preview_path, "fallback_used": False, "warning": None},
    )
    source = SimpleNamespace(
        digest="a" * 64,
        schema_version=1,
        all_files=(),
    )
    bundle = _assemble_reports_adapter(
        spec=StaticNotebookRunSpec(input_folder="captures/room"),
        run_id="run-2",
        source_inventory=source,
        selection=selection,
        reconstruction=reconstruction,
        training=training,
        polish=polish,
        metadata_preview=metadata,
        hardware=HardwareInfo("NVIDIA L4", 24.0, True, 100.0, True),
        timings={"training_seconds": 1.0},
        bundle_root=tmp_path / "bundle",
    )

    validate_bundle(bundle, run_id="run-2")
    report = json.loads((bundle / "quality_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["label"] == "training_view_checks"
    assert report["orientation"] == {"applied": True}
    assert (bundle / "world" / "collider.json").is_file()
    assert not any(
        path.suffix in {".html", ".js", ".wasm"} for path in bundle.rglob("*")
    )
