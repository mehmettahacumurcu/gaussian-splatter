from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.static_pipeline.runner import HardwareInfo, NotebookRuntimePaths
from experiments.learned_quality.contracts import LearnedQualityRunSpec
from experiments.learned_quality.runner import (
    LearnedQualityContext,
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
