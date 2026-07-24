from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend.notebooks.training_config import resolve_static_training_config
from experiments.learned_quality.ablation_training import make_ablation_static_spec
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    to_static_run_spec,
)
from experiments.learned_quality.legacy_control_export import (
    select_legacy_control_variant,
)
from experiments.learned_quality.legacy_control_long_run import (
    CHECKPOINT_ITERATIONS,
    LegacyControlLongRunResult,
    LegacyControlLongRunSpec,
    LocalSnapshotWriter,
    build_resume_checkpoint,
    make_legacy_control_long_spec,
    require_native_1080p,
    run_legacy_control_long,
    validate_legacy_control_long_hardware,
)


def _base_spec() -> object:
    return to_static_run_spec(
        LearnedQualityRunSpec(
            input_folder="myroom_test",
            recovery_mode="round0_output_first_v1",
        )
    )


def test_long_run_spec_is_local_only_and_rejects_output_folders() -> None:
    spec = LegacyControlLongRunSpec(input_folder="  myroom_test  ")

    assert spec.input_folder == "myroom_test"
    assert spec.runtime_profile == "a100_legacy_control_native_1080p_30k_local"
    assert set(spec.model_fields_set) == {"input_folder"}

    for suffix in (
        "_legacy_control_5k_result",
        "_learned_test_result",
        "_training_ablation",
    ):
        with pytest.raises(ValueError, match="[Cc]hoose the input folder"):
            LegacyControlLongRunSpec(input_folder=f"myroom_test{suffix}")


def test_hardware_requires_a100_high_ram_and_large_local_disk() -> None:
    validate_legacy_control_long_hardware(
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=180.0,
            host_ram_gb=167.0,
        )
    )

    for bad in (
        SimpleNamespace(
            gpu_name="NVIDIA L4",
            vram_gb=23.0,
            disk_free_gb=180.0,
            host_ram_gb=167.0,
        ),
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=90.0,
            host_ram_gb=167.0,
        ),
        SimpleNamespace(
            gpu_name="NVIDIA A100-SXM4-80GB",
            vram_gb=80.0,
            disk_free_gb=180.0,
            host_ram_gb=83.0,
        ),
    ):
        with pytest.raises(RuntimeError):
            validate_legacy_control_long_hardware(bad)


def test_long_spec_changes_only_duration_and_native_resolution_contract() -> None:
    base = _base_spec()
    short = make_ablation_static_spec(base, select_legacy_control_variant())
    long = make_legacy_control_long_spec(base)

    short_cfg, short_resolved = resolve_static_training_config(
        short,
        source_long_edge=1_920,
        native_image_size=(1_920, 1_080),
    )
    long_cfg, resolved = resolve_static_training_config(
        long,
        source_long_edge=1_920,
        native_image_size=(1_920, 1_080),
    )

    assert resolved.n_iters == 30_000
    assert resolved.native_resolution is True
    assert resolved.image_resolution == (1_920, 1_080)
    assert resolved.resolution_long_edge_cap == 1_920
    assert resolved.multires_schedule == ()
    assert resolved.max_gaussians == 6_000_000
    assert resolved.density_start_iter == 500
    assert resolved.density_end_iter == 4_500
    assert resolved.density_interval == 100
    assert resolved.opacity_reset_interval == 3_000

    # Preserve the objective that actually passed at 5K.
    assert resolved.lambda_ssim == short_resolved.lambda_ssim
    assert resolved.lambda_lpips == short_resolved.lambda_lpips
    assert resolved.lambda_depth == short_resolved.lambda_depth == 0.0
    assert long_cfg.train.lambda_aniso == short_cfg.train.lambda_aniso
    assert long_cfg.train.densify_grad_threshold == (
        short_cfg.train.densify_grad_threshold
    )
    assert long_cfg.train.prune_min_opacity == short_cfg.train.prune_min_opacity
    assert long_cfg.train.prune_max_scale == short_cfg.train.prune_max_scale


def test_native_resolution_requires_one_exact_1920x1080_camera_set() -> None:
    require_native_1080p(
        {
            "frame_000000.png": {"width": 1_920, "height": 1_080},
            "frame_000001.png": {"width": 1_920, "height": 1_080},
        }
    )

    for cameras in (
        {},
        {"frame.png": {"width": 1_280, "height": 720}},
        {
            "a.png": {"width": 1_920, "height": 1_080},
            "b.png": {"width": 1_080, "height": 1_920},
        },
    ):
        with pytest.raises(ValueError, match="1920x1080"):
            require_native_1080p(cameras)


def _fake_trainer() -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return SimpleNamespace(
        gs=SimpleNamespace(
            sh_degree=3,
            state_for_save=lambda: {"means": torch.zeros((2, 3))},
        ),
        deform=SimpleNamespace(state_dict=lambda: {"weight": torch.ones(1)}),
        optimizer=optimizer,
        scheduler=scheduler,
        scene_extent=2.5,
    )


def test_resume_checkpoint_contains_optimizer_and_rng_state() -> None:
    trainer = _fake_trainer()
    generator = torch.Generator(device="cpu").manual_seed(1701)

    payload = build_resume_checkpoint(
        trainer,
        iteration=5_000,
        n_iters=30_000,
        camera_generator=generator,
    )

    assert payload["iter"] == 5_000
    assert payload["n_iters"] == 30_000
    assert payload["gs"]["means"].shape == (2, 3)
    assert payload["optimizer"]["param_groups"]
    assert payload["scheduler"]
    assert torch.equal(payload["camera_generator_state"], generator.get_state())
    assert payload["torch_rng_state"].dtype == torch.uint8
    assert payload["scene_extent"] == 2.5
    assert payload["sh_degree"] == 3


def test_snapshot_writer_exports_exact_atomic_boundaries_and_preserves_earlier(
    tmp_path: Path,
) -> None:
    trainer = _fake_trainer()
    generator = torch.Generator(device="cpu").manual_seed(1701)
    export_calls: list[Path] = []
    save_calls: list[Path] = []

    def exporter(
        _gs: object,
        _deform: object,
        output_dir: Path,
        **_kwargs: object,
    ) -> list[Path]:
        output = Path(output_dir)
        export_calls.append(output)
        assert output.name.startswith(".ply-")
        output.mkdir(parents=True)
        ply = output / "frame_0000.ply"
        ply.write_bytes(b"ply")
        return [ply]

    def saver(payload: object, path: Path) -> None:
        save_calls.append(Path(path))
        assert Path(path).name.startswith(".legacy_control_")
        Path(path).write_bytes(b"checkpoint")

    writer = LocalSnapshotWriter(
        output_root=tmp_path,
        camera_generator=generator,
        exporter=exporter,
        checkpoint_saver=saver,
        free_bytes=lambda _path: 200 * 1024**3,
    )

    for iteration in CHECKPOINT_ITERATIONS:
        writer(trainer, iteration, (1_920, 1_080), 3)

    expected_plys = tuple(
        tmp_path / "ply" / f"legacy_control_{iteration:06d}.ply"
        for iteration in CHECKPOINT_ITERATIONS
    )
    expected_checkpoints = tuple(
        tmp_path / "checkpoints" / f"legacy_control_{iteration:06d}.pt"
        for iteration in CHECKPOINT_ITERATIONS
    )
    assert writer.ply_paths == expected_plys
    assert writer.checkpoint_paths == expected_checkpoints
    assert all(path.read_bytes() == b"ply" for path in expected_plys)
    assert all(path.read_bytes() == b"checkpoint" for path in expected_checkpoints)
    assert len(export_calls) == len(save_calls) == 6
    assert not any(path.name.startswith(".") for path in (tmp_path / "ply").iterdir())
    assert not any(
        path.name.startswith(".") for path in (tmp_path / "checkpoints").iterdir()
    )

    with pytest.raises(ValueError, match="snapshot iteration"):
        writer(trainer, 5_001, (1_920, 1_080), 3)
    with pytest.raises(ValueError, match="1920x1080"):
        writer(trainer, 5_000, (1_280, 720), 3)

    failed = LocalSnapshotWriter(
        output_root=tmp_path / "failed",
        camera_generator=generator,
        exporter=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("export failed")
        ),
        checkpoint_saver=saver,
        free_bytes=lambda _path: 200 * 1024**3,
    )
    (failed.output_root / "ply").mkdir(parents=True)
    earlier = failed.output_root / "ply" / "legacy_control_005000.ply"
    earlier.write_bytes(b"earlier")
    with pytest.raises(RuntimeError, match="export failed"):
        failed(trainer, 10_000, (1_920, 1_080), 3)
    assert earlier.read_bytes() == b"earlier"
    assert not (failed.output_root / "ply" / "legacy_control_010000.ply").exists()


def test_outer_run_stages_once_and_returns_only_local_snapshots(
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
        disk_free_gb=180.0,
        host_ram_gb=167.0,
        colmap_gpu_sift=True,
    )
    stage_calls: list[dict[str, object]] = []
    execute_calls: list[dict[str, object]] = []

    def stage(**kwargs: object) -> SimpleNamespace:
        stage_calls.append(kwargs)
        destination = Path(kwargs["destination"])
        destination.mkdir(parents=True)
        return SimpleNamespace(
            root=destination,
            selection=SimpleNamespace(),
            reconstruction=SimpleNamespace(),
            source_digest=inventory.digest,
            pretraining_fingerprint="b" * 64,
            source_revision="source-revision",
        )

    def execute(staged: object, **kwargs: object) -> LegacyControlLongRunResult:
        execute_calls.append({"staged": staged, **kwargs})
        local_root = Path(kwargs["local_root"])
        plys = tuple(
            local_root / "ply" / f"legacy_control_{iteration:06d}.ply"
            for iteration in CHECKPOINT_ITERATIONS
        )
        checkpoints = tuple(
            local_root / "checkpoints" / f"legacy_control_{iteration:06d}.pt"
            for iteration in CHECKPOINT_ITERATIONS
        )
        for path in (*plys, *checkpoints):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return LegacyControlLongRunResult(
            run_id=str(kwargs["run_id"]),
            local_root=local_root,
            ply_paths=plys,
            checkpoint_paths=checkpoints,
        )

    result = run_legacy_control_long(
        LegacyControlLongRunSpec(input_folder="myroom_test"),
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
        pretraining_fingerprint_resolver=lambda *_args, **_kwargs: "b" * 64,
        execute_staged=execute,
    )

    assert len(stage_calls) == 1
    assert len(execute_calls) == 1
    assert Path(stage_calls[0]["destination"]).is_relative_to(work)
    assert not Path(stage_calls[0]["destination"]).is_relative_to(drive)
    assert stage_calls[0]["minimum_free_bytes"] == 100 * 1024**3
    assert result.local_root.is_relative_to(work)
    assert all(path.is_relative_to(work) for path in result.ply_paths)
    assert all(path.is_relative_to(work) for path in result.checkpoint_paths)
    assert not any("report" in part for path in result.ply_paths for part in path.parts)
    assert not any(path.is_relative_to(drive) for path in result.ply_paths)
