from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.learned_quality.ablation import primary_variants
from experiments.learned_quality.ablation_training import (
    AblationExperimentResult,
    AblationGateFailure,
    AblationPipelineRunner,
    collect_structural_snapshot,
    make_ablation_static_spec,
    prepare_ablation_scene,
    run_ablation_experiment,
    summarize_render_metrics,
)
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    to_static_run_spec,
)
from tests.experiments.learned_quality.test_training import _fixture


def _variant(experiment_id: str):
    return next(
        variant
        for variant in primary_variants()
        if variant.experiment_id == experiment_id
    )


@pytest.mark.parametrize(
    ("experiment_id", "expects_seeds", "expects_masks", "expects_depth", "expects_density"),
    (
        ("legacy_control", False, False, False, False),
        ("dense_seeds_only", True, False, False, False),
        ("masks_only", False, True, False, False),
        ("depth_only", False, False, True, False),
        ("adaptive_density_only", False, False, False, True),
        ("full_learned", True, True, True, True),
    ),
)
def test_scene_preparation_changes_only_the_requested_learned_features(
    tmp_path: Path,
    experiment_id: str,
    expects_seeds: bool,
    expects_masks: bool,
    expects_depth: bool,
    expects_density: bool,
) -> None:
    artifacts, model, scene, originals, _ = _fixture(tmp_path)
    points_path = scene / "colmap" / "sparse" / "0" / "points3D.txt"
    sparse_rows = tuple(
        line
        for line in points_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )

    prepared = prepare_ablation_scene(
        artifacts,
        accepted_model_dir=model,
        scene_dir=scene,
        variant=_variant(experiment_id),
    )

    final_rows = tuple(
        line
        for line in points_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert len(final_rows) == len(sparse_rows) + (2 if expects_seeds else 0)
    assert (prepared.validity_mask is not None) is expects_masks
    assert (prepared.depth_dir is not None) is expects_depth
    assert prepared.adaptive_density is expects_density
    assert prepared.original_frames == originals
    assert tuple(path.name for path in prepared.camera_frames) == tuple(
        frame.image_name for frame in originals
    )
    assert all((scene / "frames" / frame.image_name).read_bytes() == frame.path.read_bytes() for frame in originals)


def test_render_metrics_distinguish_masked_and_unmasked_quality() -> None:
    gt = torch.zeros((4, 4, 3), dtype=torch.float32)
    rendered = gt.clone()
    rendered[0, 0] = 1.0
    validity = torch.ones((4, 4), dtype=torch.float32)
    validity[0, 0] = 0.0

    metrics = summarize_render_metrics(rendered, gt, validity)

    assert metrics["psnr_masked"] == pytest.approx(100.0)
    assert metrics["psnr_unmasked"] < 20.0
    assert metrics["l1_masked"] == pytest.approx(0.0)
    assert metrics["l1_unmasked"] > 0.0
    assert 0.0 <= metrics["ssim_unmasked"] <= 1.0
    assert 0.0 <= metrics["ssim_masked"] <= 1.0


def test_structural_snapshot_exposes_white_scale_anisotropy_and_quantiles() -> None:
    count = 200
    means = torch.zeros((count, 3), dtype=torch.float32)
    means[:, 0] = torch.linspace(-1.0, 1.0, count)
    scales = torch.full((count, 3), 0.001, dtype=torch.float32)
    scales[:2, 0] = 0.05
    scales[:2, 1:] = 0.001
    opacities = torch.full((count, 1), 0.5, dtype=torch.float32)
    c0 = 0.28209479177387814
    sh_dc = torch.zeros((count, 1, 3), dtype=torch.float32)
    sh_dc[:20] = (1.0 - 0.5) / c0
    sh_rest = torch.zeros((count, 15, 3), dtype=torch.float32)
    gs = SimpleNamespace(
        means=means,
        get_scales=scales,
        get_opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        num_points=count,
    )

    snapshot = collect_structural_snapshot(SimpleNamespace(gs=gs))

    assert snapshot.metrics.visible_white_fraction == pytest.approx(0.10)
    assert snapshot.metrics.high_anisotropy_fraction == pytest.approx(0.01)
    assert snapshot.metrics.oversized_fraction == pytest.approx(0.01)
    assert snapshot.metrics.nonfinite_count == 0
    assert snapshot.robust_scene_extent == pytest.approx(0.995, rel=0.02)
    assert snapshot.opacity_quantiles["q50"] == pytest.approx(0.5)
    assert snapshot.sh_dc_abs_quantiles["q100"] > 1.0
    assert snapshot.max_anisotropy == pytest.approx(50.0)


def test_structural_snapshot_counts_nonfinite_parameters() -> None:
    tensor = torch.tensor([[0.0, float("nan"), 0.0]])
    gs = SimpleNamespace(
        means=tensor,
        get_scales=torch.ones((1, 3)),
        get_opacities=torch.ones((1, 1)),
        sh_dc=torch.zeros((1, 1, 3)),
        sh_rest=torch.zeros((1, 0, 3)),
        num_points=1,
    )

    snapshot = collect_structural_snapshot(SimpleNamespace(gs=gs))

    assert snapshot.metrics.nonfinite_count == 1


@pytest.mark.parametrize(
    ("experiment_id", "expects_masks", "expects_depth", "expects_density_probe"),
    (
        ("legacy_control", False, False, False),
        ("full_learned", True, True, True),
    ),
)
def test_pipeline_runner_forwards_a_fresh_deterministic_experiment(
    tmp_path: Path,
    experiment_id: str,
    expects_masks: bool,
    expects_depth: bool,
    expects_density_probe: bool,
) -> None:
    artifacts, model, scene, _, _ = _fixture(tmp_path)
    captured: dict[str, object] = {}
    original_density = object()

    def pipeline_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        trainer = SimpleNamespace(density=original_density)
        kwargs["trainer_customizer"](trainer)
        captured["installed_density"] = trainer.density
        return {"training": "done"}

    runner = AblationPipelineRunner(
        artifacts,
        accepted_model_dir=model,
        variant=_variant(experiment_id),
        pipeline_runner=pipeline_runner,
        diagnostic_root=tmp_path / "diagnostics",
        seed=1701,
    )

    status = runner(
        video_path=scene / "video.mp4",
        scene_name="scene",
        skip_foundation=False,
    )

    assert status == {"training": "done"}
    assert captured["skip_foundation"] is True
    train_kwargs = captured["trainer_train_kwargs"]
    assert ("validity_mask" in train_kwargs) is expects_masks
    assert ("density_quality_probe" in train_kwargs) is expects_density_probe
    assert train_kwargs["diagnostic_iterations"] == _variant(experiment_id).checkpoints
    assert isinstance(train_kwargs["camera_generator"], torch.Generator)
    assert callable(train_kwargs["diagnostic_callback"])
    assert (scene / "depth").is_dir() is expects_depth
    if expects_density_probe:
        assert captured["installed_density"] is not original_density
    else:
        assert captured["installed_density"] is original_density


def test_ablation_spec_is_fixed_720p_and_uses_the_variant_iteration_budget() -> None:
    base = to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test"))

    spec = make_ablation_static_spec(base, _variant("full_learned"))

    assert spec.quality.n_iters == 5_000
    assert spec.quality.max_gaussians == base.quality.max_gaussians
    assert spec.quality.advanced.foundation is False
    assert spec.quality.advanced.run_eval is False
    assert spec.quality.advanced.resolution_long_edge_cap == 1_280
    assert spec.quality.advanced.multires_schedule == [(0, 720)]
    assert spec.quality.advanced.density_start_iter == 500
    assert spec.quality.advanced.density_end_iter == 4_500


def _checkpoint(experiment_id: str, iteration: int = 500):
    from experiments.learned_quality.ablation import (
        AblationCheckpoint,
        StructuralMetrics,
    )

    return AblationCheckpoint(
        experiment_id=experiment_id,
        iteration=iteration,
        psnr_unmasked=20.0,
        psnr_masked=21.0,
        ssim_unmasked=0.8,
        ssim_masked=0.9,
        l1_unmasked=0.1,
        l1_masked=0.05,
        gaussian_count=100,
        structural=StructuralMetrics(0.0, 0.0, 0.0, 0),
    )


def test_run_ablation_experiment_returns_the_local_ply_and_checkpoints(
    tmp_path: Path,
) -> None:
    variant = _variant("legacy_control")
    checkpoints = tuple(
        _checkpoint(variant.experiment_id, iteration)
        for iteration in variant.checkpoints
    )
    bundle = object()
    prepared = SimpleNamespace(reconstruction=bundle)
    reconstruction = SimpleNamespace(
        bundle=bundle,
        artifacts=object(),
        accepted_model_dir=tmp_path / "model",
        acceptance=None,
    )
    raw_ply = tmp_path / "frame_0000.ply"
    raw_ply.write_bytes(b"ply")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    def validated_runner(_prepared, spec, **kwargs):
        runner = kwargs["pipeline_runner"]
        runner.collector = SimpleNamespace(
            checkpoints=list(checkpoints),
            decisions=[
                SimpleNamespace(stop=False, reasons=()) for _ in checkpoints
            ],
        )
        return SimpleNamespace(
            raw_ply_path=raw_ply,
            status={"ok": True},
            resolved_config=SimpleNamespace(n_iters=spec.quality.n_iters),
            run_manifest_path=manifest,
        )

    result = run_ablation_experiment(
        prepared,
        to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test")),
        reconstruction,
        variant,
        diagnostic_root=tmp_path / "diagnostics",
        validated_training_runner=validated_runner,
        pipeline_runner=lambda **_kwargs: {},
    )

    assert isinstance(result, AblationExperimentResult)
    assert result.passed is True
    assert result.stopped_early is False
    assert result.raw_ply_path == raw_ply
    assert result.checkpoints == checkpoints


def test_run_ablation_experiment_turns_a_gate_failure_into_a_receipt(
    tmp_path: Path,
) -> None:
    variant = _variant("masks_only")
    checkpoint = _checkpoint(variant.experiment_id)
    decision = SimpleNamespace(stop=True, reasons=("visible_white_fraction",))
    bundle = object()
    prepared = SimpleNamespace(reconstruction=bundle)
    reconstruction = SimpleNamespace(
        bundle=bundle,
        artifacts=object(),
        accepted_model_dir=tmp_path / "model",
        acceptance=None,
    )

    def validated_runner(*_args, **_kwargs):
        raise AblationGateFailure(checkpoint, decision)

    result = run_ablation_experiment(
        prepared,
        to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test")),
        reconstruction,
        variant,
        diagnostic_root=tmp_path / "diagnostics",
        validated_training_runner=validated_runner,
        pipeline_runner=lambda **_kwargs: {},
    )

    assert result.passed is False
    assert result.stopped_early is True
    assert result.stop_reasons == ("visible_white_fraction",)
    assert result.raw_ply_path is None
    assert result.checkpoints == (checkpoint,)
