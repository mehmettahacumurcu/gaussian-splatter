from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.model.gaussian_model import GaussianModel
from experiments.learned_quality.contracts import LearnedQualityRunSpec, to_static_run_spec
from experiments.learned_quality.floor_recovery_training import (
    DiagnosticRenderView,
    FLOOR_RECOVERY_CHECKPOINTS,
    FloorCheckpointMetrics,
    FloorDiagnosticProbe,
    initialize_floor_seed_slice,
    make_floor_recovery_static_spec,
    prepare_floor_recovery_scene,
    run_floor_recovery_training,
    seed_floor_recovery_randomness,
)
from experiments.learned_quality.floor_recovery import FloorSeedArtifact
from tests.experiments.learned_quality.test_training import _fixture


def _trainer() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    original = np.array(
        ((-1.0, 0.2, 0.0), (1.0, 0.3, 0.0), (0.0, 0.5, 1.0)),
        dtype=np.float32,
    )
    floor_xyz = np.array(
        ((-0.2, 0.0, 1.5), (0.0, 0.0, 1.5), (0.2, 0.0, 1.5)),
        dtype=np.float32,
    )
    floor_rgb = np.array(
        ((110, 100, 90), (120, 110, 100), (130, 120, 110)), dtype=np.uint8
    )
    points = torch.from_numpy(np.concatenate((original, floor_xyz)))
    colors = torch.from_numpy(
        np.concatenate(
            (np.full((len(original), 3), 0.5, dtype=np.float32), floor_rgb / 255.0)
        )
    ).float()
    gs = GaussianModel(points, init_colors=colors, sh_degree=3, fourier_K=0)
    return SimpleNamespace(gs=gs), floor_xyz, floor_rgb


def _snapshot_original(gs: GaussianModel, count: int) -> dict[str, torch.Tensor]:
    return {
        name: value.detach()[:count].clone()
        for name, value in (
            ("means", gs.means),
            ("scales", gs.scales),
            ("quats", gs.quats),
            ("opacities", gs.opacities),
            ("sh_dc", gs.sh_dc),
            ("sh_rest", gs.sh_rest),
        )
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floor_artifact(tmp_path: Path) -> FloorSeedArtifact:
    root = tmp_path / "floor-artifact"
    root.mkdir()
    xyz = np.array(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)), np.float32)
    rgb = np.array(((100, 90, 80), (110, 100, 90)), np.uint8)
    npz = root / "floor_seeds.npz"
    np.savez(
        npz,
        xyz=xyz,
        rgb=rgb,
        confidence=np.array((0.9, 0.8), np.float32),
        view_support=np.array((3, 4), np.uint16),
        source_frame_index=np.array((0, 1), np.int32),
        hole_cell_id=np.array((2, 3), np.int32),
    )
    fingerprint = "c" * 64
    metadata = root / "floor_seeds.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "learned_quality.floor_recovery.v1",
                "content_fingerprint": fingerprint,
                "npz_sha256": _sha(npz),
                "point_count": 2,
            }
        ),
        encoding="utf-8",
    )
    plane = root / "floor_plane.json"
    plane.write_text(
        json.dumps(
            {
                "schema": "learned_quality.floor_recovery.v1",
                "normal": [0.0, 1.0, 0.0],
                "offset": 0.0,
                "inlier_tolerance": 0.02,
            }
        ),
        encoding="utf-8",
    )
    holes = root / "floor_holes.json"
    holes.write_text(
        json.dumps(
            {
                "schema": "learned_quality.floor_recovery.v1",
                "cell_width": 0.04,
            }
        ),
        encoding="utf-8",
    )
    return FloorSeedArtifact(npz, metadata, plane, holes, 2, fingerprint)


def test_floor_seed_initializer_changes_only_verified_trailing_slice() -> None:
    trainer, floor_xyz, floor_rgb = _trainer()
    before = _snapshot_original(trainer.gs, 3)

    initialize_floor_seed_slice(
        trainer,
        original_count=3,
        floor_xyz=floor_xyz,
        floor_rgb=floor_rgb,
        maximum_scale=0.025,
    )

    for name, expected in before.items():
        assert torch.equal(getattr(trainer.gs, name)[:3], expected)
    assert torch.allclose(
        trainer.gs.opacities[3:], torch.full((3, 1), -4.0)
    )
    assert torch.all(trainer.gs.get_scales[3:] <= 0.025 + 1e-7)
    assert torch.all(trainer.gs.quats[3:, 0] == 1.0)
    assert torch.all(trainer.gs.quats[3:, 1:] == 0.0)
    assert torch.all(trainer.gs.sh_rest[3:] == 0.0)


def test_floor_probe_does_not_count_low_opacity_initial_seeds_as_recovered() -> None:
    trainer, floor_xyz, floor_rgb = _trainer()
    initialize_floor_seed_slice(
        trainer,
        original_count=3,
        floor_xyz=floor_xyz,
        floor_rgb=floor_rgb,
        maximum_scale=0.025,
    )
    probe = FloorDiagnosticProbe(
        floor_xyz=floor_xyz,
        plane_normal=np.array((0.0, 1.0, 0.0), dtype=np.float32),
        plane_offset=0.0,
        plane_tolerance=0.02,
        maximum_scale=0.025,
    )

    view = DiagnosticRenderView(
        K=torch.tensor(((4.0, 0.0, 3.0), (0.0, 4.0, 2.0), (0.0, 0.0, 1.0))),
        w2c=torch.eye(4),
        alpha=torch.zeros((5, 7)),
        depth=torch.full((5, 7), 2.0),
        perturbed_w2c=torch.eye(4),
        perturbed_alpha=torch.zeros((5, 7)),
        perturbed_depth=torch.full((5, 7), 2.0),
    )

    payload = probe(trainer, 0, (7, 5), 3, (view,))

    assert payload["occupied_hole_fraction"] == 0.0
    assert payload["floor_alpha_coverage"] == 0.0
    assert payload["residual_hole_fraction"] == 1.0


def test_floor_probe_measures_rendered_floor_alpha_and_plane_depth() -> None:
    _, floor_xyz, _ = _trainer()
    probe = FloorDiagnosticProbe(
        floor_xyz=floor_xyz,
        plane_normal=np.array((0.0, 1.0, 0.0), dtype=np.float32),
        plane_offset=0.0,
        plane_tolerance=0.02,
        maximum_scale=0.025,
    )
    K = torch.tensor(
        ((8.0, 0.0, 4.0), (0.0, 8.0, 3.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    alpha = torch.ones((7, 9), dtype=torch.float32)
    depth = torch.full((7, 9), 1.5, dtype=torch.float32)
    perturbed_w2c = torch.eye(4, dtype=torch.float32)
    perturbed_w2c[0, 3] = 0.02
    view = DiagnosticRenderView(
        K=K,
        w2c=torch.eye(4, dtype=torch.float32),
        alpha=alpha,
        depth=depth,
        perturbed_w2c=perturbed_w2c,
        perturbed_alpha=alpha,
        perturbed_depth=depth,
    )

    payload = probe(object(), 5_000, (9, 7), 3, (view,))

    assert payload["floor_alpha_coverage"] == 1.0
    assert payload["residual_hole_fraction"] == 0.0
    assert payload["plane_depth_relative_error"] == pytest.approx(0.0)
    assert payload["perturbed_depth_disagreement_ratio"] == pytest.approx(1.0)


def test_floor_seed_initializer_rejects_count_position_and_color_mismatch() -> None:
    trainer, floor_xyz, floor_rgb = _trainer()
    with pytest.raises(RuntimeError, match="count mismatch"):
        initialize_floor_seed_slice(
            trainer,
            original_count=4,
            floor_xyz=floor_xyz,
            floor_rgb=floor_rgb,
            maximum_scale=0.025,
        )

    trainer, floor_xyz, floor_rgb = _trainer()
    with pytest.raises(RuntimeError, match="slice does not match"):
        initialize_floor_seed_slice(
            trainer,
            original_count=3,
            floor_xyz=floor_xyz[::-1].copy(),
            floor_rgb=floor_rgb,
            maximum_scale=0.025,
        )

    trainer, floor_xyz, floor_rgb = _trainer()
    with pytest.raises(RuntimeError, match="colors do not match"):
        initialize_floor_seed_slice(
            trainer,
            original_count=3,
            floor_xyz=floor_xyz,
            floor_rgb=floor_rgb[::-1].copy(),
            maximum_scale=0.025,
        )


def test_floor_recovery_spec_is_bounded_depth_plus_legacy_density() -> None:
    base = to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test"))
    spec = make_floor_recovery_static_spec(base)

    assert spec.quality.n_iters == 5_000
    assert spec.quality.advanced.lambda_depth is None
    assert spec.quality.advanced.density_start_iter == 500
    assert spec.quality.advanced.density_end_iter == 4_500
    assert spec.quality.advanced.multires_schedule == [(0, 720)]
    assert FLOOR_RECOVERY_CHECKPOINTS == (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)


def test_floor_recovery_randomness_is_reproducible() -> None:
    seed_floor_recovery_randomness(1701)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    seed_floor_recovery_randomness(1701)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert first == second


def test_floor_scene_uses_legacy_rgb_depth_and_only_appends_verified_seeds(
    tmp_path: Path,
) -> None:
    artifacts, model, scene, originals, _ = _fixture(tmp_path)
    prepared = prepare_floor_recovery_scene(
        artifacts,
        accepted_model_dir=model,
        scene_dir=scene,
        floor_artifact=_floor_artifact(tmp_path),
    )

    assert prepared.validity_mask is None
    assert prepared.adaptive_density is False
    assert prepared.original_frames == originals
    assert prepared.floor_xyz.shape == (2, 3)
    assert len(tuple(prepared.depth_dir.glob("*_depth.npy"))) == 2
    rows = [
        line
        for line in (
            scene / "colmap" / "sparse" / "0" / "points3D.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(rows) == prepared.original_point_count + 2


def test_run_floor_training_requires_every_global_and_floor_checkpoint(
    tmp_path: Path,
) -> None:
    bundle = object()
    prepared = SimpleNamespace(reconstruction=bundle)
    reconstruction = SimpleNamespace(
        bundle=bundle,
        artifacts=object(),
        accepted_model_dir=tmp_path / "model",
        acceptance=None,
    )
    checkpoints = tuple(
        SimpleNamespace(
            iteration=iteration,
            depth_median=2.0,
            perturbed_depth_median=2.0,
        )
        for iteration in FLOOR_RECOVERY_CHECKPOINTS
    )
    floor = tuple(
        FloorCheckpointMetrics(iteration, 0.5, 0.5, 0.1, 1.0, 0.5)
        for iteration in FLOOR_RECOVERY_CHECKPOINTS
    )
    raw_ply = tmp_path / "output" / "point_cloud" / "iteration_5000" / "ply.ply"
    raw_ply.parent.mkdir(parents=True)
    raw_ply.write_bytes(b"ply")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    def validated_runner(_prepared, spec, **kwargs):
        runner = kwargs["pipeline_runner"]
        runner.collector = SimpleNamespace(checkpoints=list(checkpoints))
        runner.floor_probe = SimpleNamespace(checkpoints=list(floor))
        assert spec.quality.n_iters == 5_000
        return SimpleNamespace(
            raw_ply_path=raw_ply,
            run_manifest_path=manifest,
            status={"ok": True},
        )

    result = run_floor_recovery_training(
        prepared,
        to_static_run_spec(LearnedQualityRunSpec(input_folder="myroom_test")),
        reconstruction,
        _floor_artifact(tmp_path),
        diagnostic_root=tmp_path / "diagnostics",
        pipeline_runner=lambda **_kwargs: {},
        validated_training_runner=validated_runner,
    )

    assert result.raw_ply_path == raw_ply
    assert result.floor_checkpoints == floor
