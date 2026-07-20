from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.static_pipeline.contracts import GateDecision
from backend.static_pipeline.training import ReconstructionValidation
from experiments.learned_quality.contracts import (
    FrameArtifact,
    GeometryAcceptance,
    LearnedArtifacts,
    LearnedReconstructionOutput,
)
from experiments.learned_quality.density import AdaptiveDensityController
from experiments.learned_quality.geometry import evaluate_output_first_checks
from experiments.learned_quality.training import (
    make_experiment_pipeline_runner,
    make_output_first_reconstruction_validator,
)
from tests.static_pipeline.fixtures import write_colmap_text_model
from tests.static_pipeline.test_training import _prepared


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 4), (value, value, value)).save(path)


def _fixture(
    tmp_path: Path,
    *,
    photometric_decision: str = "accepted",
    registered_indices: tuple[int, ...] | None = None,
):
    source_frames = tmp_path / "source-frames"
    training_frames = tmp_path / "training-frames"
    mask_root = tmp_path / "masks"
    depth_root = tmp_path / "depth"
    originals = []
    corrected = []
    fused = []
    validated_depth = []
    for index in range(2):
        name = f"frame_{index:06d}.png"
        frame_id = f"frame-{index}"
        original_path = source_frames / name
        training_path = training_frames / name
        _image(original_path, 20 + index)
        _image(training_path, 80 + index)
        original = FrameArtifact(name, frame_id, original_path, _sha(original_path))
        training = FrameArtifact(name, frame_id, training_path, _sha(training_path))
        originals.append(original)
        corrected.append(training)

        validity_path = mask_root / name
        validity_path.parent.mkdir(parents=True, exist_ok=True)
        validity = np.full((4, 6), 255, dtype=np.uint8)
        validity[0, index] = 0
        Image.fromarray(validity, mode="L").save(validity_path)
        fused.append(
            SimpleNamespace(
                frame=original,
                training_validity_path=validity_path,
                training_validity_sha256=_sha(validity_path),
                width=6,
                height=4,
            )
        )

        depth_path = depth_root / f"frame_{index:06d}_depth.npy"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, np.full((4, 6), index + 1, dtype=np.float32))
        validated_depth.append(
            SimpleNamespace(
                frame=original,
                depth_path=depth_path,
                depth_sha256=_sha(depth_path),
                width=6,
                height=4,
            )
        )

    seeds_path = tmp_path / "dense_seeds.npz"
    np.savez(
        seeds_path,
        xyz=np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], np.float32),
        rgb=np.array([[1, 2, 3], [4, 5, 6]], np.uint8),
        confidence=np.array([0.9, 0.8], np.float32),
        view_support=np.array([3, 4], np.uint16),
    )
    dense = SimpleNamespace(
        npz_path=seeds_path,
        npz_sha256=_sha(seeds_path),
        point_count=2,
    )
    photometric = SimpleNamespace(
        decision=photometric_decision,
        original_frames=tuple(originals),
        training_frames=(
            tuple(corrected) if photometric_decision == "accepted" else tuple(originals)
        ),
        training_rgb_digest="b" * 64,
    )
    artifacts = LearnedArtifacts(
        masks=SimpleNamespace(frames=tuple(fused)),
        photometric=photometric,
        depth=SimpleNamespace(
            frames=tuple(validated_depth),
            dense_seeds=dense,
        ),
        dense_seeds=dense,
    )
    registered_names = tuple(frame.image_name for frame in originals)
    if registered_indices is not None:
        registered_names = tuple(
            originals[index].image_name for index in registered_indices
        )
    model = write_colmap_text_model(
        tmp_path / "accepted-model",
        registered_names,
        (0.2, 0.3),
        (2, 2),
    )
    scene = tmp_path / "data" / "scene"
    scene_frames = scene / "frames"
    scene_model = scene / "colmap" / "sparse" / "0"
    scene_frames.mkdir(parents=True)
    scene_model.mkdir(parents=True)
    for frame in originals:
        (scene_frames / frame.image_name).write_bytes(frame.path.read_bytes())
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (scene_model / name).write_bytes((model / name).read_bytes())
    return artifacts, model, scene, tuple(originals), tuple(corrected)


def test_output_first_training_validator_requires_exact_acceptance_join(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    model_dir = prepared.reconstruction.accepted_model_dir
    metrics = replace(
        prepared.reconstruction.decision.dominant,
        model_dir=model_dir,
        registered_count=658,
        registered_ratio=0.8225,
        registered_share=1.0,
        temporal_coverage_s=99.0,
        max_interior_gap_s=9.59,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=1.217,
        p95_reprojection_error_px=2.194,
        median_track_length=5.0,
        sparse_point_count=65_367,
        valid_names_intrinsics_and_poses=True,
    )
    decision = GateDecision(
        passed=False,
        dominant=metrics,
        failures=("registered_ratio", "interior_gap", "median_reprojection"),
        uncovered_intervals=(),
        retry_recommended=True,
    )
    bundle = replace(
        prepared.reconstruction,
        decision=decision,
        decisions=(decision,),
    )
    prepared = replace(prepared, reconstruction=bundle)
    model_hashes = {
        name: _sha(model_dir / name)
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest=prepared.selection_digest,
        model_hashes=model_hashes,
        strict_failures=decision.failures,
        metrics=metrics,
        checks=evaluate_output_first_checks(decision),
        colmap_fingerprint="b" * 64,
    )
    reconstruction = LearnedReconstructionOutput(
        bundle=bundle,
        frames_dir=prepared.frames_dir,
        artifacts=LearnedArtifacts(),
        geometry_candidates=(),
        acceptance=acceptance,
    )
    validation = ReconstructionValidation(
        prepared=prepared,
        manifest=bundle.selected_manifest,
        current_decision=decision,
        model_hashes=model_hashes,
    )
    validator = make_output_first_reconstruction_validator(reconstruction)

    validator(validation)

    mutated = dict(model_hashes)
    mutated["images.txt"] = "0" * 64
    with pytest.raises(ValueError, match="model hashes"):
        validator(replace(validation, model_hashes=mutated))


def test_runner_installs_evidence_only_inside_training_scene(tmp_path: Path) -> None:
    artifacts, model, scene, originals, corrected = _fixture(tmp_path)
    source_model_before = {
        path.name: path.read_bytes() for path in model.iterdir() if path.is_file()
    }
    calls: list[dict[str, object]] = []

    def base_runner(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        assert kwargs["skip_foundation"] is True
        assert (scene / "frames" / originals[0].image_name).read_bytes() == corrected[
            0
        ].path.read_bytes()
        points = (scene / "colmap" / "sparse" / "0" / "points3D.txt").read_text()
        assert "10 11 12 1 2 3 0" in points
        assert "20 21 22 4 5 6 0" in points
        assert (scene / "depth" / "frame_000000_depth.npy").is_file()

        trainer = SimpleNamespace(gs=SimpleNamespace(num_points=123), density=None)
        kwargs["trainer_customizer"](trainer)
        assert isinstance(trainer.density, AdaptiveDensityController)
        extra = kwargs["trainer_train_kwargs"]
        assert len(extra["validity_mask"]) == 2
        assert float(extra["validity_mask"][0][0, 0]) == 0.0
        assert callable(extra["density_quality_probe"])
        return {"training": "ok"}

    runner = make_experiment_pipeline_runner(
        artifacts,
        accepted_model_dir=model,
        pipeline_runner=base_runner,
    )
    status = runner(
        video_path=scene / "video.mp4",
        scene_name="scene",
        cfg=SimpleNamespace(),
        skip_foundation=False,
    )

    assert status == {"training": "ok"}
    assert runner.final_gaussian_count == 123
    assert runner.density_history_path.is_file()
    assert {
        path.name: path.read_bytes() for path in model.iterdir() if path.is_file()
    } == source_model_before
    assert all(
        frame.path.read_bytes() != corrected[index].path.read_bytes()
        for index, frame in enumerate(originals)
    )
    assert len(calls) == 1


def test_training_handoff_aligns_probe_and_masks_to_registered_subset(
    tmp_path: Path,
) -> None:
    artifacts, model, scene, originals, _ = _fixture(
        tmp_path,
        registered_indices=(1,),
    )
    calls: list[dict[str, object]] = []

    def base_runner(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        extra = kwargs["trainer_train_kwargs"]
        assert len(extra["validity_mask"]) == 1
        assert float(extra["validity_mask"][0][0, 0]) == 1.0
        assert float(extra["validity_mask"][0][0, 1]) == 0.0
        assert callable(extra["density_quality_probe"])
        trainer = SimpleNamespace(gs=SimpleNamespace(num_points=3), density=None)
        kwargs["trainer_customizer"](trainer)
        return {}

    runner = make_experiment_pipeline_runner(
        artifacts,
        accepted_model_dir=model,
        pipeline_runner=base_runner,
    )

    runner(video_path=scene / "video.mp4", scene_name="scene", cfg=SimpleNamespace())

    assert tuple(frame.image_name for frame in originals) == (
        "frame_000000.png",
        "frame_000001.png",
    )
    assert len(calls) == 1


def test_rejected_photometric_evidence_keeps_original_rgb(tmp_path: Path) -> None:
    artifacts, model, scene, originals, _ = _fixture(
        tmp_path, photometric_decision="rejected"
    )

    def base_runner(**kwargs: object) -> dict[str, object]:
        trainer = SimpleNamespace(gs=SimpleNamespace(num_points=3), density=None)
        kwargs["trainer_customizer"](trainer)
        return {}

    runner = make_experiment_pipeline_runner(
        artifacts,
        accepted_model_dir=model,
        pipeline_runner=base_runner,
    )
    runner(video_path=scene / "video.mp4", scene_name="scene", cfg=SimpleNamespace())

    assert all(
        (scene / "frames" / frame.image_name).read_bytes() == frame.path.read_bytes()
        for frame in originals
    )
    assert runner.fallbacks == ("photometric_rejected_original_rgb",)


def test_validated_depth_may_keep_its_processed_resolution(tmp_path: Path) -> None:
    artifacts, model, scene, _, _ = _fixture(tmp_path)
    for depth in artifacts.depth.frames:
        np.save(depth.depth_path, np.full((2, 3), 1.0, dtype=np.float32))
        depth.depth_sha256 = _sha(depth.depth_path)
        depth.width = 3
        depth.height = 2

    def base_runner(**kwargs: object) -> dict[str, object]:
        installed = np.load(scene / "depth" / "frame_000000_depth.npy")
        assert installed.shape == (2, 3)
        trainer = SimpleNamespace(gs=SimpleNamespace(num_points=3), density=None)
        kwargs["trainer_customizer"](trainer)
        return {}

    runner = make_experiment_pipeline_runner(
        artifacts,
        accepted_model_dir=model,
        pipeline_runner=base_runner,
    )
    runner(video_path=scene / "video.mp4", scene_name="scene", cfg=SimpleNamespace())


def test_join_failure_happens_before_any_training_scene_mutation(
    tmp_path: Path,
) -> None:
    artifacts, model, scene, originals, _ = _fixture(tmp_path)
    bad_mask = SimpleNamespace(
        **{
            **vars(artifacts.masks.frames[0]),
            "frame": FrameArtifact(
                originals[0].image_name,
                "wrong-frame-id",
                originals[0].path,
                originals[0].sha256,
            ),
        }
    )
    artifacts = LearnedArtifacts(
        masks=SimpleNamespace(frames=(bad_mask, artifacts.masks.frames[1])),
        photometric=artifacts.photometric,
        depth=artifacts.depth,
        dense_seeds=artifacts.dense_seeds,
    )
    before = {
        path.relative_to(scene).as_posix(): path.read_bytes()
        for path in scene.rglob("*")
        if path.is_file()
    }
    called = False

    def base_runner(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    runner = make_experiment_pipeline_runner(
        artifacts,
        accepted_model_dir=model,
        pipeline_runner=base_runner,
    )
    with pytest.raises(ValueError, match="exact frame join"):
        runner(
            video_path=scene / "video.mp4", scene_name="scene", cfg=SimpleNamespace()
        )

    after = {
        path.relative_to(scene).as_posix(): path.read_bytes()
        for path in scene.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not called
