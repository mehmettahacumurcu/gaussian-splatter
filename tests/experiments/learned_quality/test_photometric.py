from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.flow import (
    RigidFrameEvidence,
    RigidSceneEvidence,
    StaticTrack,
    TrackObservation,
)
from experiments.learned_quality.masks import (
    FusedMaskFrame,
    MaskFusionEvidence,
    MaskFusionPolicy,
)


def _photometric_module():
    return importlib.import_module("experiments.learned_quality.photometric")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    )
    return _sha256(path)


def _write_rgb(path: Path, pixels: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return _sha256(path)


def _write_mask(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8) * 255, mode="L").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return _sha256(path)


def _write_depth(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return _sha256(path)


def _latent_rgb(track_id: int) -> np.ndarray:
    return np.array(
        (
            0.20 + 0.50 * ((track_id * 7) % 23) / 22.0,
            0.18 + 0.54 * ((track_id * 11 + 3) % 29) / 28.0,
            0.22 + 0.48 * ((track_id * 13 + 5) % 31) / 30.0,
        ),
        dtype=np.float64,
    )


def _make_inputs(
    root: Path,
    corrections: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    *,
    track_count: int = 24,
    hard_masks: tuple[np.ndarray, ...] | None = None,
    uncertain_masks: tuple[np.ndarray, ...] | None = None,
    outliers: tuple[tuple[int, int, tuple[int, int, int]], ...] = (),
    observation_frames_by_track: tuple[tuple[int, ...], ...] | None = None,
    reverse_tracks: bool = False,
    background_values: tuple[tuple[int, int, int], ...] | None = None,
    image_suffix: str = ".png",
) -> tuple[
    tuple[FrameArtifact, ...],
    RigidSceneEvidence,
    MaskFusionEvidence,
]:
    height, width = 20, 24
    frame_count = len(corrections)
    if hard_masks is None:
        hard_masks = tuple(np.zeros((height, width), dtype=bool) for _ in corrections)
    if uncertain_masks is None:
        uncertain_masks = tuple(
            np.zeros((height, width), dtype=bool) for _ in corrections
        )
    if background_values is None:
        background_values = tuple((96, 104, 112) for _ in corrections)
    assert (
        len(hard_masks) == len(uncertain_masks) == len(background_values) == frame_count
    )

    coordinates = tuple(
        (2 + 3 * (index % 7), 2 + 3 * (index // 7)) for index in range(track_count)
    )
    frames: list[FrameArtifact] = []
    rigid_frames: list[RigidFrameEvidence] = []
    fused_frames: list[FusedMaskFrame] = []
    mask_inventory: list[dict[str, object]] = []
    depth_inventory: list[str] = []
    for frame_index, ((gain, bias), background) in enumerate(
        zip(corrections, background_values, strict=True)
    ):
        image_name = f"frame_{frame_index:06d}{image_suffix}"
        frame_id = f"frame-{frame_index:06d}"
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[...] = np.asarray(background, dtype=np.uint8)
        gain_array = np.asarray(gain, dtype=np.float64)
        bias_array = np.asarray(bias, dtype=np.float64)
        for track_id, (x, y) in enumerate(coordinates):
            raw = (_latent_rgb(track_id) - bias_array) / gain_array
            quantized = np.rint(np.clip(raw, 0.0, 1.0) * 255.0).astype(np.uint8)
            pixels[y : y + 2, x : x + 2] = quantized
        for outlier_frame, track_id, value in outliers:
            if outlier_frame == frame_index:
                x, y = coordinates[track_id]
                pixels[y : y + 2, x : x + 2] = np.asarray(value, dtype=np.uint8)
        source_path = root / "source" / image_name
        frame = FrameArtifact(
            image_name=image_name,
            frame_id=frame_id,
            path=source_path,
            sha256=_write_rgb(source_path, pixels),
        )
        frames.append(frame)

        depth_path = root / "depth" / f"{frame_id}.npy"
        depth_sha256 = _write_depth(
            depth_path, np.full((height, width), 2.0 + frame_index, dtype=np.float32)
        )
        depth_inventory.append(depth_sha256)
        rigid_frames.append(
            RigidFrameEvidence(
                frame=frame,
                width=width,
                height=height,
                registered=True,
                w2c_4x4=(
                    1.0,
                    0.0,
                    0.0,
                    float(frame_index) * 0.01,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                pinhole_fx_fy_cx_cy=(20.0, 20.0, 12.0, 10.0),
                depth_path=depth_path,
                depth_sha256=depth_sha256,
            )
        )

        hard = np.asarray(hard_masks[frame_index], dtype=bool)
        uncertain = np.asarray(uncertain_masks[frame_index], dtype=bool)
        empty = np.zeros_like(hard)
        map_root = root / "masks" / frame_id
        arrays = {
            "semantic_confirmed": hard,
            "sky_confirmed": empty,
            "motion_confirmed": empty,
            "uncertain": uncertain,
            "hard_exclude": hard,
            "colmap_keep": ~hard,
            "training_validity": ~hard,
        }
        paths = {name: map_root / f"{name}.png" for name in arrays}
        digests = {
            name: _write_mask(paths[name], value) for name, value in arrays.items()
        }
        fused_frames.append(
            FusedMaskFrame(
                frame=frame,
                width=width,
                height=height,
                semantic_confirmed_path=paths["semantic_confirmed"],
                semantic_confirmed_sha256=digests["semantic_confirmed"],
                sky_confirmed_path=paths["sky_confirmed"],
                sky_confirmed_sha256=digests["sky_confirmed"],
                motion_confirmed_path=paths["motion_confirmed"],
                motion_confirmed_sha256=digests["motion_confirmed"],
                uncertain_path=paths["uncertain"],
                uncertain_sha256=digests["uncertain"],
                hard_exclude_path=paths["hard_exclude"],
                hard_exclude_sha256=digests["hard_exclude"],
                colmap_keep_path=paths["colmap_keep"],
                colmap_keep_sha256=digests["colmap_keep"],
                training_validity_path=paths["training_validity"],
                training_validity_sha256=digests["training_validity"],
                exclusion_fraction_before_trim=float(np.mean(hard)),
                exclusion_fraction_after_trim=float(np.mean(hard)),
                decision="accepted",
                warnings=(),
            )
        )
        mask_inventory.append(
            {"frame_id": frame_id, "digests": dict(sorted(digests.items()))}
        )

    if observation_frames_by_track is None:
        observation_frames_by_track = tuple(
            tuple(range(frame_count)) for _ in range(track_count)
        )
    assert len(observation_frames_by_track) == track_count
    tracks = [
        StaticTrack(
            track_id=track_id,
            xyz=(float(track_id) * 0.01, 0.0, 2.0),
            mean_reprojection_error=0.1,
            observations=tuple(
                TrackObservation(
                    frame_id=frame.frame_id,
                    x=float(coordinates[track_id][0]),
                    y=float(coordinates[track_id][1]),
                )
                for frame_index, frame in enumerate(frames)
                if frame_index in observation_frames_by_track[track_id]
            ),
        )
        for track_id in range(track_count)
    ]
    if reverse_tracks:
        tracks.reverse()
    geometry_digest = hashlib.sha256(b"synthetic-geometry-v1").hexdigest()
    depth_digest = hashlib.sha256(
        json.dumps(depth_inventory, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    scene = RigidSceneEvidence(
        frames=tuple(rigid_frames),
        static_tracks=tuple(tracks),
        geometry_digest=geometry_digest,
        depth_digest=depth_digest,
    )
    mask_set_digest = hashlib.sha256(
        json.dumps(mask_inventory, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    mask_manifest_path = root / "masks" / "manifest.json"
    mask_manifest_sha256 = _write_json(
        mask_manifest_path,
        {"frames": mask_inventory, "mask_set_digest": mask_set_digest},
    )
    masks = MaskFusionEvidence(
        policy=MaskFusionPolicy(0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.0, 0.0, 0.5),
        frames=tuple(fused_frames),
        mask_set_digest=mask_set_digest,
        manifest_path=mask_manifest_path,
        manifest_sha256=mask_manifest_sha256,
    )
    return tuple(frames), scene, masks


def test_public_photometric_contract_is_frozen_exact_and_has_no_defaults() -> None:
    photometric = _photometric_module()

    assert photometric.GAIN_MIN == 0.75
    assert photometric.GAIN_MAX == 1.33
    assert photometric.BIAS_MIN == -0.10
    assert photometric.BIAS_MAX == 0.10
    assert photometric.MAX_ADDITIONAL_CLIP_FRACTION == 0.005
    assert tuple(
        field.name for field in dataclasses.fields(photometric.PhotometricPolicy)
    ) == (
        "robust_loss_delta",
        "temporal_smoothness_weight",
        "heldout_stride",
    )
    assert all(
        field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        for field in dataclasses.fields(photometric.PhotometricPolicy)
    )
    assert tuple(
        field.name for field in dataclasses.fields(photometric.RgbAffineTransform)
    ) == ("frame_id", "gain_rgb", "bias_rgb")
    assert tuple(
        field.name for field in dataclasses.fields(photometric.PhotometricEvidence)
    ) == (
        "decision",
        "reference_frame_id",
        "transforms",
        "original_frames",
        "training_frames",
        "original_rgb_digest",
        "training_rgb_digest",
        "heldout_error_before",
        "heldout_error_after",
        "maximum_additional_clip_fraction",
        "rejection_reasons",
        "report_path",
        "report_sha256",
        "manifest_path",
        "manifest_sha256",
    )
    assert photometric.PhotometricPolicy.__dataclass_params__.frozen
    assert photometric.RgbAffineTransform.__dataclass_params__.frozen
    assert photometric.PhotometricEvidence.__dataclass_params__.frozen
    assert tuple(
        inspect.signature(
            photometric.fit_and_validate_photometric_transforms
        ).parameters
    ) == (
        "frames",
        "scene",
        "masks",
        "output_dir",
        "reference_frame_id",
        "policy",
    )
    signature = inspect.signature(photometric.fit_and_validate_photometric_transforms)
    assert (
        signature.parameters["reference_frame_id"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY


def test_photometric_policy_rejects_malformed_values() -> None:
    photometric = _photometric_module()

    for value in (0.0, -0.1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="robust_loss_delta"):
            photometric.PhotometricPolicy(value, 0.0, 4)
    for value in (-0.1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="temporal_smoothness_weight"):
            photometric.PhotometricPolicy(0.05, value, 4)
    for value in (0, 1, -1, 2.0, True):
        with pytest.raises(ValueError, match="heldout_stride"):
            photometric.PhotometricPolicy(0.05, 0.0, value)


def test_recovers_known_rgb_affines_with_exact_reference_gauge_and_holdout(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    original_bytes = tuple(frame.path.read_bytes() for frame in frames)
    original_scene = scene
    original_masks = masks

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "accepted"
    assert result.rejection_reasons == ()
    assert result.transforms[0].frame_id == frames[0].frame_id
    assert result.transforms[0].gain_rgb == (1.0, 1.0, 1.0)
    assert result.transforms[0].bias_rgb == (0.0, 0.0, 0.0)
    for transform, (expected_gain, expected_bias) in zip(
        result.transforms[1:], corrections[1:], strict=True
    ):
        np.testing.assert_allclose(transform.gain_rgb, expected_gain, atol=0.025)
        np.testing.assert_allclose(transform.bias_rgb, expected_bias, atol=0.012)
    assert result.heldout_error_after < result.heldout_error_before
    assert result.original_frames == frames
    assert result.training_frames != frames
    assert tuple(frame.frame_id for frame in result.training_frames) == tuple(
        frame.frame_id for frame in frames
    )
    assert tuple(frame.image_name for frame in result.training_frames) == tuple(
        frame.image_name for frame in frames
    )
    assert result.original_rgb_digest != result.training_rgb_digest
    for original, training in zip(frames, result.training_frames, strict=True):
        assert training.path != original.path
        assert training.path.is_relative_to(tmp_path / "photometric" / "training_rgb")
        assert _sha256(training.path) == training.sha256
        with Image.open(training.path) as image:
            assert image.mode == "RGB"
            assert image.format == "PNG"
            assert image.info == {}
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["heldout"]["track_ids"] == [0, 4, 8, 12, 16, 20]
    assert report["training"]["track_ids"] == [
        track_id for track_id in range(24) if track_id % 4 != 0
    ]
    assert report["eligibility"]["scene_static_tracks_are_producer_prefiltered"] is True
    assert tuple(frame.path.read_bytes() for frame in frames) == original_bytes
    assert scene == original_scene
    assert masks == original_masks
    assert result.report_sha256 == _sha256(result.report_path)
    assert result.manifest_sha256 == _sha256(result.manifest_path)


def test_huber_irls_resists_a_large_training_track_outlier(tmp_path: Path) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.12, 0.91, 1.07), (-0.025, 0.035, -0.02)),
        ((0.89, 1.13, 0.94), (0.04, -0.03, 0.025)),
    )
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        outliers=((1, 1, (255, 0, 255)),),
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.005, 0.0, 4),
    )

    assert result.decision == "accepted"
    for transform, (expected_gain, expected_bias) in zip(
        result.transforms[1:], corrections[1:], strict=True
    ):
        np.testing.assert_allclose(transform.gain_rgb, expected_gain, atol=0.035)
        np.testing.assert_allclose(transform.bias_rgb, expected_bias, atol=0.018)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["solver"]["robust_loss"] == "huber_irls"
    assert report["solver"]["robust_loss_delta"] == 0.005


def test_solver_clamps_every_nonreference_channel_to_explicit_bounds(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.60, 1.55, 1.50), (0.0, 0.0, 0.0)),
        ((0.88, 1.14, 0.96), (0.035, -0.025, 0.02)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.01, 0.0, 4),
    )

    for transform in result.transforms:
        assert all(
            photometric.GAIN_MIN <= value <= photometric.GAIN_MAX
            for value in transform.gain_rgb
        )
        assert all(
            photometric.BIAS_MIN <= value <= photometric.BIAS_MAX
            for value in transform.bias_rgb
        )
    assert photometric.GAIN_MAX in result.transforms[1].gain_rgb
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["solver"]["gain_bounds"] == [0.75, 1.33]
    assert manifest["solver"]["bias_bounds"] == [-0.1, 0.1]


def test_temporal_smoothness_reduces_adjacent_transform_roughness(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.22, 0.83, 1.16), (-0.04, 0.045, -0.03)),
        ((0.82, 1.24, 0.86), (0.055, -0.045, 0.04)),
        ((1.20, 0.85, 1.18), (-0.035, 0.04, -0.035)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)

    unsmoothed = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "unsmoothed",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.01, 0.0, 4),
    )
    smoothed = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "smoothed",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.01, 25.0, 4),
    )

    def roughness(result: object) -> float:
        packed = [
            np.asarray(transform.gain_rgb + transform.bias_rgb, dtype=np.float64)
            for transform in result.transforms
        ]
        return float(
            sum(
                np.linalg.norm(current - previous)
                for previous, current in zip(packed[:-1], packed[1:], strict=True)
            )
        )

    assert roughness(smoothed) < roughness(unsmoothed)
    report = json.loads(smoothed.report_path.read_text(encoding="utf-8"))
    assert report["solver"]["temporal_edge_count"] == len(frames) - 1
    assert report["solver"]["temporal_smoothness_weight"] == 25.0


def test_global_track_latents_solve_frame_without_direct_reference_overlap(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.08, 0.94, 1.04), (-0.02, 0.025, -0.015)),
        ((0.91, 1.11, 0.96), (0.035, -0.025, 0.02)),
        ((1.16, 0.88, 1.09), (-0.04, 0.045, -0.025)),
    )
    observation_frames = tuple(
        (0, 1, 2) if track_id < 12 else (1, 2, 3) for track_id in range(24)
    )
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        observation_frames_by_track=observation_frames,
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.01, 0.0, 4),
    )

    assert result.decision == "accepted"
    np.testing.assert_allclose(
        result.transforms[-1].gain_rgb, corrections[-1][0], atol=0.035
    )
    np.testing.assert_allclose(
        result.transforms[-1].bias_rgb, corrections[-1][1], atol=0.018
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["support_graph"]["connected_to_reference"] is True
    assert report["support_graph"]["direct_reference_overlap_required"] is False


def test_accepted_training_artifacts_append_png_without_changing_canonical_name(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        image_suffix=".jpg",
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "accepted"
    assert tuple(frame.image_name for frame in result.training_frames) == tuple(
        frame.image_name for frame in frames
    )
    assert tuple(frame.path.name for frame in result.training_frames) == (
        "frame_000000.jpg.png",
        "frame_000001.jpg.png",
        "frame_000002.jpg.png",
    )
    for frame in result.training_frames:
        with Image.open(frame.path) as image:
            assert image.format == "PNG"


def test_no_static_samples_rejects_and_reuses_original_frames_exactly(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    excluded = tuple(np.ones((20, 24), dtype=bool) for _ in corrections)
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        hard_masks=excluded,
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "rejected"
    assert result.rejection_reasons == (
        "no_static_samples",
        "insufficient_static_samples",
    )
    assert result.training_frames is result.original_frames
    assert result.training_frames == frames
    assert result.training_rgb_digest == result.original_rgb_digest
    assert not (tmp_path / "photometric" / "training_rgb").exists()
    assert all(
        transform.gain_rgb == (1.0, 1.0, 1.0) and transform.bias_rgb == (0.0, 0.0, 0.0)
        for transform in result.transforms
    )


def test_disconnected_static_support_rejects_without_publishing_corrected_rgb(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = tuple(
        ((1.0 + 0.02 * index,) * 3, (-0.005 * index,) * 3) for index in range(6)
    )
    observations = tuple(
        (0, 1, 2) if track_id < 6 else (3, 4, 5) for track_id in range(12)
    )
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        track_count=12,
        observation_frames_by_track=observations,
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "rejected"
    assert "disconnected_static_support" in result.rejection_reasons
    assert result.training_frames == frames
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["support_graph"]["connected_to_reference"] is False


def test_no_strict_heldout_improvement_rejects_identity_scene(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    identity = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        (identity, identity, identity),
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "rejected"
    assert result.rejection_reasons == ("no_strict_heldout_improvement",)
    assert result.heldout_error_before == 0.0
    assert result.heldout_error_after == pytest.approx(0.0, abs=1e-12)
    assert result.training_frames == frames


def test_additional_clipping_above_half_percent_rejects_correction(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.20, 1.18, 1.16), (-0.02, -0.02, -0.02)),
        ((0.90, 1.12, 0.94), (0.04, -0.03, 0.02)),
    )
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        background_values=((245, 245, 245),) * 3,
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert result.decision == "rejected"
    assert result.rejection_reasons == ("additional_clipping_exceeded",)
    assert result.maximum_additional_clip_fraction > 0.005
    assert result.training_frames == frames
    assert result.training_rgb_digest == result.original_rgb_digest


@pytest.mark.parametrize("mask_kind", ("hard", "uncertain"))
def test_masked_track_footprints_never_influence_photometric_fit(
    tmp_path: Path,
    mask_kind: str,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.12, 0.91, 1.07), (-0.025, 0.035, -0.02)),
        ((0.89, 1.13, 0.94), (0.04, -0.03, 0.025)),
    )
    masks_by_frame = [np.zeros((20, 24), dtype=bool) for _ in corrections]
    masks_by_frame[1][2:4, 5:7] = True
    kwargs = {f"{mask_kind}_masks": tuple(masks_by_frame)}
    frames, scene, masks = _make_inputs(
        tmp_path / "inputs",
        corrections,
        outliers=((1, 1, (255, 0, 255)),),
        **kwargs,
    )

    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(1.0, 0.0, 4),
    )

    assert result.decision == "accepted"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert 1 not in report["training"]["track_ids"]
    for transform, (expected_gain, expected_bias) in zip(
        result.transforms[1:], corrections[1:], strict=True
    ):
        np.testing.assert_allclose(transform.gain_rgb, expected_gain, atol=0.035)
        np.testing.assert_allclose(transform.bias_rgb, expected_bias, atol=0.018)


def test_outputs_are_byte_deterministic_after_relocating_identical_inputs(
    tmp_path: Path,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    first = _make_inputs(tmp_path / "first-inputs", corrections)
    second = _make_inputs(tmp_path / "second-inputs", corrections)

    for inputs, output_name in ((first, "first-output"), (second, "second-output")):
        frames, scene, masks = inputs
        photometric.fit_and_validate_photometric_transforms(
            frames,
            scene,
            masks,
            tmp_path / output_name,
            reference_frame_id=frames[0].frame_id,
            policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
        )

    def inventory(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert inventory(tmp_path / "first-output") == inventory(tmp_path / "second-output")


def test_publication_failure_cleans_private_staging_and_leaves_no_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    output_dir = tmp_path / "photometric"
    real_write = photometric._write_rgb_png_fsync
    writes = 0

    def fail_second_write(path: Path, pixels: np.ndarray) -> str:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("injected publication failure")
        return real_write(path, pixels)

    monkeypatch.setattr(photometric, "_write_rgb_png_fsync", fail_second_write)

    with pytest.raises(RuntimeError, match="injected publication failure"):
        photometric.fit_and_validate_photometric_transforms(
            frames,
            scene,
            masks,
            output_dir,
            reference_frame_id=frames[0].frame_id,
            policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}.staging-*"))


def test_no_replace_promotion_preserves_competing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    output_dir = tmp_path / "photometric"
    real_promote = photometric.promote_directory

    def racing_promote(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "competitor.txt").write_text("winner", encoding="utf-8")
        real_promote(staging, target)

    monkeypatch.setattr(photometric, "promote_directory", racing_promote)

    with pytest.raises(FileExistsError):
        photometric.fit_and_validate_photometric_transforms(
            frames,
            scene,
            masks,
            output_dir,
            reference_frame_id=frames[0].frame_id,
            policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
        )

    assert (output_dir / "competitor.txt").read_text(encoding="utf-8") == "winner"
    assert not (output_dir / "manifest.json").exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}.staging-*"))


def test_publication_fsyncs_staged_tree_before_promotion_and_parent_afterward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    output_dir = tmp_path / "photometric"
    real_promote = photometric.promote_directory
    events: list[str] = []

    def record_tree(path: Path) -> None:
        assert path.parent == tmp_path
        events.append("tree")

    def record_directory(path: Path) -> None:
        assert path == tmp_path
        events.append("parent")

    def record_promote(staging: Path, target: Path) -> None:
        assert events == ["tree"]
        real_promote(staging, target)
        events.append("promote")

    monkeypatch.setattr(photometric, "_fsync_directory_tree", record_tree)
    monkeypatch.setattr(photometric, "_fsync_directory", record_directory)
    monkeypatch.setattr(photometric, "promote_directory", record_promote)

    photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        output_dir,
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    assert events == ["tree", "promote", "parent"]


@pytest.mark.parametrize(
    "fault",
    (
        "source_digest",
        "scene_order",
        "mask_order",
        "mask_digest",
        "geometry_digest",
        "invalid_reference",
        "existing_output",
    ),
)
def test_join_digest_and_destination_faults_fail_before_any_output_write(
    tmp_path: Path,
    fault: str,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    reference_frame_id = frames[0].frame_id
    output_dir = tmp_path / "photometric"
    if fault == "source_digest":
        frames = (dataclasses.replace(frames[0], sha256="f" * 64), *frames[1:])
    elif fault == "scene_order":
        scene = dataclasses.replace(scene, frames=tuple(reversed(scene.frames)))
    elif fault == "mask_order":
        masks = dataclasses.replace(masks, frames=tuple(reversed(masks.frames)))
    elif fault == "mask_digest":
        masks.frames[0].hard_exclude_path.write_bytes(
            masks.frames[0].hard_exclude_path.read_bytes() + b"drift"
        )
    elif fault == "geometry_digest":
        scene = dataclasses.replace(scene, geometry_digest="not-a-digest")
    elif fault == "invalid_reference":
        reference_frame_id = "missing-frame"
    elif fault == "existing_output":
        output_dir.mkdir()
        (output_dir / "owner.txt").write_text("existing", encoding="utf-8")
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(fault)

    with pytest.raises((ValueError, FileExistsError)):
        photometric.fit_and_validate_photometric_transforms(
            frames,
            scene,
            masks,
            output_dir,
            reference_frame_id=reference_frame_id,
            policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
        )

    if fault == "existing_output":
        assert (output_dir / "owner.txt").read_text(encoding="utf-8") == "existing"
    else:
        assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}.staging-*"))


def test_upstream_digest_is_rechecked_after_staging_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    output_dir = tmp_path / "photometric"
    real_write = photometric._write_rgb_png_fsync
    mutated = False

    def mutate_after_first_write(path: Path, pixels: np.ndarray) -> str:
        nonlocal mutated
        digest = real_write(path, pixels)
        if not mutated:
            masks.manifest_path.write_bytes(masks.manifest_path.read_bytes() + b"drift")
            mutated = True
        return digest

    monkeypatch.setattr(photometric, "_write_rgb_png_fsync", mutate_after_first_write)

    with pytest.raises(ValueError, match="sha256"):
        photometric.fit_and_validate_photometric_transforms(
            frames,
            scene,
            masks,
            output_dir,
            reference_frame_id=frames[0].frame_id,
            policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(f".{output_dir.name}.staging-*"))


def test_result_contract_is_deeply_immutable(tmp_path: Path) -> None:
    photometric = _photometric_module()
    corrections = (
        ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.10, 0.90, 1.05), (-0.03, 0.04, -0.02)),
        ((0.86, 1.16, 0.95), (0.05, -0.04, 0.03)),
    )
    frames, scene, masks = _make_inputs(tmp_path / "inputs", corrections)
    result = photometric.fit_and_validate_photometric_transforms(
        frames,
        scene,
        masks,
        tmp_path / "photometric",
        reference_frame_id=frames[0].frame_id,
        policy=photometric.PhotometricPolicy(0.03, 0.0, 4),
    )

    def assert_deeply_immutable(value: object) -> None:
        assert not isinstance(value, (dict, list, set, np.ndarray))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            assert value.__dataclass_params__.frozen
            for field in dataclasses.fields(value):
                assert_deeply_immutable(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                assert_deeply_immutable(item)

    assert_deeply_immutable(result)


def test_cpu_import_does_not_load_torch_or_learned_model_packages() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import experiments.learned_quality.photometric; "
                "blocked=('torch','transformers','sam2','sea_raft'); "
                "assert not any(name == root or name.startswith(root + '.') "
                "for name in sys.modules for root in blocked)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
