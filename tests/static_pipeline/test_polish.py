from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

import backend.static_pipeline.polish as polish_module
from backend.preprocess.frame_alignment import RegisteredFrame
from backend.static_pipeline.contracts import (
    FrameRecord,
    GateDecision,
    ModelMetrics,
    PolishPolicy,
    ReconstructionBundle,
    RenderViewMetric,
    SelectionManifest,
    SelectionPolicy,
)
from backend.static_pipeline.polish import _build_polish_report, validate_static_ply


def _write_standard_ply(
    path: Path,
    *,
    count: int = 4,
    positions: np.ndarray | None = None,
    nan_position: bool = False,
    zero_quaternion: bool = False,
    rest_indices: tuple[int, ...] | None = None,
    scale_logit: float = -4.0,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> Path:
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        count = len(positions)
    if rest_indices is None:
        rest_indices = tuple(range(45))
    dtype = (
        [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
        + [(f"f_dc_{index}", "f4") for index in range(3)]
        + [(f"f_rest_{index}", "f4") for index in rest_indices]
        + [("opacity", "f4")]
        + [(f"scale_{index}", "f4") for index in range(3)]
        + [(f"rot_{index}", "f4") for index in range(4)]
    )
    vertices = np.zeros(count, dtype=dtype)
    if positions is None:
        vertices["x"] = np.linspace(-0.25, 0.25, count, dtype=np.float32)
        vertices["z"] = 2.0
    else:
        vertices["x"], vertices["y"], vertices["z"] = positions.T
    vertices["opacity"] = 2.0
    for index in range(3):
        vertices[f"scale_{index}"] = scale_logit
    for index, value in enumerate(quaternion):
        vertices[f"rot_{index}"] = value
    if nan_position:
        vertices["x"][0] = np.nan
    if zero_quaternion:
        for index in range(4):
            vertices[f"rot_{index}"][0] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
    return path


def _rewrite_vertices(path: Path, mutate: object) -> None:
    ply = PlyData.read(str(path), mmap=False)
    vertices = np.array(ply["vertex"].data, copy=True)
    mutate(vertices)  # type: ignore[operator]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sparse_bounds(model_dir: Path) -> None:
    model_dir.mkdir()
    corners = (
        (-6.0, -1.0, 1.0),
        (-6.0, -1.0, 3.0),
        (-6.0, 1.0, 1.0),
        (-6.0, 1.0, 3.0),
        (6.0, -1.0, 1.0),
        (6.0, -1.0, 3.0),
        (6.0, 1.0, 1.0),
        (6.0, 1.0, 3.0),
    )
    rows = ["# sparse bounds"]
    rows.extend(
        f"{index} {x} {y} {z} 128 128 128 0.1"
        for index, (x, y, z) in enumerate(corners, start=1)
    )
    (model_dir / "points3D.txt").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _registered_scene(
    tmp_path: Path,
    *,
    camera_tx: tuple[float, ...] = (0.0,),
) -> tuple[ReconstructionBundle, tuple[RegisteredFrame, ...]]:
    frames: list[FrameRecord] = []
    image_paths: list[Path] = []
    for index in range(len(camera_tx)):
        image_path = tmp_path / f"frame_{index:06d}.png"
        Image.new("RGB", (64, 64), (96, 96, 96)).save(image_path)
        image_paths.append(image_path)
        frames.append(
            FrameRecord(
                frame_id=f"{index + 1:024x}",
                source_relative_path="capture.mov",
                source_index=index,
                source_pts=index,
                timestamp_s=float(index),
                output_name=image_path.name,
                sha256=_sha256_file(image_path),
                selected=True,
                metrics=None,
                selection_score=None,
                reasons=("smart",),
            )
        )
    manifest = SelectionManifest(
        schema_version=1,
        source_digest="a" * 64,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=300,
            resolution_long_edge_cap=1280,
        ),
        frames=tuple(frames),
        image_set_digest="b" * 64,
    )
    model_dir = tmp_path / "accepted-model"
    _write_sparse_bounds(model_dir)
    metrics = ModelMetrics(
        model_dir=model_dir,
        registered_names=frozenset(path.name for path in image_paths),
        registered_count=len(image_paths),
        registered_ratio=1.0,
        registered_share=1.0,
        temporal_coverage_s=0.0,
        max_interior_gap_s=0.0,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=0.1,
        p95_reprojection_error_px=0.1,
        median_track_length=4.0,
        sparse_point_count=8,
        valid_names_intrinsics_and_poses=True,
    )
    decision = GateDecision(
        passed=True,
        dominant=metrics,
        failures=(),
        uncovered_intervals=(),
        retry_recommended=False,
    )
    reconstruction = ReconstructionBundle(
        selected_manifest=manifest,
        accepted_model_dir=model_dir,
        decision=decision,
        attempts=(),
        decisions=(decision,),
    )
    intrinsic = np.array(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    registered: list[RegisteredFrame] = []
    for frame, image_path, translation in zip(
        frames,
        image_paths,
        camera_tx,
        strict=True,
    ):
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[0, 3] = translation
        registered.append(
            RegisteredFrame(
                frame_id=frame.frame_id,
                image_name=image_path.name,
                image_path=image_path,
                depth_path=None,
                K=np.array(intrinsic, copy=True),
                w2c=world_to_camera,
                timestamp_s=frame.timestamp_s,
            )
        )
    return reconstruction, tuple(registered)


def test_conservative_real_candidate_is_accepted_and_normalized(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        count=8,
        scale_logit=-6.0,
        quaternion=(2.0, 0.0, 0.0, 0.0),
    )
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def evaluator(
        ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        calls.append((ply_path, tuple(view.image_name for view in views)))
        return (RenderViewMetric(views[0].image_name, 30.0, 0.95),)

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    assert report.accepted is True
    assert report.selected_path == report.candidate_path == candidate
    assert report.reasons == ()
    assert raw.read_bytes() == raw_bytes
    assert calls == [
        (raw, ("frame_000000.png",)),
        (candidate, ("frame_000000.png",)),
    ]
    validated = validate_static_ply(candidate)
    quaternions = np.column_stack(
        [validated.vertices[f"rot_{index}"] for index in range(4)]
    )
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)


def test_render_evaluator_failure_keeps_immutable_raw(tmp_path: Path) -> None:
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        count=8,
        scale_logit=-6.0,
    )
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def failing_evaluator(
        _ply_path: Path,
        _views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        raise RuntimeError("injected render failure")

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=failing_evaluator,
    )

    assert report.accepted is False
    assert report.selected_path == raw
    assert report.reasons == ("render_evaluation",)
    assert raw.read_bytes() == raw_bytes
    validate_static_ply(candidate)


def test_candidate_path_must_not_alias_raw(tmp_path: Path) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", scale_logit=-6.0)
    raw_bytes = raw.read_bytes()
    reconstruction, registered_frames = _registered_scene(tmp_path)

    with pytest.raises(ValueError, match="candidate|raw|distinct|alias"):
        polish_module.polish_static_ply(
            raw,
            raw,
            reconstruction,
            registered_frames,
            render_evaluator=lambda _path, _views: (),
        )

    assert raw.read_bytes() == raw_bytes


def test_existing_candidate_is_never_overwritten(tmp_path: Path) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", scale_logit=-6.0)
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    candidate.write_bytes(b"existing-winner")
    reconstruction, registered_frames = _registered_scene(tmp_path)

    with pytest.raises(FileExistsError, match="candidate|exists"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=lambda _path, _views: (),
        )

    assert raw.read_bytes() == raw_bytes
    assert candidate.read_bytes() == b"existing-winner"


def test_frustum_pruning_keeps_point_visible_only_in_second_camera(
    tmp_path: Path,
) -> None:
    positions = np.column_stack(
        (
            np.concatenate((np.linspace(-0.5, 0.5, 18), (2.0, 5.0))),
            np.zeros(20),
            np.full(20, 2.0),
        )
    )
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        positions=positions,
        scale_logit=-6.0,
    )
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(
        tmp_path,
        camera_tx=(0.0, -2.0),
    )

    def evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    validated = validate_static_ply(candidate)
    candidate_x = np.asarray(validated.vertices["x"])
    assert validated.count == 19
    assert np.any(np.isclose(candidate_x, 2.0))
    assert not np.any(np.isclose(candidate_x, 5.0))


def test_render_sampling_is_uniform_deterministic_and_capped_at_twelve(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        count=8,
        scale_logit=-6.0,
    )
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(
        tmp_path,
        camera_tx=(0.0,) * 13,
    )
    sampled_names: list[tuple[str, ...]] = []

    def evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        names = tuple(view.image_name for view in views)
        sampled_names.append(names)
        return tuple(RenderViewMetric(name, 30.0, 0.95) for name in names)

    polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    expected_indices = tuple(position * 12 // 11 for position in range(12))
    expected_names = tuple(f"frame_{index:06d}.png" for index in expected_indices)
    assert sampled_names == [expected_names, expected_names]


def test_candidate_prunes_low_opacity_scale_anisotropy_and_sparse_bounds(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        count=20,
        scale_logit=-6.0,
    )

    def add_outliers(vertices: np.ndarray) -> None:
        vertices["f_dc_0"] = np.arange(len(vertices), dtype=np.float32)
        vertices["opacity"][0] = -10.0
        vertices["scale_0"][1] = -2.0
        vertices["scale_0"][2] = -4.0
        vertices["scale_1"][2] = -10.0
        vertices["scale_2"][2] = -10.0
        vertices["z"][3] = 4.0

    _rewrite_vertices(raw, add_outliers)
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    validated = validate_static_ply(candidate)
    kept_ids = set(np.asarray(validated.vertices["f_dc_0"], dtype=int))
    assert validated.count == 16
    assert kept_ids.isdisjoint({0, 1, 2, 3})


def test_outer_crop_margin_is_retained_with_smoothstep_opacity_fade(
    tmp_path: Path,
) -> None:
    positions = np.column_stack(
        (
            np.linspace(-0.25, 0.25, 20),
            np.zeros(20),
            np.concatenate((np.full(19, 2.0), (3.05,))),
        )
    )
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        positions=positions,
        scale_logit=-6.0,
    )
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    validated = validate_static_ply(candidate)
    candidate_alpha = 1.0 / (1.0 + np.exp(-validated.vertices["opacity"].astype(float)))
    raw_alpha = 1.0 / (1.0 + np.exp(-2.0))
    assert validated.count == 20
    assert candidate_alpha[-1] == pytest.approx(raw_alpha * 0.5, rel=1e-5)
    np.testing.assert_allclose(candidate_alpha[:-1], raw_alpha, rtol=1e-6)
    assert report.opacity_mass_loss == pytest.approx(0.025, rel=1e-5)


def test_exact_outer_crop_boundary_keeps_finite_opacity(tmp_path: Path) -> None:
    positions = np.column_stack(
        (
            np.linspace(-0.25, 0.25, 20),
            np.zeros(20),
            np.concatenate((np.full(19, 2.0), (3.1,))),
        )
    )
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        positions=positions,
        scale_logit=-6.0,
    )
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    validated = validate_static_ply(candidate)
    assert validated.count == 20
    assert np.isfinite(validated.vertices["opacity"][-1])


def test_candidate_promotion_race_preserves_winner_and_cleans_owned_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    promotions: list[tuple[Path, Path]] = []

    def racing_promote(staged: Path, destination: Path) -> None:
        promotions.append((staged, destination))
        destination.write_bytes(b"race-winner")
        raise FileExistsError("injected promotion race")

    monkeypatch.setattr(
        polish_module,
        "_atomic_promote_no_replace",
        racing_promote,
        raising=False,
    )

    with pytest.raises(FileExistsError, match="promotion race"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=lambda _path, _views: (),
        )

    assert len(promotions) == 1
    assert promotions[0][1] == candidate
    assert candidate.read_bytes() == b"race-winner"
    assert raw.read_bytes() == raw_bytes
    assert not list(tmp_path.glob("candidate.ply.tmp-*"))


def test_valid_candidate_replacement_during_evaluation_is_an_integrity_error(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    replacement = _write_standard_ply(
        tmp_path / "replacement.ply",
        count=8,
        scale_logit=-7.0,
    )
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def replacing_evaluator(
        ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        if ply_path == candidate:
            candidate.write_bytes(replacement.read_bytes())
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    with pytest.raises((RuntimeError, ValueError), match="integrity|changed|identity"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=replacing_evaluator,
        )

    assert raw.read_bytes() == raw_bytes


def test_registered_ground_truth_hash_must_match_selection_manifest(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    Image.new("RGB", (64, 64), (12, 34, 56)).save(registered_frames[0].image_path)
    evaluator_called = False

    def evaluator(
        _ply_path: Path,
        _views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        nonlocal evaluator_called
        evaluator_called = True
        return ()

    with pytest.raises(ValueError, match="hash|manifest|ground truth"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=evaluator,
        )

    assert evaluator_called is False
    assert not candidate.exists()


def test_default_evaluator_restores_exporter_sh_layout_and_camera_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        count=2,
        scale_logit=-6.0,
        quaternion=(2.0, 0.0, 0.0, 0.0),
    )

    def encode_sh(vertices: np.ndarray) -> None:
        for channel in range(3):
            vertices[f"f_dc_{channel}"] = float(channel + 1)
        for index in range(45):
            vertices[f"f_rest_{index}"] = float(index + 10)

    _rewrite_vertices(raw, encode_sh)
    _, registered_frames = _registered_scene(tmp_path)
    captured: dict[str, object] = {}

    def fake_render_view(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, dict]:
        captured.update(kwargs)
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        return (
            torch.full((height, width, 3), 0.4, dtype=torch.float32),
            torch.ones((height, width, 1), dtype=torch.float32),
            {},
        )

    import backend.model.renderer as renderer_module

    monkeypatch.setattr(renderer_module, "render_view", fake_render_view)

    metrics = polish_module.evaluate_static_ply_views(
        raw,
        registered_frames,
        device="cpu",
    )

    assert len(metrics) == 1
    assert metrics[0].image_name == "frame_000000.png"
    assert np.isfinite(metrics[0].psnr_db)
    assert np.isfinite(metrics[0].ssim)
    assert captured["width"] == 64
    assert captured["height"] == 64
    assert captured["sh_degree"] == 3
    colors = captured["colors"].detach().cpu().numpy()  # type: ignore[union-attr]
    assert colors.shape == (2, 16, 3)
    np.testing.assert_allclose(
        colors[:, 0, :],
        np.tile((1.0, 2.0, 3.0), (2, 1)),
    )
    np.testing.assert_allclose(colors[0, 1:, 0], np.arange(10.0, 25.0))
    np.testing.assert_allclose(colors[0, 1:, 1], np.arange(25.0, 40.0))
    np.testing.assert_allclose(colors[0, 1:, 2], np.arange(40.0, 55.0))
    quaternions = captured["quats"].detach().cpu().numpy()  # type: ignore[union-attr]
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)


def test_empty_geometry_candidate_returns_raw_without_rendering(tmp_path: Path) -> None:
    positions = np.column_stack(
        (
            np.linspace(4.5, 5.5, 8),
            np.zeros(8),
            np.full(8, 2.0),
        )
    )
    raw = _write_standard_ply(
        tmp_path / "raw.ply",
        positions=positions,
        scale_logit=-6.0,
    )
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    evaluator_called = False

    def evaluator(
        _ply_path: Path,
        _views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        nonlocal evaluator_called
        evaluator_called = True
        return ()

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    assert report.accepted is False
    assert report.selected_path == raw
    assert report.candidate_path is None
    assert report.reasons == (
        "candidate_empty",
        "removed_fraction",
        "opacity_mass_loss",
    )
    assert evaluator_called is False
    assert raw.read_bytes() == raw_bytes
    assert not candidate.exists()


def test_evaluator_metrics_must_name_the_exact_sampled_views(tmp_path: Path) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)

    def wrong_name_evaluator(
        _ply_path: Path,
        _views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        return (RenderViewMetric("different-frame.png", 30.0, 0.95),)

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=wrong_name_evaluator,
    )

    assert report.accepted is False
    assert report.selected_path == raw
    assert report.reasons == ("render_metrics_incomplete",)


def test_ground_truth_replacement_during_evaluation_is_an_integrity_error(
    tmp_path: Path,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    image_path = registered_frames[0].image_path
    calls = 0

    def replacing_evaluator(
        _ply_path: Path,
        views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            Image.new("RGB", (64, 64), (1, 2, 3)).save(image_path)
        return tuple(RenderViewMetric(view.image_name, 30.0, 0.95) for view in views)

    with pytest.raises((RuntimeError, ValueError), match="integrity|changed|identity"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=replacing_evaluator,
        )


def test_generated_candidate_validation_failure_returns_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    raw_bytes = raw.read_bytes()
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    evaluator_called = False

    def invalid_candidate(*_args: object, **_kwargs: object) -> object:
        raise ValueError("injected candidate validation failure")

    def evaluator(
        _ply_path: Path,
        _views: tuple[RegisteredFrame, ...],
    ) -> tuple[RenderViewMetric, ...]:
        nonlocal evaluator_called
        evaluator_called = True
        return ()

    monkeypatch.setattr(polish_module, "_write_normalized_candidate", invalid_candidate)

    report = polish_module.polish_static_ply(
        raw,
        candidate,
        reconstruction,
        registered_frames,
        render_evaluator=evaluator,
    )

    assert report.accepted is False
    assert report.selected_path == raw
    assert report.candidate_path is None
    assert report.reasons == ("candidate_validation",)
    assert evaluator_called is False
    assert raw.read_bytes() == raw_bytes


def test_sparse_points_change_during_candidate_build_is_an_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _write_standard_ply(tmp_path / "raw.ply", count=8, scale_logit=-6.0)
    candidate = tmp_path / "candidate.ply"
    reconstruction, registered_frames = _registered_scene(tmp_path)
    points_path = reconstruction.accepted_model_dir / "points3D.txt"

    import backend.preprocess.parse_colmap as parse_colmap

    real_loader = parse_colmap.load_points3d_from_model

    def mutating_loader(model_dir: str | Path) -> tuple[np.ndarray, ...]:
        loaded = real_loader(model_dir)
        with points_path.open("a", encoding="utf-8") as handle:
            handle.write("# changed during polish\n")
        return loaded

    monkeypatch.setattr(parse_colmap, "load_points3d_from_model", mutating_loader)

    with pytest.raises((RuntimeError, ValueError), match="integrity|changed|identity"):
        polish_module.polish_static_ply(
            raw,
            candidate,
            reconstruction,
            registered_frames,
            render_evaluator=lambda _path, _views: (),
        )

    assert not candidate.exists()


@pytest.mark.parametrize(
    ("rest_count", "expected_degree"),
    [(0, 0), (9, 1), (24, 2), (45, 3)],
)
def test_exact_supported_sh_schemas_pass_validation(
    tmp_path: Path,
    rest_count: int,
    expected_degree: int,
) -> None:
    exact = _write_standard_ply(
        tmp_path / f"degree-{rest_count}.ply",
        rest_indices=tuple(range(rest_count)),
    )

    validated = validate_static_ply(exact)

    assert validated.sh_degree == expected_degree


@pytest.mark.parametrize(
    "rest_indices",
    [
        tuple((*range(10), *range(11, 46))),
        tuple(range(10)),
        tuple(range(46)),
    ],
    ids=("gapped", "unsupported-degree", "above-degree-three"),
)
def test_gapped_or_unsupported_sh_schema_fails_validation(
    tmp_path: Path,
    rest_indices: tuple[int, ...],
) -> None:
    invalid = _write_standard_ply(
        tmp_path / "invalid-sh.ply",
        rest_indices=rest_indices,
    )

    with pytest.raises(ValueError, match="f_rest|SH|spherical"):
        validate_static_ply(invalid)


def test_finite_log_scale_that_overflows_after_exp_fails_validation(
    tmp_path: Path,
) -> None:
    invalid = _write_standard_ply(tmp_path / "overflow-scale.ply", scale_logit=100.0)

    with pytest.raises(ValueError, match="scale|finite"):
        validate_static_ply(invalid)


def test_nonunit_nonzero_quaternion_is_valid_raw_input(tmp_path: Path) -> None:
    nonunit = _write_standard_ply(
        tmp_path / "nonunit-quaternion.ply",
        quaternion=(2.0, 0.0, 0.0, 0.0),
    )

    validate_static_ply(nonunit)


@pytest.mark.parametrize(
    ("nan_position", "zero_quaternion"),
    [(True, False), (False, True)],
)
def test_nonfinite_fields_and_bad_quaternions_fail_validation(
    tmp_path: Path,
    nan_position: bool,
    zero_quaternion: bool,
) -> None:
    bad = _write_standard_ply(
        tmp_path / "bad.ply",
        nan_position=nan_position,
        zero_quaternion=zero_quaternion,
    )

    with pytest.raises(ValueError, match="finite|quaternion"):
        validate_static_ply(bad)


def _metrics(
    *,
    mean_psnr_drop: float = 0.0,
    mean_ssim_drop: float = 0.0,
    single_drop: float = 0.0,
) -> tuple[tuple[RenderViewMetric, ...], tuple[RenderViewMetric, ...]]:
    raw_psnr = (30.0, 30.0)
    candidate_psnr = (
        raw_psnr[0] - mean_psnr_drop - single_drop,
        raw_psnr[1] - mean_psnr_drop + single_drop,
    )
    raw = (
        RenderViewMetric("frame_000000.png", raw_psnr[0], 0.95),
        RenderViewMetric("frame_000001.png", raw_psnr[1], 0.95),
    )
    candidate = (
        RenderViewMetric(
            "frame_000000.png",
            candidate_psnr[0],
            0.95 - mean_ssim_drop,
        ),
        RenderViewMetric(
            "frame_000001.png",
            candidate_psnr[1],
            0.95 - mean_ssim_drop,
        ),
    )
    return raw, candidate


@pytest.mark.parametrize(
    ("removed", "opacity_loss", "mean_psnr_drop", "mean_ssim_drop", "single_drop"),
    [
        (0.16, 0.01, 0.0, 0.0, 0.0),
        (0.01, 0.06, 0.0, 0.0, 0.0),
        (0.01, 0.01, 0.26, 0.0, 0.0),
        (0.01, 0.01, 0.0, 0.006, 0.0),
        (0.01, 0.01, 0.0, 0.0, 1.01),
    ],
)
def test_any_acceptance_limit_keeps_raw(
    tmp_path: Path,
    removed: float,
    opacity_loss: float,
    mean_psnr_drop: float,
    mean_ssim_drop: float,
    single_drop: float,
) -> None:
    raw = tmp_path / "raw.ply"
    candidate = tmp_path / "candidate.ply"
    raw_metrics, candidate_metrics = _metrics(
        mean_psnr_drop=mean_psnr_drop,
        mean_ssim_drop=mean_ssim_drop,
        single_drop=single_drop,
    )

    report = _build_polish_report(
        raw_path=raw,
        candidate_path=candidate,
        original_count=10_000,
        kept_count=round(10_000 * (1.0 - removed)),
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=1_000.0 * (1.0 - opacity_loss),
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=PolishPolicy(),
    )

    assert report.accepted is False
    assert report.selected_path == report.raw_path == raw
    assert report.reasons


def test_conservative_candidate_is_selected(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    candidate = tmp_path / "candidate.ply"
    raw_metrics, candidate_metrics = _metrics(
        mean_psnr_drop=0.10,
        mean_ssim_drop=0.002,
        single_drop=0.5,
    )

    report = _build_polish_report(
        raw_path=raw,
        candidate_path=candidate,
        original_count=10_000,
        kept_count=9_200,
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=980.0,
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=PolishPolicy(),
    )

    assert report.accepted is True
    assert report.selected_path == report.candidate_path == candidate
    assert report.reasons == ()


_POLICY_NUMERIC_FIELDS = (
    "max_removed_fraction",
    "max_opacity_mass_loss",
    "max_mean_psnr_drop_db",
    "max_mean_ssim_drop",
    "max_single_view_psnr_drop_db",
    "min_opacity",
    "max_relative_scale",
    "max_anisotropy",
    "crop_margin_fraction",
)


@pytest.mark.parametrize("field_name", _POLICY_NUMERIC_FIELDS)
@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), True],
    ids=("nan", "positive-infinity", "negative-infinity", "boolean"),
)
def test_policy_rejects_nonfinite_and_boolean_numeric_values(
    field_name: str,
    invalid_value: float | bool,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field_name):
        PolishPolicy(**{field_name: invalid_value})


def test_exact_acceptance_thresholds_select_candidate(tmp_path: Path) -> None:
    raw_metrics, candidate_metrics = _metrics(
        mean_psnr_drop=0.25,
        mean_ssim_drop=0.005,
        single_drop=0.75,
    )

    report = _build_polish_report(
        raw_path=tmp_path / "raw.ply",
        candidate_path=tmp_path / "candidate.ply",
        original_count=10_000,
        kept_count=8_500,
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=950.0,
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=PolishPolicy(),
    )

    assert report.accepted is True
    assert report.selected_path == report.candidate_path
    assert report.reasons == ()


@pytest.mark.parametrize(
    (
        "kept_count",
        "candidate_opacity",
        "psnr_drop",
        "ssim_drop",
        "single_drop",
        "reason",
    ),
    [
        (849_999, 1_000.0, 0.0, 0.0, 0.0, "removed_fraction"),
        (1_000_000, 949.999, 0.0, 0.0, 0.0, "opacity_mass_loss"),
        (1_000_000, 1_000.0, 0.250001, 0.0, 0.0, "mean_psnr_drop"),
        (1_000_000, 1_000.0, 0.0, 0.005001, 0.0, "mean_ssim_drop"),
        (1_000_000, 1_000.0, 0.0, 0.0, 1.000001, "single_view_psnr_drop"),
    ],
)
def test_just_over_each_acceptance_threshold_has_stable_reason_key(
    tmp_path: Path,
    kept_count: int,
    candidate_opacity: float,
    psnr_drop: float,
    ssim_drop: float,
    single_drop: float,
    reason: str,
) -> None:
    raw_metrics, candidate_metrics = _metrics(
        mean_psnr_drop=psnr_drop,
        mean_ssim_drop=ssim_drop,
        single_drop=single_drop,
    )

    report = _build_polish_report(
        raw_path=tmp_path / "raw.ply",
        candidate_path=tmp_path / "candidate.ply",
        original_count=1_000_000,
        kept_count=kept_count,
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=candidate_opacity,
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=PolishPolicy(),
    )

    assert report.accepted is False
    assert report.selected_path == report.raw_path
    assert report.reasons == (reason,)


def test_multiple_failures_have_canonical_reason_order(tmp_path: Path) -> None:
    raw = (
        RenderViewMetric("frame_000000.png", 30.0, 0.95),
        RenderViewMetric("frame_000001.png", 30.0, 0.95),
    )
    candidate = (
        RenderViewMetric("frame_000000.png", 28.0, 0.90),
        RenderViewMetric("frame_000001.png", 28.0, 0.90),
    )

    report = _build_polish_report(
        raw_path=tmp_path / "raw.ply",
        candidate_path=tmp_path / "candidate.ply",
        original_count=10_000,
        kept_count=8_000,
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=900.0,
        raw_metrics=raw,
        candidate_metrics=candidate,
        policy=PolishPolicy(),
    )

    assert report.accepted is False
    assert report.reasons == (
        "removed_fraction",
        "opacity_mass_loss",
        "mean_psnr_drop",
        "mean_ssim_drop",
        "single_view_psnr_drop",
    )


def _invalid_metric_case(
    case: str,
) -> tuple[
    tuple[RenderViewMetric, ...],
    tuple[RenderViewMetric, ...],
    str,
]:
    raw, candidate = _metrics()
    if case == "incomplete":
        return raw, candidate[:1], "render_metrics_incomplete"
    if case == "swapped":
        return raw, tuple(reversed(candidate)), "render_metrics_incomplete"
    if case == "duplicate":
        duplicate = (candidate[0], candidate[0])
        return raw, duplicate, "render_metrics_incomplete"
    if case == "nonfinite":
        invalid = object.__new__(RenderViewMetric)
        object.__setattr__(invalid, "image_name", candidate[0].image_name)
        object.__setattr__(invalid, "psnr_db", float("nan"))
        object.__setattr__(invalid, "ssim", candidate[0].ssim)
        nonfinite = (
            invalid,
            candidate[1],
        )
        return raw, nonfinite, "render_metrics_nonfinite"
    raise AssertionError(f"unknown metric case: {case}")


@pytest.mark.parametrize(
    "case",
    ["incomplete", "swapped", "duplicate", "nonfinite"],
)
def test_invalid_render_metrics_select_raw_with_json_safe_report(
    tmp_path: Path,
    case: str,
) -> None:
    raw_metrics, candidate_metrics, expected_reason = _invalid_metric_case(case)

    report = _build_polish_report(
        raw_path=tmp_path / "raw.ply",
        candidate_path=tmp_path / "candidate.ply",
        original_count=10_000,
        kept_count=10_000,
        raw_opacity_mass=1_000.0,
        candidate_opacity_mass=1_000.0,
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=PolishPolicy(),
    )

    assert report.accepted is False
    assert report.selected_path == report.raw_path
    assert report.reasons == (expected_reason,)
    json.dumps(report.render_metrics, allow_nan=False)
