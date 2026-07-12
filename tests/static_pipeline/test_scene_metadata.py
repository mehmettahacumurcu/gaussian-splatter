from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

import backend.static_pipeline.scene_metadata as scene_metadata
from backend.image_to_scene.orientation import WorldOrientation
from backend.static_pipeline.contracts import (
    FrameRecord,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
    SelectionPolicy,
)


@dataclass(frozen=True)
class _Scene:
    ply_path: Path
    reconstruction: ReconstructionBundle
    cameras: tuple[str, ...]


def _write_ply(path: Path) -> Path:
    dtype = (
        [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
        + [(f"f_dc_{index}", "f4") for index in range(3)]
        + [("opacity", "f4")]
        + [(f"scale_{index}", "f4") for index in range(3)]
        + [(f"rot_{index}", "f4") for index in range(4)]
    )
    rng = np.random.default_rng(7)
    vertices = np.zeros(600, dtype=dtype)
    vertices["x"] = rng.uniform(-2.0, 2.0, len(vertices))
    vertices["y"] = np.concatenate(
        (rng.normal(0.0, 0.01, 450), rng.uniform(0.5, 2.0, 150))
    )
    vertices["z"] = rng.uniform(1.0, 5.0, len(vertices))
    vertices["opacity"] = 2.0
    for index in range(3):
        vertices[f"scale_{index}"] = -6.0
    vertices["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
    return path


def _write_model(model_dir: Path, names: tuple[str, ...]) -> None:
    model_dir.mkdir()
    (model_dir / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 64 64 50 50 32 32\n",
        encoding="utf-8",
    )
    translations = (2.0, 0.0, -3.0)
    image_lines = ["# images", f"# Number of images: {len(names)}"]
    for image_id, (name, translation) in enumerate(
        zip(names, translations, strict=True),
        start=1,
    ):
        image_lines.extend(
            (
                f"{image_id} 1 0 0 0 {translation} 0 0 1 {name}",
                "",
            )
        )
    (model_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    corners = (
        (-2.0, 0.0, 1.0),
        (-2.0, 0.0, 5.0),
        (2.0, 0.0, 1.0),
        (2.0, 0.0, 5.0),
        (-2.0, 2.0, 1.0),
        (-2.0, 2.0, 5.0),
        (2.0, 2.0, 1.0),
        (2.0, 2.0, 5.0),
    )
    point_lines = ["# points"]
    point_lines.extend(
        f"{point_id} {x} {y} {z} 128 128 128 0.1"
        for point_id, (x, y, z) in enumerate(corners, start=1)
    )
    (model_dir / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n",
        encoding="utf-8",
    )


def _scene(tmp_path: Path) -> _Scene:
    names = tuple(f"frame_{index:06d}.png" for index in range(3))
    frames = tuple(
        FrameRecord(
            frame_id=f"{index + 1:024x}",
            source_relative_path="capture.mov",
            source_index=index,
            source_pts=index,
            timestamp_s=float(index),
            output_name=name,
            sha256=f"{index + 1:064x}",
            selected=True,
            metrics=None,
            selection_score=None,
            reasons=("smart",),
        )
        for index, name in enumerate(names)
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
        frames=frames,
        image_set_digest="b" * 64,
    )
    model_dir = tmp_path / "accepted-model"
    _write_model(model_dir, names)
    metrics = ModelMetrics(
        model_dir=model_dir,
        registered_names=frozenset(names),
        registered_count=len(names),
        registered_ratio=1.0,
        registered_share=1.0,
        temporal_coverage_s=2.0,
        max_interior_gap_s=1.0,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=0.1,
        p95_reprojection_error_px=0.2,
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
    return _Scene(
        ply_path=_write_ply(tmp_path / "selected.ply"),
        reconstruction=reconstruction,
        cameras=names,
    )


def _patch_plane(
    monkeypatch,
    *,
    plane_inlier: float,
    agreement_deg: float,
) -> None:
    radians = np.radians(agreement_deg)
    plane_up = np.array([np.sin(radians), -np.cos(radians), 0.0])
    estimate = WorldOrientation(
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        up_raw=plane_up,
        tilt_deg=agreement_deg,
        plane_inlier_frac=plane_inlier,
        above_below_ratio=20.0,
        sky_agrees=None,
    )
    monkeypatch.setattr(
        scene_metadata,
        "estimate_world_orientation",
        lambda *_args, **_kwargs: estimate,
    )


def test_camera_up_flips_antipodal_signs_and_rejects_angular_outlier() -> None:
    cameras = {
        "a.png": {"R": np.eye(3)},
        "b.png": {"R": np.diag((1.0, -1.0, -1.0))},
        "c.png": {"R": np.eye(3)},
        "outlier.png": {
            "R": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        },
    }

    estimated = scene_metadata.estimate_camera_up(cameras)

    np.testing.assert_allclose(estimated, (0.0, -1.0, 0.0), atol=1e-8)


def test_low_plane_confidence_omits_orientation_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.49, agreement_deg=5.0)

    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )

    assert metadata["orientation"]["applied"] is False
    assert "transform" not in metadata["orientation"]


def test_camera_plane_disagreement_omits_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.8, agreement_deg=21.0)

    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )

    assert metadata["orientation"]["applied"] is False
    assert "transform" not in metadata["orientation"]


def test_default_view_is_real_registered_camera_near_median(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.8, agreement_deg=10.0)

    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )

    assert metadata["orientation"]["applied"] is True
    assert metadata["default_camera"]["image_name"] == "frame_000001.png"
    assert metadata["default_camera"]["image_name"] in scene.cameras
    assert metadata["default_camera"]["source"] == "registered_colmap_camera"
    assert metadata["environment"]["sky"] == "viewer_recommendation_only"
    assert all(
        "K" in camera and "w2c" in camera for camera in metadata["registered_cameras"]
    )


def test_write_metadata_emits_json_only_optional_world_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.8, agreement_deg=10.0)
    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )
    metadata_path = tmp_path / "scene_metadata.json"
    world_dir = tmp_path / "world"

    written = scene_metadata.write_scene_metadata(
        metadata_path,
        metadata,
        world_dir=world_dir,
    )

    assert written == {
        "scene_metadata": metadata_path,
        "collider": world_dir / "collider.json",
        "trajectory": world_dir / "trajectory.json",
        "request": world_dir / "request.json",
    }
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert {path.name for path in world_dir.iterdir()} == {
        "collider.json",
        "trajectory.json",
        "request.json",
    }
    for path in written.values():
        assert path.suffix == ".json"
        json.loads(path.read_text(encoding="utf-8"))
    collider = json.loads(written["collider"].read_text(encoding="utf-8"))
    assert "boundingWalls" in collider
    assert "bounds" not in collider


def test_preview_renderer_failure_uses_deterministic_sparse_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.8, agreement_deg=10.0)
    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )
    preview_path = tmp_path / "preview.png"

    def failing_renderer(*_args, **_kwargs):
        raise RuntimeError("injected GPU render failure")

    result = scene_metadata.render_preview(
        scene.ply_path,
        scene.reconstruction,
        metadata,
        preview_path,
        render_backend=failing_renderer,
    )

    assert result == {
        "path": preview_path,
        "fallback_used": True,
        "warning": "preview_renderer_failed_sparse_fallback",
    }
    with Image.open(preview_path) as preview:
        assert preview.format == "PNG"
        assert preview.size == (64, 64)
        pixels = np.asarray(preview)
    assert np.any(pixels != 0)


def test_preview_uses_renderer_pixels_without_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = _scene(tmp_path)
    _patch_plane(monkeypatch, plane_inlier=0.8, agreement_deg=10.0)
    metadata = scene_metadata.build_scene_metadata(
        scene.ply_path,
        scene.reconstruction,
    )
    preview_path = tmp_path / "preview.png"

    def successful_renderer(
        _ply_path: Path,
        _camera,
        width: int,
        height: int,
    ) -> np.ndarray:
        return np.full((height, width, 3), 0.25, dtype=np.float32)

    result = scene_metadata.render_preview(
        scene.ply_path,
        scene.reconstruction,
        metadata,
        preview_path,
        render_backend=successful_renderer,
    )

    assert result == {
        "path": preview_path,
        "fallback_used": False,
        "warning": None,
    }
    with Image.open(preview_path) as preview:
        pixels = np.asarray(preview)
    assert np.all(pixels == 64)
