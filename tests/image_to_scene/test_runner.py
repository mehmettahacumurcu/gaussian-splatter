# tests/image_to_scene/test_runner.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image


@pytest.fixture
def fake_depth_npy(tmp_path):
    """Provide a (H, W) float32 .npy file that estimate_depth would write."""
    def make(out_dir, frame_stem, H=64, W=64, value=2.0):
        out = Path(out_dir) / f"{frame_stem}_depth.npy"
        np.save(out, np.full((H, W), value, dtype=np.float32))
        return out
    return make


def _mock_estimate_depth_side_effect(fake_depth_npy):
    """Return a side-effect for estimate_depth that writes a constant 2.0 depth."""
    def _side_effect(frames_dir, output_dir, **kwargs):
        from pathlib import Path
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for img_path in sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg")):
            stem = img_path.stem
            out = output_dir / f"{stem}_depth.npy"
            # Read image to know the H, W
            with Image.open(img_path) as im:
                W, H = im.size
            np.save(out, np.full((H, W), 2.0, dtype=np.float32))
            written.append(out)
        return written
    return _side_effect


def test_run_image_to_scene_full_pipeline(tmp_path, mocker, fake_depth_npy):
    """End-to-end orchestration with heavy components mocked."""
    from backend.image_to_scene.runner import run_image_to_scene

    # Stage input image
    src = tmp_path / "input.png"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(src)

    # Mock estimate_depth + release_models
    mocker.patch(
        "backend.image_to_scene.runner.estimate_depth",
        side_effect=_mock_estimate_depth_side_effect(fake_depth_npy),
    )
    mocker.patch("backend.image_to_scene.runner.release_models")

    # Mock SDInpainter so no diffusion runs
    fake_inpainter = mocker.Mock()
    fake_inpainter.inpaint.side_effect = lambda rgb, mask: rgb  # returns input unchanged
    fake_inpainter.load = mocker.Mock()
    fake_inpainter.unload = mocker.Mock()
    mocker.patch(
        "backend.image_to_scene.runner.SDInpainter",
        return_value=fake_inpainter,
    )

    # Mock Trainer4DGS so no training runs
    mock_trainer_class = mocker.patch("backend.image_to_scene.runner.Trainer4DGS")
    mock_trainer_class.return_value.train.return_value = {"final_iter": 2}

    # Mock export_to_ply so no .ply is generated; just touch a file
    def _fake_export(gs, deform, output_dir, **kwargs):
        out = Path(output_dir) / "frame_0000.ply"
        out.write_text("# fake PLY", encoding="utf-8")
        return [out]
    mocker.patch(
        "backend.image_to_scene.runner.export_to_ply",
        side_effect=_fake_export,
    )

    # Mock run_outpaint_loop and render_pose to avoid GPU/gsplat dependency
    mocker.patch(
        "backend.image_to_scene.runner.run_outpaint_loop",
        return_value=mocker.Mock(views_processed=2, views_rejected=0, gaussians_added=10),
    )
    mocker.patch(
        "backend.image_to_scene.runner.render_pose",
        return_value=(
            np.zeros((64, 64, 3), dtype=np.float32),
            np.zeros((64, 64), dtype=np.float32),
            np.zeros((64, 64), dtype=np.float32),
        ),
    )

    # Use the fast profile (15 views, 1500 iters) — but override more aggressively
    from backend.image_to_scene.config import ImageToSceneConfig
    cfg = ImageToSceneConfig(n_views=3, train_iterations=2, image_max_dim=64)

    worlds_root = tmp_path / "worlds"
    out = run_image_to_scene(
        image_path=src,
        scene_name="smoke-test",
        cfg=cfg,
        worlds_root=worlds_root,
    )

    # Verify return shape
    assert "scene_name" in out
    assert "ply_path" in out
    assert "collider_json_path" in out
    assert "trajectory_json_path" in out
    assert "request_json_path" in out
    # All declared paths exist on disk
    assert Path(out["ply_path"]).exists()
    assert Path(out["collider_json_path"]).exists()
    assert Path(out["trajectory_json_path"]).exists()
    assert Path(out["request_json_path"]).exists()

    # Envelope dirs were created
    scene_dir = worlds_root / "smoke-test"
    assert (scene_dir / "source").exists()
    assert (scene_dir / "output" / "world").exists()
    assert (scene_dir / "project.json").exists()
    assert (scene_dir / "image.json").exists()
    # The input was staged
    staged_files = list((scene_dir / "source").glob("0-*"))
    assert any(f.suffix == ".png" for f in staged_files)
    # image.json has expected schema
    image_json_data = json.loads((scene_dir / "image.json").read_text(encoding="utf-8"))
    assert image_json_data["schema_version"] == 1
    assert image_json_data["world"] == "smoke-test"
    # Collider JSON well-formed
    collider_data = json.loads(Path(out["collider_json_path"]).read_text(encoding="utf-8"))
    assert collider_data["schema_version"] == 1


def test_run_image_to_scene_rejects_missing_image(tmp_path):
    from backend.image_to_scene.runner import run_image_to_scene
    with pytest.raises(FileNotFoundError):
        run_image_to_scene(
            image_path=tmp_path / "no-such-file.png",
            scene_name="nope",
            cfg=None,
            worlds_root=tmp_path / "worlds",
        )


def test_run_image_to_scene_accepts_string_profile(tmp_path, mocker):
    """cfg can be a string like 'fast' or 'default' — profile() resolves it."""
    from backend.image_to_scene.runner import run_image_to_scene

    src = tmp_path / "input.png"
    Image.new("RGB", (32, 32), color=(50, 50, 50)).save(src)

    mocker.patch("backend.image_to_scene.runner.estimate_depth",
                 side_effect=_mock_estimate_depth_side_effect(None))
    mocker.patch("backend.image_to_scene.runner.release_models")
    inp = mocker.Mock()
    inp.inpaint.side_effect = lambda r, m: r
    mocker.patch("backend.image_to_scene.runner.SDInpainter", return_value=inp)
    mocker.patch("backend.image_to_scene.runner.Trainer4DGS")
    mocker.patch(
        "backend.image_to_scene.runner.run_outpaint_loop",
        return_value=mocker.Mock(views_processed=2, views_rejected=0, gaussians_added=5),
    )
    mocker.patch(
        "backend.image_to_scene.runner.render_pose",
        return_value=(
            np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros((32, 32), dtype=np.float32),
            np.zeros((32, 32), dtype=np.float32),
        ),
    )
    def _fake_export(gs, deform, output_dir, **kwargs):
        out = Path(output_dir) / "frame_0000.ply"
        out.write_text("# fake", encoding="utf-8")
        return [out]
    mocker.patch("backend.image_to_scene.runner.export_to_ply", side_effect=_fake_export)

    out = run_image_to_scene(
        image_path=src, scene_name="prof-test",
        cfg="fast",  # string profile name
        worlds_root=tmp_path / "worlds",
    )
    assert Path(out["ply_path"]).exists()
