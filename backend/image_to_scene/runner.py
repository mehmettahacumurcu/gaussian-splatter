"""Public entry for sub-project B + Trainer4DGS adapter.

run_image_to_scene(image_path, scene_name, cfg, progress_callback, worlds_root) -> dict
  — end-to-end single-image full-scene splat reconstruction

train_on_generated_views(model, poses, images, K, n_iterations, output_dir, progress_callback)
  — Trainer4DGS wrapper for the generated views (Task 7.1)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import json
import shutil
import tempfile
import time
import numpy as np
import torch
from PIL import Image

# Apply the MSVC-compat shim BEFORE anything triggers gsplat JIT compile.
from . import _gsplat_msvc_shim as _gsplat_msvc_shim  # noqa: F401

# Module-level imports of upstream modules — patched in tests via mocker.patch
from backend.model.gaussian_model import GaussianModel
from backend.model.deformation import DeformationField
from backend.model.trainer import Trainer4DGS
from backend.preprocess.depth_estimate import estimate_depth, release_models
from backend.export.to_splat import export_to_ply
from backend.edit.inpainter import SDInpainter

from .config import ImageToSceneConfig, profile as cfg_profile
from .disk_layout import ensure_envelope, artifact_path, request_path
from .intrinsics import CameraIntrinsics, intrinsics_for_image
from .seed import image_to_pointcloud, init_gaussian_model_from_seed
from .trajectory import CameraPose, generate_bounded_room_trajectory
from .outpaint_loop import run_outpaint_loop, render_pose, _w2c_from_pose
from .collider import derive_minimal_collider, write_collider_json


# ---------------------------------------------------------------------------
# Phase 7 — Trainer4DGS adapter (already implemented)
# ---------------------------------------------------------------------------

def train_on_generated_views(
    model: GaussianModel,
    poses: list[CameraPose],
    images: list[np.ndarray],   # each (H, W, 3) uint8
    K: CameraIntrinsics,
    n_iterations: int,
    output_dir: Path,
    progress_callback: Callable | None = None,
    log_interval: int = 50,
) -> dict:
    """Wrap Trainer4DGS.train() for our generated view set.

    Trainer4DGS reads images from disk, so we first persist each image to
    `<output_dir>/train_frames/<i>.png`. Then we build a single (3, 3) K tensor
    and a per-frame (4, 4) w2c. Static_mode=True bypasses 4D features.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "train_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for i, img in enumerate(images):
        p = frames_dir / f"frame_{i:04d}.png"
        Image.fromarray(img).save(p)
        frame_paths.append(p)

    # Build K and per-frame w2c. The trainer expects torch tensors on the
    # device the model lives on.
    device = next(model.parameters()).device
    K_tensor = torch.from_numpy(K.as_matrix()).float().to(device)

    w2c_list: list[torch.Tensor] = [_w2c_from_pose(p, device) for p in poses]

    # DeformationField required even in static_mode (trainer constructs but
    # bypasses). Default args are fine.
    deform = DeformationField()
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # B uses fourier_K=0 (static, no time-varying motion). Trainer4DGS's default
    # deform_pos_mode='hybrid' requires fourier_K>0, so override to "mlp" which
    # doesn't need Fourier coefficients. In static_mode the trainer bypasses
    # deformation entirely anyway.
    trainer = Trainer4DGS(gs=model, deform=deform, device=device_str,
                          deform_pos_mode="mlp")

    result = trainer.train(
        frame_paths=frame_paths,
        cam_K=K_tensor,
        cam_w2c_per_frame=w2c_list,
        n_iters=n_iterations,
        image_size=(K.width, K.height),  # Trainer4DGS docstring uses (W, H)
        ckpt_dir=output_dir / "ckpt",
        log_interval=log_interval,
        progress_callback=progress_callback,
        static_mode=True,
    )

    return {"final_iteration": n_iterations, "trainer_result": result}


# ---------------------------------------------------------------------------
# Phase 9 — Full pipeline orchestration
# ---------------------------------------------------------------------------

def _make_depth_fn(tmp_root: Path, device: str = "cuda"):
    """Closure that takes (H, W, 3) uint8 RGB and returns (H, W) float32 depth.

    Uses the existing batch-oriented estimate_depth via a single-image-per-call
    disk roundtrip. tmp_root must exist; we write a temp file under it each call.
    """
    frames_dir = tmp_root / "depth_in"
    depths_dir = tmp_root / "depth_out"
    frames_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    counter = {"i": 0}

    def depth_fn(rgb_uint8: np.ndarray) -> np.ndarray:
        i = counter["i"]
        counter["i"] += 1
        # estimate_depth() only globs `frame_*.png`, so the filename MUST start
        # with `frame_` and end with `.png`.
        stem = f"frame_{i:05d}"
        img_path = frames_dir / f"{stem}.png"
        Image.fromarray(rgb_uint8).save(img_path)
        estimate_depth(frames_dir=str(frames_dir), output_dir=str(depths_dir), device=device, overwrite=True)
        npy = depths_dir / f"{stem}_depth.npy"
        depth = np.load(npy).astype(np.float32)
        # Clean up so the directory does not grow unbounded.
        img_path.unlink(missing_ok=True)
        npy.unlink(missing_ok=True)
        return depth

    return depth_fn


def run_image_to_scene(
    image_path: str | Path,
    scene_name: str,
    cfg: ImageToSceneConfig | str | None = None,
    progress_callback: Callable | None = None,
    worlds_root: Path | None = None,
) -> dict:
    """End-to-end single-image full-scene reconstruction.

    Args:
        image_path: Path to the input image (PNG/JPG).
        scene_name: Slug for worlds/<slug>/.
        cfg: ImageToSceneConfig, profile name string ("default"/"fast"/"quality"), or None.
        progress_callback: Optional (phase, fraction, message, details) -> None.
        worlds_root: Override for the worlds/ directory; defaults to <cwd>/worlds.

    Returns dict with paths to outputs.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    if cfg is None:
        cfg = cfg_profile("default")
    elif isinstance(cfg, str):
        cfg = cfg_profile(cfg)

    worlds_root = Path(worlds_root) if worlds_root else (Path.cwd() / "worlds")
    worlds_root = worlds_root.resolve()
    env = ensure_envelope(worlds_root, scene_name)

    cb = progress_callback or (lambda *a, **k: None)
    started = time.time()

    # ── Stage input ──────────────────────────────────────────────────────────
    src_index = 0
    src_ext = image_path.suffix.lower() or ".png"
    src_stage = artifact_path(env.source, src_index, env.slug, src_ext)
    if not src_stage.exists():
        shutil.copyfile(image_path, src_stage)
    cb("stage_input", 0.05, f"staged {src_stage.name}", {})

    # ── Minimal image.json (spec self-review patch) ──────────────────────────
    if not env.image_json.exists():
        try:
            rel = src_stage.relative_to(worlds_root.parent)
            src_rel = str(rel)
        except ValueError:
            src_rel = str(src_stage)
        env.image_json.write_text(json.dumps({
            "schema_version": 1,
            "world": env.slug,
            "source_images": [src_rel],
            "scene_name": env.slug.replace("-", " ").title(),
            "short_caption": "",
            "literal_description": "",
            "environment": "",
            "visual_style": "",
            "lighting": "",
            "atmosphere": "",
            "ambient_sound": "",
            "objects": [],
        }, indent=2), encoding="utf-8")

    # ── Phase 1: depth + seed ────────────────────────────────────────────────
    cb("depth", 0.10, "running MiDaS on input", {})
    K = intrinsics_for_image(src_stage, fallback_fov_deg=cfg.default_fov_deg)

    # Depth on the input image. estimate_depth() only globs `frame_*.png`, so the
    # source file must be re-encoded as a PNG named `frame_XXXX.png` before being
    # passed in.
    with tempfile.TemporaryDirectory(prefix="b_seed_") as td:
        td = Path(td)
        in_dir = td / "in"
        in_dir.mkdir()
        seed_input = in_dir / "frame_0000.png"
        # Re-encode whatever input format we got (jpg/png/webp) into a PNG.
        with Image.open(src_stage) as _im:
            _im.convert("RGB").save(seed_input, "PNG")
        out_dir = td / "out"
        estimate_depth(frames_dir=str(in_dir), output_dir=str(out_dir),
                       device=("cuda" if torch.cuda.is_available() else "cpu"),
                       overwrite=True)
        depth0_path = out_dir / "frame_0000_depth.npy"
        depth0 = np.load(depth0_path).astype(np.float32)

    points, colors = image_to_pointcloud(src_stage, depth0, K)
    cb("seed", 0.20, f"seed pointcloud: {points.shape[0]} pts", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_gaussian_model_from_seed(points, colors, sh_degree=0).to(device)

    # ── Phase 2: trajectory ──────────────────────────────────────────────────
    poses = generate_bounded_room_trajectory(
        n_views=cfg.n_views,
        bubble_radius_m=cfg.bubble_radius_m,
        n_orbit_rings=cfg.n_orbit_rings,
        pitch_range_deg=cfg.pitch_range_deg,
        yaw_full_360=cfg.yaw_full_360,
        fov_horizontal_deg=cfg.default_fov_deg,
    )
    cb("trajectory", 0.25, f"{len(poses)} poses", {})

    # ── Phase 3: outpaint loop ───────────────────────────────────────────────
    cb("outpaint_loop_start", 0.30, "loading SDInpainter", {})
    # Uses SDInpainter's default model_id ("runwayml/stable-diffusion-inpainting"),
    # which is already cached locally. diffusers is pinned to 0.30.x because newer
    # versions (0.32+) block loading .bin files without torch 2.6+; the runwayml
    # HF mirror ships only .bin, so we keep diffusers older to avoid that wall.
    inpainter = SDInpainter(device=device)

    with tempfile.TemporaryDirectory(prefix="b_loop_") as tmp_root:
        tmp_root = Path(tmp_root)
        depth_fn = _make_depth_fn(tmp_root, device=device)

        stats = run_outpaint_loop(
            model=model, poses=poses, K=K, inpainter=inpainter,
            depth_fn=depth_fn, cfg=cfg, progress_callback=cb,
        )
    cb("outpaint_loop_done", 0.55,
       f"+{stats.gaussians_added} gaussians, {stats.views_rejected} rejected", {})

    # Release MiDaS before training.
    try:
        release_models()
    except Exception:
        pass

    # ── Phase 4: render targets for trainer ──────────────────────────────────
    cb("render_targets", 0.60, "rendering training targets", {})
    final_images = []
    for pose in poses:
        rgb_rendered, _, _ = render_pose(model, pose, K)
        final_images.append((np.clip(rgb_rendered, 0, 1) * 255).astype(np.uint8))

    # ── Phase 5: final 3DGS training ─────────────────────────────────────────
    train_out = env.output_world
    train_stats = train_on_generated_views(
        model=model, poses=poses, images=final_images, K=K,
        n_iterations=cfg.train_iterations,
        output_dir=train_out, progress_callback=cb,
    )
    cb("training_done", 0.85, "training complete", train_stats)

    # ── Phase 6: collider ────────────────────────────────────────────────────
    pts_np = model.means.detach().cpu().numpy().astype(np.float32)
    collider = derive_minimal_collider(pts_np, eye_height_m=1.7)
    out_index = 0
    collider_path = artifact_path(train_out, out_index, "world-collider", ".json")
    write_collider_json(collider_path, collider)
    cb("collider", 0.90, "collider derived", {})

    # ── Phase 7: export PLY ──────────────────────────────────────────────────
    # export_to_ply writes frame_0000.ply into the given dir; move to indexed name.
    with tempfile.TemporaryDirectory(prefix="b_ply_") as ply_tmp:
        ply_tmp = Path(ply_tmp)
        written = export_to_ply(gs=model, deform=None, output_dir=ply_tmp,
                                num_timestamps=1, device=device)
        if not written:
            raise RuntimeError("export_to_ply returned no files")
        ply_path = artifact_path(train_out, out_index, "world", ".ply")
        shutil.move(str(written[0]), str(ply_path))
    cb("export_ply", 0.95, f"ply: {ply_path.name}", {})

    # ── Phase 8: sidecars ────────────────────────────────────────────────────
    traj_path = artifact_path(train_out, out_index, "world-trajectory", ".json")
    traj_path.write_text(json.dumps({
        "schema_version": 1,
        "fov_horizontal_deg": cfg.default_fov_deg,
        "poses": [
            {"position": p.position.tolist(), "rotation": p.rotation.tolist()}
            for p in poses
        ],
        "spawn_position": collider.spawn_position.tolist(),
        "spawn_look_direction": collider.spawn_look_direction.tolist(),
    }, indent=2), encoding="utf-8")

    req_path = request_path(train_out, out_index, "world")
    req_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "image-to-scene",
        "scene": env.slug,
        "source_image": str(src_stage),
        "config": cfg.__dict__,
        "stats": {
            "n_views": len(poses),
            "outpaint_views_rejected": stats.views_rejected,
            "outpaint_gaussians_added": stats.gaussians_added,
            "final_gaussian_count": int(model.means.shape[0]),
            "wall_clock_s": time.time() - started,
        },
    }, indent=2, default=str), encoding="utf-8")

    cb("done", 1.0, "image-to-scene complete", {})
    return {
        "scene_name": env.slug,
        "ply_path": str(ply_path),
        "collider_json_path": str(collider_path),
        "trajectory_json_path": str(traj_path),
        "request_json_path": str(req_path),
        "wall_clock_s": time.time() - started,
    }
