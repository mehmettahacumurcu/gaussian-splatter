"""Public entry for sub-project B + Trainer4DGS adapter.

run_image_to_scene(image_path, scene_name, cfg, progress_callback) -> dict
  — implemented in Task 9.1

train_on_generated_views(model, poses, images, K, n_iterations, output_dir, progress_callback)
  — implemented here (Task 7.1)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from PIL import Image

from backend.model.gaussian_model import GaussianModel
from backend.model.deformation import DeformationField
from backend.model.trainer import Trainer4DGS

from .intrinsics import CameraIntrinsics
from .trajectory import CameraPose
from .outpaint_loop import _w2c_from_pose


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

    trainer = Trainer4DGS(gs=model, deform=deform, device=device_str)

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
# Phase 9 — full pipeline orchestration (placeholder until Task 9.1)
# ---------------------------------------------------------------------------

def run_image_to_scene(
    image_path: str | Path,
    scene_name: str,
    cfg: Any = None,
    progress_callback: Callable | None = None,
) -> dict:
    raise NotImplementedError("Task 9.1 implements this.")
