"""Integration test: full object-deletion edit on the cube fixture.

Asserts:
  1. Edit job completes without exception
  2. New ckpt has fewer Gaussians than source (cube was deleted)
  3. meta.json has correct parent_ckpt
  4. inpaint_anchors_tmp/ does not exist after success

Run: python scripts/integration_test_edit.py

GPU REQUIRED. SAM 2 + LaMa must be installed:
  pip install git+https://github.com/facebookresearch/sam2.git
  pip install simple-lama-inpainting
And SAM 2 checkpoint downloaded to ~/.cache/sam2/sam2.1_hiera_small.pt.

Build the fixture first:
  python scripts/build_cube_fixture.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.edit.runner import EditJobRunner
from backend.model.gaussian_model import GaussianModel


def main() -> int:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "cube_scene"
    if not fixture_dir.exists():
        print(f"ERROR: fixture not found at {fixture_dir}")
        print(f"Run: python scripts/build_cube_fixture.py first")
        return 1

    source_ckpt = fixture_dir / "output" / "ckpt" / "ckpt_final.pt"
    if not source_ckpt.exists():
        print(f"ERROR: source ckpt missing at {source_ckpt}")
        return 1

    # Source gaussian count
    src_state = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    src_gs = GaussianModel.from_checkpoint(src_state["gs"])
    n_source = src_gs.num_points
    print(f"Source ckpt: N={n_source}")

    # Click on the cube. The cube is centered at origin so it should appear
    # roughly centered in frame 0. Click at (0.5, 0.5).
    click_xy = (0.5, 0.5)
    frame_idx = 0

    progress_log: list[tuple[str, float, str]] = []

    def progress_cb(phase: str, frac: float, msg: str):
        progress_log.append((phase, frac, msg))
        print(f"  [{phase}] {frac:.2f} {msg}")

    cancel_check = lambda: False  # never cancel in integration test

    runner = EditJobRunner(
        scene_dir=fixture_dir,
        source_ckpt=source_ckpt,
        frame_idx=frame_idx,
        click_xy=click_xy,
        quality_mode="A",  # LaMa, fast
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )

    t0 = time.time()
    new_ckpt_path = runner.run()
    elapsed = time.time() - t0
    print(f"Edit completed in {elapsed:.1f}s, new ckpt at {new_ckpt_path}")

    # === Assertions ===
    assert new_ckpt_path.exists(), f"New ckpt not produced at {new_ckpt_path}"

    # 1. New ckpt has fewer gaussians (cube was deleted)
    new_state = torch.load(new_ckpt_path, map_location="cpu", weights_only=False)
    new_gs = GaussianModel.from_checkpoint(new_state["gs"])
    n_new = new_gs.num_points
    print(f"New ckpt: N={n_new} (source was {n_source})")
    assert n_new < n_source, (
        f"Expected fewer gaussians after delete, got {n_new} >= {n_source}"
    )

    # 2. meta.json has correct parent_ckpt + edit_op + mode
    meta_path = runner.edit_dir / "meta.json"
    assert meta_path.exists(), f"meta.json not found at {meta_path}"
    meta = json.loads(meta_path.read_text())
    assert meta["parent_ckpt"] == str(source_ckpt), (
        f"parent_ckpt mismatch: {meta['parent_ckpt']!r} != {str(source_ckpt)!r}"
    )
    assert meta["edit_op"] == "delete", f"edit_op should be 'delete', got {meta['edit_op']!r}"
    assert meta["mode"] == "A", f"mode should be 'A', got {meta['mode']!r}"
    assert meta["n_deleted"] >= 50, f"n_deleted should be >= 50, got {meta['n_deleted']}"

    # 3. tmp dir is gone (cleaned up after Phase 4 atomic rename)
    tmp_dir = runner.edit_dir / "inpaint_anchors_tmp"
    assert not tmp_dir.exists(), f"Stale tmp dir found at {tmp_dir}"

    # 4. inpainted_frames dir exists with the right count
    inpainted_dir = runner.edit_dir / "inpainted_frames"
    assert inpainted_dir.exists(), f"inpainted_frames dir not found at {inpainted_dir}"
    n_inpainted = len(list(inpainted_dir.glob("frame_*.png")))
    assert n_inpainted == 10, f"Expected 10 inpainted frames, got {n_inpainted}"

    print("ok: integration_test_edit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
