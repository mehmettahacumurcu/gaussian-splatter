# backend/edit/runner.py
"""EditJobRunner — orchestrates the 5-phase edit pipeline.

Phases:
  1. SAM 2 click-to-mask (~3 s)
  2. SAM 2 video propagation across all training frames (~30 s @ 750 frames)
  3. Gaussian classification (~10 s)
  4. Sparse-view inpaint + depth-warp propagation (~1 min A / ~5-10 min B)
  5. Local refit (~3-10 min)

Sequential model loading (SAM 2 → unload → inpainter → unload → main pipeline)
keeps peak VRAM under 8 GB.
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .gauss_classifier import classify_gaussians_for_deletion
from .inpainter import InpainterBase, LaMaInpainter, SDInpainter
from .refit import run_refit
from .sam_service import SAM2Service
from .warp import warp_anchor_to_target


class EditJobRunner:
    def __init__(
        self,
        scene_dir: Path,
        source_ckpt: Path,
        frame_idx: int,
        click_xy: tuple[float, float],
        quality_mode: str,  # "A" | "B"
        progress_cb: Callable[[str, float, str], None],
        cancel_check: Callable[[], bool],
    ):
        self.scene_dir = scene_dir
        self.source_ckpt = source_ckpt
        self.frame_idx = frame_idx
        self.click_xy = click_xy
        self.quality_mode = quality_mode
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check

        # Resolve next edit_NNN dir
        edits_dir = scene_dir / "output" / "edits"
        edits_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(p.name.split("_")[1])
            for p in edits_dir.glob("edit_*")
            if p.name.startswith("edit_") and p.name.split("_")[1].isdigit()
        ]
        next_n = (max(existing) + 1) if existing else 1
        self.edit_dir = edits_dir / f"edit_{next_n:03d}"
        self.edit_dir.mkdir(exist_ok=True)

    def _check_cancel(self, phase: str) -> None:
        if self.cancel_check():
            self.progress_cb(phase, 1.0, "Cancelled")
            raise RuntimeError(f"Edit cancelled during {phase}")

    def run(self) -> Path:
        # ==== Phase 1: Mask in clicked frame ====
        self.progress_cb("segment", 0.0, "Loading SAM 2")
        self._check_cancel("segment")
        sam = SAM2Service()
        frames_dir = self.scene_dir / "frames"
        all_frames = sorted(frames_dir.glob("frame_*.png"))
        frame_path = all_frames[self.frame_idx]
        img = np.array(Image.open(frame_path).convert("RGB"))
        seed_mask = sam.predict_from_click(img, self.click_xy)

        cov = float(seed_mask.mean())
        if cov < 0.001 or cov > 0.5:
            sam.unload()
            raise ValueError(
                f"Mask coverage {cov:.4f} out of bounds (0.001..0.5). "
                f"Click missed object or SAM grabbed the whole scene. Try again."
            )

        Image.fromarray(seed_mask * 255).save(self.edit_dir / "mask_preview.png")
        self.progress_cb("segment", 1.0, f"Seed mask coverage {cov:.2%}")

        # ==== Phase 2: Video propagation ====
        self.progress_cb("propagate", 0.0, "Propagating mask across views")
        self._check_cancel("propagate")
        mask_stack_np = sam.propagate_video(frames_dir, self.frame_idx, seed_mask)
        sam.unload()
        torch.cuda.empty_cache()
        self.progress_cb("propagate", 1.0, f"{mask_stack_np.shape[0]} masks generated")

        # ==== Phase 3: Gaussian classification ====
        self.progress_cb("classify", 0.0, "Loading source ckpt")
        self._check_cancel("classify")
        from ..model.gaussian_model import GaussianModel
        from ..preprocess.parse_colmap import parse_cameras

        ckpt = torch.load(self.source_ckpt, map_location="cuda", weights_only=False)
        gs = GaussianModel.from_checkpoint(ckpt["gs"]).to("cuda")

        cams = parse_cameras(self.scene_dir / "colmap")
        cam_names_sorted = sorted(cams.keys())
        first_cam = cams[cam_names_sorted[0]]
        K_native = torch.from_numpy(first_cam["K"]).float()
        w2cs = [torch.from_numpy(cams[n]["w2c"]).float() for n in cam_names_sorted]

        # Scale K to mask resolution
        H_mask, W_mask = mask_stack_np.shape[1], mask_stack_np.shape[2]
        native_w, native_h = int(first_cam["width"]), int(first_cam["height"])
        sx, sy = W_mask / native_w, H_mask / native_h
        K_mask = K_native.clone()
        K_mask[0, 0] *= sx; K_mask[0, 2] *= sx
        K_mask[1, 1] *= sy; K_mask[1, 2] *= sy

        masks_t = torch.from_numpy(mask_stack_np).bool()
        # GaussianModel uses .means as the position parameter (confirmed in gaussian_model.py:53)
        delete_flags = classify_gaussians_for_deletion(
            means=gs.means.detach(),
            K=K_mask, w2cs=w2cs, masks=masks_t,
            threshold=0.6, image_size=(W_mask, H_mask),
        )
        n_deleted = int(delete_flags.sum().item())
        if n_deleted < 50:
            raise ValueError(
                f"Only {n_deleted} gaussians flagged for deletion. "
                f"Selection too sparse — try a more central object."
            )
        self.progress_cb("classify", 1.0, f"{n_deleted} gaussians flagged")

        del gs, ckpt
        torch.cuda.empty_cache()

        # ==== Phase 4: Sparse-view inpaint + warp ====
        self.progress_cb("inpaint", 0.0, "Loading inpainter")
        inpainter: InpainterBase = LaMaInpainter() if self.quality_mode == "A" else SDInpainter()
        inpainter.load()

        anchors_tmp = self.edit_dir / "inpaint_anchors_tmp"
        anchors_final = self.edit_dir / "inpaint_anchors"
        if anchors_tmp.exists():
            shutil.rmtree(anchors_tmp)
        anchors_tmp.mkdir()

        T = len(all_frames)
        anchor_step = 8
        anchor_indices = list(range(0, T, anchor_step))

        # Lazy-import scipy only if available; fall back to a manual dilation if not
        def _dilate_mask(m: np.ndarray, iters: int = 4) -> np.ndarray:
            try:
                from scipy.ndimage import binary_dilation
                return binary_dilation(m.astype(bool), iterations=iters).astype(np.uint8)
            except ImportError:
                # Manual 4-connected dilation, slower but no scipy dep
                out = m.astype(bool)
                for _ in range(iters):
                    nxt = out.copy()
                    nxt[1:, :]  |= out[:-1, :]
                    nxt[:-1, :] |= out[1:, :]
                    nxt[:, 1:]  |= out[:, :-1]
                    nxt[:, :-1] |= out[:, 1:]
                    out = nxt
                return out.astype(np.uint8)

        for i, ai in enumerate(anchor_indices):
            self._check_cancel("inpaint")
            img = np.array(Image.open(all_frames[ai]).convert("RGB"))
            mask = mask_stack_np[ai]
            if mask.sum() == 0:
                # Nothing to inpaint here, but write the original so warps have a target
                Image.fromarray(img).save(anchors_tmp / f"anchor_{ai:06d}.png")
                continue
            mask_feathered = _dilate_mask(mask, iters=4)
            inpainted = inpainter.inpaint(img, mask_feathered)
            Image.fromarray(inpainted).save(anchors_tmp / f"anchor_{ai:06d}.png")
            self.progress_cb(
                "inpaint", (i + 1) / max(len(anchor_indices), 1) * 0.7,
                f"Anchor {i+1}/{len(anchor_indices)}",
            )

        inpainter.unload()
        torch.cuda.empty_cache()

        # Atomic rename of tmp dir
        if anchors_final.exists():
            shutil.rmtree(anchors_final)
        anchors_tmp.rename(anchors_final)

        # ==== Phase 4b: Build full inpainted frame stack ====
        # v1 fallback approach: for non-anchor frames where the mask is non-empty,
        # use the nearest anchor's inpainted RGB as a substitute (no warp). The
        # full depth-warp implementation is in Task C4 — until C4 lands, the
        # fallback gives a valid (if blurrier) inpainted frame stack for refit.
        self.progress_cb("inpaint", 0.85, "Building full inpainted frames")
        full_frames_dir = self.edit_dir / "inpainted_frames"
        full_frames_dir.mkdir(exist_ok=True)
        for fi in range(T):
            self._check_cancel("inpaint")
            img = np.array(Image.open(all_frames[fi]).convert("RGB"))
            mask = mask_stack_np[fi]
            if mask.sum() == 0:
                Image.fromarray(img).save(full_frames_dir / f"frame_{fi:06d}.png")
                continue

            if fi in anchor_indices:
                inpainted = np.array(Image.open(anchors_final / f"anchor_{fi:06d}.png").convert("RGB"))
            else:
                # Nearest anchor fallback (C4 will replace with depth-warp).
                nearest_anchor = min(anchor_indices, key=lambda a: abs(a - fi))
                inpainted = np.array(Image.open(anchors_final / f"anchor_{nearest_anchor:06d}.png").convert("RGB"))

            out = img.copy()
            mask_b = mask.astype(bool)
            out[mask_b] = inpainted[mask_b]
            Image.fromarray(out).save(full_frames_dir / f"frame_{fi:06d}.png")
        self.progress_cb("inpaint", 1.0, "Inpaint complete")

        # ==== Phase 5: Local refit ====
        self.progress_cb("refit", 0.0, "Starting local refit")
        n_iters = 3000 if self.quality_mode == "A" else 5000

        # The refit loads its own ckpt + builds its own GaussianModel — we passed
        # delete_flags computed above so the same Gaussians get removed there.
        new_ckpt = run_refit(
            source_ckpt_path=self.source_ckpt,
            inpainted_frames_dir=full_frames_dir,
            inpainted_mask_stack=masks_t,
            delete_flags=delete_flags,
            output_ckpt_dir=self.edit_dir,
            n_iters=n_iters,
            cancel_check=self.cancel_check,
            progress_cb=lambda p, msg: self.progress_cb("refit", p, msg),
        )

        # ==== Save metadata ====
        meta = {
            "parent_ckpt": str(self.source_ckpt),
            "edit_op": "delete",
            "click_xy": list(self.click_xy),
            "frame_idx": self.frame_idx,
            "mode": self.quality_mode,
            "n_deleted": n_deleted,
            "n_anchors": len(anchor_indices),
            "n_frames": T,
            "created_at": time.time(),
        }
        (self.edit_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        return new_ckpt
