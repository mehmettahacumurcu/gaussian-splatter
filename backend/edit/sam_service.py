"""SAM 2 wrapper for click-to-mask + multi-view propagation.

Lazy-loads the model on first use, keeps it resident across edits in the
same process. Caller must explicitly call .unload() before loading the
inpainter (memory budget on 8 GB cards).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SAM2Service:
    """SAM 2 wrapper. Default 'small' variant ~600 MB resident."""

    def __init__(self, model_size: str = "small", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model: Optional[object] = None
        self._predictor: Optional[object] = None
        self._video_predictor: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Import inside method so this file is importable without sam2 installed
        # (e.g., during CPU-only unit tests of other edit modules).
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg_map = {
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
        }
        ckpt_map = {
            "small": "sam2.1_hiera_small.pt",
            "base":  "sam2.1_hiera_base_plus.pt",
        }
        cfg = cfg_map[self.model_size]
        ckpt = Path.home() / ".cache" / "sam2" / ckpt_map[self.model_size]
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found at {ckpt}. Download from "
                f"https://github.com/facebookresearch/sam2#download-checkpoints"
            )
        self._model = build_sam2(cfg, str(ckpt), device=self.device)
        self._predictor = SAM2ImagePredictor(self._model)
        self._video_predictor = build_sam2_video_predictor(cfg, str(ckpt), device=self.device)
        print(f"[sam_service] SAM 2 ({self.model_size}) yuklendi (device={self.device})")

    @torch.no_grad()
    def predict_from_click(
        self,
        image_rgb: np.ndarray,  # (H, W, 3) uint8
        click_xy: tuple[float, float],  # normalized [0, 1]
    ) -> np.ndarray:
        """Single-positive-click → binary mask (H, W) uint8."""
        self._ensure_loaded()
        H, W = image_rgb.shape[:2]
        u, v = click_xy
        x_px = float(u) * (W - 1)
        y_px = float(v) * (H - 1)
        self._predictor.set_image(image_rgb)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[x_px, y_px]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=False,
        )
        m = masks[0]
        return (m > 0.5).astype(np.uint8)

    @torch.no_grad()
    def propagate_video(
        self,
        frames_dir: Path,
        seed_frame_idx: int,
        seed_mask: np.ndarray,  # (H, W) uint8
    ) -> np.ndarray:
        """Propagate a seed mask across all frames in the directory.
        Returns (T, H, W) uint8 array. Frames sorted by name (matches
        COLMAP-name-sort convention used elsewhere).
        """
        self._ensure_loaded()
        state = self._video_predictor.init_state(video_path=str(frames_dir))
        self._video_predictor.add_new_mask(
            inference_state=state,
            frame_idx=seed_frame_idx,
            obj_id=1,
            mask=seed_mask.astype(np.uint8),
        )
        out_masks: dict[int, np.ndarray] = {}
        for frame_idx, obj_ids, mask_logits in self._video_predictor.propagate_in_video(state):
            m = (mask_logits[0].cpu().numpy() > 0).astype(np.uint8)
            out_masks[frame_idx] = m

        T = len(out_masks)
        if T == 0:
            raise RuntimeError("SAM 2 video propagation returned 0 masks")
        sample = next(iter(out_masks.values()))
        H, W = sample.shape
        stack = np.zeros((T, H, W), dtype=np.uint8)
        for fi in sorted(out_masks.keys()):
            stack[fi] = out_masks[fi]
        return stack

    def unload(self) -> None:
        """Free GPU memory before loading the inpainter."""
        if self._model is not None:
            del self._model
            del self._predictor
            del self._video_predictor
            self._model = None
            self._predictor = None
            self._video_predictor = None
            torch.cuda.empty_cache()
            print("[sam_service] SAM 2 unloaded, VRAM released")
