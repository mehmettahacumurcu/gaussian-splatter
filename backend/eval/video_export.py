"""4D Quality v6.1 — Video export utility.

Render output frames'i (CHW float [0,1] veya HWC uint8) MP4'e yazar.
Backend dependencies: imageio + imageio-ffmpeg (opsiyonel: opencv-python).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


def write_video(
    frames: list,
    out_path: Path,
    fps: int = 30,
    quality: int = 8,
) -> Path:
    """Frame listesi → MP4.

    Args:
      frames: list of torch.Tensor (H, W, 3) float [0,1] OR np.ndarray (H, W, 3) uint8
      out_path: hedef .mp4 dosya yolu
      fps: frames per second
      quality: imageio quality (1=düşük, 10=en iyi). Default 8.
    Returns:
      out_path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to (H, W, 3) uint8
    np_frames = []
    for f in frames:
        if isinstance(f, torch.Tensor):
            arr = f.detach().cpu().numpy()
        else:
            arr = np.asarray(f)
        if arr.dtype != np.uint8:
            arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)  # CHW -> HWC
        np_frames.append(arr)

    try:
        import imageio
        imageio.mimsave(
            str(out_path),
            np_frames,
            fps=fps,
            quality=quality,
            macro_block_size=1,  # arbitrary frame size support
        )
    except ImportError:
        # Fallback to opencv-python
        import cv2
        H, W = np_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        for arr in np_frames:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    return out_path
