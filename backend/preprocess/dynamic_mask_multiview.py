"""Phase 1.5 — Multi-view per-camera dynamic mask.

Her cam icin frame-difference + morfoloji ile motion mask uretir.
Output layout:

    data/<scene>/masks_multiview/cam00/mask_0001.png
    data/<scene>/masks_multiview/cam01/mask_0001.png
    ...

Single-view dynamic_mask.compute_dynamic_masks() reuse edilir. SAM2
entegrasyonu Phase 4'te (heavy, opsiyonel).

Multi-view trainer her (cam, t) cifti icin mask supervision alir
(`lambda_mask_motion`). Su an cache hazir, training pipeline'i
Phase 2'de bunu consume edecek.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .dynamic_mask import compute_dynamic_masks


def compute_dynamic_masks_multiview(
    frames_mv_dir: Path,
    output_mv_dir: Path,
    overwrite: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, List[Path]]:
    """Her cam icin ayri motion mask uretir.

    Args:
        frames_mv_dir: data/<scene>/frames_multiview/ — cam00/, cam01/, ...
        output_mv_dir: data/<scene>/masks_multiview/ — ayni yapida olusturulur
        overwrite:     mevcut mask'leri ezer mi?
        on_progress:   Optional[fn(frac, msg)]

    Returns:
        {cam_name: [Path, ...]}
    """
    frames_mv_dir = Path(frames_mv_dir)
    output_mv_dir = Path(output_mv_dir)
    output_mv_dir.mkdir(parents=True, exist_ok=True)

    if not frames_mv_dir.exists():
        raise FileNotFoundError(f"frames_mv klasoru yok: {frames_mv_dir}")

    cam_dirs = sorted(
        d for d in frames_mv_dir.iterdir()
        if d.is_dir() and d.name.startswith("cam")
    )
    if not cam_dirs:
        raise FileNotFoundError(f"cam* alt klasoru yok: {frames_mv_dir}")

    print(f"\n[masks_mv] {len(cam_dirs)} cam icin motion mask")

    out: Dict[str, List[Path]] = {}
    for i, cam_dir in enumerate(cam_dirs):
        cam_name = cam_dir.name
        cam_out = output_mv_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        n_frames = len(list(cam_dir.glob("frame_*.png")))
        msg = f"Mask {cam_name} ({i + 1}/{len(cam_dirs)}, {n_frames} frame)"
        print(f"\n[masks_mv] {msg}")
        if on_progress:
            on_progress(i / len(cam_dirs), msg)

        paths = compute_dynamic_masks(cam_dir, cam_out, overwrite=overwrite)
        out[cam_name] = paths

    if on_progress:
        on_progress(1.0, f"Mask tamam: {len(cam_dirs)} cam")

    total = sum(len(v) for v in out.values())
    print(f"\n[masks_mv] ✓ Total {total} masks uretildi ({len(cam_dirs)} cams)")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-view per-cam dynamic mask")
    p.add_argument("frames_mv_dir", type=str)
    p.add_argument("output_mv_dir", type=str)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    compute_dynamic_masks_multiview(
        Path(args.frames_mv_dir), Path(args.output_mv_dir),
        overwrite=args.overwrite,
    )
