"""Phase 1.4 — Multi-view per-camera depth estimation.

N3V multi-view sahnelerde her kamera icin ayri MiDaS/Metric3D depth map
uretir. Output layout:

    data/<scene>/depth_multiview/cam00/frame_0000_depth.npy
    data/<scene>/depth_multiview/cam01/frame_0000_depth.npy
    ...

Multi-view trainer her (cam, t) cifti icin depth supervision alir
(Phase 2'de aktif edilecek). Su an cache'lenir, training pipeline'i
hazirlar.

Single-view depth_estimate.estimate_depth() reuse edilir (model cache
shared). Cam'lar arasi torch.cuda.empty_cache() VRAM kontrolu icin.

Premium profile: metric3d_vit_large (DPT_Large) — ~1.4 GB VRAM, ~4-7 sn
per frame on RTX 3060 Ti. 21 cam x 60 frame x 5 sn = ~105 dk.

Standard/high profile: metric3d_vit_small (MiDaS_small) — hizli, ~80 MB.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from .depth_estimate import estimate_depth, release_models


def estimate_depth_multiview(
    frames_mv_dir: Path,
    output_mv_dir: Path,
    model_name: str = "metric3d_vit_small",
    device: str = "cuda",
    overwrite: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
    cleanup_between_cams: bool = True,
) -> Dict[str, List[Path]]:
    """Her cam icin ayri depth uretir.

    Args:
        frames_mv_dir: data/<scene>/frames_multiview/ — icinde cam00/, cam01/, ...
        output_mv_dir: data/<scene>/depth_multiview/ — ayni yapida olusturulur
        model_name:    metric3d_vit_small / metric3d_vit_large / DPT_*
        device:        "cuda" or "cpu"
        overwrite:     mevcut .npy dosyalarini ezsin mi?
        on_progress:   Optional[fn(frac, msg)] — cam-level coarse progress
        cleanup_between_cams: cam'lar arasi torch.cuda.empty_cache (default True)

    Returns:
        {cam_name: [Path, ...]} — her cam icin uretilen depth dosya listesi
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

    print(f"\n[depth_mv] {len(cam_dirs)} cam icin depth tahmini "
          f"(model={model_name}, device={device})")

    out: Dict[str, List[Path]] = {}
    for i, cam_dir in enumerate(cam_dirs):
        cam_name = cam_dir.name
        cam_out = output_mv_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        n_frames = len(list(cam_dir.glob("frame_*.png")))
        msg = f"Depth {cam_name} ({i + 1}/{len(cam_dirs)}, {n_frames} frame)"
        print(f"\n[depth_mv] {msg}")
        if on_progress:
            on_progress(i / len(cam_dirs), msg)

        paths = estimate_depth(
            cam_dir, cam_out,
            model_name=model_name,
            device=device,
            overwrite=overwrite,
        )
        out[cam_name] = paths

        # VRAM kontrolu — model cache'te kalsin (sonraki cam icin reuse),
        # sadece intermediate buffer'lari temizle.
        if cleanup_between_cams and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if on_progress:
        on_progress(1.0, f"Depth tamam: {len(cam_dirs)} cam")

    total = sum(len(v) for v in out.values())
    print(f"\n[depth_mv] ✓ Total {total} depth maps uretildi "
          f"({len(cam_dirs)} cams, model={model_name})")
    return out


def is_depth_mv_complete(
    frames_mv_dir: Path,
    output_mv_dir: Path,
    tolerance: int = 5,
) -> bool:
    """Hizli sanity check — her cam icin >= (frame_count - tolerance) depth dosyasi var mi?

    is_step_cached'in glob'u "cam*/*.png" patterned, ama biz .npy uretiyoruz.
    Bu helper depth_mv tamamliklik kontrolune yardimci.
    """
    frames_mv_dir = Path(frames_mv_dir)
    output_mv_dir = Path(output_mv_dir)
    if not frames_mv_dir.exists() or not output_mv_dir.exists():
        return False
    cam_dirs = sorted(
        d for d in frames_mv_dir.iterdir()
        if d.is_dir() and d.name.startswith("cam")
    )
    for cam_dir in cam_dirs:
        cam_out = output_mv_dir / cam_dir.name
        if not cam_out.exists():
            return False
        n_frames = len(list(cam_dir.glob("frame_*.png")))
        n_depth = len(list(cam_out.glob("*_depth.npy")))
        if n_depth < n_frames - tolerance:
            return False
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-view per-cam depth")
    p.add_argument("frames_mv_dir", type=str)
    p.add_argument("output_mv_dir", type=str)
    p.add_argument("--model", default="metric3d_vit_small")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    estimate_depth_multiview(
        Path(args.frames_mv_dir), Path(args.output_mv_dir),
        model_name=args.model, device=args.device, overwrite=args.overwrite,
    )
    release_models()
