"""Phase 1.8 — RAFT optical flow preprocessing.

torchvision'un dahili RAFT modelini kullanir (no extra deps gerekli).
Her ardisik frame_i -> frame_{i+1} icin (2, H, W) flow tensor uretir.

Output:
    data/<scene>/flow/forward_0000.pt    (single-view)
    data/<scene>/flow_multiview/cam00/forward_0000.pt    (multi-view)

Cache marker yazilir (Phase 1.3 settings_hash dahil).

Trainer'da flow loss: rendered piksel akisini gt flow'a yaklastirir.
Implementasyon Phase 2'de — su an cache hazirlanir.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch


_RAFT_CACHE: dict = {}


def release_models() -> None:
    """RAFT modelini bosalt (CoTracker'dan once VRAM)."""
    import gc
    global _RAFT_CACHE
    n = len(_RAFT_CACHE)
    for k in list(_RAFT_CACHE.keys()):
        m = _RAFT_CACHE.pop(k)
        try:
            m.cpu()
            del m
        except Exception:
            pass
    _RAFT_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if n > 0:
        print(f"✓ RAFT model cache released ({n} model)")


def _load_raft(device: str):
    """torchvision RAFT_Large modelini yukle."""
    key = ("raft_large", device)
    if key in _RAFT_CACHE:
        return _RAFT_CACHE[key]
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except ImportError as e:
        raise ImportError(
            "torchvision >= 0.13 gerekli (raft_large). pip install -U torchvision"
        ) from e
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _RAFT_CACHE[key] = (model, weights.transforms())
    return _RAFT_CACHE[key]


def _read_frame(path: Path) -> torch.Tensor:
    """PNG -> [3, H, W] float in [0, 1]."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t


def estimate_flow(
    frames_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    overwrite: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> List[Path]:
    """frame_i -> frame_{i+1} icin RAFT flow uretir.

    Output: output_dir/forward_<i:04d>.pt — torch.Tensor [2, H, W] (dx, dy)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 2:
        raise FileNotFoundError(f"En az 2 frame gerekli: {frames_dir} ({len(frames)} bulundu)")

    if not torch.cuda.is_available() and device == "cuda":
        print("⚠ CUDA yok, CPU'ya dusuluyor (RAFT cok yavas)")
        device = "cpu"

    model, transforms = _load_raft(device)

    out_paths: List[Path] = []
    pairs = list(zip(frames[:-1], frames[1:]))
    for i, (fa, fb) in enumerate(pairs):
        out_path = output_dir / f"forward_{i:04d}.pt"
        if out_path.exists() and not overwrite:
            out_paths.append(out_path)
            continue

        a = _read_frame(fa).unsqueeze(0).to(device)
        b = _read_frame(fb).unsqueeze(0).to(device)
        a_t, b_t = transforms(a, b)
        with torch.no_grad():
            flows = model(a_t, b_t)  # list of flows, last = final
            flow = flows[-1][0].cpu()  # [2, H, W]
        torch.save(flow.half(), out_path)  # half precision yeterli
        out_paths.append(out_path)

        if on_progress:
            on_progress((i + 1) / len(pairs), f"flow {i+1}/{len(pairs)}")
        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            print(f"  flow {i+1}/{len(pairs)}")

    print(f"✓ {len(out_paths)} flow uretildi -> {output_dir}")
    return out_paths


def estimate_flow_multiview(
    frames_mv_dir: Path,
    output_mv_dir: Path,
    device: str = "cuda",
    overwrite: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, List[Path]]:
    """Per-cam optical flow."""
    frames_mv_dir = Path(frames_mv_dir)
    output_mv_dir = Path(output_mv_dir)
    output_mv_dir.mkdir(parents=True, exist_ok=True)

    cam_dirs = sorted(
        d for d in frames_mv_dir.iterdir()
        if d.is_dir() and d.name.startswith("cam")
    )
    if not cam_dirs:
        raise FileNotFoundError(f"cam* yok: {frames_mv_dir}")

    out: Dict[str, List[Path]] = {}
    for i, cam_dir in enumerate(cam_dirs):
        cam_name = cam_dir.name
        cam_out = output_mv_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)
        msg = f"Flow {cam_name} ({i+1}/{len(cam_dirs)})"
        print(f"\n[flow_mv] {msg}")
        if on_progress:
            on_progress(i / len(cam_dirs), msg)
        paths = estimate_flow(cam_dir, cam_out, device=device, overwrite=overwrite)
        out[cam_name] = paths
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if on_progress:
        on_progress(1.0, f"Flow tamam ({len(cam_dirs)} cam)")
    total = sum(len(v) for v in out.values())
    print(f"[flow_mv] ✓ Total {total} flow uretildi ({len(cam_dirs)} cam)")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RAFT optical flow")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--multiview", action="store_true",
                   help="frames_dir multi-cam layout (cam00/, cam01/, ...)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.multiview:
        estimate_flow_multiview(Path(args.frames_dir), Path(args.output_dir),
                                device=args.device, overwrite=args.overwrite)
    else:
        estimate_flow(Path(args.frames_dir), Path(args.output_dir),
                      device=args.device, overwrite=args.overwrite)
    release_models()
