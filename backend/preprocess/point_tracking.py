"""Faz 3b — CoTracker ile piksellerin video boyunca takibi.

CoTracker referans:
  https://github.com/facebookresearch/co-tracker
  torch.hub adı: facebookresearch/co-tracker
  Modeller: cotracker2, cotracker2_online, cotracker3
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path


def _load_video_tensor(frames_dir: Path, max_long_edge: int = 720) -> torch.Tensor:
    """frame_*.png → (1, T, 3, H, W) float tensor [0, 255]."""
    import cv2
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"PNG frame bulunamadı: {frames_dir}")

    arrs = []
    target_hw = None
    for fp in frames:
        img = cv2.imread(str(fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if max_long_edge and max(h, w) > max_long_edge:
            s = max_long_edge / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        if target_hw is None:
            target_hw = img.shape[:2]
        elif img.shape[:2] != target_hw:
            img = cv2.resize(img, (target_hw[1], target_hw[0]))
        arrs.append(img)

    video = np.stack(arrs, axis=0)                          # (T, H, W, 3)
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    return tensor.unsqueeze(0)                              # (1, T, 3, H, W)


def track_points(
    frames_dir: str | Path,
    output_path: str | Path,
    grid_size: int = 30,
    model_name: str = "cotracker2",
    device: str = "cuda",
    max_long_edge: int = 720,
) -> Path:
    """
    Bir grid noktasını video boyunca takip et.

    Args:
        frames_dir: PNG karelerin bulunduğu klasör
        output_path: .pt dosyası — {"tracks": (1, T, N, 2), "visibility": (1, T, N)}
        grid_size: NxN noktadan oluşan ızgara (toplam N² nokta)
        model_name: cotracker2 | cotracker3
        device: cuda | cpu
        max_long_edge: VRAM tasarrufu için yeniden boyutlandırma

    Returns:
        output_path
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("⚠ CUDA yok, CPU'ya düşülüyor (çok yavaş olacak)")
        device = "cpu"

    print(f"→ CoTracker yükleniyor: {model_name}")
    model = torch.hub.load("facebookresearch/co-tracker", model_name)
    model = model.to(device).eval()

    print(f"→ Video tensoru yükleniyor (max_long_edge={max_long_edge})")
    video = _load_video_tensor(Path(frames_dir), max_long_edge).to(device)

    print(f"→ Tracking ({grid_size}x{grid_size} grid = {grid_size**2} nokta)")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, grid_size=grid_size)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tracks":     pred_tracks.cpu(),       # (1, T, N, 2)  piksel xy
        "visibility": pred_visibility.cpu(),   # (1, T, N)
        "video_shape": tuple(video.shape),     # (1, T, 3, H, W)
    }, out)

    n_pts = pred_tracks.shape[2]
    n_frames = pred_tracks.shape[1]
    print(f"✓ {n_pts} nokta {n_frames} frame boyunca izlendi → {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CoTracker ile piksel takibi")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_path", type=str)
    p.add_argument("--grid-size", type=int, default=30)
    p.add_argument("--model", default="cotracker2", choices=["cotracker2", "cotracker3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-long-edge", type=int, default=720)
    args = p.parse_args()

    track_points(args.frames_dir, args.output_path, args.grid_size,
                 args.model, args.device, args.max_long_edge)
