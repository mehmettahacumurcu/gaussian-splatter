"""Faz 3a — Metric3D-v2 ile her kare için metrik derinlik tahmini.

Metric3D-v2 referans:
  https://github.com/YvanYin/Metric3D
  torch.hub adı: yvanyin/metric3d
  Modeller: metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import List


_MODEL_CACHE = {}


def _load_model(model_name: str = "metric3d_vit_small", device: str = "cuda"):
    """Metric3D modelini hub'dan yükle (cache'li)."""
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    print(f"→ Metric3D yükleniyor: {model_name} ({device})")
    model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
    model = model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _preprocess_image(img_path: Path, target_size: tuple[int, int] = (616, 1064)) -> tuple[torch.Tensor, tuple[int, int]]:
    """PNG → normalize edilmiş tensor (1, 3, H, W) ve orijinal boyut."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Metric3D için özel boyut + padding
    th, tw = target_size
    scale = min(th / h, tw / w)
    rh, rw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # Padding
    pad_h = th - rh
    pad_w = tw - rw
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53],
    )

    # Normalize (ImageNet stats)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_norm = (img_padded.astype(np.float32) - mean) / std

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (h, w)


def estimate_depth(
    frames_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "metric3d_vit_small",
    device: str = "cuda",
    overwrite: bool = False,
) -> List[Path]:
    """
    Her PNG kare için (H, W) float32 derinlik haritası üret.

    Çıktı: output_dir/<frame_stem>_depth.npy
    """
    frames_dir = Path(frames_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"PNG frame bulunamadı: {frames_dir}")

    if not torch.cuda.is_available() and device == "cuda":
        print("⚠ CUDA yok, CPU'ya düşülüyor (çok yavaş olacak)")
        device = "cpu"

    model = _load_model(model_name, device)
    out_paths: List[Path] = []

    for frame_path in frames:
        out_path = out / f"{frame_path.stem}_depth.npy"
        if out_path.exists() and not overwrite:
            out_paths.append(out_path)
            continue

        img_tensor, (h, w) = _preprocess_image(frame_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred_depth, *_ = model.inference({"input": img_tensor})

        # Padded boyuttan orijinale dön
        pred_depth = pred_depth.squeeze().cpu().numpy()
        # Metric3D zaten metrik (metre cinsinden) döner
        # Resize back
        import cv2
        depth_resized = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        np.save(out_path, depth_resized.astype(np.float32))
        out_paths.append(out_path)

    print(f"✓ {len(out_paths)} derinlik haritası üretildi → {out}")
    return out_paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Metric3D-v2 derinlik tahmini")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--model", default="metric3d_vit_small",
                   choices=["metric3d_vit_small", "metric3d_vit_large", "metric3d_vit_giant2"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    estimate_depth(args.frames_dir, args.output_dir, args.model, args.device, args.overwrite)
