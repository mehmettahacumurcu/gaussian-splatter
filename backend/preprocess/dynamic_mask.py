"""Faz 3c — Hareketli/statik ayrımı: optik akış tabanlı binary maskeler.

Notlar:
- Plandaki ideal: SAM2 + akış birleşimi (kategori bilgisiyle).
- Bu prototip versiyonu: dış bağımlılık olmadan çalışsın diye OpenCV'nin
  Farneback dense optical flow'una düşer. SAM2 entegrasyonu opsiyonel
  olarak `compute_dynamic_masks_sam2(...)` altında bırakıldı.
- Çıktı: mask_0001.png ... mask_0NNN.png (uint8: 0=statik, 255=hareketli)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List


def _farneback_flow_magnitude(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """İki gri kare arasında Farneback dense flow → magnitude (H, W) float32."""
    import cv2
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.linalg.norm(flow, axis=-1)  # (H, W)
    return mag


def compute_dynamic_masks(
    frames_dir: str | Path,
    output_dir: str | Path,
    threshold: float | None = None,
    blur_kernel: int = 11,
    overwrite: bool = False,
) -> List[Path]:
    """
    Her kare için 'hareketli' piksel maskesi üret (Farneback fallback).

    Args:
        frames_dir: PNG kareler
        output_dir: Çıktı maskelerinin yazılacağı klasör
        threshold: Magnitude eşiği (None → her kareye 75. yüzdelik adaptif)
        blur_kernel: Gaussian blur tek sayı (gürültü azaltma)
        overwrite: True → mevcut maskeleri yeniden hesapla

    Returns:
        Üretilen mask_*.png dosyalarının sıralı listesi.
    """
    import cv2
    frames_dir = Path(frames_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 2:
        raise RuntimeError(f"Akış için en az 2 frame gerekli, bulunan: {len(frames)}")

    out_paths: List[Path] = []
    prev_gray = None

    for i, fp in enumerate(frames):
        mask_path = out / f"mask_{i+1:04d}.png"
        if mask_path.exists() and not overwrite:
            out_paths.append(mask_path)
            continue

        img = cv2.imread(str(fp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # İlk kare için sonraki kareyle akış al → maske oluştur
            next_img = cv2.imread(str(frames[i + 1]))
            next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
            mag = _farneback_flow_magnitude(gray, next_gray)
        else:
            mag = _farneback_flow_magnitude(prev_gray, gray)

        if blur_kernel and blur_kernel >= 3:
            mag = cv2.GaussianBlur(mag, (blur_kernel, blur_kernel), 0)

        thr = threshold if threshold is not None else float(np.percentile(mag, 75)) * 1.5
        mask = (mag > thr).astype(np.uint8) * 255

        # Morfolojik temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cv2.imwrite(str(mask_path), mask)
        out_paths.append(mask_path)
        prev_gray = gray

    print(f"✓ {len(out_paths)} dinamik maske üretildi (Farneback) → {out}")
    return out_paths


def compute_dynamic_masks_sam2(
    frames_dir: str | Path,
    output_dir: str | Path,
    sam2_checkpoint: str | None = None,
    flow_threshold: float = 1.5,
) -> List[Path]:
    """
    SAM2 + optik akış: hareketli SAM segmentlerini birleştir.

    Bu fonksiyon SAM2 paketi kuruluysa (sam2 pip paketi) çalışır.
    Kurulum: pip install git+https://github.com/facebookresearch/segment-anything-2.git
    """
    raise NotImplementedError(
        "SAM2 entegrasyonu opsiyonel — şu an Farneback fallback kullanılıyor.\n"
        "Tam entegrasyon için: SAM2 ile per-frame mask üret, optik akış ile "
        "hareketli olanları seç, birleştir."
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dinamik maske (Farneback)")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--blur", type=int, default=11)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    compute_dynamic_masks(
        args.frames_dir, args.output_dir,
        threshold=args.threshold, blur_kernel=args.blur, overwrite=args.overwrite,
    )
