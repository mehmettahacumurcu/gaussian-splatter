"""Faz 3a — MiDaS (DPT) ile monoküler derinlik tahmini.

Metric3D-v2 mmcv/mmengine bağımlılıkları yüzünden Windows'ta zahmetli.
MiDaS (Intel-ISL) torch.hub'da, sadece timm + torchvision gerektirir.

Çıktı: output_dir/<frame_stem>_depth.npy  — (H, W) float32 metre-cinsinden
  (MiDaS inverse depth döner, biz 1/x ile depth'e çeviriyoruz.
   Scale-invariant L1 loss metre doğrulamaz, yine de intuitive olsun diye.)

Eski Metric3D model isimleri otomatik MiDaS karşılıklarına map edilir:
  metric3d_vit_small  → DPT_Small
  metric3d_vit_large  → DPT_Large
  metric3d_vit_giant2 → DPT_Large  (giant karşılığı yok)
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import List


_MODEL_CACHE: dict = {}


def _resolve_model_name(name: str) -> str:
    """
    Eski Metric3D isimlerini MiDaS hub'da GERÇEKTEN var olan isimlere çevir.
    Intel-ISL MiDaS hub callable'ları (https://github.com/isl-org/MiDaS/blob/master/hubconf.py):
      - MiDaS, MiDaS_small         (klasik + küçük)
      - DPT_Large, DPT_Hybrid      (DPT ailesi)
      - DPT_BEiT_L_512 vb.         (daha yeni BEiT/SwinV2)
    DPT_Small diye bir callable YOK — onun yerine MiDaS_small.
    """
    mapping = {
        "metric3d_vit_small":  "MiDaS_small",   # ~80 MB, hızlı
        "metric3d_vit_large":  "DPT_Large",     # ~1.4 GB, kaliteli
        "metric3d_vit_giant2": "DPT_Large",     # giant karşılığı yok
    }
    if name in mapping:
        return mapping[name]
    valid_hub_names = {
        "MiDaS", "MiDaS_small",
        "DPT_Large", "DPT_Hybrid",
        "DPT_BEiT_L_512", "DPT_BEiT_L_384", "DPT_BEiT_B_384",
        "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256",
        "DPT_Swin_L_384", "DPT_LeViT_224",
    }
    if name in valid_hub_names:
        return name
    print(f"⚠ Bilinmeyen depth model '{name}', MiDaS_small fallback")
    return "MiDaS_small"


def _load_model(model_name: str, device: str):
    """MiDaS model + transform'u hub'dan yükle (cache'li)."""
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    print(f"→ MiDaS yukleniyor: {model_name} ({device}) — ilk sefer download eder")
    model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    model = model.to(device).eval()

    transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    # Transform seçimi model type'ına göre
    if "DPT_BEiT" in model_name or "DPT_SwinV2" in model_name or "DPT_Swin_L" in model_name:
        # Yeni BEiT/Swin tabanlı DPT'ler için beit_512_transform / swin_384_transform vs.
        # Hub'da transforms objesinin attr'ları: beit512, swin384 vs.
        tname = {
            "DPT_BEiT_L_512":    "beit512_transform",
            "DPT_BEiT_L_384":    "beit384_transform",
            "DPT_BEiT_B_384":    "beit384_transform",
            "DPT_SwinV2_L_384":  "swin384_transform",
            "DPT_SwinV2_B_384":  "swin384_transform",
            "DPT_SwinV2_T_256":  "swin256_transform",
            "DPT_Swin_L_384":    "swin384_transform",
            "DPT_LeViT_224":     "levit_transform",
        }.get(model_name, "dpt_transform")
        transform = getattr(transforms_hub, tname, transforms_hub.dpt_transform)
    elif model_name in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms_hub.dpt_transform
    elif model_name == "MiDaS_small":
        transform = transforms_hub.small_transform
    else:  # "MiDaS" classic
        transform = transforms_hub.default_transform

    _MODEL_CACHE[key] = (model, transform)
    return model, transform


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
    import cv2
    frames_dir = Path(frames_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"PNG frame bulunamadi: {frames_dir}")

    if not torch.cuda.is_available() and device == "cuda":
        print("⚠ CUDA yok, CPU'ya dusuluyor (cok yavas olacak)")
        device = "cpu"

    midas_name = _resolve_model_name(model_name)
    model, transform = _load_model(midas_name, device)

    out_paths: List[Path] = []
    for i, frame_path in enumerate(frames):
        out_path = out / f"{frame_path.stem}_depth.npy"
        if out_path.exists() and not overwrite:
            out_paths.append(out_path)
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"⚠ Okunamadi: {frame_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # MiDaS → inverse depth. Depth'e cevir (1/x ile, scale keyfi).
        inv_depth = prediction.cpu().numpy().astype(np.float32)
        depth = 1.0 / np.maximum(inv_depth, 1e-3)
        np.save(out_path, depth.astype(np.float32))
        out_paths.append(out_path)

        if (i + 1) % 20 == 0 or (i + 1) == len(frames):
            print(f"  depth {i+1}/{len(frames)}")

    print(f"✓ {len(out_paths)} derinlik haritasi uretildi (MiDaS {midas_name}) → {out}")
    return out_paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MiDaS derinlik tahmini")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--model", default="MiDaS_small",
                   help="MiDaS_small | DPT_Hybrid | DPT_Large | DPT_BEiT_L_512 ...")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    estimate_depth(args.frames_dir, args.output_dir, args.model, args.device, args.overwrite)
