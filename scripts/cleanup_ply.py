"""PLY cleanup — oversized ve extremely-anisotropic gaussian'ları drop et.

Neden:
    cutlemon_full_recovered ckpt_15000'den export edildi. Density control
    tam o iter'de durduğundan, INRIA'nın klasik "reset_opacity + final prune"
    döngüsü hiç çalışmadı. Sonuç: %38 gaussian ya "bloat" (scale > scene*%2)
    ya da "streak" (anisotropy > 30x). Bu %38 GPU'yu ezer ve ekranda
    radial streakler gösterir.

Kullanım:
    python scripts/cleanup_ply.py cutlemon_full_recovered
        → data/cutlemon_full_recovered_clean/output/ply/ yaratır

    python scripts/cleanup_ply.py cutlemon_full_recovered \
        --max-scale-frac 0.03 --max-anisotropy 20

Çıktı:
    ✓ 57378 → 35473 gaussian (38% drop), rendering ~2x hızlı olmalı
"""
from __future__ import annotations
import sys
import shutil
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parent.parent


def cleanup_ply_file(
    src: Path,
    dst: Path,
    keep_mask: Optional[np.ndarray] = None,
    max_scale_frac: float = 0.03,
    max_anisotropy: float = 30.0,
    scene_extent: Optional[float] = None,
) -> tuple[int, int, np.ndarray]:
    """
    Tek bir PLY'yi oku, outlier'ları drop et, yaz.

    Args:
        keep_mask: Eğer verilirse bu mask'i kullan (ilk frame'den hesaplanıp
                   tüm frame'lerde aynı mask uygulanmalı — aynı gaussian set).
        max_scale_frac: scene_extent'in bu oranından büyük scale drop edilir
        max_anisotropy: max_scale / min_scale oranı bundan büyükse drop

    Returns:
        (original_count, kept_count, keep_mask)
    """
    ply = PlyData.read(str(src))
    v = ply["vertex"]
    N = len(v)

    if keep_mask is None:
        # İlk çağrı: keep_mask hesapla
        ls = np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], axis=-1)
        scales = np.exp(ls)
        max_s = scales.max(axis=-1)
        min_s = scales.min(axis=-1)
        aniso = max_s / np.maximum(min_s, 1e-6)

        if scene_extent is None:
            pos = np.stack([np.asarray(v[a]) for a in "xyz"], axis=-1)
            center = pos.mean(axis=0)
            scene_extent = float(np.linalg.norm(pos - center, axis=-1).max())

        size_ok = max_s < (scene_extent * max_scale_frac)
        aniso_ok = aniso < max_anisotropy
        keep_mask = size_ok & aniso_ok

    # Mask'i uygula, yeni structured array yaz
    dst.parent.mkdir(parents=True, exist_ok=True)
    new_data = v.data[keep_mask]
    new_element = PlyElement.describe(new_data, "vertex")
    PlyData([new_element]).write(str(dst))

    return N, int(keep_mask.sum()), keep_mask


def main():
    parser = argparse.ArgumentParser(description="PLY cleanup — outlier prune")
    parser.add_argument("scene", help="Sahne adı (örn. cutlemon_full_recovered)")
    parser.add_argument("--max-scale-frac", type=float, default=0.03,
                        help="scene_extent × bu orandan büyük scale drop (default 0.03 = %%3)")
    parser.add_argument("--max-anisotropy", type=float, default=30.0,
                        help="max/min scale oranı bunu aşarsa drop (default 30.0)")
    parser.add_argument("--suffix", default="_clean",
                        help="Yeni sahne suffix'i (default: _clean)")
    args = parser.parse_args()

    src_dir = REPO_ROOT / "data" / args.scene / "output" / "ply"
    if not src_dir.exists():
        print(f"[ERR] {src_dir} yok")
        sys.exit(1)

    dst_scene = args.scene + args.suffix
    dst_dir = REPO_ROOT / "data" / dst_scene / "output" / "ply"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(src_dir.glob("frame_*.ply"))
    if not ply_files:
        print(f"[ERR] {src_dir} içinde PLY yok")
        sys.exit(2)

    print(f"→ {len(ply_files)} PLY bulundu, keep_mask hesaplanıyor (frame 0'dan)")

    # frame 0'dan mask hesapla (tüm frame'lerde aynı gaussian set var)
    first = ply_files[0]
    # scene extent frame 0'dan hesaplanır
    ply = PlyData.read(str(first))
    v = ply["vertex"]
    pos = np.stack([np.asarray(v[a]) for a in "xyz"], axis=-1)
    center = pos.mean(axis=0)
    scene_extent = float(np.linalg.norm(pos - center, axis=-1).max())
    print(f"  scene_extent: {scene_extent:.2f}")
    print(f"  max_scale cap: {scene_extent * args.max_scale_frac:.3f}")
    print(f"  max_anisotropy: {args.max_anisotropy}")

    # İlk frame cleanup + mask üret
    N, K, mask = cleanup_ply_file(
        first, dst_dir / first.name,
        max_scale_frac=args.max_scale_frac,
        max_anisotropy=args.max_anisotropy,
        scene_extent=scene_extent,
    )
    print(f"  frame 0: {N} → {K} ({100*K/N:.1f}% kept, {N-K} dropped)")

    # Kalan frame'lerde aynı mask kullan
    for p in ply_files[1:]:
        _, _, _ = cleanup_ply_file(p, dst_dir / p.name, keep_mask=mask)

    print(f"\n✓ {len(ply_files)} frame cleaned → {dst_dir}")
    print(f"  Viewer'da yükle: '{dst_scene}'")
    print(f"  Beklenen performance: ~{N/K:.1f}x daha hızlı render")


if __name__ == "__main__":
    main()
