"""Karisik 'ply_mixed_inconsistent' klasorunden eski premium overnight run'in
hayatta kalan 70 frame'ini (frame_0030 ... frame_0099) ayri bir scene
klasorune ayikla + frame_0000'den baslayarak yeniden numaralandir.

Usage:
    python scripts/extract_baseline_ply.py

Sonuc:
    data/flame_steak_baseline/output/ply/frame_0000.ply  (= eski frame_0030)
    ...
    data/flame_steak_baseline/output/ply/frame_0069.ply  (= eski frame_0099)

Frontend bu yeni scene'i 'flame_steak_baseline' olarak yukleyebilir.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "data" / "flame_steak" / "output" / "ply_mixed_inconsistent"
DST_DIR = PROJECT_ROOT / "data" / "flame_steak_baseline" / "output" / "ply"

# Eski premium run'in hayatta kalan kismi: yeni run sadece ilk 30 frame
# (frame_0000 ... frame_0029) overwrite etmisti; frame_0030 ... frame_0099
# eski premium overnight'tan kalan kaliteli set.
OLD_FIRST = 30
OLD_LAST = 99  # inclusive


def main():
    if not SRC_DIR.exists():
        print(f"✗ Kaynak klasor yok: {SRC_DIR}")
        print(f"  Once 'ren ply ply_mixed_inconsistent' calistirildi mi?")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0
    for new_idx, old_idx in enumerate(range(OLD_FIRST, OLD_LAST + 1)):
        src = SRC_DIR / f"frame_{old_idx:04d}.ply"
        dst = DST_DIR / f"frame_{new_idx:04d}.ply"
        if not src.exists():
            print(f"  ⚠ Eksik: {src.name}")
            n_missing += 1
            continue
        shutil.copy2(src, dst)
        n_copied += 1
        if (new_idx + 1) % 10 == 0 or new_idx == OLD_LAST - OLD_FIRST:
            print(f"  copied {new_idx + 1}/{OLD_LAST - OLD_FIRST + 1}")

    print(f"\n✓ {n_copied} frame ayiklandi -> {DST_DIR}")
    if n_missing > 0:
        print(f"  ({n_missing} eksik dosya atlandi)")
    print(f"\n  Frontend'de scene name: 'flame_steak_baseline'")
    print(f"  Toplam frame: {n_copied}")


if __name__ == "__main__":
    main()
