"""Re-mint existing world bundles with gravity alignment (worldRotation).

Reconstruction frames are arbitrary (COLMAP gauge / generator convention) —
measured 2026-07-04: garden inverted + pitched 31 deg, myroom-v2 on its side,
fixture-a pitched 90 deg. This re-runs wrap_scene_as_world on each bundle's
own 0-world.ply, which re-derives the collider in viewer space (+Y up),
rewrites the trajectory/request sidecars, and records worldRotation in the
collider JSON for the viewer. The .ply itself is untouched.

Run inside the gs4d env (the runner module imports torch):
    python scripts/align_world_up.py --all
    python scripts/align_world_up.py garden myroom-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Gravity-align existing world bundles")
    p.add_argument("slugs", nargs="*", help="world slugs under worlds/")
    p.add_argument("--all", action="store_true",
                   help="align every bundle that has output/world/0-world.ply")
    p.add_argument("--worlds-root", default=str(PROJECT_ROOT / "worlds"))
    args = p.parse_args()

    worlds_root = Path(args.worlds_root)
    if args.all:
        slugs = sorted(
            d.name for d in worlds_root.iterdir()
            if (d / "output" / "world" / "0-world.ply").exists()
        )
    else:
        slugs = args.slugs
    if not slugs:
        p.error("give world slugs or --all")

    from backend.image_to_scene.runner import wrap_scene_as_world  # torch dep

    for slug in slugs:
        ply = worlds_root / slug / "output" / "world" / "0-world.ply"
        if not ply.exists():
            print(f"[skip]  {slug}: no output/world/0-world.ply")
            continue
        print(f"[align] {slug}")
        info = wrap_scene_as_world(ply_path=ply, world_slug=slug,
                                   worlds_root=worlds_root)
        print(f"        collider -> {info['collider_json_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
