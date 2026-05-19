"""Wrap an existing trained 3DGS .ply as a walkable world.

This is the Option-A path for sub-project B: instead of trying to generate a
single-image scene with LucidDreamer-style synthesis (which produces low-quality
results locally), take a .ply produced by the project's own static 3DGS pipeline
(multi-image or video → splat) and layer physics + first-person walking on top.

Usage:
    py -3 scripts/wrap_existing_ply.py \\
        --ply data/myroom_v2/output/ply/frame_0000.ply \\
        --scene myroom-v2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.image_to_scene.runner import wrap_scene_as_world  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to an existing trained .ply")
    parser.add_argument("--scene", required=True, help="World slug (target of worlds/<slug>)")
    parser.add_argument("--eye-height", type=float, default=1.7, help="Spawn height (m)")
    args = parser.parse_args()

    out = wrap_scene_as_world(
        ply_path=args.ply,
        world_slug=args.scene,
        eye_height_m=args.eye_height,
    )
    print("\nWrapped scene:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
