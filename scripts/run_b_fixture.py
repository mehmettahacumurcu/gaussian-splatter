"""Run sub-project B against a fixture image. CLI entry for manual acceptance runs.

Usage:
    py -3 scripts/run_b_fixture.py --scene fixture-a-render --profile fast
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.image_to_scene import run_image_to_scene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="slug of worlds/<slug>")
    parser.add_argument("--profile", default="default", choices=["fast", "default", "quality"])
    args = parser.parse_args()

    source_dir = Path("worlds") / args.scene / "source"
    images = (
        sorted(source_dir.glob("0-*.png"))
        + sorted(source_dir.glob("0-*.jpg"))
        + sorted(source_dir.glob("0-*.jpeg"))
        + sorted(source_dir.glob("0-*.webp"))
    )
    if not images:
        sys.exit(f"No source image found in {source_dir}")

    started = time.time()

    def cb(phase: str, frac: float, msg: str = "", details: dict | None = None) -> None:
        elapsed = time.time() - started
        print(f"[{elapsed:6.1f}s {frac * 100:5.1f}%] {phase:24s} {msg}")

    out = run_image_to_scene(
        image_path=images[0],
        scene_name=args.scene,
        cfg=args.profile,
        progress_callback=cb,
    )
    print("\nResult:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
