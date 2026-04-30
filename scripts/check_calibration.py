"""Calibration sanity check — kameraların sahne merkezine bakıp bakmadığını kontrol et."""
import json
import sys
from pathlib import Path
import numpy as np


def main():
    calib_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/flame_steak/calibration.json")
    c = json.load(open(calib_path))
    cams = c["cameras"]

    positions = np.array([np.linalg.inv(np.array(cm["w2c"]))[:3, 3] for cm in cams])
    centroid = positions.mean(0)
    print(f"format: {c.get('format')}")
    print(f"n_cameras: {len(cams)}")
    print(f"centroid: {centroid}")
    print()
    print("Per-cam dot(forward, to_centroid):")
    print("-" * 70)

    dots = []
    for cm in cams:
        c2w = np.linalg.inv(np.array(cm["w2c"]))
        pos = c2w[:3, 3]
        fwd = c2w[:3, 2]
        to_c = centroid - pos
        n = np.linalg.norm(to_c)
        if n < 1e-9:
            continue
        to_c /= n
        d = float(np.dot(fwd, to_c))
        dots.append(d)
        print(f"  {cm['name']}: pos={pos.round(2)}, fwd={fwd.round(2)}, dot={d:+.3f}")

    print("-" * 70)
    arr = np.array(dots)
    print(f"Mean dot: {arr.mean():+.3f}")
    print(f"Min dot:  {arr.min():+.3f}")
    print(f"Max dot:  {arr.max():+.3f}")
    print(f"  >0.7 = good (looking at scene)")
    print(f"  ~0   = sideways")
    print(f"  <0   = backwards (axis flip needed)")


if __name__ == "__main__":
    main()
