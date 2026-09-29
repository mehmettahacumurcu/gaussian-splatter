"""Golden cases shared by backend bake tests and frontend preview tests.

Regenerate after an intentional convention change:
    python -m tests.compose.golden
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from backend.compose.bake import ColorAdjust, Crop, color_matrix, crop_mask, quat_to_matrix

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "compose_golden.json"


def _unit(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


TRANSFORMS = [
    {"position": [0.5, -1.0, 2.0], "quaternion": _unit([0.2, 0.3, -0.1, 0.9]), "scale": 1.7},
    {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], "scale": 0.5},
]
POINTS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.3, 0.25, 1.5]]
CROPS = [{"center": [0.1, 0.2, 0.0], "halfSize": [0.5, 0.3, 0.4],
          "quaternion": [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]}]
CROP_POINTS = [[0.1, 0.2, 0.0], [0.5, 0.2, 0.0], [0.1, 0.45, 0.0], [-0.2, 0.5, 0.0],
               [0.8, 0.2, 0.0], [0.1, 0.2, 0.39], [0.1, 0.2, 0.41]]
COLORS = [{"exposure": 0.5, "tint": [1.1, 0.9, 0.8], "saturation": 0.6},
          {"exposure": -1.0, "tint": [1.0, 1.0, 1.0], "saturation": 1.5}]
RGBS = [[0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.5, 0.5, 0.5]]


def build_golden() -> dict:
    transform_cases = []
    for t in TRANSFORMS:
        r = quat_to_matrix(tuple(t["quaternion"]))
        pts = np.asarray(POINTS, dtype=np.float64)
        expected = t["scale"] * (pts @ r.T) + np.asarray(t["position"])
        transform_cases.append({"transform": t, "points": POINTS, "expected": expected.tolist()})
    crop_cases = []
    for c in CROPS:
        crop = Crop(center=tuple(c["center"]), half_size=tuple(c["halfSize"]),
                    quaternion_xyzw=tuple(c["quaternion"]))
        inside = crop_mask(np.asarray(CROP_POINTS, dtype=np.float64), crop)
        crop_cases.append({"crop": c, "points": CROP_POINTS, "inside": inside.tolist()})
    color_cases = []
    for c in COLORS:
        m = color_matrix(ColorAdjust(exposure=c["exposure"], tint=tuple(c["tint"]),
                                     saturation=c["saturation"]))
        color_cases.append({"color": c, "rgb": RGBS, "expected": (np.asarray(RGBS) @ m.T).tolist()})
    return {"transform_cases": transform_cases, "crop_cases": crop_cases, "color_cases": color_cases}


def main() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build_golden(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
