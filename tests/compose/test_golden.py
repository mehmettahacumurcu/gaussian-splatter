from __future__ import annotations

import json

import numpy as np

from backend.compose.bake import Crop, Placement, transform_cloud
from backend.compose.plyio import GaussianCloud
from tests.compose.golden import GOLDEN_PATH, build_golden


def _cloud_at(points):
    n = len(points)
    return GaussianCloud(
        means=np.asarray(points, np.float32), log_scales=np.zeros((n, 3), np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)), opacities=np.zeros(n, np.float32),
        sh_dc=np.zeros((n, 3), np.float32), sh_rest=np.zeros((n, 0, 3), np.float32),
    )


def _assert_close(a, b, path="root"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        assert a is b, path
    elif isinstance(a, (int, float)):
        assert isinstance(b, (int, float)), path
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=path)
    else:
        assert a == b, path


def test_committed_fixture_matches_generator():
    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    _assert_close(committed, json.loads(json.dumps(build_golden())))


def test_bake_reproduces_golden_transforms():
    golden = build_golden()
    for case in golden["transform_cases"]:
        t = case["transform"]
        out = transform_cloud(_cloud_at(case["points"]), Placement(
            position=tuple(t["position"]), quaternion_xyzw=tuple(t["quaternion"]), scale=t["scale"]))
        np.testing.assert_allclose(out.means, case["expected"], atol=1e-5)


def test_bake_reproduces_golden_crops():
    golden = build_golden()
    for case in golden["crop_cases"]:
        c = case["crop"]
        crop = Crop(center=tuple(c["center"]), half_size=tuple(c["halfSize"]),
                    quaternion_xyzw=tuple(c["quaternion"]))
        out = transform_cloud(_cloud_at(case["points"]), Placement(crop=crop))
        kept = [p for p, inside in zip(case["points"], case["inside"]) if inside]
        np.testing.assert_allclose(out.means, kept, atol=1e-6)
