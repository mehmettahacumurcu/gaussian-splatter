"""Coverage on a synthetic scene with gsplat (skipped without CUDA/MSVC)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from backend.compose.plyio import read_ply
from backend.coverage import compute
from backend.coverage.colmap import read_cameras
from backend.coverage.runner import run_coverage
from backend.coverage.store import CoverageStore
from tests.coverage.helpers import look_at_w2c, plane_cloud, write_model, write_splat

pytestmark = pytest.mark.gpu

# Nine cameras 1 unit above the right half of a 2x2 plane (x >= 0), image
# "down" along +y; the left edge (x = -1) is out of every frustum. The eval_
# view (held out) looks at the left half.
TRAIN = [look_at_w2c([x, y, 1.0], [x, y, 0.0]) for x in (0.5, 0.6, 0.7) for y in (-0.1, 0.0, 0.1)]
NAMES = [f"train_{i}" for i in range(len(TRAIN))] + ["eval_0"]
POSES = TRAIN + [look_at_w2c([-0.6, 0.0, 1.0], [-0.6, 0.0, 0.0])]


@pytest.fixture(scope="module")
def cuda():
    try:
        import torch
        import gsplat  # noqa: F401

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = compute.splat_tensors(plane_cloud(4))
        compute.visibility(t, torch.eye(4, device="cuda"), torch.eye(3, device="cuda") * 10, 8, 8)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ImportError) as exc:
        pytest.skip(f"gsplat CUDA rendering unavailable: {type(exc).__name__}: {exc}")


def test_counts_follow_the_frusta(cuda, tmp_path):
    cloud = plane_cloud(40)
    cams = read_cameras(write_model(tmp_path, NAMES[:-1], POSES[:-1]))
    cov = compute.accumulate(compute.splat_tensors(cloud), cams)
    x, y = cloud.means[:, 0], cloud.means[:, 1]
    centre = (np.abs(x - 0.6) < 0.2) & (np.abs(y) < 0.1)
    assert (cov["count"][centre] == len(TRAIN)).all()
    assert (cov["count"][x < -0.5] == 0).all()
    # Straight-down views of a plane barely differ in direction.
    assert cov["spread"][centre].max() < 0.1


def test_run_coverage_writes_heatmap_views_and_meta(cuda, tmp_path):
    ply = write_splat(tmp_path / "run" / "splat.ply", plane_cloud(40))
    sparse = write_model(tmp_path / "run" / "sparse" / "0", NAMES, POSES)
    store = CoverageStore(tmp_path / "data")
    phases = []
    out = run_coverage(store, "t-1", "test", ply, sparse, lambda p, v, m, d: phases.append(p))
    assert out["coverage_id"] == "t-1"
    assert phases[0] == "coverage_load" and phases[-1] == "coverage_write"

    meta = store.get("t-1")
    assert meta["cameras"] == len(NAMES) and meta["test_views"] == 1
    np.testing.assert_allclose(meta["up"], [0, -1, 0], atol=1e-6)
    views = {v["name"]: v for v in store.views("t-1")}
    # The held-out view looks at the uncovered left half: mostly weak.
    assert views["eval_0"]["test"] and views["eval_0"]["weak"] > 0.8
    assert views["train_4"]["weak"] < 0.2 and views["train_4"]["score"] > 0.5

    heat = read_ply(store.heat_path("t-1"))
    assert heat.count == 1600 and heat.sh_rest.shape[1] == 0
    data = np.load(store.dir("t-1") / "coverage.npz")
    assert set(data.files) == {"count", "spread", "px_per_unit", "score"}
    assert not list(store.root.glob(".*.tmp"))


def test_misaligned_splat_is_rejected(cuda, tmp_path):
    cloud = plane_cloud(20)
    cloud.means += np.array([100, 0, 0], np.float32)
    ply = write_splat(tmp_path / "splat.ply", cloud)
    sparse = write_model(tmp_path / "sparse" / "0", NAMES[:-1], POSES[:-1])
    store = CoverageStore(tmp_path / "data")
    with pytest.raises(ValueError, match="koordinat"):
        run_coverage(store, "bad-1", "bad", ply, sparse, lambda *a: None)
    assert store.get("bad-1") is None and not list(store.root.glob(".*.tmp"))
