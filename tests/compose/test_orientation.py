"""Per-asset up estimation: endpoint, cache, SceneDoc.up validation, create_scene."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import backend.compose.orientation as orientation_mod
from backend.compose.models import SceneDoc
from backend.compose.plyio import GaussianCloud, write_ply
from tests.compose.helpers import minimal_glb
from tests.compose.test_routes import _setup, _upload


def floor_cloud(seed: int = 0) -> GaussianCloud:
    """~5000 points on z=0 plus ~2000 above it: the floor normal is +z."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-2, 2, 5000), rng.uniform(-2, 2, 5000), np.zeros(5000)])
    above = np.column_stack([rng.uniform(-2, 2, 2000), rng.uniform(-2, 2, 2000),
                             rng.uniform(0.2, 1.5, 2000)])
    means = np.vstack([floor, above]).astype(np.float32)
    n = len(means)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return GaussianCloud(
        means=means,
        log_scales=np.full((n, 3), -3.0, dtype=np.float32),
        quats=quats,
        opacities=np.full(n, 4.0, dtype=np.float32),
        sh_dc=np.zeros((n, 3), dtype=np.float32),
        sh_rest=np.zeros((n, 0, 3), dtype=np.float32),
    )


def _floor_asset(tmp_path, client, name="floor.ply"):
    data = write_ply(floor_cloud(), tmp_path / "src" / name).read_bytes()
    r = _upload(client, name, data)
    assert r.status_code == 201
    return r.json()


def _scene_body(scene_id="s_1", up=None, **extra):
    body = {
        "version": 1, "id": scene_id, "name": "x",
        "objects": [{"id": "o_1", "kind": "splat", "asset": "a_1", "name": "b", "role": "base"}],
        **extra,
    }
    if up is not None:
        body["up"] = up
    return body


# -- SceneDoc.up validation ----------------------------------------------------
def test_up_accepts_unit_vector_and_null_and_defaults_to_none():
    assert SceneDoc.model_validate(_scene_body(up=[0, 0, 1])).up == (0.0, 0.0, 1.0)
    assert SceneDoc.model_validate(_scene_body(up=None)).up is None
    assert SceneDoc.model_validate(_scene_body()).up is None


def test_up_is_normalised_within_tolerance():
    doc = SceneDoc.model_validate(_scene_body(up=[0, 0, 1.0005]))
    assert doc.up == pytest.approx((0.0, 0.0, 1.0))
    assert sum(c * c for c in doc.up) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[0, 0, 2], [0, 0, 0], [1, 1, 1]])
def test_up_rejects_non_unit(bad):
    with pytest.raises(ValueError):
        SceneDoc.model_validate(_scene_body(up=bad))


def test_put_up_validation_and_round_trip(tmp_path):
    client, _ = _setup(tmp_path)
    base = _floor_asset(tmp_path, client)
    doc = client.post("/compose/scenes", json={"name": "x", "base_asset": base["id"]}).json()
    url = f"/compose/scenes/{doc['id']}"

    doc["up"] = [0, 0, 1]
    r = client.put(url, json=doc)
    assert r.status_code == 200 and r.json()["up"] == [0.0, 0.0, 1.0]
    assert client.get(url).json()["up"] == [0.0, 0.0, 1.0]

    doc["up"] = [0, 0, 2]
    assert client.put(url, json=doc).status_code == 422

    body = json.dumps({**doc, "up": ["NaN_PLACEHOLDER", 0, 1]}).replace('"NaN_PLACEHOLDER"', "NaN")
    r = client.put(url, content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422

    doc["up"] = None
    assert client.put(url, json=doc).json()["up"] is None
    assert client.get(url).json()["up"] is None


# -- endpoint -------------------------------------------------------------------
def test_orientation_endpoint_finds_floor_normal(tmp_path):
    client, _ = _setup(tmp_path)
    base = _floor_asset(tmp_path, client)
    r = client.get(f"/compose/assets/{base['id']}/orientation")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"up", "tilt_deg", "plane_inlier_frac", "above_below_ratio"}
    up = np.array(body["up"])
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-6)
    assert up @ np.array([0, 0, 1]) > 0.9


def test_orientation_works_for_pipeline_assets(tmp_path):
    client, _ = _setup(tmp_path)
    write_ply(floor_cloud(), tmp_path / "room" / "output" / "ply" / "frame_0000.ply")
    r = client.get("/compose/assets/scene__room/orientation")
    assert r.status_code == 200 and r.json()["up"][2] > 0.9


def test_orientation_mesh_is_400_and_unknown_is_404(tmp_path):
    client, _ = _setup(tmp_path)
    mesh = _upload(client, "chair.glb", minimal_glb()).json()
    assert client.get(f"/compose/assets/{mesh['id']}/orientation").status_code == 400
    assert client.get("/compose/assets/a_missing/orientation").status_code == 404
    assert client.get("/compose/assets/scene__nope/orientation").status_code == 404


# -- cache ----------------------------------------------------------------------
def test_orientation_cache_hit_and_invalidation(tmp_path, monkeypatch):
    client, _ = _setup(tmp_path)
    base = _floor_asset(tmp_path, client)
    calls = []
    real = orientation_mod.estimate_world_orientation

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(orientation_mod, "estimate_world_orientation", counting)
    url = f"/compose/assets/{base['id']}/orientation"

    first = client.get(url).json()
    second = client.get(url).json()
    assert len(calls) == 1 and first == second
    cache_files = list((tmp_path / "compose" / "cache").glob("*.orientation.json"))
    assert len(cache_files) == 1

    ply = tmp_path / "compose" / "assets" / f"{base['id']}.ply"
    st = ply.stat()
    os.utime(ply, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert client.get(url).json()["up"] == pytest.approx(first["up"])
    assert len(calls) == 2

    # A corrupt cache entry is ignored and rewritten, not fatal.
    cache_files[0].write_text("{not json", encoding="utf-8")
    assert client.get(url).status_code == 200
    assert len(calls) == 3


# -- create_scene ---------------------------------------------------------------
def test_create_scene_sets_up_from_floor(tmp_path):
    client, _ = _setup(tmp_path)
    base = _floor_asset(tmp_path, client)
    r = client.post("/compose/scenes", json={"name": "Oda", "base_asset": base["id"]})
    assert r.status_code == 201
    up = np.array(r.json()["up"])
    assert up @ np.array([0, 0, 1]) > 0.9
    assert client.get(f"/compose/scenes/{r.json()['id']}").json()["up"] == r.json()["up"]


def test_create_scene_survives_estimator_failure(tmp_path, monkeypatch):
    client, _ = _setup(tmp_path)
    base = _floor_asset(tmp_path, client)

    def boom(*args, **kwargs):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(orientation_mod, "estimate_world_orientation", boom)
    r = client.post("/compose/scenes", json={"name": "Oda", "base_asset": base["id"]})
    assert r.status_code == 201
    assert r.json()["up"] is None
