from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.compose.plyio import read_ply, write_ply
from backend.compose.routes import build_compose_router
from tests.compose.helpers import minimal_glb, random_cloud


class FakeManager:
    """Runs submitted jobs synchronously and records their results."""

    def __init__(self):
        self.jobs: dict[str, SimpleNamespace] = {}
        self.results: dict[str, dict] = {}

    def create(self, scene: str, smoke_test: bool = False):
        job = SimpleNamespace(id=f"job{len(self.jobs)}", scene=scene)
        self.jobs[job.id] = job
        return job

    def submit(self, job_id, runner):
        self.results[job_id] = runner(lambda *args, **kwargs: None)


def _setup(tmp_path):
    manager = FakeManager()
    app = FastAPI()
    app.include_router(build_compose_router(tmp_path, lambda: manager))
    return TestClient(app), manager


def _ply_bytes(tmp_path, n=20, seed=0, name="x.ply"):
    return write_ply(random_cloud(n, seed=seed), tmp_path / "src" / name).read_bytes()


def _upload(client, filename, data):
    return client.post("/compose/assets", files={"file": (filename, data, "application/octet-stream")})


def test_upload_ply_and_glb_are_listed(tmp_path):
    client, _ = _setup(tmp_path)
    r1 = _upload(client, "statue.ply", _ply_bytes(tmp_path))
    r2 = _upload(client, "chair.glb", minimal_glb())
    assert r1.status_code == 201 and r1.json()["kind"] == "splat" and r1.json()["name"] == "statue"
    assert r2.status_code == 201 and r2.json()["kind"] == "mesh"
    ids = {a["id"] for a in client.get("/compose/assets").json()}
    assert {r1.json()["id"], r2.json()["id"]} <= ids


def test_upload_rejects_wrong_extension_and_corrupt_files(tmp_path):
    client, _ = _setup(tmp_path)
    assert _upload(client, "notes.txt", b"hi").status_code == 400
    bad = _upload(client, "broken.ply", b"ply\nformat ascii 1.0\nend_header\n")
    assert bad.status_code == 400
    leftovers = list((tmp_path / "compose" / "assets").glob("*"))
    assert leftovers == []


def test_upload_filename_cannot_escape_assets_dir(tmp_path):
    client, _ = _setup(tmp_path)
    r = _upload(client, "../../evil.ply", _ply_bytes(tmp_path))
    assert r.status_code == 201
    assert r.json()["name"] == "evil"
    assert not (tmp_path.parent / "evil.ply").exists()
    assert (tmp_path / "compose" / "assets" / f"{r.json()['id']}.ply").is_file()


def test_pipeline_results_are_assets_and_downloadable(tmp_path):
    client, _ = _setup(tmp_path)
    write_ply(random_cloud(5), tmp_path / "garden" / "output" / "ply" / "frame_0000.ply")
    assets = {a["id"]: a for a in client.get("/compose/assets").json()}
    assert assets["scene__garden"]["source"] == "pipeline"
    r = client.get("/compose/assets/scene__garden/file")
    assert r.status_code == 200 and r.content.startswith(b"ply")
    assert client.get("/compose/assets/scene__..%2F/file").status_code == 404
    assert client.get("/compose/assets/nope/file").status_code == 404


def test_scene_create_save_load_roundtrip(tmp_path):
    client, _ = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    doc = client.post("/compose/scenes", json={"name": "Oda", "base_asset": base["id"]}).json()
    assert [o["role"] for o in doc["objects"]] == ["base"]

    statue = _upload(client, "statue.ply", _ply_bytes(tmp_path, seed=1)).json()
    doc["objects"].append({
        "id": "o_statue", "kind": "splat", "asset": statue["id"], "name": "statue",
        "transform": {"position": [1, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 0.5},
        "crop": {"center": [0, 0, 0], "halfSize": [1, 1, 1], "quaternion": [0, 0, 0, 1]},
        "color": {"exposure": 0.5, "tint": [1, 1, 1], "saturation": 1},
    })
    saved = client.put(f"/compose/scenes/{doc['id']}", json=doc)
    assert saved.status_code == 200, saved.text
    loaded = client.get(f"/compose/scenes/{doc['id']}").json()
    assert loaded["objects"][1]["transform"]["scale"] == 0.5
    listed = client.get("/compose/scenes").json()
    assert listed[0]["id"] == doc["id"] and listed[0]["object_count"] == 2


def test_scene_validation_errors(tmp_path):
    client, _ = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    mesh = _upload(client, "chair.glb", minimal_glb()).json()
    assert client.post("/compose/scenes", json={"name": "x", "base_asset": mesh["id"]}).status_code == 400
    assert client.post("/compose/scenes", json={"name": "x", "base_asset": "a_missing"}).status_code == 400
    doc = client.post("/compose/scenes", json={"name": "x", "base_asset": base["id"]}).json()
    url = f"/compose/scenes/{doc['id']}"

    moved = {**doc, "objects": [{**doc["objects"][0], "transform": {"position": [1, 0, 0]}}]}
    r = client.put(url, json=moved)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][0] == "body"
    two_bases = {**doc, "objects": doc["objects"] + [{**doc["objects"][0], "id": "o_b2"}]}
    assert client.put(url, json=two_bases).status_code == 422
    mesh_crop = {**doc, "objects": doc["objects"] + [{
        "id": "o_m", "kind": "mesh", "asset": mesh["id"], "name": "m",
        "crop": {"center": [0, 0, 0], "halfSize": [1, 1, 1]}}]}
    assert client.put(url, json=mesh_crop).status_code == 422
    kind_mismatch = {**doc, "objects": doc["objects"] + [{
        "id": "o_k", "kind": "splat", "asset": mesh["id"], "name": "k"}]}
    assert client.put(url, json=kind_mismatch).status_code == 400
    assert client.put(f"/compose/scenes/s_other", json=doc).status_code == 400
    assert client.get("/compose/scenes/s_nope").status_code == 404
    assert client.get("/compose/scenes/..").status_code == 404


def test_export_bakes_merged_ply_meshes_and_zip(tmp_path):
    client, manager = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path, n=30, seed=1)).json()
    statue = _upload(client, "statue.ply", _ply_bytes(tmp_path, n=40, seed=2)).json()
    chair = _upload(client, "chair.glb", minimal_glb()).json()
    doc = client.post("/compose/scenes", json={"name": "Bahçe", "base_asset": base["id"]}).json()
    doc["objects"] += [
        {"id": "o_statue", "kind": "splat", "asset": statue["id"], "name": "statue",
         "transform": {"position": [5, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 1},
         "crop": {"center": [0, 0, 0], "halfSize": [0.5, 0.5, 0.5], "quaternion": [0, 0, 0, 1]}},
        {"id": "o_hidden", "kind": "splat", "asset": statue["id"], "name": "hidden", "visible": False},
        {"id": "o_chair", "kind": "mesh", "asset": chair["id"], "name": "chair",
         "transform": {"position": [0, 1, 0], "quaternion": [0, 0, 0, 1], "scale": 1}},
    ]
    assert client.put(f"/compose/scenes/{doc['id']}", json=doc).status_code == 200

    r = client.post(f"/compose/scenes/{doc['id']}/export")

    assert r.status_code == 202
    result = manager.results[r.json()["job_id"]]
    assert result["download_url"] == f"/compose/exports/{doc['id']}/download"
    assert manager.jobs[r.json()["job_id"]].scene == f"compose-{doc['id']}"
    statue_src = read_ply(tmp_path / "compose" / "assets" / f"{statue['id']}.ply")
    inside = int(np.all(np.abs(statue_src.means) <= 0.5, axis=1).sum())
    merged = read_ply(tmp_path / "compose" / "exports" / doc["id"] / "merged.ply")
    assert merged.count == 30 + inside == result["gaussians"]

    z = client.get(result["download_url"])
    assert z.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(z.content)).namelist())
    assert names == {"merged.ply", "meshes/o_chair.glb", "scene.json"}


def test_export_rejects_missing_assets(tmp_path):
    client, manager = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    doc = client.post("/compose/scenes", json={"name": "x", "base_asset": base["id"]}).json()
    (tmp_path / "compose" / "assets" / f"{base['id']}.ply").unlink()
    r = client.post(f"/compose/scenes/{doc['id']}/export")
    assert r.status_code == 400 and base["id"] in r.json()["detail"]
    assert manager.jobs == {}
    assert client.get(f"/compose/exports/{doc['id']}/download").status_code == 404


# -- hardening ------------------------------------------------------------------

def _scene_with_statue(client, tmp_path, name="Bahce", seed=2):
    base = _upload(client, "room.ply", _ply_bytes(tmp_path, n=30, seed=1)).json()
    statue = _upload(client, "statue.ply", _ply_bytes(tmp_path, n=40, seed=seed)).json()
    doc = client.post("/compose/scenes", json={"name": name, "base_asset": base["id"]}).json()
    doc["objects"].append({
        "id": "o_statue", "kind": "splat", "asset": statue["id"], "name": "statue",
        "transform": {"position": [5, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 1}})
    assert client.put(f"/compose/scenes/{doc['id']}", json=doc).status_code == 200
    return doc


@pytest.mark.parametrize("field,token", [
    ("quaternion", '[NaN, 0, 0, 1]'),
    ("position", '[Infinity, 0, 0]'),
    ("scale", 'Infinity'),
    ("halfSize", '[NaN, 1, 1]'),
    ("exposure", 'NaN'),
])
def test_non_finite_numbers_are_rejected(tmp_path, field, token):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    statue = doc["objects"][1]
    if field == "halfSize":
        statue["crop"] = {"halfSize": "__X__"}
    elif field == "exposure":
        statue["color"] = {"exposure": "__X__"}
    else:
        statue["transform"][field] = "__X__"
    body = json.dumps(doc).replace('"__X__"', token)
    r = client.put(f"/compose/scenes/{doc['id']}", content=body,
                   headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text


def test_unreadable_scene_file_is_409_not_500(tmp_path):
    client, manager = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    (tmp_path / "compose" / "scenes" / f"{doc['id']}.json").write_text("{garbage", encoding="utf-8")
    r = client.get(f"/compose/scenes/{doc['id']}")
    assert r.status_code == 409 and "unreadable" in r.json()["detail"]
    assert client.post(f"/compose/scenes/{doc['id']}/export").status_code == 409
    assert client.get(f"/compose/exports/{doc['id']}/download").status_code == 409
    assert manager.jobs == {}
    assert client.get("/compose/scenes").json() == []


def test_put_unknown_scene_is_404(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    ghost = {**doc, "id": "s_ghost"}
    assert client.put("/compose/scenes/s_ghost", json=ghost).status_code == 404
    assert not (tmp_path / "compose" / "scenes" / "s_ghost.json").exists()


def test_unicode_pipeline_scene_is_usable_as_asset_and_base(tmp_path):
    client, _ = _setup(tmp_path)
    write_ply(random_cloud(5), tmp_path / "bahçe" / "output" / "ply" / "frame_0000.ply")
    asset_id = "scene__bahçe"
    assert asset_id in {a["id"] for a in client.get("/compose/assets").json()}
    r = client.get(f"/compose/assets/{quote(asset_id)}/file")
    assert r.status_code == 200 and r.content.startswith(b"ply")
    doc = client.post("/compose/scenes", json={"name": "b", "base_asset": asset_id})
    assert doc.status_code == 201, doc.text
    assert doc.json()["objects"][0]["asset"] == asset_id


def _export(client, doc_id):
    r = client.post(f"/compose/scenes/{doc_id}/export")
    assert r.status_code == 202, r.text
    return r


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_failed_export_keeps_previous_export_and_leaves_no_temp(tmp_path, monkeypatch):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    _export(client, doc["id"])
    exports = tmp_path / "compose" / "exports"
    zip_before = (exports / f"{doc['id']}.zip").read_bytes()
    tree_before = _tree(exports / doc["id"])

    doc["objects"][1]["transform"]["position"] = [9, 9, 9]
    assert client.put(f"/compose/scenes/{doc['id']}", json=doc).status_code == 200

    def boom(path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("backend.compose.exporter.zipfile.ZipFile", boom)
        with pytest.raises(OSError, match="disk full"):
            client.post(f"/compose/scenes/{doc['id']}/export")

    assert (exports / f"{doc['id']}.zip").read_bytes() == zip_before
    assert _tree(exports / doc["id"]) == tree_before
    assert sorted(p.name for p in exports.iterdir()) == sorted([doc["id"], f"{doc['id']}.zip"])


def test_export_can_be_repeated_and_replaces_previous(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    _export(client, doc["id"])
    exports = tmp_path / "compose" / "exports"
    first = (exports / doc["id"] / "merged.ply").read_bytes()
    doc["objects"][1]["transform"]["position"] = [9, 9, 9]
    client.put(f"/compose/scenes/{doc['id']}", json=doc)
    _export(client, doc["id"])
    assert (exports / doc["id"] / "merged.ply").read_bytes() != first
    assert sorted(p.name for p in exports.iterdir()) == sorted([doc["id"], f"{doc['id']}.zip"])
    z = zipfile.ZipFile(exports / f"{doc['id']}.zip")
    assert z.read("merged.ply") == (exports / doc["id"] / "merged.ply").read_bytes()
    z.close()


def test_export_cleans_stale_leftovers_from_killed_runs(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    sid = doc["id"]
    exports = tmp_path / "compose" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / f"{sid}.aaaaaaaa.tmp").mkdir()
    (exports / f"{sid}.aaaaaaaa.tmp" / "merged.ply").write_bytes(b"x")
    (exports / f"{sid}.bbbbbbbb.zip.tmp").write_bytes(b"x")
    (exports / f"{sid}.cccccccc.old").mkdir()
    other = exports / "s_other.dddddddd.tmp"  # another scene's leftovers are not ours to touch
    other.mkdir()
    _export(client, sid)
    assert sorted(p.name for p in exports.iterdir()) == sorted([sid, f"{sid}.zip", other.name])


def test_export_restores_newest_backup_when_export_dir_is_missing(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    sid = doc["id"]
    exports = tmp_path / "compose" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    old, older = exports / f"{sid}.11111111.old", exports / f"{sid}.22222222.old"
    for d, content in ((older, b"older"), (old, b"newer")):
        d.mkdir()
        (d / "marker.txt").write_bytes(content)
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(old, (2_000_000, 2_000_000))
    from backend.compose.exporter import _clean_stale_exports
    from backend.compose.store import ComposeStore
    _clean_stale_exports(ComposeStore(tmp_path), sid)
    assert (exports / sid / "marker.txt").read_bytes() == b"newer"
    assert sorted(p.name for p in exports.iterdir()) == [sid]
    _export(client, sid)
    assert (exports / sid / "merged.ply").is_file()
    assert not (exports / sid / "marker.txt").exists()


def test_export_honours_cancellation(tmp_path):
    class CancelManager(FakeManager):
        def cancel_requested(self, job_id):
            return True

    manager = CancelManager()
    app = FastAPI()
    app.include_router(build_compose_router(tmp_path, lambda: manager))
    client = TestClient(app)
    doc = _scene_with_statue(client, tmp_path)
    with pytest.raises(RuntimeError, match="Export cancelled"):
        client.post(f"/compose/scenes/{doc['id']}/export")
    assert list((tmp_path / "compose" / "exports").iterdir()) == []


def test_atomic_write_leaves_no_temp_files(tmp_path):
    client, _ = _setup(tmp_path)
    _scene_with_statue(client, tmp_path)
    for d in ("assets", "scenes"):
        assert [p for p in (tmp_path / "compose" / d).iterdir() if p.suffix in (".tmp", ".part")] == []


def test_put_with_malformed_json_body_is_422(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    r = client.put(f"/compose/scenes/{doc['id']}", content="{not json",
                   headers={"content-type": "application/json"})
    assert r.status_code == 422
