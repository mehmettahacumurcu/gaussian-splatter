from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.coverage import routes as coverage_routes
from backend.coverage.routes import build_coverage_router
from tests.coverage.helpers import look_at_w2c, plane_cloud, write_model, write_splat


class FakeManager:
    """Runs submitted jobs synchronously."""

    def __init__(self):
        self.results = {}

    def create(self, scene: str, smoke_test: bool = False):
        return SimpleNamespace(id=f"job{len(self.results)}", scene=scene)

    def submit(self, job_id, runner):
        self.results[job_id] = runner(lambda *a, **k: None)


def fake_run(store, result_id, name, ply, sparse, on_progress, cancelled):
    """Writes what run_coverage writes, without the GPU."""
    tmp = store.tmp_dir(result_id)
    write_splat(tmp / "heat.ply", plane_cloud(3))
    views = [{"index": 0, "name": "train_0001.png", "test": False, "c2w": list(np.eye(4).ravel()),
              "fov_y": 50.0, "aspect": 1.5, "score": 0.9, "weak": 0.1}]
    (tmp / "views.json").write_text(json.dumps(views), encoding="utf-8")
    images = sparse.parent.parent / "images"
    meta = {"id": result_id, "name": name, "created_at": time.time(), "ply_path": str(ply),
            "sparse_dir": str(sparse), "images_dir": str(images) if images.is_dir() else None,
            "cameras": 1, "test_views": 0, "up": [0, 0, 1], "summary": {"weak_share": 0.1}}
    (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    store.publish(result_id, tmp)
    return {"coverage_id": result_id}


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage_routes, "run_coverage", fake_run)
    manager = FakeManager()
    app = FastAPI()
    app.include_router(build_coverage_router(tmp_path / "data", lambda: manager))
    return TestClient(app), manager


def _run_folder(root):
    write_splat(root / "splat.ply", plane_cloud(10))
    write_model(root / "sparse" / "0", ["train_0001.png"], [look_at_w2c([0, 0, 2], [0, 0, 0])])
    (root / "images").mkdir()
    (root / "images" / "train_0001.png").write_bytes(b"\x89PNG fake")
    return root


def test_start_list_get_files_and_delete(tmp_path, monkeypatch):
    client, manager = _client(tmp_path, monkeypatch)
    run = _run_folder(tmp_path / "ROOM2")

    res = client.post("/coverage", json={"path": str(run)})
    assert res.status_code == 202, res.text
    body = res.json()
    cid = body["coverage_id"]
    assert cid.startswith("room2-") and body["ply_path"].endswith("splat.ply")
    assert manager.results[body["job_id"]] == {"coverage_id": cid}

    assert [m["id"] for m in client.get("/coverage").json()] == [cid]
    detail = client.get(f"/coverage/{cid}").json()
    assert detail["name"] == "ROOM2" and detail["views"][0]["name"] == "train_0001.png"

    normal = client.get(f"/coverage/{cid}/splat/normal")
    assert normal.status_code == 200 and normal.content == (run / "splat.ply").read_bytes()
    assert client.get(f"/coverage/{cid}/splat/heat").status_code == 200
    assert client.get(f"/coverage/{cid}/splat/other").status_code == 404
    assert client.get(f"/coverage/{cid}/photo/0").content == b"\x89PNG fake"
    assert client.get(f"/coverage/{cid}/photo/5").status_code == 404

    assert client.delete(f"/coverage/{cid}").json() == {"ok": True}
    assert client.get(f"/coverage/{cid}").status_code == 404
    assert client.get("/coverage").json() == []


def test_bad_input_is_a_400_with_the_reason(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    res = client.post("/coverage", json={"path": str(tmp_path / "missing")})
    assert res.status_code == 400 and "bulunamadı" in res.json()["detail"]


def test_unknown_or_malicious_ids_are_404(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    assert client.get("/coverage/nope-123").status_code == 404
    assert client.get("/coverage/..%2F..%2Fetc").status_code == 404
    assert client.delete("/coverage/nope-123").status_code == 404
