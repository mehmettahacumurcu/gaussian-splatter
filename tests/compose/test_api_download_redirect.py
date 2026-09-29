from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.job_manager as job_manager_module
from backend.api import app
from backend.job_manager import JobManager


@pytest.fixture()
def client_and_manager(tmp_path, monkeypatch):
    manager = JobManager(data_dir=tmp_path)
    monkeypatch.setattr(job_manager_module, "_default_manager", manager)
    with TestClient(app) as client:
        yield client, manager


def test_compose_job_download_redirects_and_keeps_query(client_and_manager):
    client, manager = client_and_manager
    job = manager.create(scene="compose-s_x")
    manager._mark_completed(job.id, result={"download_url": "/compose/exports/s_x/download"})

    r = client.get(f"/download/{job.id}?token=abc", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/compose/exports/s_x/download?token=abc"

    r = client.get(f"/download/{job.id}", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/compose/exports/s_x/download"


def test_download_still_404_for_unknown_and_409_for_unfinished_jobs(client_and_manager):
    client, manager = client_and_manager
    assert client.get("/download/nope", follow_redirects=False).status_code == 404
    job = manager.create(scene="compose-s_x")
    assert client.get(f"/download/{job.id}", follow_redirects=False).status_code == 409
