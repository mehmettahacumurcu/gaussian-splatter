from __future__ import annotations

import nbformat
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.notebooks.routes import static_notebook_router
from backend.notebooks.source import NotebookSource


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        "backend.notebooks.routes.resolve_notebook_source",
        lambda: NotebookSource(
            repo_url="https://github.com/mehmettahacumurcu/gaussian-splatter.git",
            commit_sha="a" * 40,
            generator_id="4dgs-studio.static-notebook",
            generator_version=1,
        ),
    )
    app = FastAPI()
    app.include_router(static_notebook_router)
    return TestClient(app)


def test_presets_route_is_backend_owned(monkeypatch) -> None:
    response = _client(monkeypatch).get("/notebooks/static/presets")
    assert response.status_code == 200
    assert response.json()["default_profile"] == "balanced_l4"
    assert [profile["id"] for profile in response.json()["profiles"]] == [
        "balanced_l4",
        "high",
        "premium",
        "ultra",
    ]


def test_generate_route_returns_attachment(monkeypatch) -> None:
    response = _client(monkeypatch).post(
        "/notebooks/static",
        json={"schema_version": 1, "input_folder": "captures/room"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ipynb+json")
    assert response.headers["content-disposition"] == (
        'attachment; filename="room_static_splat.ipynb"'
    )
    notebook = nbformat.reads(response.content.decode("utf-8"), as_version=4)
    assert notebook.nbformat == 4


def test_generate_route_rejects_invalid_body(monkeypatch) -> None:
    response = _client(monkeypatch).post(
        "/notebooks/static",
        json={"schema_version": 1, "input_folder": "../escape"},
    )
    assert response.status_code == 422


def test_pipeline_catalog_and_download(monkeypatch):
    client = _client(monkeypatch)
    catalog = client.get('/notebooks/static/pipelines').json()
    assert catalog['presets']['spirula']['quality']['max_gaussians'] == 6000000
    response = client.post('/notebooks/static/pipeline', json={
        'pipeline': 'spirula', 'input_mode': 'video', 'input_path': 'captures/room.MOV',
        'preset': 'quality', 'iterations': 65000})
    assert response.status_code == 200
    assert 'room_spirula_quality.ipynb' in response.headers['content-disposition']
    notebook = nbformat.reads(response.text, 4)
    assert notebook.metadata.pipeline.spec.iterations == 65000
    response = client.post('/notebooks/static/pipeline', json={
        'pipeline': 'hybrid', 'input_mode': 'video', 'input_path': 'captures/room.MOV'})
    assert response.status_code == 422


def test_preprocessing_download_and_preset_validation(monkeypatch):
    client = _client(monkeypatch)
    catalog = client.get('/notebooks/static/preprocess').json()
    assert catalog['fields']['geometry_model']['when'] == 'generate_geometry'
    payload = {'inputs': [{'kind': 'photos', 'path': 'MyDrive/captures/room'}],
               'settings': {'quality': 'extreme', 'generate_geometry': False}}
    validated = client.post('/notebooks/static/preprocess/validate', json=payload)
    assert validated.status_code == 200
    assert validated.json()['inputs'][0]['path'] == 'captures/room'
    response = client.post('/notebooks/static/preprocess', json=payload)
    assert response.status_code == 200
    notebook = nbformat.reads(response.text, 4)
    assert notebook.metadata.pipeline.id == 'spirula_preprocess'
    assert 'preprocessing.ipynb' in response.headers['content-disposition']
    assert client.post('/notebooks/static/preprocess', json={**payload, 'training_steps': 30000}).status_code == 422
