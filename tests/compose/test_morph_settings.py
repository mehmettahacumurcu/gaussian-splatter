from __future__ import annotations

import io
import json
import zipfile

import pytest
from pydantic import ValidationError

from backend.compose.models import SceneDoc
from tests.compose.test_routes import _scene_with_statue, _setup


def _recipe(source_id="o_base", target_id="o_statue"):
    return {
        "sourceId": source_id, "targetId": target_id, "mode": "shape",
        "duration": 9.5, "dissolve": 0.35, "wave": 0.4, "arc": 0.2,
        "targetBlend": 0.85, "seed": 4294967295, "autoAlign": True,
    }


def _document(**changes):
    return {
        "version": 1, "id": "s_morph", "name": "Morph", "objects": [
            {"id": "o_base", "kind": "splat", "asset": "a_base", "name": "Base", "role": "base"},
            {"id": "o_statue", "kind": "splat", "asset": "a_statue", "name": "Statue"},
            {"id": "o_mesh", "kind": "mesh", "asset": "a_mesh", "name": "Mesh"},
        ], **changes,
    }


def test_legacy_v1_and_null_morph_remain_valid():
    assert SceneDoc.model_validate(_document()).morph is None
    assert SceneDoc.model_validate(_document(morph=None)).morph is None
    # Older optional recipes can omit new controls and get explicit defaults.
    minimal = SceneDoc.model_validate(_document(morph={"sourceId": "o_base"})).morph
    assert minimal.targetId is None
    assert minimal.mode == "shape" and minimal.duration == 6 and minimal.seed == 42
    assert minimal.autoAlign is False
    with pytest.raises(ValidationError):
        SceneDoc.model_validate(_document(version=2))


@pytest.mark.parametrize("field,value", [
    ("mode", "unknown"), ("duration", 0.49), ("duration", 120.1),
    ("dissolve", -0.1), ("wave", 1.01), ("arc", -1), ("targetBlend", 1.01),
    ("seed", -1), ("seed", 4294967296), ("seed", 1.5), ("seed", True),
    ("autoAlign", "true"), ("sourceId", "../base"), ("sourceId", "o_missing"),
    ("targetId", "o_mesh"), ("targetId", "o_base"),
    ("playing", True), ("enabled", True), ("t", 0.5),
])
def test_morph_settings_validation(field, value):
    with pytest.raises(ValidationError):
        SceneDoc.model_validate(_document(morph={**_recipe(), field: value}))


@pytest.mark.parametrize("field", ["duration", "dissolve", "wave", "arc", "targetBlend", "seed"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_morph_numbers_must_be_finite(field, value):
    with pytest.raises(ValidationError):
        SceneDoc.model_validate(_document(morph={**_recipe(), field: value}))


@pytest.mark.parametrize("mode", ["shape", "cloud"])
def test_morph_save_open_roundtrip_and_clear_after_delete(tmp_path, mode):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    url = f"/compose/scenes/{doc['id']}"
    recipe = {**_recipe(doc["objects"][0]["id"]), "mode": mode}
    doc["morph"] = recipe
    saved = client.put(url, json=doc)
    assert saved.status_code == 200, saved.text
    assert saved.json()["morph"] == recipe
    assert client.get(url).json()["morph"] == recipe
    disk = tmp_path / "compose" / "scenes" / f"{doc['id']}.json"
    assert json.loads(disk.read_text(encoding="utf-8"))["morph"] == recipe

    # The composer removes the whole recipe when either referenced object goes.
    doc["objects"] = doc["objects"][:1]
    doc["morph"] = None
    assert client.put(url, json=doc).status_code == 200
    assert client.get(url).json()["morph"] is None


def test_morph_partial_pair_and_legacy_scene_open(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    url = f"/compose/scenes/{doc['id']}"
    doc["morph"] = _recipe(doc["objects"][0]["id"], None)
    assert client.put(url, json=doc).status_code == 200
    assert client.get(url).json()["morph"] == doc["morph"]

    doc.pop("morph")
    disk = tmp_path / "compose" / "scenes" / f"{doc['id']}.json"
    disk.write_text(json.dumps(doc), encoding="utf-8")
    opened = client.get(url)
    assert opened.status_code == 200
    assert opened.json()["morph"] is None and opened.json()["version"] == 1


def test_morph_invalid_api_payload_is_422_and_preserves_saved_scene(tmp_path):
    client, _ = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    url = f"/compose/scenes/{doc['id']}"
    before = client.get(url).json()
    for override in ({"duration": float("nan")}, {"wave": float("inf")}, {"targetId": "o_missing"}):
        doc["morph"] = {**_recipe(doc["objects"][0]["id"]), **override}
        response = client.put(url, content=json.dumps(doc), headers={"content-type": "application/json"})
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"][0] == "body"
        assert client.get(url).json() == before


def test_export_carries_morph_recipe_but_static_asset_bytes_do_not_change(tmp_path):
    client, manager = _setup(tmp_path)
    doc = _scene_with_statue(client, tmp_path)
    url = f"/compose/scenes/{doc['id']}"

    def export_files():
        response = client.post(f"{url}/export")
        assert response.status_code == 202
        result = manager.results[response.json()["job_id"]]
        with zipfile.ZipFile(io.BytesIO(client.get(result["download_url"]).content)) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    before = export_files()
    doc["morph"] = _recipe(doc["objects"][0]["id"])
    assert client.put(url, json=doc).status_code == 200
    after = export_files()
    assert after["merged.ply"] == before["merged.ply"]
    assert json.loads(after["scene.json"])["morph"] == doc["morph"]
    assert json.loads(after["scene.json"])["version"] == 1
