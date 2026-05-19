import pytest
from pathlib import Path
from backend.image_to_scene.disk_layout import (
    parse_indexed_name, next_index, slug_safe, IndexedName,
)
from backend.image_to_scene.disk_layout import (
    WorldEnvelope, ensure_envelope, artifact_path, request_path,
)


def test_parse_visible_name():
    p = parse_indexed_name("0-world.ply")
    assert p == IndexedName(index=0, slug="world", extension=".ply", hidden=False)


def test_parse_hidden_request():
    p = parse_indexed_name(".3-world-request.json")
    assert p == IndexedName(index=3, slug="world-request", extension=".json", hidden=True)


def test_parse_non_indexed_returns_none():
    assert parse_indexed_name("README.md") is None
    assert parse_indexed_name("abc-world.png") is None  # index not numeric


def test_next_index_empty_dir(tmp_path):
    assert next_index(tmp_path, slug="world") == 0


def test_next_index_finds_highest(tmp_path):
    (tmp_path / "0-world.json").touch()
    (tmp_path / "2-world.json").touch()
    (tmp_path / ".2-world-request.json").touch()
    assert next_index(tmp_path, slug="world") == 3


def test_next_index_isolates_by_slug(tmp_path):
    (tmp_path / "5-other.json").touch()
    assert next_index(tmp_path, slug="world") == 0


def test_slug_safe_lowercases_and_hyphenates():
    assert slug_safe("My Test Scene!") == "my-test-scene"


def test_slug_safe_collapses_dashes():
    assert slug_safe("hello---world") == "hello-world"


def test_slug_safe_rejects_empty():
    with pytest.raises(ValueError):
        slug_safe("")
    with pytest.raises(ValueError):
        slug_safe("!!!")


def test_ensure_envelope_creates_dirs(tmp_worlds_dir):
    env = ensure_envelope(tmp_worlds_dir, "test-scene")
    assert env.root.exists()
    assert env.source.exists()
    assert env.output_world.exists()
    assert env.project_json.exists()
    import json
    data = json.loads(env.project_json.read_text(encoding="utf-8"))
    assert data["slug"] == "test-scene"


def test_ensure_envelope_idempotent(tmp_worlds_dir):
    env1 = ensure_envelope(tmp_worlds_dir, "test-scene")
    env2 = ensure_envelope(tmp_worlds_dir, "test-scene")
    assert env1.root == env2.root


def test_artifact_path(tmp_worlds_dir):
    env = ensure_envelope(tmp_worlds_dir, "scn")
    p = artifact_path(env.output_world, index=0, slug="world", extension=".ply")
    assert p == env.output_world / "0-world.ply"


def test_request_path(tmp_worlds_dir):
    env = ensure_envelope(tmp_worlds_dir, "scn")
    p = request_path(env.output_world, index=0, slug="world")
    assert p == env.output_world / ".0-world-request.json"
