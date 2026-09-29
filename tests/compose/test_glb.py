from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from backend.compose.glb import GlbFormatError, placement_matrix, validate_glb, wrap_with_transform
from tests.compose.helpers import minimal_glb


def _chunks(data: bytes):
    _, _, length = struct.unpack_from("<III", data, 0)
    assert length == len(data)
    off, out = 12, []
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        out.append((ctype, data[off + 8: off + 8 + clen]))
        off += 8 + clen
    return out


def test_wrap_adds_root_node_with_matrix(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb())
    m = placement_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 2.0)

    wrap_with_transform(src, tmp_path / "out" / "o.glb", m)

    data = (tmp_path / "out" / "o.glb").read_bytes()
    chunks = _chunks(data)
    gltf = json.loads(chunks[0][1])
    root = gltf["nodes"][-1]
    assert root["name"] == "compose_root"
    assert root["children"] == [0]
    assert gltf["scenes"][0]["nodes"] == [len(gltf["nodes"]) - 1]
    assert root["matrix"] == [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1]
    assert chunks[1][1] == _chunks(minimal_glb())[1][1]
    assert len(chunks[0][1]) % 4 == 0
    validate_glb(tmp_path / "out" / "o.glb")


def test_wrap_without_scenes_uses_parentless_nodes(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb({"asset": {"version": "2.0"},
                                 "nodes": [{"children": [1]}, {}, {}]}))
    wrap_with_transform(src, tmp_path / "o.glb", np.eye(4))
    gltf = json.loads(_chunks((tmp_path / "o.glb").read_bytes())[0][1])
    assert gltf["nodes"][-1]["children"] == [0, 2]
    assert gltf["scene"] == 0


def test_placement_matrix_rotation():
    s = math.sqrt(0.5)
    m = placement_matrix((0.0, 0.0, 0.0), (0.0, 0.0, s, s), 1.0)
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)


@pytest.mark.parametrize("payload", [b"nope" * 10, struct.pack("<III", 0x46546C67, 1, 12)])
def test_validate_rejects_bad_files(tmp_path, payload):
    p = tmp_path / "bad.glb"
    p.write_bytes(payload)
    with pytest.raises(GlbFormatError):
        validate_glb(p)


def _glb_from_chunks(chunks) -> bytes:
    total = 12 + sum(8 + len(b) for _, b in chunks)
    out = struct.pack("<III", 0x46546C67, 2, total)
    for ctype, body in chunks:
        out += struct.pack("<II", len(body), ctype) + body
    return out


def _json_chunk(obj) -> tuple[int, bytes]:
    js = json.dumps(obj).encode("utf-8")
    return 0x4E4F534A, js + b" " * (-len(js) % 4)


def test_wrap_multi_scene_keeps_only_default_scene(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb({"asset": {"version": "2.0"}, "scene": 1,
                                 "scenes": [{"nodes": [0]}, {"nodes": [0]}],
                                 "nodes": [{}, {}]}))
    wrap_with_transform(src, tmp_path / "o.glb", np.eye(4))
    gltf = json.loads(_chunks((tmp_path / "o.glb").read_bytes())[0][1])
    assert len(gltf["scenes"]) == 1
    assert gltf["scene"] == 0
    root_idx = len(gltf["nodes"]) - 1
    assert gltf["scenes"][0]["nodes"] == [root_idx]
    assert gltf["nodes"][root_idx]["name"] == "compose_root"
    assert gltf["nodes"][root_idx]["children"] == [0]


@pytest.mark.parametrize("gltf", [
    [1, 2, 3],
    {"nodes": []},
    {"asset": {}, "nodes": []},
    {"asset": {"version": "1.0"}},
    {"asset": {"version": "2.0"}, "scene": "0", "scenes": [{"nodes": []}]},
    {"asset": {"version": "2.0"}, "scene": 1, "scenes": [{"nodes": []}]},
    {"asset": {"version": "2.0"}, "scene": 0},
    {"asset": {"version": "2.0"}, "scenes": {"a": 1}},
    {"asset": {"version": "2.0"}, "nodes": [1]},
    {"asset": {"version": "2.0"}, "scenes": [{"nodes": [5]}], "nodes": [{}]},
    {"asset": {"version": "2.0"}, "scenes": [{"nodes": ["0"]}], "nodes": [{}]},
    {"asset": {"version": "2.0"}, "nodes": [{"children": [3]}]},
    {"asset": {"version": "2.0"}, "nodes": [{"children": 0}]},
    {"asset": {"version": "2.0"}, "buffers": [{"uri": "external.bin", "byteLength": 4}]},
    {"asset": {"version": "2.0"}, "images": [{"uri": "tex.png"}]},
])
def test_validate_rejects_malformed_gltf(tmp_path, gltf):
    p = tmp_path / "bad.glb"
    p.write_bytes(minimal_glb(gltf))
    with pytest.raises(GlbFormatError):
        validate_glb(p)
    with pytest.raises(GlbFormatError):
        wrap_with_transform(p, tmp_path / "o.glb", np.eye(4))


def test_validate_accepts_data_uri_buffers(tmp_path):
    p = tmp_path / "ok.glb"
    p.write_bytes(minimal_glb({"asset": {"version": "2.0"}, "nodes": [{}],
                               "buffers": [{"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 3}]}))
    validate_glb(p)


def test_placement_matrix_rejects_non_finite():
    with pytest.raises(ValueError):
        placement_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError):
        placement_matrix((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1.0)


def test_wrap_rejects_non_finite_matrix(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb())
    m = np.eye(4)
    m[0, 0] = float("nan")
    with pytest.raises(ValueError):
        wrap_with_transform(src, tmp_path / "o.glb", m)


def test_wrap_without_bin_chunk_roundtrips(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(_glb_from_chunks([_json_chunk({"asset": {"version": "2.0"}, "scene": 0,
                                                   "scenes": [{"nodes": [0]}], "nodes": [{}]})]))
    wrap_with_transform(src, tmp_path / "o.glb", np.eye(4))
    chunks = _chunks((tmp_path / "o.glb").read_bytes())
    assert len(chunks) == 1
    assert json.loads(chunks[0][1])["nodes"][-1]["children"] == [0]


def test_wrap_preserves_unrelated_top_level_keys(tmp_path):
    extra = {
        "animations": [{"channels": [], "samplers": []}],
        "skins": [{"joints": [0]}],
        "extensionsUsed": ["KHR_materials_unlit"],
    }
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb({"asset": {"version": "2.0"}, "scene": 0,
                                 "scenes": [{"nodes": [0]}], "nodes": [{}], **extra}))
    wrap_with_transform(src, tmp_path / "o.glb", np.eye(4))
    gltf = json.loads(_chunks((tmp_path / "o.glb").read_bytes())[0][1])
    for key, value in extra.items():
        assert gltf[key] == value


def test_validate_rejects_truncated_chunk(tmp_path):
    p = tmp_path / "t.glb"
    p.write_bytes(minimal_glb()[:-6])
    with pytest.raises(GlbFormatError):
        validate_glb(p)
    # header claims a JSON chunk longer than the file
    p.write_bytes(struct.pack("<III", 0x46546C67, 2, 40) + struct.pack("<II", 1000, 0x4E4F534A) + b"{}" * 8)
    with pytest.raises(GlbFormatError):
        validate_glb(p)


def test_validate_rejects_non_json_first_chunk(tmp_path):
    p = tmp_path / "b.glb"
    p.write_bytes(_glb_from_chunks([(0x004E4942, b"\x00" * 8)]))
    with pytest.raises(GlbFormatError):
        validate_glb(p)
    p.write_bytes(_glb_from_chunks([(0x4E4F534A, b"not json!")]))
    with pytest.raises(GlbFormatError):
        validate_glb(p)
