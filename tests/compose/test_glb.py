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
