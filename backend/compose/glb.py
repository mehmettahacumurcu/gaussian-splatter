"""Minimal GLB (binary glTF 2.0) helpers: validate, and wrap a scene in a transform node."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from .bake import quat_to_matrix

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


class GlbFormatError(ValueError):
    """The file is not a usable binary glTF 2.0 file."""


def _check_header(magic: int, version: int, length: int, size: int) -> None:
    if magic != GLB_MAGIC:
        raise GlbFormatError("Not a GLB file (bad magic)")
    if version != 2:
        raise GlbFormatError(f"Unsupported glTF version {version}")
    if length > size:
        raise GlbFormatError("GLB file is truncated")


def _read_chunks(data: bytes) -> list[tuple[int, memoryview]]:
    if len(data) < 20:
        raise GlbFormatError("File too small to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    _check_header(magic, version, length, len(data))
    view = memoryview(data)
    chunks: list[tuple[int, memoryview]] = []
    off = 12
    while off + 8 <= length:
        clen, ctype = struct.unpack_from("<II", data, off)
        body = view[off + 8: off + 8 + clen]
        if len(body) != clen or off + 8 + clen > length:
            raise GlbFormatError("GLB chunk is truncated")
        chunks.append((ctype, body))
        off += 8 + clen
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise GlbFormatError("GLB must start with a JSON chunk")
    return chunks


def _read_json_chunk(path: Path) -> bytes:
    """Read only the header and the first (JSON) chunk of a GLB file."""
    size = os.path.getsize(path)
    if size < 20:
        raise GlbFormatError("File too small to be a GLB")
    with open(path, "rb") as f:
        head = f.read(20)
        magic, version, length = struct.unpack_from("<III", head, 0)
        _check_header(magic, version, length, size)
        clen, ctype = struct.unpack_from("<II", head, 12)
        if ctype != CHUNK_JSON:
            raise GlbFormatError("GLB must start with a JSON chunk")
        if 20 + clen > length:
            raise GlbFormatError("GLB chunk is truncated")
        body = f.read(clen)
    if len(body) != clen:
        raise GlbFormatError("GLB chunk is truncated")
    return body


def _is_index_list(value, limit: int) -> bool:
    return isinstance(value, list) and all(
        isinstance(i, int) and not isinstance(i, bool) and 0 <= i < limit for i in value
    )


def _check_gltf(gltf) -> dict:
    """Structural checks that later code (and the exporter) relies on."""
    if not isinstance(gltf, dict):
        raise GlbFormatError("glTF JSON must be an object")
    asset = gltf.get("asset")
    version = asset.get("version") if isinstance(asset, dict) else None
    if not isinstance(version, str) or not version.startswith("2."):
        raise GlbFormatError("Only glTF 2.x is supported (missing or invalid asset.version)")
    for key in ("scenes", "nodes", "buffers", "images"):
        if key in gltf and not (isinstance(gltf[key], list) and all(isinstance(x, dict) for x in gltf[key])):
            raise GlbFormatError(f"'{key}' must be a list of objects")
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    if "scene" in gltf:
        scene = gltf["scene"]
        if isinstance(scene, bool) or not isinstance(scene, int) or not 0 <= scene < len(scenes):
            raise GlbFormatError("'scene' must be an index into 'scenes'")
    for scene in scenes:
        if "nodes" in scene and not _is_index_list(scene["nodes"], len(nodes)):
            raise GlbFormatError("Scene 'nodes' must be a list of valid node indices")
    for node in nodes:
        if "children" in node and not _is_index_list(node["children"], len(nodes)):
            raise GlbFormatError("Node 'children' must be a list of valid node indices")
    for key in ("buffers", "images"):
        for item in gltf.get(key, []):
            uri = item.get("uri")
            if uri is None:
                continue
            if not isinstance(uri, str) or not uri.startswith("data:"):
                raise GlbFormatError(
                    "External resources are not supported; export as a self-contained .glb")
    return gltf


def _parse_json(chunk) -> dict:
    try:
        return _check_gltf(json.loads(bytes(chunk).decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbFormatError(f"Invalid GLB JSON chunk: {exc}") from exc


def validate_glb(path: str | Path) -> dict:
    """Return the parsed glTF JSON of a well-formed, self-contained glTF 2.x GLB.

    Guarantees: valid header/JSON chunk within the file size, an object with ``asset.version`` 2.x,
    well-typed ``scene``/``scenes``/``nodes`` with in-range indices, and no external buffer/image URIs.
    Reads only the header and the JSON chunk, never the binary payload.
    """
    return _parse_json(_read_json_chunk(Path(path)))


def placement_matrix(position, quaternion_xyzw, scale: float) -> np.ndarray:
    m = np.eye(4)
    with np.errstate(all="ignore"):
        m[:3, :3] = float(scale) * quat_to_matrix(tuple(quaternion_xyzw))
        m[:3, 3] = position
    if not np.all(np.isfinite(m)):
        raise ValueError("Placement produces a non-finite transform matrix")
    return m


def wrap_with_transform(src: str | Path, dst: str | Path, matrix: np.ndarray) -> Path:
    """Copy ``src`` to ``dst`` with every scene root re-parented under one node carrying ``matrix``.

    Only the default scene is kept in the output.
    """
    chunks = _read_chunks(Path(src).read_bytes())
    gltf = _parse_json(chunks[0][1])
    nodes = gltf.setdefault("nodes", [])
    scenes = gltf.get("scenes")
    scene_idx = gltf.get("scene", 0)
    if scenes and 0 <= scene_idx < len(scenes):
        default_scene = scenes[scene_idx]
        roots = list(default_scene.get("nodes", []))
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
        default_scene = {"nodes": []}
    wrapper: dict = {
        "name": "compose_root",
        "matrix": [float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)],  # column-major
    }
    if roots:
        wrapper["children"] = roots
    nodes.append(wrapper)
    default_scene["nodes"] = [len(nodes) - 1]
    gltf["scenes"] = [default_scene]
    gltf["scene"] = 0

    try:
        json_bytes = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise ValueError(f"Cannot write non-finite numbers into glTF JSON: {exc}") from exc
    json_bytes += b" " * (-len(json_bytes) % 4)
    body = [(CHUNK_JSON, json_bytes)] + chunks[1:]
    total = 12 + sum(8 + len(b) for _, b in body)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    for ctype, b in body:
        out += struct.pack("<II", len(b), ctype)
        out += b
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    return dst
